"""Section 4.3 shape fixtures -- production-shape, one-layer packed host buffers.

Gives the P3/P4/P8 Op tests real NVFP4 payloads at the PRODUCTION expert
geometry (512 experts, hidden 2560, inter 640, fused expert-major matrices)
without an Engine, the real artifact, or a GPU:

* allocation-free geometry queries (``block_scale_geometry`` / ``encoded_size``
  / per-expert plane offsets and alignment proofs);
* ``write_packed_experts`` streams the source expert-by-expert (deterministic
  per-expert seeds) and writes ONE valid (N, K) NVFP4 payload -- codes plane
  || scales plane || 4-byte divisor -- to a host file.

Bit-exactness contract: ``quantize_nvfp4`` is per-row / per-16-group given a
single global divisor (its own row-chunked path relies on this), so pass 1
collects the global ``max(|W|)`` over all experts, and pass 2 re-quantizes each
expert with that explicit divisor, appending its codes/scales rows to the
single payload planes. The result is byte-identical to quantizing and
encoding the full (N, K) matrix in one shot. Peak host memory is the output
file plus one expert chunk (tens of MB), never the f32 source matrix.

Never 48 layers x 512 experts in a unit test: this helper builds ONE layer's
expert block; the Op tests take what they need from it (``§4.3``).
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from tools.artifact.layouts import block_scale_geometry
from tools.convert.qwen3_8_flash_next.quantize_nvfp4 import (
    weight_scale_divisor_word,
    quantize_nvfp4,
)

NVFP4 = "NVFP4"
_ROLES = ("gate_up", "down")


def _seed_for(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


@dataclass(frozen=True, slots=True)
class ExpertGeometry:
    """One layer's fused expert block at a given scale."""

    experts: int
    hidden: int
    inter: int
    scale_rows: int = 128  # blockscale N-tile (layout m128x4)

    # --- shapes ---------------------------------------------------------------

    def role_shape(self, role: str) -> tuple[int, int]:
        if role == "gate_up":
            return (self.experts * 2 * self.inter, self.hidden)
        if role == "down":
            return (self.experts * self.hidden, self.inter)
        raise ValueError(f"role must be one of {_ROLES}, got {role!r}")

    @property
    def gate_up_shape(self) -> tuple[int, int]:
        return self.role_shape("gate_up")

    @property
    def down_shape(self) -> tuple[int, int]:
        return self.role_shape("down")

    def rows_per_expert(self, role: str) -> int:
        return self.role_shape(role)[0] // self.experts

    def validate(self) -> None:
        for role in _ROLES:
            n, k = self.role_shape(role)
            if n % 128 != 0 or k % 64 != 0:
                raise ValueError(f"{role}: NVFP4 geometry violated, shape {(n, k)}")
            rpe = self.rows_per_expert(role)
            if rpe % self.scale_rows != 0:
                raise ValueError(
                    f"{role}: per-expert stride {rpe} rows is not a multiple of "
                    f"the {self.scale_rows}-row blockscale N-tile; (layer, expert_id) "
                    "slices would not be layout-aligned"
                )

    # --- allocation-free geometry queries --------------------------------------

    def geometry(self, role: str):
        return block_scale_geometry(NVFP4, self.role_shape(role))

    def expert_offsets(self, role: str, expert: int) -> dict[str, int]:
        """Plane offsets/bytes of one expert's rows inside the fused payload.

        Expert-major: expert ``e`` owns rows ``[e*rpe, (e+1)*rpe)`` of BOTH the
        codes plane and the scales plane; ``rpe`` is a multiple of the 128-row
        N-tile, so both are scale-tile aligned.
        """

        if not 0 <= expert < self.experts:
            raise ValueError(f"expert {expert} out of range [0, {self.experts})")
        n, k = self.role_shape(role)
        geo = self.geometry(role)
        rpe = self.rows_per_expert(role)
        return {
            "code_offset": expert * rpe * (k // 2),
            "code_bytes": rpe * (k // 2),
            "scale_offset": geo.scale_plane_offset + expert * rpe * (k // 16),
            "scale_bytes": rpe * (k // 16),
        }


PRODUCTION_EXPERTS = ExpertGeometry(experts=512, hidden=2560, inter=640)
PRODUCTION_EXPERTS.validate()


def production_expert_geometry() -> ExpertGeometry:
    """The locked production geometry (P11/P12 reference): 512 x top-10 x 640."""

    return PRODUCTION_EXPERTS


# --- deterministic source + streaming writer ---------------------------------


def _expert_source(
    geometry: ExpertGeometry, role: str, expert: int, seed_name: str
) -> torch.Tensor:
    k = geometry.role_shape(role)[1]
    rpe = geometry.rows_per_expert(role)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_seed_for(f"{seed_name}:{role}:{expert}"))
    return (torch.randn((rpe, k), dtype=torch.float32, generator=gen) * 0.05).bfloat16()


def write_packed_experts(
    path: str | Path,
    geometry: ExpertGeometry = PRODUCTION_EXPERTS,
    role: str = "gate_up",
    seed_name: str = "flash_next_P1",
) -> dict[str, object]:
    """Write one (N, K) fused-expert NVFP4 payload to *path* (host buffer).

    Two passes: (1) global max(|W|) over the deterministic per-expert sources,
    (2) per-expert quantization with the explicit canonical divisor, appending
    codes/scales rows to the single payload planes.
    """

    if role not in _ROLES:
        raise ValueError(f"role must be one of {_ROLES}, got {role!r}")
    geometry.validate()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    n, k = geometry.role_shape(role)
    geo = geometry.geometry(role)
    rpe = geometry.rows_per_expert(role)
    codes_row_bytes = k // 2
    scales_row_bytes = k // 16

    # Pass 1: global max over the deterministic bf16 sources (exact in f32).
    gmax = 0.0
    for expert in range(geometry.experts):
        src = _expert_source(geometry, role, expert, seed_name)
        gmax = max(gmax, float(src.abs().max().item()))
    divisor_word = weight_scale_divisor_word(gmax)
    divisor_bytes = struct.pack("<I", divisor_word)
    divisor = struct.unpack("<f", divisor_bytes)[0]

    # Pass 2: quantize per expert with the explicit divisor; append planes.
    with out.open("wb") as handle:
        for expert in range(geometry.experts):
            src = _expert_source(geometry, role, expert, seed_name)
            qz = quantize_nvfp4(src, weight_scale_divisor=divisor)
            r0 = expert * rpe
            handle.seek(r0 * codes_row_bytes)
            handle.write(qz.packed_codes.contiguous().numpy().tobytes())
            handle.seek(geo.scale_plane_offset + r0 * scales_row_bytes)
            handle.write(qz.natural_scales.contiguous().numpy().tobytes())
        handle.seek(geo.weight_divisor_offset)
        handle.write(divisor_bytes)
        handle.flush()

    return {
        "path": str(out),
        "role": role,
        "experts": geometry.experts,
        "shape": [n, k],
        "rows_per_expert": rpe,
        "scale_plane_offset": geo.scale_plane_offset,
        "weight_divisor_offset": geo.weight_divisor_offset,
        "payload_bytes": geo.payload_bytes,
        "global_max_abs": gmax,
        "divisor": divisor,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def geometry_report(geometry: ExpertGeometry = PRODUCTION_EXPERTS) -> dict[str, object]:
    """Allocation-free description of the fused-expert payload geometry."""

    geometry.validate()
    report: dict[str, object] = {
        "experts": geometry.experts,
        "hidden": geometry.hidden,
        "inter": geometry.inter,
        "roles": {},
    }
    for role in _ROLES:
        geo = geometry.geometry(role)
        report["roles"][role] = {
            "shape": list(geometry.role_shape(role)),
            "rows_per_expert": geometry.rows_per_expert(role),
            "groups_per_row": geo.groups_per_row,
            "k_tiles": geo.k_tiles,
            "code_plane_bytes": geo.code_plane_bytes,
            "scale_plane_offset": geo.scale_plane_offset,
            "scale_plane_bytes": geo.scale_plane_bytes,
            "weight_divisor_offset": geo.weight_divisor_offset,
            "payload_bytes": geo.payload_bytes,
            "expert0": geometry.expert_offsets(role, 0),
            "expert_last": geometry.expert_offsets(role, geometry.experts - 1),
        }
    return report


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    geo_cmd = sub.add_parser("geometry", help="print the allocation-free geometry report")
    write_cmd = sub.add_parser("write", help="write one fused-expert NVFP4 payload")
    write_cmd.add_argument("--out", default="out/flash_next_dev/shape/experts_gate_up.nvfp4")
    write_cmd.add_argument("--role", choices=_ROLES, default="gate_up")
    args = parser.parse_args(argv)
    if args.cmd == "geometry":
        print(json.dumps(geometry_report(), indent=2))
    else:
        report = write_packed_experts(args.out, role=args.role)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
