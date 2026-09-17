"""STEP 10 -- numerical reconstruction check for the Flash-Next ``.ninfer``.

Two independent layers, both reusing the converter's OWN code (no re-derivation
of the encode -- ``Do not guess, reuse existing conventions``):

(a) **BYTE-EXACT.**  For each checked object, re-encode it from the official BF16
    source with the converter's exact encoder
    (``convert_nvfp4._encode_payload``) and compare it byte-for-byte against the
    bytes stored in the ``.ninfer`` at that object's offset.  Every encode
    (NVFP4 / FP8 / BF16-FP32 pass-through / FP32 divisor placeholder / resource
    copy) is deterministic, so a re-encoded payload MUST be byte-identical to the
    stored one; any difference is a write / read corruption or a non-deterministic
    encode.  This is the write-integrity + determinism half.

(b) **ROUND-TRIP FIDELITY.**  For the two lossy formats (NVFP4, FP8), materialize
    the source tensor, quantize it with the registered quantizer and dequantize it
    with the registered dequantizer, and assert the reconstruction stays within
    the format's error bound.  This is the source-approximation half: it proves
    the quantized payload is a faithful approximation of the official weights,
    not merely a self-consistent byte string.  The bounds are calibrated by
    ``.logs/numerical_bound_probe.py`` (randn BF16, stable across shapes):
    NVFP4 rms_rel ~ 0.095 / cosine ~ 0.995; FP8 rms_rel ~ 0.027 /
    cosine ~ 0.9997.

Together the two layers reconstruct the chain  source -> quantize -> (dequantize)
~= source, with the ``.ninfer`` holding the exact quantize bytes.  Run
``verify_nvfp4`` (structure) first; this is the *value* layer on top.

A full ``--all`` check re-materializes + re-quantizes every object from the
multi-hundred-GB source (as expensive as the convert itself) and therefore runs
only the byte-exact layer; the round-trip layer runs on the bounded default
sample (a handful of objects fully exercises the single quantizer code path, so
there is no point repeating it 1600 times).

Canonical invocation::

    python -m tools.convert.qwen3_8_flash_next.verify_numerical \
        models/qwen3_8_flash_next.ninfer \
        --model Z:/ninfer_flash_next_tmp/flash_next_bf16 [--device cpu] \
        [--all] [--per-format N] [--json]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ResourceObject,
    TensorObject,
)
from tools.artifact.layouts import encoded_size
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import inventory as family_inventory
from tools.convert.qwen3_8_flash_next import config as cfg
from tools.convert.qwen3_8_flash_next import convert_nvfp4
from tools.convert.qwen3_8_flash_next import inventory_nvfp4 as inventory
from tools.convert.qwen3_8_flash_next import quantize_fp8_row as fp8
from tools.convert.qwen3_8_flash_next import quantize_nvfp4 as nvfp4
from tools.convert.qwen3_8_flash_next import recipe_nvfp4 as recipe_mod

__all__ = [
    "FP8_MAX_RMS_REL",
    "FP8_MIN_COSINE",
    "NVFP4_MAX_RMS_REL",
    "NVFP4_MIN_COSINE",
    "NumericalError",
    "ObjectCheck",
    "NumericalSummary",
    "round_trip_metrics",
    "select_objects",
    "verify",
    "main",
]

# Round-trip bounds (calibrated by .logs/numerical_bound_probe.py; ~2.5x margin on
# the measured rms_rel, comfortable floors on the measured cosine).
NVFP4_MAX_RMS_REL = 0.25
NVFP4_MIN_COSINE = 0.97
FP8_MAX_RMS_REL = 0.10
FP8_MIN_COSINE = 0.998


class NumericalError(ValueError):
    """The artifact payload does not numerically reconstruct the source."""


@dataclass(frozen=True, slots=True)
class ObjectCheck:
    name: str
    kind: str  # "tensor" | "resource"
    fmt: str
    bytes: int
    byte_exact: bool
    first_diff: int  # -1 if identical, else the first differing index
    round_trip_rms_rel: float | None
    round_trip_cosine: float | None


@dataclass(frozen=True, slots=True)
class NumericalSummary:
    model_id: str
    checked: int
    byte_exact_all: bool
    round_trip_checked: int
    max_rms_rel: float
    min_cosine: float
    objects: tuple
    note: str


def _first_diff(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def select_objects(all_: bool, per_format: int = 1) -> list:
    """The inventory specs to check, deterministically ordered by name.

    ``all_`` returns every object.  Otherwise a bounded representative sample:
    the ``per_format`` smallest (by ``encoded_size``) tensor per numeric format
    (fast to materialize) plus exactly one divisor object and one resource -- the
    two non-recipe encode paths -- so every path is exercised.
    """

    specs = list(inventory.OBJECT_SPECS)
    if all_:
        return specs
    if per_format < 1:
        raise NumericalError("per_format must be >= 1")

    tensor_by_fmt: dict[str, list] = {}
    for spec in specs:
        if isinstance(spec, family_inventory.TensorSpec):
            tensor_by_fmt.setdefault(spec.format, []).append(spec)

    chosen: list = []
    for fmt in sorted(tensor_by_fmt):
        by_size = sorted(
            tensor_by_fmt[fmt],
            key=lambda s: encoded_size(s.layout, s.format, s.shape),
        )
        chosen.extend(by_size[:per_format])
    for spec in specs:
        if spec.name in recipe_mod.DIVISOR_NAMES:
            chosen.append(spec)
            break
    for spec in specs:
        if isinstance(spec, family_inventory.ResourceSpec):
            chosen.append(spec)
            break
    return sorted(chosen, key=lambda s: s.name)


def round_trip_metrics(bf16: torch.Tensor, fmt: str) -> tuple[float, float]:
    """(rms_rel, cosine) of the registered quantize -> dequantize round trip."""

    if fmt == inventory.NVFP4:
        qz = nvfp4.quantize_nvfp4(bf16)
        recon = nvfp4.dequantize_nvfp4(qz)
    elif fmt == inventory.FP8:
        qz = fp8.quantize_fp8_row(bf16)
        recon = fp8.dequantize_fp8_row(qz)
    else:
        raise NumericalError(f"no round-trip defined for format {fmt!r}")
    src = bf16.float()
    err = (recon - src).abs()
    rms_rel = float(
        err.pow(2).mean().sqrt() / src.pow(2).mean().sqrt().clamp_min(1e-12)
    )
    flat_r = recon.flatten()
    flat_s = src.flatten()
    cosine = float(
        (flat_r @ flat_s) / (flat_r.norm() * flat_s.norm()).clamp_min(1e-12)
    )
    return rms_rel, cosine


def _in_bound(fmt: str, rms_rel: float, cosine: float) -> tuple[bool, str]:
    if fmt == inventory.NVFP4:
        ok = rms_rel <= NVFP4_MAX_RMS_REL and cosine >= NVFP4_MIN_COSINE
        bound = (
            f"rms_rel<={NVFP4_MAX_RMS_REL} cosine>={NVFP4_MIN_COSINE}"
        )
    else:
        ok = rms_rel <= FP8_MAX_RMS_REL and cosine >= FP8_MIN_COSINE
        bound = f"rms_rel<={FP8_MAX_RMS_REL} cosine>={FP8_MIN_COSINE}"
    return ok, bound


def verify(
    artifact_path: str | Path,
    model_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    all_: bool = False,
    per_format: int = 1,
) -> NumericalSummary:
    """Run the byte-exact (all objects) + round-trip (sample) reconstruction check."""

    out = Path(artifact_path)
    model = Path(model_dir)
    if out.name != convert_nvfp4.OUTPUT_BASENAME:
        raise NumericalError(
            f"artifact basename must be {convert_nvfp4.OUTPUT_BASENAME!r}"
        )
    dev = pick_device(device)
    objects = select_objects(all_, per_format)

    with ShardReader(model) as reader, Artifact.open(out) as artifact:
        if artifact.identity != ArtifactIdentity(
            inventory.MODEL_ID, inventory.WEIGHTS_ID
        ):
            raise NumericalError(
                f"artifact identity is {artifact.identity!r}, expected "
                f"({inventory.MODEL_ID!r}, {inventory.WEIGHTS_ID!r})"
            )
        resources = {
            r.name: r.data
            for r in family_conversion.load_resources(model, inventory.RESOURCE_SPECS)
        }
        divisor = torch.tensor(
            cfg.INPUT_SCALE_DIVISOR_PLACEHOLDER, dtype=torch.float32
        )

        checks: list[ObjectCheck] = []
        round_trip_rms: list[float] = []
        round_trip_cos: list[float] = []
        for spec in objects:
            name = spec.name
            actual = bytes(artifact.payload(name))
            expected = convert_nvfp4._encode_payload(
                spec, reader, resources, divisor, dev
            )
            byte_exact = len(actual) == len(expected) and _first_diff(
                actual, expected
            ) == -1
            if not byte_exact:
                first_diff = _first_diff(actual, expected)
                raise NumericalError(
                    f"{name}: stored {len(actual)} bytes != re-encoded "
                    f"{len(expected)} bytes (first difference @ {first_diff})"
                )

            rt_rms: float | None = None
            rt_cos: float | None = None
            is_tensor = isinstance(spec, family_inventory.TensorSpec)
            if (
                not all_
                and is_tensor
                and spec.format in (inventory.NVFP4, inventory.FP8)
            ):
                source = recipe_mod.materialize(spec, reader, dev)
                rt_rms, rt_cos = round_trip_metrics(source, spec.format)
                ok, bound = _in_bound(spec.format, rt_rms, rt_cos)
                if not ok:
                    raise NumericalError(
                        f"{name}: round trip rms_rel={rt_rms:.4f} "
                        f"cosine={rt_cos:.6f} outside {bound}"
                    )
                round_trip_rms.append(rt_rms)
                round_trip_cos.append(rt_cos)

            kind = "tensor" if is_tensor else "resource"
            fmt = spec.format if is_tensor else spec.encoding
            checks.append(
                ObjectCheck(
                    name=name,
                    kind=kind,
                    fmt=fmt,
                    bytes=len(actual),
                    byte_exact=True,
                    first_diff=-1,
                    round_trip_rms_rel=rt_rms,
                    round_trip_cosine=rt_cos,
                )
            )

    if round_trip_rms:
        max_rms = max(round_trip_rms)
        min_cos = min(round_trip_cos)
    else:
        max_rms = 0.0
        min_cos = 1.0
    return NumericalSummary(
        model_id=inventory.MODEL_ID,
        checked=len(checks),
        byte_exact_all=True,
        round_trip_checked=len(round_trip_rms),
        max_rms_rel=max_rms,
        min_cosine=min_cos,
        objects=tuple(checks),
        note=(
            "Every checked object's stored bytes are byte-identical to the "
            "converter's re-encode of the official source; each checked NVFP4/"
            "FP8 object dequantizes within its calibrated bound. A structurally "
            "valid + numerically faithful artifact is STILL not a runnable model "
            "(BLOCKER: RUNTIME KERNEL REQUIRED for the Flash-Next architecture)."
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--all",
        action="store_true",
        help="byte-exact check every object (skips the round-trip layer)",
    )
    parser.add_argument(
        "--per-format", type=int, default=1,
        help="per-format tensor count in the default sample (default 1)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        s = verify(
            args.artifact,
            args.model,
            device=args.device,
            all_=args.all,
            per_format=args.per_format,
        )
    except NumericalError as exc:
        print(f"[FAIL] {exc}")
        return 1

    if args.json:
        print(json.dumps(asdict(s), indent=2, sort_keys=True))
    else:
        print(
            f"[PASS] {s.model_id}: {s.checked} objects byte-exact; "
            f"{s.round_trip_checked} round-trip-checked "
            f"(max rms_rel={s.max_rms_rel:.4f}, min cosine={s.min_cosine:.6f})"
        )
        for c in s.objects:
            rt = (
                f"  rms_rel={c.round_trip_rms_rel:.4f} "
                f"cosine={c.round_trip_cosine:.5f}"
                if c.round_trip_rms_rel is not None
                else ""
            )
            print(f"  {c.name}  [{c.kind}/{c.fmt}] {c.bytes}B{rt}")
        print(f"\n  note: {s.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
