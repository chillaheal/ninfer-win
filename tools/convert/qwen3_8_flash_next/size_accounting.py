"""STEP 12 -- RTX 5090 size accounting for the Flash-Next NVFP4 foundation.

Computes the on-disk weight footprint of BOTH deliverables and compares it to a
single RTX 5090's 32 GB of VRAM:

* the ``.ninfer`` container -- the exact sum of ``encoded_size(layout, format,
  shape)`` over all 1602 inventory tensor objects, grouped by numeric format
  (the same primitive the structural validator uses for per-object byte checks);
* the separate ``.ngram`` PLE store -- 128 shards at the shared row-scale
  geometry (``ngram_store.shard_geometry``), each padded to the NVMe alignment.

This is a *static* accounting: it needs only the inventory + the ngram geometry,
never the multi-hundred-gigabyte payloads, so it runs before either artifact
exists.  The verdict is deliberately honest -- the combined weights are many
times the VRAM, so the model does NOT fit resident on one RTX 5090.  That is
expected and is precisely why the offloading workstreams (CPU / NVMe paging,
expert caching, prefetch) are out of scope tonight: the STOP condition is the
validated artifact + the separate ngram store, not a runnable model.

Canonical invocation::

    python -m tools.convert.qwen3_8_flash_next.size_accounting [--vram-mib 32607] [--json]
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from typing import Sequence

from tools.artifact.layouts import encoded_size

from . import config as cfg
from . import inventory_nvfp4 as inventory
from . import ngram_store as NS

__all__ = [
    "NinferSize",
    "NgramSize",
    "SizeSummary",
    "accounting",
    "ninfer_payload",
    "ngram_payload",
    "main",
]


@dataclass(frozen=True, slots=True)
class NinferSize:
    tensor_objects: int
    payload_bytes: int
    per_format_bytes: dict  # format -> total encoded bytes (scale planes included)
    per_format_counts: dict  # format -> object count


@dataclass(frozen=True, slots=True)
class NgramSize:
    n_shards: int
    rows: int
    cols: int
    block_rows: int
    shard_payload_bytes: int
    shard_stride_bytes: int  # align_up(payload_bytes, NGRAM_ALIGN)
    payload_bytes: int  # n_shards * shard_stride_bytes


@dataclass(frozen=True, slots=True)
class SizeSummary:
    model_id: str
    ninfer: NinferSize
    ngram: NgramSize
    combined_bytes: int
    vram_bytes: int
    fits_in_vram: bool
    multiple_of_vram: float
    note: str


def ninfer_payload() -> NinferSize:
    """Exact ``.ninfer`` tensor payload: sum of ``encoded_size`` per object."""

    per_format_bytes: Counter[str] = Counter()
    per_format_counts: Counter[str] = Counter()
    total = 0
    for spec in inventory.TENSOR_SPECS:
        nbytes = encoded_size(spec.layout, spec.format, spec.shape)
        per_format_bytes[spec.format] += nbytes
        per_format_counts[spec.format] += 1
        total += nbytes
    return NinferSize(
        tensor_objects=len(inventory.TENSOR_SPECS),
        payload_bytes=total,
        per_format_bytes=dict(per_format_bytes),
        per_format_counts=dict(per_format_counts),
    )


def ngram_payload() -> NgramSize:
    """Exact ``.ngram`` payload: 128 shards, each padded to the NVMe alignment."""

    geo = NS.shard_geometry()
    stride = NS.align_up(geo.payload_bytes, NS.NGRAM_ALIGN)
    return NgramSize(
        n_shards=cfg.NGRAM_SPLIT,
        rows=geo.n,
        cols=geo.k,
        block_rows=NS.BLOCK_ROWS,
        shard_payload_bytes=geo.payload_bytes,
        shard_stride_bytes=stride,
        payload_bytes=cfg.NGRAM_SPLIT * stride,
    )


def accounting(vram_mib: int = 32607) -> SizeSummary:
    """Combine the two artifacts and compare the weight footprint to VRAM."""

    ni = ninfer_payload()
    ng = ngram_payload()
    combined = ni.payload_bytes + ng.payload_bytes
    vram_bytes = vram_mib * 1024 * 1024
    fits = combined <= vram_bytes
    note = (
        "Weights are a STATIC foundation, not a runnable claim. The combined "
        "footprint is far above one RTX 5090's VRAM, so the model cannot be "
        "resident on-device; runtime offloading (CPU/NVMe paging, expert "
        "caching, prefetch) is the out-of-scope follow-up (STOP = validated "
        "artifact + separate ngram store)."
        if not fits
        else "Combined weights fit within the reported VRAM."
    )
    return SizeSummary(
        model_id=inventory.MODEL_ID,
        ninfer=ni,
        ngram=ng,
        combined_bytes=combined,
        vram_bytes=vram_bytes,
        fits_in_vram=fits,
        multiple_of_vram=combined / vram_bytes,
        note=note,
    )


def _gib(n: int) -> float:
    return n / (1024 ** 3)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vram-mib", type=int, default=32607,
        help="RTX 5090 VRAM in MiB (nvidia-smi measured 32607 on this box)",
    )
    parser.add_argument("--json", action="store_true", help="emit the machine-readable report")
    args = parser.parse_args(argv)

    s = accounting(args.vram_mib)
    if args.json:
        print(json.dumps(asdict(s), indent=2, sort_keys=True))
        return 0

    print(f"[{s.model_id}] RTX 5090 size accounting (static, no payload reads)")
    print(f"\n  .ninfer: {s.ninfer.tensor_objects} tensor objects, "
          f"{_gib(s.ninfer.payload_bytes):8.3f} GiB payload")
    for fmt in sorted(s.ninfer.per_format_bytes, key=lambda f: -s.ninfer.per_format_bytes[f]):
        print(f"     {fmt:24s} {s.ninfer.per_format_counts[fmt]:5d} objs  "
              f"{_gib(s.ninfer.per_format_bytes[fmt]):9.3f} GiB")
    print(f"  .ngram : {s.ngram.n_shards} shards x ({s.ngram.rows}x{s.ngram.cols}) "
          f"block_rows={s.ngram.block_rows}, "
          f"{_gib(s.ngram.payload_bytes):8.3f} GiB payload")
    print(f"\n  combined: {_gib(s.combined_bytes):8.3f} GiB  "
          f"= {s.multiple_of_vram:5.2f} x RTX 5090 VRAM ({_gib(s.vram_bytes):.3f} GiB)")
    print(f"  fits resident on one RTX 5090: {s.fits_in_vram}")
    print(f"\n  note: {s.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
