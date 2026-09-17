"""STEP 5/6/11 -- fail-loudly validator for the separate Flash-Next ``.ngram`` store.

The 128 PLE n-gram embedding shards (51.2B params) are NOT in the main ``.ninfer``
container; they live here in a paging-oriented ``.ngram`` store (see
``ngram_store.py``) so the main artifact stays a single mmap-friendly file.  This
module validates that store and mirrors the ``verify_nvfp4`` convention (raise
``VerificationError`` on the first violation, return a frozen dataclass summary,
emit JSON from ``main``).

Structural layer (reused from ``ngram_store``, never re-implemented): the
``NNGRAM`` magic, store version, per-shard offset alignment + scale-plane
consistency + in-file extent, file-extent == last-shard extent, and the three
I64 config tensors (order / shape / int64 range) -- all enforced by
``ngram_store.validate_ngram_file``.

On top of that this adds the *identity / geometry* contract that proves the store
is the Flash-Next PLE and nothing else:

* ``model_id`` / ``quant_id`` match the registered constants;
* ``n_shards`` / ``rows_per_shard`` / ``cols_per_shard`` / ``block_rows`` match the
  documented geometry, and the shard-record count equals ``n_shards``;
* (opt-in ``--verify-checksums``) the whole-file + every per-shard SHA-256 match
  the write-time sidecar ``<store>.checksums.json``.

Canonical invocation::

    python -m tools.convert.qwen3_8_flash_next.verify_ngram \
        models/qwen3_8_flash_next.ngram [--verify-checksums] [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from . import config as cfg
from . import inventory_nvfp4 as inventory
from . import ngram_store

__all__ = [
    "VerificationError",
    "VerificationSummary",
    "verify_flash_next_ngram",
    "verify_ngram",
    "main",
]


class VerificationError(ValueError):
    """The store does not satisfy the registered Flash-Next PLE geometry contract."""


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    model_id: str
    quant_id: str
    n_shards: int
    rows_per_shard: int
    cols_per_shard: int
    block_rows: int
    version: int
    payload_offset: int
    shard_stride: int
    shard_bytes: int
    checksums_verified: bool
    shards_hashed: int


def _error(message: str) -> None:
    raise VerificationError(message)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksums(path: Path) -> int:
    """Re-hash the whole file + every shard against the write-time sidecar."""

    sidecar = path.with_name(path.name + ".checksums.json")
    if not sidecar.is_file():
        _error("checksums sidecar missing; content integrity not verified")
    report = json.loads(sidecar.read_text(encoding="utf-8"))
    actual_size = path.stat().st_size
    if int(report.get("file_bytes", -1)) != actual_size:
        _error(f"sidecar file_bytes {report.get('file_bytes')} != actual {actual_size}")
    if _sha256_file(path) != report.get("whole_file_sha256"):
        _error("whole-file sha256 mismatch vs sidecar")
    hashed = 0
    with ngram_store.NgramReader(path) as reader:
        for entry in report.get("shards", []):
            if not reader.verify_shard_checksum(int(entry["id"]), str(entry["sha256"])):
                _error(f"shard {entry['id']} sha256 mismatch vs sidecar")
            hashed += 1
    return hashed


def verify_ngram(
    path: str | Path,
    *,
    expected_model_id: str,
    expected_quant_id: str,
    expected_n_shards: int,
    expected_rows: int,
    expected_cols: int,
    expected_block_rows: int,
    verify_checksums: bool = False,
) -> VerificationSummary:
    """Validate a ``.ngram`` store: structural (reused) + identity/geometry + opt-in checksums."""

    p = Path(path)
    try:
        directory, shards = ngram_store.validate_ngram_file(p)
    except (ngram_store.NgramStoreError, OSError) as exc:
        _error(f"STRUCTURAL: {exc}")

    for key, expected in (
        ("model_id", expected_model_id),
        ("quant_id", expected_quant_id),
        ("n_shards", expected_n_shards),
        ("rows_per_shard", expected_rows),
        ("cols_per_shard", expected_cols),
        ("block_rows", expected_block_rows),
    ):
        if directory.get(key) != expected:
            _error(f"{key}: {directory.get(key)!r} != expected {expected!r}")
    if len(shards) != expected_n_shards:
        _error(f"shard records: {len(shards)} != expected {expected_n_shards}")

    shards_hashed = 0
    if verify_checksums:
        shards_hashed = _verify_checksums(p)

    return VerificationSummary(
        model_id=str(directory["model_id"]),
        quant_id=str(directory["quant_id"]),
        n_shards=int(directory["n_shards"]),
        rows_per_shard=int(directory["rows_per_shard"]),
        cols_per_shard=int(directory["cols_per_shard"]),
        block_rows=int(directory["block_rows"]),
        version=int(directory["version"]),
        payload_offset=int(directory["payload_offset"]),
        shard_stride=int(directory["shard_stride"]),
        shard_bytes=int(shards[0].bytes) if shards else 0,
        checksums_verified=verify_checksums,
        shards_hashed=shards_hashed,
    )


def verify_flash_next_ngram(path: str | Path, verify_checksums: bool = False) -> VerificationSummary:
    """Production ``.ngram`` validation (the documented Flash-Next PLE geometry)."""

    return verify_ngram(
        path,
        expected_model_id=inventory.MODEL_ID,
        expected_quant_id=ngram_store.QUANT_ID,
        expected_n_shards=cfg.NGRAM_SPLIT,
        expected_rows=cfg.NGRAM_SHARD[0],
        expected_cols=cfg.NGRAM_SHARD[1],
        expected_block_rows=ngram_store.BLOCK_ROWS,
        verify_checksums=verify_checksums,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="re-hash every shard + the whole file against the sidecar (slow; reads the whole store)",
    )
    parser.add_argument("--json", action="store_true", help="emit the machine-readable report")
    args = parser.parse_args(argv)

    try:
        summary = verify_flash_next_ngram(args.artifact, verify_checksums=args.verify_checksums)
    except VerificationError as exc:
        print(f"[FAIL] {exc}")
        return 1

    if args.json:
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    else:
        print(
            f"[PASS] {args.artifact}: n_shards={summary.n_shards} "
            f"rows={summary.rows_per_shard} cols={summary.cols_per_shard} "
            f"block_rows={summary.block_rows} payload_offset={summary.payload_offset}"
        )
        print(f"       model_id={summary.model_id} quant_id={summary.quant_id} shard_bytes={summary.shard_bytes}")
        if summary.checksums_verified:
            print(f"       checksums: verified {summary.shards_hashed} shards + whole file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
