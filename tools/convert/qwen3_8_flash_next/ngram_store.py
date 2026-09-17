"""Separate SSD demand-paging store for the Flash-Next PLE n-gram table.

The 128 PLE n-gram embedding shards (51.2B elements, 102.4 GB in BF16) and the
3 I64 PLE control tables are NOT representable in the ``.ninfer`` container
(the ``.ninfer`` numeric formats are BF16/FP32/I32 only and the table is a
paged lookup, not a resident matrix), so they live in a dedicated
``qwen3_8_flash_next.ngram`` artifact.

Quantization (data-grounded, 2026-09-11 probe on real shard_0 values):
  * source distribution: zero-mean, std ~7.5e-3, per-row absmax 0.010-0.051
    (rows vary ~5x), fully dense (0% zeros).
  * chosen quant: **FP8 E4M3FN with a per-row BF16 scale** -- the EXACT
    profile the model's own token embedding uses in the ``.ninfer``
    (config.py: "FP8 row-scale ... the embedding and output head"). Measured
    round-trip on a 65536-row shard_0 sample: 2.64% RMS / 8.6% max relative
    error (standard FP8 embedding error). Halves the footprint (102.4 GB ->
    51.84 GB), which is the whole point of a demand-paged NVMe table, and the
    dequant is a single ``code * scale`` multiply (cheap on the host).
  * ``quant_id`` is explicit so a lossless BF16 or NVFP4 re-quant is a
    trivial re-emit without re-downloading the checkpoint.

Container discipline mirrors the NInfer v2 ``.ninfer`` writer
(``tools.artifact.container``): an 8-byte magic + a self-describing JSON
directory at the front + 4096-aligned payload. The front directory is
deterministic in ``(model_id, quant_id, geometry, config_tensors)`` -- it
carries NO content hashes -- so a run can resume from the first incomplete
shard exactly like the ``.ninfer`` ``ResumableArtifactWriter``. Per-shard and
whole-file SHA-256 checksums are written to a sidecar
``<out>.checksums.json`` (the NInfer ``.conversion.json`` report convention),
not into the container (which must stay precomputable for resume).

Layout::

    [ PREFIX <8sQ: NGRAM_MAGIC, json_bytes ]
    [ JSON directory (json_bytes) ]
    [ zero pad to 4096 ]                       -> payload_offset
    [ shard 0 payload ] [ pad ] [ shard 1 payload ] ... [ shard 127 payload ]

Each shard payload is the exact ``layouts.encode_fp8_row_scaled`` output for
shape ``(rows, cols)``: a code plane (rows*cols E4M3FN bytes) followed by a
scale plane (rows*2 BF16 bytes) at a 256-aligned offset. Demand-paging blocks
are ``BLOCK_ROWS`` (128) contiguous rows: the codes block is
``128*cols = 20480 = 5*4096`` bytes -- 4096-aligned, so an NVMe read of a block
is a single aligned transfer. Block ``b`` of shard ``s`` reads:
  * codes  at  shards[s].offset       + b * 20480
  * scales at  shards[s].scale_offset + b * 256
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch

from tools.artifact.layouts import (
    PLANE_ALIGNMENT,
    align_up,
    encode_fp8_row_scaled,
    row_scale_geometry,
)
from . import config as cfg
from .quantize_fp8_row import quantize_fp8_row

# --- container identity -----------------------------------------------------
NGRAM_MAGIC = b"NNGRAM\x00\x01"
NGRAM_VERSION = 1
NGRAM_ALIGN = 4096
PREFIX = struct.Struct("<8sQ")
PREFIX_BYTES = PREFIX.size  # 16

# Demand-paging block: 128 rows -> 128*160 = 20480 = 5*4096 bytes of codes,
# the smallest 4096-aligned block (cols=160 is not a power of two).
BLOCK_ROWS = 128

QUANT_ID = "fp8e4m3fn_row_bf16s-v1"
NGRAM_BASENAME = "qwen3_8_flash_next.ngram"

# The 3 I64 PLE control tables, by short name -> checkpoint source tensor name.
_CONFIG_SOURCE = "model.language_model.layers.1.ple.ple_embedding"
NGRAM_CONFIG_SOURCES = {
    "layer_multipliers": f"{_CONFIG_SOURCE}.layer_multipliers",
    "ngram_heads_offsets": f"{_CONFIG_SOURCE}.ngram_heads_offsets",
    "ngram_heads_vocab_sizes": f"{_CONFIG_SOURCE}.ngram_heads_vocab_sizes",
}
NGRAM_CONFIG_ORDER = ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")
NGRAM_CONFIG_SHAPES = cfg.NGRAM_CONFIG_SHAPES  # {"layer_multipliers": (3,), ...}

_ROWS = cfg.NGRAM_SHARD[0]
_COLS = cfg.NGRAM_SHARD[1]
_N_SHARDS = cfg.NGRAM_SPLIT

# I64 is signed 64-bit; values must fit (the contract is "63-bit" in the source).
_INT64_MIN = -(2 ** 63)
_INT64_MAX = 2 ** 63 - 1


class NgramStoreError(RuntimeError):
    """The store is structurally inconsistent or a shard failed verification."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# --- geometry ---------------------------------------------------------------

def shard_geometry(rows: int = _ROWS, cols: int = _COLS):
    """The exact per-shard payload geometry from the shared row-scale layout."""

    return row_scale_geometry("FP8_E4M3FN_ROW_BF16S", (rows, cols))


def _validate_geometry(rows: int, cols: int, block_rows: int) -> None:
    if rows <= 0 or cols <= 0 or block_rows <= 0:
        raise NgramStoreError("rows/cols/block_rows must be positive")
    # The codes block must land on a 4096 boundary (aligned NVMe read).
    block_codes_bytes = block_rows * cols
    if block_codes_bytes % NGRAM_ALIGN != 0:
        raise NgramStoreError(
            f"block_rows*cols = {block_codes_bytes} is not {NGRAM_ALIGN}-aligned"
        )


@dataclass(frozen=True, slots=True)
class ShardRecord:
    id: int
    offset: int  # absolute file offset of the codes plane (payload start)
    scale_offset: int  # absolute file offset of the scale plane
    bytes: int  # full shard payload size


# --- config tensors ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ConfigTensor:
    name: str
    shape: tuple[int, ...]
    values: tuple[int, ...]

    def _check_int64(self) -> None:
        for v in self.values:
            if not isinstance(v, int) or v < _INT64_MIN or v > _INT64_MAX:
                raise NgramStoreError(
                    f"config {self.name} value {v!r} is not a signed 64-bit int"
                )

    def to_int64(self) -> torch.Tensor:
        self._check_int64()
        return torch.tensor(list(self.values), dtype=torch.int64).reshape(self.shape)


def validate_config_tensors(configs: Sequence[ConfigTensor]) -> None:
    """Order + shape + int64-range checks on the 3 PLE control tables."""

    names = tuple(c.name for c in configs)
    if names != NGRAM_CONFIG_ORDER:
        raise NgramStoreError(f"config tensor order mismatch: {names}")
    for c in configs:
        if c.shape != NGRAM_CONFIG_SHAPES[c.name]:
            raise NgramStoreError(
                f"config {c.name} shape {c.shape} != expected {NGRAM_CONFIG_SHAPES[c.name]}"
            )
        if len(c.values) != int(torch.tensor(c.shape).prod()):
            raise NgramStoreError(f"config {c.name} value count does not match shape")
        c._check_int64()


# --- JSON directory ---------------------------------------------------------

def build_directory(
    model_id: str,
    quant_id: str,
    configs: Sequence[ConfigTensor],
    *,
    rows: int = _ROWS,
    cols: int = _COLS,
    block_rows: int = BLOCK_ROWS,
    n_shards: int = _N_SHARDS,
) -> bytes:
    """Deterministically build the front JSON directory (no content hashes)."""

    _validate_geometry(rows, cols, block_rows)
    validate_config_tensors(configs)

    # Solve the (payload_offset -> dir length -> payload_offset) fixed point. The
    # directory embeds payload_offset and every shard offset (payload_offset +
    # i*stride), so its JSON length depends on the digit width of those ints.
    # The stride term dominates that width, so the length stabilises in one or
    # two iterations; iterate to the fixed point and fail loudly if it does not.
    payload_offset = 0
    directory = None
    for _ in range(16):
        directory = _directory_bytes(model_id, quant_id, configs, rows, cols, block_rows, n_shards, payload_offset)
        candidate = align_up(PREFIX_BYTES + len(directory), NGRAM_ALIGN)
        if candidate == payload_offset:
            break
        payload_offset = candidate
    else:
        raise NgramStoreError("directory length did not reach a fixed point")
    return directory


def _directory_bytes(
    model_id, quant_id, configs, rows, cols, block_rows, n_shards, payload_offset
) -> bytes:
    geo = shard_geometry(rows, cols)
    shards = []
    for i in range(n_shards):
        off = payload_offset + i * align_up(geo.payload_bytes, NGRAM_ALIGN)
        shards.append({
            "id": i,
            "offset": off,
            "scale_offset": off + geo.scale_plane_offset,
            "bytes": geo.payload_bytes,
        })
    block_codes_bytes = block_rows * cols
    directory = {
        "version": NGRAM_VERSION,
        "model_id": model_id,
        "quant_id": quant_id,
        "n_shards": n_shards,
        "rows_per_shard": rows,
        "cols_per_shard": cols,
        "block_rows": block_rows,
        "payload_offset": payload_offset,
        "shard_stride": align_up(geo.payload_bytes, NGRAM_ALIGN),
        "planes": {
            "code_plane_bytes": geo.code_plane_bytes,
            "scale_plane_offset": geo.scale_plane_offset,
            "scale_plane_bytes": geo.scale_plane_bytes,
            "payload_bytes": geo.payload_bytes,
            "plane_alignment": PLANE_ALIGNMENT,
        },
        "block": {
            "rows": block_rows,
            "cols": cols,
            "code_row_bytes": cols,
            "block_codes_bytes": block_codes_bytes,
            "scale_row_bytes": 2,
            "block_scale_bytes": block_rows * 2,
            "blocks_per_shard": -(-rows // block_rows),  # ceil
            "note": (
                "block b of shard s: codes at shards[s].offset + b*"
                f"{block_codes_bytes} ({block_codes_bytes}="
                f"{block_codes_bytes // NGRAM_ALIGN}*{NGRAM_ALIGN}, aligned); "
                "scales at shards[s].scale_offset + b*"
                f"{block_rows * 2}"
            ),
        },
        "ngram": {
            "size": cfg.NGRAM_SIZE,
            "heads": cfg.NGRAM_HEADS,
            "heads_per_ngram": cfg.HEADS_PER_NGRAM,
            "vocab_base": cfg.NGRAM_VOCAB_BASE,
        },
        "config_tensors": [
            {"name": c.name, "dtype": "I64", "shape": list(c.shape), "values": list(c.values)}
            for c in configs
        ],
        "shards": shards,
    }
    return json.dumps(directory, sort_keys=True, separators=(",", ":")).encode("utf-8")


# --- writer -----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Progress:
    directory_sha256: str
    completed: int

    @classmethod
    def read(cls, path: Path) -> "_Progress | None":
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(directory_sha256=str(value["directory_sha256"]), completed=int(value["completed"]))

    def write(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"directory_sha256": self.directory_sha256, "completed": self.completed}),
            encoding="utf-8",
        )
        tmp.replace(path)


class NgramWriter:
    """Resumable single-pass writer for the ``.ngram`` store."""

    def __init__(
        self,
        path: str | Path,
        directory: bytes,
        *,
        n_shards: int,
        overwrite: bool = False,
    ):
        self.path = Path(path)
        self.directory = directory
        self.directory_sha256 = _sha256(directory)
        self.n_shards = n_shards
        dir_obj = json.loads(directory.decode("utf-8"))
        self.payload_offset = int(dir_obj["payload_offset"])
        self.shards: tuple[ShardRecord, ...] = tuple(
            ShardRecord(id=s["id"], offset=s["offset"], scale_offset=s["scale_offset"], bytes=s["bytes"])
            for s in dir_obj["shards"]
        )
        if len(self.shards) != n_shards:
            raise NgramStoreError("directory shard count != n_shards")
        self.shard_bytes = self.shards[0].bytes
        self.progress_path = self.path.with_name(self.path.name + ".progress.json")

        started = self._resolve_start(overwrite)
        self._file = self.path.open("r+b")
        self._next = started
        self._cursor = self.shards[started].offset if self.shards and started < n_shards else self._end_offset()
        self._finish_progress(started)

    @property
    def completed(self) -> int:
        return self._next

    def _end_offset(self) -> int:
        last = self.shards[-1]
        return last.offset + last.bytes

    def _resolve_start(self, overwrite: bool) -> int:
        if not self.path.exists() or overwrite:
            self._write_header()
            return 0
        with self.path.open("rb") as fh:
            head = fh.read(PREFIX_BYTES)
            if len(head) < PREFIX_BYTES:
                raise NgramStoreError("existing store shorter than the prefix")
            magic, json_bytes = PREFIX.unpack(head)
            if magic != NGRAM_MAGIC:
                raise NgramStoreError("existing store magic is not NNGRAM v1")
            stored = fh.read(json_bytes)
        if stored != self.directory:
            raise NgramStoreError("existing store has a different directory; re-run with --overwrite")
        progress = _Progress.read(self.progress_path)
        if progress is None or progress.directory_sha256 != self.directory_sha256:
            raise NgramStoreError("store exists but has no matching progress record; re-run with --overwrite")
        if progress.completed > self.n_shards:
            raise NgramStoreError("progress record exceeds the shard count")
        return progress.completed

    def _write_header(self) -> None:
        with self.path.open("wb") as fh:
            fh.write(PREFIX.pack(NGRAM_MAGIC, len(self.directory)))
            fh.write(self.directory)
            fh.write(b"\x00" * (self.payload_offset - PREFIX_BYTES - len(self.directory)))
        self.progress_path.unlink(missing_ok=True)

    def _finish_progress(self, completed: int) -> None:
        _Progress(self.directory_sha256, completed).write(self.progress_path)

    def write_shard(self, shard_id: int, payload: bytes) -> None:
        if shard_id != self._next:
            raise NgramStoreError(f"expected shard {self._next}, got {shard_id}")
        if self._next >= self.n_shards:
            raise NgramStoreError("store already has every shard")
        if len(payload) != self.shard_bytes:
            raise NgramStoreError(
                f"shard {shard_id} has {len(payload)} bytes; expected {self.shard_bytes}"
            )
        target = self.shards[self._next].offset
        if target > self._cursor:
            self._file.seek(self._cursor)
            self._file.write(b"\x00" * (target - self._cursor))
        self._file.seek(target)
        self._file.write(payload)
        self._file.flush()
        self._cursor = target + len(payload)
        self._next += 1
        self._finish_progress(self._next)

    def finish(self) -> None:
        if self._next != self.n_shards:
            raise NgramStoreError(f"store is missing shard {self._next}")
        self._file.truncate(self._end_offset())
        self._file.flush()
        self._file.close()
        self.progress_path.unlink(missing_ok=True)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "NgramWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            try:
                self.finish()
            finally:
                self.close()
        else:
            self.close()


def encode_shard(bf16: torch.Tensor) -> bytes:
    """Quantize one BF16 shard [N,K] to its exact FP8 row-scale payload bytes."""

    qz = quantize_fp8_row(bf16)
    return encode_fp8_row_scaled(qz.code_words, qz.row_scales, tuple(bf16.shape))


def write_checksums_sidecar(path: str | Path, directory: bytes, shard_hashes: dict[int, str]) -> dict:
    """Write ``<out>.checksums.json`` (per-shard + directory + whole-file SHA-256)."""

    p = Path(path)
    whole = _sha256_file(p)
    report = {
        "file": p.name,
        "directory_sha256": _sha256(directory),
        "whole_file_sha256": whole,
        "file_bytes": p.stat().st_size,
        "shards": [
            {"id": int(k), "sha256": v} for k, v in sorted(shard_hashes.items())
        ],
    }
    sidecar = p.with_name(p.name + ".checksums.json")
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(sidecar)
    return report


# --- reader / validator -----------------------------------------------------

def parse_directory(path: str | Path) -> tuple[dict, tuple[ShardRecord, ...]]:
    p = Path(path)
    with p.open("rb") as fh:
        head = fh.read(PREFIX_BYTES)
        if len(head) < PREFIX_BYTES:
            raise NgramStoreError("store shorter than the prefix")
        magic, json_bytes = PREFIX.unpack(head)
        if magic != NGRAM_MAGIC:
            raise NgramStoreError("magic is not NNGRAM v1")
        raw = fh.read(json_bytes)
    directory = json.loads(raw.decode("utf-8"))
    if int(directory["version"]) != NGRAM_VERSION:
        raise NgramStoreError(f"unsupported store version {directory['version']}")
    shards = tuple(
        ShardRecord(id=s["id"], offset=s["offset"], scale_offset=s["scale_offset"], bytes=s["bytes"])
        for s in directory["shards"]
    )
    return directory, shards


def validate_ngram_file(path: str | Path) -> tuple[dict, tuple[ShardRecord, ...]]:
    """Structurally validate a finished ``.ngram`` store (no content hash)."""

    directory, shards = parse_directory(path)
    p = Path(path)
    file_size = p.stat().st_size
    payload_offset = int(directory["payload_offset"])
    if payload_offset > file_size:
        raise NgramStoreError("declared payload_offset exceeds the file")
    for rec in shards:
        if rec.offset < payload_offset or rec.offset % NGRAM_ALIGN != 0:
            raise NgramStoreError(f"shard {rec.id} offset {rec.offset} misaligned")
        if rec.scale_offset != rec.offset + int(directory["planes"]["scale_plane_offset"]):
            raise NgramStoreError(f"shard {rec.id} scale_offset inconsistent")
        end = rec.offset + rec.bytes
        if end > file_size:
            raise NgramStoreError(f"shard {rec.id} extends beyond the file")
    last = shards[-1]
    if file_size != last.offset + last.bytes:
        raise NgramStoreError("file size does not match the last shard extent")
    # config tensors are intact + int64
    configs = _configs_from_directory(directory)
    validate_config_tensors(configs)
    return directory, shards


def _configs_from_directory(directory: dict) -> tuple[ConfigTensor, ...]:
    out = []
    for entry in directory["config_tensors"]:
        out.append(
            ConfigTensor(
                name=entry["name"],
                shape=tuple(int(d) for d in entry["shape"]),
                values=tuple(int(v) for v in entry["values"]),
            )
        )
    return tuple(out)


class NgramReader:
    """Read shards / demand-paging blocks / dequantize from a finished store."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.directory, self.shards = validate_ngram_file(self.path)
        self.rows = int(self.directory["rows_per_shard"])
        self.cols = int(self.directory["cols_per_shard"])
        self.block_rows = int(self.directory["block_rows"])
        self._handle = self.path.open("rb")

    def config_tensors(self) -> dict[str, torch.Tensor]:
        return {c.name: c.to_int64() for c in _configs_from_directory(self.directory)}

    def _shard(self, shard_id: int) -> ShardRecord:
        if shard_id < 0 or shard_id >= len(self.shards):
            raise NgramStoreError(f"shard {shard_id} out of range 0..{len(self.shards)-1}")
        return self.shards[shard_id]

    def read_block(self, shard_id: int, block_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one aligned block: (codes uint8 [<=block_rows, cols], scales bf16 [<=block_rows])."""

        rec = self._shard(shard_id)
        nblocks = -(-self.rows // self.block_rows)
        if block_id < 0 or block_id >= nblocks:
            raise NgramStoreError(f"block {block_id} out of range 0..{nblocks-1}")
        rows = min(self.block_rows, self.rows - block_id * self.block_rows)
        code_bytes = rows * self.cols
        scale_bytes = rows * 2
        self._handle.seek(rec.offset + block_id * self.block_rows * self.cols)
        codes = self._handle.read(code_bytes)
        if len(codes) != code_bytes:
            raise NgramStoreError("short codes read")
        self._handle.seek(rec.scale_offset + block_id * self.block_rows * 2)
        scales = self._handle.read(scale_bytes)
        if len(scales) != scale_bytes:
            raise NgramStoreError("short scales read")
        return (
            torch.frombuffer(bytearray(codes), dtype=torch.uint8).reshape(rows, self.cols),
            torch.frombuffer(bytearray(scales), dtype=torch.uint16).view(torch.bfloat16),
        )

    def read_shard_codes(self, shard_id: int) -> torch.Tensor:
        rec = self._shard(shard_id)
        code_bytes = int(self.directory["planes"]["code_plane_bytes"])
        self._handle.seek(rec.offset)
        data = self._handle.read(code_bytes)
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(self.rows, self.cols)

    def read_shard_scales(self, shard_id: int) -> torch.Tensor:
        rec = self._shard(shard_id)
        scale_bytes = int(self.directory["planes"]["scale_plane_bytes"])
        self._handle.seek(rec.scale_offset)
        data = self._handle.read(scale_bytes)
        return torch.frombuffer(bytearray(data), dtype=torch.uint16).view(torch.bfloat16)

    def dequant_block(self, shard_id: int, block_id: int) -> torch.Tensor:
        codes, scales = self.read_block(shard_id, block_id)
        return codes.view(torch.float8_e4m3fn).float() * scales.float().unsqueeze(1)

    def verify_shard_checksum(self, shard_id: int, expected_sha256: str) -> bool:
        rec = self._shard(shard_id)
        h = hashlib.sha256()
        self._handle.seek(rec.offset)
        remaining = rec.bytes
        while remaining > 0:
            chunk = self._handle.read(min(1 << 24, remaining))
            if not chunk:
                raise NgramStoreError("short read during checksum")
            h.update(chunk)
            remaining -= len(chunk)
        return h.hexdigest() == expected_sha256

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "NgramReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "BLOCK_ROWS",
    "ConfigTensor",
    "NgramReader",
    "NgramStoreError",
    "NgramWriter",
    "NGRAM_BASENAME",
    "NGRAM_CONFIG_ORDER",
    "NGRAM_CONFIG_SOURCES",
    "NGRAM_MAGIC",
    "NGRAM_VERSION",
    "QUANT_ID",
    "ShardRecord",
    "build_directory",
    "encode_shard",
    "parse_directory",
    "shard_geometry",
    "validate_config_tensors",
    "validate_ngram_file",
    "write_checksums_sidecar",
]
