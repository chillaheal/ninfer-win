"""Build the separate ``.ngram`` PLE store for Qwen3.8-Flash-Next.

The 128 PLE n-gram embedding shards (51.2B elements) + 3 I64 PLE control tables
are NOT in the ``.ninfer`` artifact (paged lookup, I64 63-bit values); they are
emitted here as ``qwen3_8_flash_next.ngram`` (see ``ngram_store``).

Modes:
  * ``--dry-run``  : readiness report -- which of the 128 shards / 3 configs are
    on disk (parsed from the download log, no per-file stats on the slow drive).
  * ``--smoke N``  : convert the first N *available* shards, reading only
    ``--rows`` rows of each, into a throwaway store; real-data round-trip + FP8
    error check. Validates the real-data path before the full 51.84 GB run.
  * ``--convert``  : the full 128-shard production run (resumable).
  * ``--verify``   : open a finished store, validate structure, verify every
    shard SHA-256 against the sidecar, dequant a sample block.

Canonical invocation (full run, once the download has finished)::

    python -m tools.convert.qwen3_8_flash_next.convert_ngram \
      --model Z:/ninfer_flash_next_tmp/flash_next_bf16 \
      --convert --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import time
from pathlib import Path
from typing import Sequence

import torch
from safetensors import safe_open

from tools.artifact.layouts import encode_fp8_row_scaled
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader

from . import config as cfg
from . import inventory_nvfp4 as inventory
from . import ngram_store as NS
from .quantize_fp8_row import dequantize_fp8_row, quantize_fp8_row

RECIPE_ID = "qwen3_8_flash_next_ngram-v1"
OUTPUT_BASENAME = "qwen3_8_flash_next.ngram"

_P = "model.language_model.layers.1.ple.ple_embedding"
_NGRAM_SHARD_FMT = f"{_P}.ngram_embedding.shard_{{}}.weight"
_CONFIG_SOURCE = {
    "layer_multipliers": f"{_P}.layer_multipliers",
    "ngram_heads_offsets": f"{_P}.ngram_heads_offsets",
    "ngram_heads_vocab_sizes": f"{_P}.ngram_heads_vocab_sizes",
}

_SHARD_RE = re.compile(r"model-\d+-of-\d+\.safetensors")


def shard_name(i: int) -> str:
    return _NGRAM_SHARD_FMT.format(i)


def _read_configs(reader: ShardReader) -> tuple[NS.ConfigTensor, ...]:
    out = []
    for short in NS.NGRAM_CONFIG_ORDER:
        name = _CONFIG_SOURCE[short]
        if not reader.has(name):
            raise KeyError(f"config tensor {name!r} not in the weight_map")
        t = reader.get(name).to(torch.int64)
        shape = tuple(cfg.NGRAM_CONFIG_SHAPES[short])
        values = tuple(int(v) for v in t.reshape(-1).tolist())
        out.append(NS.ConfigTensor(short, shape, values))
    NS.validate_config_tensors(out)
    return tuple(out)


def _done_files_from_log(log_path: Path) -> set[str] | None:
    """Basename set of checkpoint files completed per the download log."""

    if not log_path.is_file():
        return None
    done: set[str] = set()
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _SHARD_RE.search(line)
        if not m:
            continue
        if "FAILED" in line:
            continue
        if " done" in line or "skip" in line:
            done.add(m.group(0))
    return done


def _readiness(model_dir: Path):
    """Map each ngram shard + config to its host file; report presence.

    Presence is derived from the download log (one small read) so we never
    stat 131 files on the slow network drive. Falls back to per-file stat if the
    log is absent.
    """

    reader = ShardReader(model_dir)
    shard_hosts = [reader.weight_map[shard_name(i)] for i in range(128)]
    config_hosts = [_CONFIG_SOURCE[s] for s in NS.NGRAM_CONFIG_ORDER]
    config_file_set = {reader.weight_map[c] for c in config_hosts}

    done = _done_files_from_log(model_dir / "download.log")
    if done is not None:
        def present(f: str) -> bool:
            return f in done
        presence_source = "download log"
    else:
        def present(f: str) -> bool:
            return (model_dir / f).is_file()
        presence_source = "per-file stat (log absent)"

    shard_avail = [i for i, h in enumerate(shard_hosts) if present(h)]
    config_avail = [c for c in config_hosts if present(reader.weight_map[c])]
    return {
        "presence_source": presence_source,
        "ngram_shards_available": len(shard_avail),
        "ngram_shards": 128,
        "shard_ids_available": shard_avail,
        "configs_available": [c.split(".")[-1] for c in config_avail],
        "configs": 3,
        "distinct_checkpoint_files": len(set(shard_hosts) | config_file_set),
        "ready": len(shard_avail) == 128 and len(config_avail) == 3,
    }


def _quantize_and_encode(bf16: torch.Tensor, device: torch.device) -> bytes:
    w = bf16.to(device)
    qz = quantize_fp8_row(w)
    codes = qz.code_words.cpu()
    scales = qz.row_scales.cpu()
    return encode_fp8_row_scaled(codes, scales, tuple(bf16.shape))


def _preflight_convert(model_dir: Path) -> tuple[ShardReader, tuple[NS.ConfigTensor, ...], bytes]:
    reader = ShardReader(model_dir)
    # Presence via the download log (one small read) -- never stat 128 files
    # individually on the slow SMB drive: each .is_file() is a metadata
    # round-trip that can take seconds, and 128 of them hung the preflight
    # for minutes with no progress (2026-09-11). Same fast check as _readiness.
    done = _done_files_from_log(model_dir / "download.log")
    if done is not None:
        present = lambda f: f in done
    else:
        present = lambda f: (model_dir / f).is_file()
    missing_shard = [i for i in range(128) if not present(reader.weight_map[shard_name(i)])]
    if missing_shard:
        raise FileNotFoundError(
            f"{len(missing_shard)} ngram shard host files missing (e.g. shard {missing_shard[0]} -> "
            f"{reader.weight_map[shard_name(missing_shard[0])]}); the download must finish first"
        )
    configs = _read_configs(reader)
    directory = NS.build_directory(inventory.MODEL_ID, NS.QUANT_ID, configs)
    return reader, configs, directory


def convert(model_dir: str | Path, out_path: str | Path, *, device="cuda", overwrite=False) -> Path:
    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(f"output basename must be {OUTPUT_BASENAME!r}")
    resolved = pick_device(device)
    reader, configs, directory = _preflight_convert(model)

    geo = NS.shard_geometry()
    print(
        f"ngram preflight: 128 shards x ({geo.n}x{geo.k}) -> {geo.payload_bytes} B/shard, "
        f"total {128 * geo.payload_bytes / 1e9:.4f} GB, quant={NS.QUANT_ID}, device={resolved}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes: dict[int, str] = {}
    t0 = time.perf_counter()
    with NS.NgramWriter(output, directory, n_shards=128, overwrite=overwrite) as writer:
        if writer.completed:
            print(f"resuming: {writer.completed}/128 shards already complete", flush=True)
        for sid in range(writer.completed, 128):
            bf16 = reader.get(shard_name(sid)).to(torch.bfloat16)
            payload = _quantize_and_encode(bf16, resolved)
            del bf16
            hashes[sid] = hashlib.sha256(payload).hexdigest()
            writer.write_shard(sid, payload)
            del payload
            dt = time.perf_counter() - t0
            print(f"[{sid + 1}/128] shard {sid}  ({dt / max(1, sid + 1 - writer.completed):.1f}s/shard avg)", flush=True)

    elapsed = time.perf_counter() - started
    report = NS.write_checksums_sidecar(output, directory, hashes)
    report_path = Path(str(output) + ".checksums.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"complete: {output.stat().st_size} bytes in {elapsed / 60:.1f} min; "
        f"checksums={report_path}", flush=True,
    )
    return report_path


def smoke(model_dir: str | Path, n_shards: int, rows: int, out_path: str | Path, *, device="cuda") -> Path:
    """Convert the first ``n_shards`` available shards (``rows`` each) to a
    throwaway store and verify the real-data round-trip."""

    model = Path(model_dir)
    resolved = pick_device(device)
    reader = ShardReader(model)
    configs = _read_configs(reader)
    done = _done_files_from_log(model / "download.log")

    avail = []
    for i in range(128):
        host = reader.weight_map[shard_name(i)]
        ok = (host in done) if done is not None else (model / host).is_file()
        if ok:
            avail.append(i)
        if len(avail) >= n_shards:
            break
    if len(avail) < n_shards:
        raise FileNotFoundError(f"only {len(avail)} shards available; need {n_shards}")

    out = Path(out_path)
    directory = NS.build_directory(inventory.MODEL_ID, NS.QUANT_ID, configs,
                                   rows=rows, cols=cfg.NGRAM_SHARD[1], n_shards=n_shards)
    out.parent.mkdir(parents=True, exist_ok=True)
    with NS.NgramWriter(out, directory, n_shards=n_shards, overwrite=True) as writer:
        for k, sid in enumerate(avail):
            with safe_open(str(model / reader.weight_map[shard_name(sid)]), framework="pt", device="cpu") as h:
                slice_t = h.get_slice(shard_name(sid))
                bf16 = slice_t[0:rows, :].to(torch.bfloat16)
            payload = _quantize_and_encode(bf16, resolved)
            writer.write_shard(k, payload)

    # verify real-data round-trip: dequant the store, compare to the source slice
    src = {}
    with NS.NgramReader(out) as r:
        nblocks = -(-rows // NS.BLOCK_ROWS)
        for k, sid in enumerate(avail):
            with safe_open(str(model / reader.weight_map[shard_name(sid)]), framework="pt", device="cpu") as h:
                bf16 = h.get_slice(shard_name(sid))[0:rows, :].to(torch.bfloat16)
            src[k] = bf16
            expect = dequantize_fp8_row(quantize_fp8_row(bf16))
            for b in range(nblocks):
                lo, hi = b * NS.BLOCK_ROWS, min((b + 1) * NS.BLOCK_ROWS, rows)
                got = r.dequant_block(k, b)
                assert torch.equal(got, expect[lo:hi]), f"smoke dequant mismatch shard {sid} block {b}"
        # report the real-data FP8 error (should match the probe's ~2.6% RMS)
        rel = []
        for k in range(n_shards):
            dq = dequantize_fp8_row(quantize_fp8_row(src[k]))
            w = src[k].float()
            denom = w.abs().clamp_min(1e-6)
            e = (dq - w).abs() / denom
            rel.append((e.pow(2).mean().sqrt(), e.max()))
    rms = torch.stack([t[0] for t in rel]).mean()
    mx = torch.stack([t[1] for t in rel]).max()
    print(
        f"smoke: {n_shards} real shard(s) {avail} x {rows} rows -> {out.stat().st_size} bytes; "
        f"FP8 RMS rel err {float(rms):.5f}, max {float(mx):.5f} (probe: RMS 0.0264, max 0.0857)",
        flush=True,
    )
    return out


def verify(model_dir: str | Path, out_path: str | Path) -> None:
    out = Path(out_path)
    directory, shards = NS.validate_ngram_file(out)
    sidecar = Path(str(out) + ".checksums.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"checksum sidecar missing: {sidecar}")
    report = json.loads(sidecar.read_text(encoding="utf-8"))
    if NS._sha256(directory_bytes(out)) != report["directory_sha256"]:
        raise RuntimeError("directory sha256 mismatch vs sidecar")
    if NS._sha256_file(out) != report["whole_file_sha256"]:
        raise RuntimeError("whole-file sha256 mismatch vs sidecar")
    with NS.NgramReader(out) as r:
        by_id = {e["id"]: e["sha256"] for e in report["shards"]}
        for s in shards:
            assert r.verify_shard_checksum(s.id, by_id[s.id]), f"shard {s.id} checksum mismatch"
        # dequant a sample block to prove the store is readable
        sample = r.dequant_block(0, 0)
        assert sample.shape == (NS.BLOCK_ROWS, directory["cols_per_shard"])
    print(
        f"verify OK: {len(shards)} shards, all SHA-256 match, whole-file + directory match; "
        f"sample block {tuple(sample.shape)} dequantized",
        flush=True,
    )


def directory_bytes(out: Path) -> bytes:
    # parse_directory re-serializes; we need the exact stored directory bytes, so
    # read them straight from the prefix (magic + json_bytes length).
    with out.open("rb") as fh:
        head = fh.read(NS.PREFIX_BYTES)
        _, json_bytes = NS.PREFIX.unpack(head)
        return fh.read(json_bytes)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("qwen3_8_flash_next.ngram"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="readiness report (no weight reads)")
    p.add_argument("--smoke", type=int, metavar="N", help="convert N available shards (rows slice) and verify")
    p.add_argument("--rows", type=int, default=4096, help="rows per shard for --smoke")
    p.add_argument("--convert", action="store_true", help="full 128-shard production run")
    p.add_argument("--verify", action="store_true", help="verify a finished store + sidecar")
    args = p.parse_args(argv)

    if args.dry_run:
        print(json.dumps(_readiness(args.model), indent=2))
        return
    if args.smoke:
        smoke(args.model, args.smoke, args.rows, args.out, device=args.device)
        return
    if args.verify:
        verify(args.model, args.out)
        return
    if args.convert:
        convert(args.model, args.out, device=args.device, overwrite=args.overwrite)
        return
    p.error("one of --dry-run / --smoke N / --convert / --verify is required")


if __name__ == "__main__":
    main()
