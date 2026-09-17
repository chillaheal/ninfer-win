# Qwen3.8 Flash-Next — 32 GB placement

Flash-Next is the 27B MoE family that runs on one 32 GB RTX 5090 by keeping only a
small expert window resident on the GPU and streaming the rest of the 512 experts
from host pinned memory. Because most of the model is resident off-GPU, the load
is often **host-RAM-bound rather than VRAM-bound**: the planner must predict
which dimension a given launch overflows *before* the GPU ever sees the real
file.

## Placement dry-run

`build/tests/flash_next_placement` plans a load from inventoried byte counts and
the launch options, runs the planner, and prints the placement table. It never
touches a device and never loads the artifact, so the same inputs always print
the same table. Exit codes: `0` = FIT, `3` = OVERFLOW (the planner named the
budget that exceeded), `2` = bad usage.

```bash
./build/tests/flash_next_placement \
  --max-context 32768 --max-concurrency 1 --kv-dtype fp8 \
  --backbone-bytes 6021195832 \
  --expert-host-bytes 69363302792 \
  --ple-bytes 51840558936 \
  --expert-window-bytes 2831155216 \
  --workspace-bytes 0 \
  --gdn-state-bytes 113246208 \
  --device-budget-bytes 34359738368 \
  --system-ram-bytes 68384739328
```

The byte counts come from the artifact inventory
(`docs/maintainer/qwen3_8_flash_next-artifact-audit.md`):

| Field | Bytes | Meaning |
|---|---:|---|
| `--backbone-bytes` | `6021195832` | GPU-resident (non-expert) weights |
| `--expert-host-bytes` | `69363302792` | all experts, host pinned |
| `--ple-bytes` | `51840558936` | PLE `.ngram`, mmap (page-cache, never counted) |
| `--expert-window-bytes` | `2831155216` | `2` layers × `1415577608` per-layer expert payload |
| `--gdn-state-bytes` | `113246208` | one device state slot (36 GDN layers × conv+SSM) |
| `--device-budget-bytes` | `34359738368` | 32 GiB |
| `--system-ram-bytes` | `68384739328` | physical RAM on this machine |

KV bytes per token are taken from the production constants (FP8 E4M3 row-256 =
`12288`, BF16 = `24576`), selected by `--kv-dtype`, so the tool and the runtime
share the same numbers.

## Recommended flags for this machine

The intended single-request configuration on a 32 GB card:

```
--max-context 32768 --kv-capacity 32768 --kv-dtype fp8
--max-concurrency 1 --device-state-slots 1 --host-state-slots 4
--host-kv-mib 8192
--expert-device-window 2 --expert-host-all --ple-mmap
```

This is the **GPU** dimension the planner must pass, and it does:

| Budget | Used | Available | Verdict |
|---|---:|---:|---|
| GPU VRAM | `9,368,250,440` B (8.72 GiB) | `34,359,738,368` B (32 GiB) | FIT |
| Host pinned | `69,363,302,792` B (64.55 GiB) | `68,384,739,328` B (63.67 GiB) | **OVERFLOW** |

## Honest finding: this machine is host-RAM-bound

The NVFP4 expert payload (`69,363,302,792` B ≈ 64.55 GiB) **exceeds this
machine's physical RAM** (`68,384,739,328` B ≈ 63.67 GiB) by ~0.91 GiB.
`--expert-host-all` therefore overflows the host pinned budget here, even though
the GPU side fits comfortably. The binding constraint on this machine is **host
RAM, not GPU VRAM**, and the PLE stays out of the count only because it is mmap'd
(page-cache, evictable) — without `--ple-mmap` the PLE alone (48.27 GiB) would
add another 48 GiB to the same pinned budget.

Consequences:

- The planner's recommended-flags test is **honest**, not green-by-assertion: the
  GPU check PASSES and the host check FAILS LOUD (it names the overflowing
  `host_pinned` budget and the required/available byte counts).
- This predicts the first real-weight load will OOM on **host pinned RAM**
  (expert host-pin), not GPU VRAM.
- A machine with ≥ ~66 GiB of usable RAM clears the host dimension for these
  flags; on this machine the expert window / host split would need to shrink, or
  the expert payload must be reduced, before `--expert-host-all` fits.

The planner is deterministic byte arithmetic with no device access, so these
verdicts hold on any machine given the same two inputs (`--device-budget-bytes`,
`--system-ram-bytes`).
