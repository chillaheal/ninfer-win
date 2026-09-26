# Flash-Next MoE expert-residency — acceleration design

Date: 2026-09-23
Status: design approved by user (autonomous mode, "fria händer ... inga frågor ... kör självständigt")

## Context

ninfer's `qwen3_8_flash_next` (Qwen3.8-Flash-Next, 176.94B MoE + 51B n-gram PLE,
6B active params/token) runs on a single RTX 5090 32 GB but is very slow:
decode ~1.2 tok/s, prefill ~5.7–6.8 tok/s. The P13 timing breakdown
(`out/flash_next_dev/P13_timing_report.md`) pins the bottleneck:

- `moe_expert_h2d` = **86.65% of GPU-busy / 86.51% of wall** (106.1 s of
  122.7 s wall).
- Per-token decode (T=1): 48 layers × top-10 = 480 expert H2D copies ×
  2,764,800 B = 1.33 GB/token.
- Effective pageable H2D = 0.77–2.86 GB/s (64.55 GiB payload > 63.67 GiB
  physical RAM → cold pages from SSD; round 1 0.77 GB/s, warm round 2
  12.34 GB/s).

Unsloth's llama.cpp runs the same model at 25–40 tok/s. The difference is the
experts' placement, not the model.

## Comparison: Unsloth vs ninfer

| Dimension | Unsloth (llama.cpp) | ninfer (us) |
|---|---|---|
| Quant | UD-IQ3_XXS GGUF (3 shards) | NVFP4 (4-bit codes + E4M3 scales, block-scale k16) |
| Expert placement | `-ncmoe 30`: 30/48 MoE layers' experts computed on CPU; only 18/48 cross PCIe | All 48 layers' experts staged H2D per round from a pageable mmap |
| n-gram PLE | SSD via mmap | SSD via mmap (51B, ~26–30 GB Q4) — matches the target |
| MTP | `--spec-type draft-mtp --spec-draft-n-max 3` (1.3–1.7×) | Not implemented (only an MTP *weight* layer, no speculative decode) |
| Flash attention | `-fa 1` | has flash attention |
| Bottleneck | — | `moe_expert_h2d` = 86.65% of wall |

**The core difference:** Unsloth computes 30/48 MoE layers on CPU, so 62.5% of
the expert weights never cross PCIe. ninfer stages all 48 layers' experts H2D
every round. That is the single dominant cost.

ninfer already has the same *placement hierarchy* the user described
("ladda experter till vram, mindre heta experter till ram, ngram till ssd"):

| Tier | Residency | Status in ninfer |
|---|---|---|
| Hot experts | VRAM | `ExpertGpuLru` (M4), default DYNAMIC (free VRAM − 2 GiB) |
| Warm experts | RAM (pinned) | `ExpertPinnedLru` (M2), opt-in `NINFER_EXPERT_PINNED_SLOTS` (default 0 = off) |
| Cold experts + n-gram PLE | SSD mmap | mmap (69 GB experts, 51.8 GB n-gram) |

So the architecture the user described is already present. The gaps:
1. **M4 (GPU LRU) correctness** — an open S5 divergence (device top-K ≠ host
   top-K from FMA divergence → device reads an un-staged window slot). M4 is
   the tier-1 default-when-fits but unverified on the real model.
2. **M2 (pinned L2)** regressed on re-prefill and is off by default. It is the
   fast miss-path for M4 but needs a decode-only gate.
3. **MTP** is absent — Unsloth's single biggest lever (1.3–1.7×).

## Root cause (measured, P13)

`moe_expert_h2d` = 86.65% of GPU-busy / 86.51% of wall. The scatter is re-done
per layer AND per round; the only reusable residency is per-layer across
rounds. The pageable page faults are on the critical path (0.92 ms/expert);
page faults are on the critical path (0.92 ms/expert); pinned is ~105
µs/expert (~26 GB/s warm), ~4× faster.

## Design: three-tier expert residency (already implemented, tune + harden)

| Tier | Residency | Class | Knob |
|---|---|---|
| Hot experts | VRAM (device) | `ExpertGpuLru` (M4) | `NINFER_EXPERT_GPU_SLOTS` (default DYNAMIC) |
| Warm experts | RAM pinned | `ExpertPinnedLru` (M2) | `NINFER_EXPERT_PINNED_SLOTS` (default 0) |
| Cold experts + n-gram PLE | SSD mmap | mmap (always on) |

Per-(layer, expert) blob = 2,764,800 B, 4 planes (gu_codes/gu_scales/dn_codes/
dn_scales at kPlaneGuCodes/GuScales/DnCodes/DnScales). `kExpertBytes` =
2,764,800 B.

### Decode (T=1, stable routing) — the target workload
- The 480-blob working set (10 experts × 48 layers) is ~1.33 GB. A 2048-slot
  GPU LRU = 5.5 GB holds it with room. After warm-up, hits → zero-DMA
  in-place read.
- P13 (M2) confirms: "production decode (T=1 incremental) retains ~95%+ of the
  ~480-key union, where a pinned hit (~26 GB/s warm) is ~4× faster than
  pageable."

### What changes
1. **Decode mode (T=1): M4 + M2 as the decode default.** When T==1 and the
   union fits the GPU LRU, use M4 (GPU LRU) with M2 pinned as the miss fast-
   path. This makes decode H2D-free (hits = in-place read; misses = one
   2.76 MB pinned→device DMA at ~26 GB/s ≈ 105 µs).
2. **Prefill (T>1, churning):** keep the M4 LRU but skip the pinned miss-path
   when the union exceeds the pinned pool (a pinned miss at disk rate is
   slower than the driver's async pageable staging). Use the pageable
   4-plane scatter for prefill.
3. **Seed residency at startup** from a prior usage histogram
   (`NINFER_EXPERT_USAGE_FILE`) so the hot experts are resident before the
   first round (avoids a cold round 1).

### Why M2 (pinned) regressed on re-prefill and the gate
A pinned MISS pays ~700 soft page faults/expert (~2.4–2.7 ms, disk rate) —
SLOWER than the driver's async pageable staging (0.92 ms/expert). Per-round
top-10 membership churns ~65% even though the union only grows (router GEMM
tiling changes FMA ordering with T → boundary flips). So a pinned LRU chases a
moving target during prefill and loses to async pageable. The gate: use the
pinned tier only for T==1 (stable routing); use pageable for prefill.

## Implementation (phased)

### Phase 1 — decode residency (this work, model-free)
- **Gate the pinned tier to T==1.** In `moe_block`, use the pinned L2 only
  when T==1 (stable routing); use pageable 4-plane scatter for prefill (T>1).
  This avoids the re-prefill regression while giving decode the pinned fast
  miss-path.
- **Enable M4 (GPU LRU) as the decode default** when the union fits the slot
  count (already the default-when-fits logic). Verify the S5 path via the
   `NINFER_EXPERT_GPU_VERIFY` self-check + `ctest -L flash_next`.
- **Seed residency** at startup from `NINFER_EXPERT_USAGE_FILE` so round 1
   isn't cold.

### Phase 2 — MTP (later, needs the model + MTP head)
- `--spec-type draft-mtp --spec-draft-n-max 3` equivalent. Needs the MTP head
  (2.6 GB) + a draft forward. 1.3–1.7×.

### Phase 3 — CPU offload of cold MoE layers (Unsloth `-ncmoe 30`) (later)
- Compute the coldest 30/48 layers' experts on CPU; their weights never cross
  PCIe. Cuts expert H2D by 62.5%. Large effort (CPU MoE GEMM path).

## Verification (no 75 GB model)
- `ctest -L flash_next` (fixtures / mini artifacts) — bit-exactness.
- H2D microbenchmark (pinned vs pageable, hit vs miss).
- `NINFER_EXPERT_GPU_VERIFY` self-check (device table vs host table, slot
  bytes vs host mmap source).
- The 75 GB `.ninfer` model is NEVER loaded in this session.

## Out of scope (this session)
- MTP (Phase 2) — needs the 2.6 GB MTP head + a draft forward.
- CPU offload of cold MoE layers (Unsloth `-ncmoe 30`) — Phase 3.

## Verification (2026-09-24, model-free)

H2D microbenchmark (fln_h2d_bench.cu, RTX 5090, 2,764,800 B payload,
50 timed copies after 5 warmup):
- pinned (cudaHostAlloc):   85.4 us/iter   (32.4 GB/s)
- pageable (malloc, warm): 104.1 us/iter  (26.6 GB/s)
- pinned / pageable ratio: 1.22x

Interpretation: the P13 moe_expert_h2d 0.77-2.86 GB/s figure is the
SSD-mmap COLD page-fault rate (69 GB cold-expert file), NOT the pageable-RAM
transfer rate (warm-RAM pageable measures 26.6 GB/s here). M4 VRAM residency is
the dominant lever (it eliminates the H2D for hot experts); M2 pinned's marginal
RAM-transfer gain is ~1.2x - its larger value is avoiding the SSD-mmap fault.
