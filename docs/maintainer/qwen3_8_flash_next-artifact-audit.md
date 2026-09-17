# Qwen3.8-Flash-Next NVFP4 — artifact audit (P0)

Header-only audit of the converted artifact. No payload was decoded and no
bytes were moved to the GPU; every number below comes from the v2 container
directory (mmap metadata) or the canonical inventory
(`tools/convert/qwen3_8_flash_next/inventory_nvfp4.py`, which validates itself
at import).

**Identity guard (spec STOP condition): PASS.** The artifact identity is
`qwen3.8-flash-next / nvfp4` — not a registered `qwen3.8-27b/*` or
`qwen3.6-*` key. The 27B registry was not touched.

## 1. Container facts (measured, `tools/artifact/inspect.py`)

| field | value |
|---|---|
| path | `models/qwen3_8_flash_next.ninfer` (env: `NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS`) |
| model_id / weights_id | `qwen3.8-flash-next` / `nvfp4` |
| file bytes | **75,397,717,504** (75.398 GB) |
| payload offset | 270,336 (4,096 prefix + 266,240 B JSON directory) |
| objects | **1,608** = 1,602 tensors + 6 resources |
| tensor payload | 75,384,498,624 B |
| resource payload | 12,948,544 B (6 `frontend/*.json/jinja`, `raw-bytes-v1`) |
| companion store | `models/qwen3_8_flash_next.ngram` — 51,840,558,936 B (128 PLE shards + 3 I64 config tables; verified by the `.ngram` workstream) |

Raw inspect output: `out/flash_next_dev/inspect_real.json`,
`out/flash_next_dev/inspect_real_objects.txt` (1,608 object rows).

## 2. Format / layout census (locked, matches `validate_inventory()`)

| format | count | layout | count |
|---|---|---|---|
| BF16 | 901 | contiguous-le-v1 | 1,169 |
| FP32 | 268 | row-split-k128-v1 | 111 |
| Q4G64_F16S | 54 | blockscale-k16-m128x4-v1 | 196 |
| Q5G64_F16S | 54 | row-scale-v1 | 126 |
| Q6G64_F16S | 1 | | |
| W8G32_F16S | 2 | | |
| NVFP4 | 196 | | |
| FP8_E4M3FN_ROW_BF16S | 126 | | |

## 3. Sections

| section | objects | payload bytes |
|---|---|---|
| `text/` (embedding + 48 layers + output head) | 1,235 | 73,561,452,544 (73.561 GB) |
| `mtp/` (1 full-attn + MoE layer) | 34 | 1,539,873,312 (1.540 GB) |
| `vision/` (shared Qwen3.6 ViT) | 333 | 283,172,768 (283 MB) |
| `frontend/` resources | 6 | 12,948,544 (12.9 MB) |

MTP and Vision are **stored but out of execution scope until P13+** (spec §2);
they are inventoried and must remain byte-stable, not executed.

## 4. Per-layer template (one GDN layer, e.g. `text/layers/0/`)

| object | shape | format |
|---|---|---|
| `attn_hyper_connection/block_inject_weight` | (4, 10240) | BF16 |
| `attn_hyper_connection/hc_norm` | (10240,) | BF16 |
| `attn_hyper_connection/input_mix_weight_down` | (320, 10240) | BF16 |
| `attn_hyper_connection/input_mix_weight_up` | (10240, 320) | BF16 |
| `gdn/a_log` | (48,) | FP32 |
| `gdn/dt_bias` | (48,) | FP32 |
| `gdn/convolution` | (4, 10240) | BF16 |
| `gdn/a_b_projection` | (96, 2560) | BF16 |
| `gdn/query_key_value_z` | (16384, 2560) | FP8 |
| `gdn/norm` | (128,) | BF16 |
| `gdn/output` | (2560, 6144) | FP8 |
| `mlp_hyper_connection/*` (4 objects) | as attn-HC | BF16 |
| `mlp/experts/gate_up` | (655,360, 2560) = 512×1280 | **NVFP4** |
| `mlp/experts/gate_up_projection/input_scale_divisor` | () | FP32 |
| `mlp/experts/down` | (1,310,720, 640) = 512×2560 | **NVFP4** |
| `mlp/experts/down_projection/input_scale_divisor` | () | FP32 |
| `mlp/experts/routing` | (512, 2560) | BF16 |
| `mlp/shared_expert/gate_up` | (1280, 2560) | NVFP4 |
| `mlp/shared_expert/down` | (2560, 640) | NVFP4 |
| `mlp/shared_expert/*_projection/input_scale_divisor` (2) | () | FP32 |
| `mlp/shared_expert_gate` | (1, 2560) | BF16 |

QSA layers (12, `L % 4 == 3`) replace `gdn/*` with:

| object | shape | format |
|---|---|---|
| `attention/query` | (12288, 2560) = 24×512 | FP8 |
| `attention/key`, `attention/value` | (512, 2560) = 2×256 | FP8 |
| `attention/output` | (2560, 6144) = 24×256 | FP8 |
| `attention/query_norm`, `attention/key_norm` | (256,) | BF16 |
| `attention/indexer/qk_proj` | (640, 2560) = (4Q+1K)×128 | BF16 |
| `attention/indexer/k_norm`, `attention/indexer/q_norm` | (128,) | BF16 |

Layer 1 only (PLE projections; the n-gram table itself is in the `.ngram`
store): `ple/convolution` (4, 10240), `ple/key_projection` (10240, 2560),
`ple/value_projection` (2560, 2560), `ple/norm_conv|key|query` (10240,) — all
BF16.

Non-layer objects: `text/token_embedding` (248,320, 2560) FP8,
`text/hyper_connection_mixer/*` (3, BF16), `text/output_head`
(248,320, 2560) FP8, `mtp/*` (34), `vision/*` (333, shared builder).

The full 1,608-row table (name, shape, format, layout, encoded nbytes,
proposed residency) is generated, not hand-maintained:
`python -m tools.flash_next_dev.audit_inventory` →
`out/flash_next_dev/audit_objects.csv`.

## 5. Proposed residency classes (input for P11)

| class | objects | bytes | notes |
|---|---|---|---|
| `gpu_resident` | 1,504 | 6,021,195,832 (6.021 GB) | backbone, shared experts, norms, HC, router, indexers, MTP, Vision, scalar scales |
| `host_experts` | 98 | 69,363,302,792 (69.363 GB) | fused 512-expert NVFP4 `gate_up`/`down`, 48 text layers + 1 MTP layer (96 + 2) |
| `mmap_ple` | (not in `.ninfer`) | 51,840,558,936 B (file) | 128 PLE shards 2,500,012×160 FP8 + 3 I64 tables, in `qwen3_8_flash_next.ngram`; page-cache-backed, never GPU |

## 6. Deltas vs the spec §2 locked numbers

| item | spec §2 | artifact / inventory | verdict |
|---|---|---|---|
| hidden / vocab / layers | 2560 / 248320 / 48 | same (`config.py`) | match |
| max position | 262144 | asserted by `validate_config` (checkpoint-side) | match (runtime param) |
| layer split | 12×(3 GDN + 1 QSA) | 36 GDN + 12 QSA (`L%4==3`), tests lock it | match |
| GDN heads / dim / conv | 16 key, 48 value, 128, conv 4, FP32 state | 16×128 / 48×128, conv (4, 10240), `a_log`/`dt_bias` FP32 | match (state is runtime) |
| QSA | 24Q / 2KV / head 256, partial rotary 0.25 | query (12288, 2560), k/v (512, 2560), head_dim 256 | match (rotary ratio is runtime) |
| indexer | MQA 4Q+1K, head 128, compress 4, budget 2048 | `qk_proj` (640, 2560), norms (128,) | match (compress/budget runtime) |
| MoE | 512 routed, top-10, 1 shared, int 640, norm_topk | 512 fused NVFP4, router (512, 2560), shared (1280/640) | match (top-k + norm_topk runtime) |
| HC | count 4, lowrank 320, separate attn+mlp per layer | HC_STREAM 10240, lowrank 320, two HC blocks per layer | match |
| PLE | layer 1, 8 bigram + 8 trigram, base 20,000,000 /128 | layer 1 only; 16 heads (8+8, trigram order 3), 128 shards × 2,500,012×160 | match |
| **expert payload size** | **~58 GB (estimate)** | **69.363 GB measured (NVFP4 incl. block scales, incl. MTP)** | **DELTA — P11 placement + RAM policy must use 69.4 GB** |
| MTP / Vision | out of scope until P13+ | present (34 + 333 objects) | stored, not executed — matches scope |
| activation scales | — | 196 `input_scale_divisor` scalars = documented placeholder (2688.0) until the runtime exists | must be recalibrated in P8+ |

## 7. Executability (honesty constraint)

The artifact is **not runnable**: the runtime kernels for the Flash-Next
operations (expert paging, PLE gather, gated residual / HC, GDN at this
geometry, QSA indexer + sparse GQA, sparse MoE 512×top-10) do not exist yet.
`tools/convert/qwen3_8_flash_next/verify_nvfp4.py` reports
`runtime_ready = false` with the per-operation
**BLOCKER: RUNTIME KERNEL REQUIRED** list. This audit proves the *stored*
artifact is structurally correct; it proves nothing about execution.

## 8. P0 verification

- `tests/targets/qwen3_8_flash_next/test_p0_inspect_fake_fixture.py` — inspect
  parser on the checked-in 4-tensor fake header (CLI `--json` and `--objects`).
- `tests/targets/qwen3_8_flash_next/test_p0_inventory_contract.py` — canonical
  inventory lists every logical parent (embed, 48×(GDN|QSA, HC×2, router,
  shared, 512-expert fused), PLE only in layer 1, lm_head, norms); counts
  locked at (1235, 34, 333, 1602, 1608).
- `tests/targets/qwen3_8_flash_next/test_p0_real_artifact.py` — real file:
  identity guard, 1,608 objects, format census; header-only (mmap metadata,
  no H2D). Skips unless `NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS` is set.
- Runner: `bash tools/flash_next_dev/run_phase.sh P0` (13/13 PASS on 2026-09-11
  with the env set).
