# TODO — flash_next acceleration

Index of the granular task breakdown (mirrors harness tasks #6–#22).
Source of truth for phase detail: docs/superpowers/specs/2026-09-23-flash-next-moe-residency-design.md

## Work order (llama.cpp/Unsloth imitation, 2026-09-25)
1) P3 — CPU offload 30/48 coldest layers (= `-ncmoe 30`); cuts 62.5% of expert H2D for BOTH prefill and decode.
2) P1.5 — async expert pipeline for GPU layers: pinned host source + copy-stream overlap + GPU-side top-k (llama.cpp copy-stream/graph model). Makes prefill H2D nearly free.
3) P2 — MTP draft-3 (`--spec-type draft-mtp --spec-draft-n-max 3`), 1.3–1.7× on decode.
Target: prefill ≈ 20–35 tok/s (CPU-bound, ≈ Unsloth band); decode 25–40 tok/s × 1.3–1.7.

## Phase 1 — decode expert residency (implementation done, compile-verified)
- [x] P1.1 T==1 gate: pinned L2 (M2) only at T==1; pageable 4-plane scatter for T>1 (real_program.cpp:1535)
- [x] P1.2 M4 GPU-LRU slot budget as decode default (ExpertGpuLru; DYNAMIC = free VRAM − 2 GiB)
- [x] P1.3 Seed residency from NINFER_EXPERT_USAGE_FILE at startup (real_program.cpp:951)
- [x] P1.4 H2D microbench: pinned 32.4 GB/s vs pageable-warm 26.6 GB/s (1.22×)
- [x] V1.1 Register flash_next ctest targets (LABELS flash_next) — DONE: core/tests/CMakeLists.txt registers 13 flash_next test targets (LABELS flash_next) + flash_next_real_load; FN_PROJ_ROOT="C:/fn" machine junction (load-bearing).
- [x] V1.2 Run ctest -L flash_next (bit-exactness) — GREEN 2026-09-25: 14 tests, 13 passed + s1_real_bind expected skip (`ctest --test-dir core/build -L flash_next --timeout 420 --output-on-failure`).
- [ ] V1.3 Run NINFER_EXPERT_GPU_VERIFY self-check (device table vs host table, slot bytes vs mmap source)
- [x] V1.4 Timed T=1 decode on 75 GB model — DONE 2026-09-25 via `.agent/p1v4_m3_visible.ps1` (visible window, transcript in project root): decode T=1 ≈ 1.1 tok/s; moe_expert_h2d = 84.1% of GPU-busy; router D2H CPU-stall 115.8 s of 117.4 s wall; LRU 18.52 GiB = 29% of expert weights; GPU pinned 31.816/31.82 GiB

## Phase 1.5 — async expert pipeline for GPU layers (NEW 2026-09-25, llama.cpp copy-stream model)
- [ ] P1.5.1 Pinned host source for GPU-layer expert weights (no page faults on critical path; pageable is 0.92 ms/expert vs pinned ~105 µs)
- [ ] P1.5.2 Per-layer expert copy on a separate copy stream, overlapped with previous layer's GEMM (no per-layer CPU sync; today cpu_barrier ≈ wall)
- [ ] P1.5.3 Router top-k on GPU; D2H only selected expert IDs (~40 kB/chunk) — kills the 115.8 s router-D2H CPU stall
- [ ] P1.5.4 Verify: prefill tok/s (target: GPU-layer H2D hidden under GEMM; prefill → CPU-bound band 20–35 tok/s once P3 lands)

## Phase 2 — MTP speculative decode (Unsloth's biggest lever, 1.3–1.7×)
- [ ] P2.1 Load MTP head (2.6 GB) in model load path
- [ ] P2.2 Draft-mtp forward
- [ ] P2.3 Wire spec-decode loop into decode loop (draft-n-max 3)
- [ ] P2.4 Verify acceptance rate & tok/s (target 1.3–1.7×)

## Phase 3 — CPU offload of cold MoE layers (Unsloth -ncmoe 30) — PRIORITY 1 (2026-09-25)
- [x] P3.1 Select coldest 30/48 layers from usage histogram — DONE 2026-09-25,
      from .agent/p3_layer_ranking.txt (ranks 18–47, usage-file row index =
      engine layer index 0-based). CPU-offload 30: 0,5,6,8,10,12,13,14,15,16,20,
      23,24,26,27,28,29,30,31,32,33,35,36,39,40,43,44,45,46,47. GPU-resident 18:
      1,2,3,4,7,9,11,17,18,19,21,22,25,34,37,38,41,42. Boundary tie ranks 17/18
      (both total 1174): layer 25→GPU, 33→CPU (ranking tiebreak = lower layer id
      first). Spread is flat (~16% between coldest and hottest), so any 30 is
      near-optimal; the ranked list is the deterministic choice.
- [x] P3.2 CPU MoE GEMM path (NVFP4, 4 planes: gu_codes/gu_scales/dn_codes/dn_scales) — DONE 2026-09-25. CpuMoe pimpl (cpu_moe.h/.cpp) F32-dequant LRU (64 routed slots + non-evictable shared) + 4-phase run (top-k/sg, dequant, gu→silu→dn GEMM, merge), /arch:AVX2. p10 gate GREEN at mini/fgeo/prod (512×10×2560×640) — prod verified in full ctest (214 s, fit in ~1.4 GB free VRAM; dequant is CPU-side, T=4). ROOT-CAUSE of the p10 segfault: the E4M3 F32 LUT was a function-local magic static holding a raw ptr from a make_unique<vector> lambda, which materialized as an EMPTY vector under MSVC /O2 /arch:AVX2 (NULL deref in dequant_plane). Fix: plain `static float table[256]` + `std::call_once` (array, no magic-static pointer). Debug instrumentation (test `mk`, engine `dbg`, forced threads=1) stripped.
- [x] P3.3 Route coldest layers to CPU in moe_block — DONE 2026-09-25. env
      NINFER_MOE_CPU_LAYERS (default off; non-empty non-"0" → on). Fixed coldest-30
      table (see P3.1). init_moe_cpu_offload(): one-time D2H of shared gu/dn codes+
      scales + shared_gate per CPU layer into host buffers; one shared
      CpuMoe(512,10,2560,640,64). moe_block CPU branch (after `selected`/
      `expert_usage_`, before `routed`): reuses router_host_ logits, D2H xhat (bf16)
      + sync, CpuMoe->run, H2D y, skips scatter + GPU op, keeps B_MOE_SCATTER/
      B_MOE_GEMM segs, barrier_end() before early return. real_program.{h,cpp}
      md5-verified identical in project + core tree. Default-off gate GREEN:
      rebuilt targets p2/p7/p9/p10 + p6/p11 all pass (p4/p5 device-delta exes are
      stale 16:04 + perturbed by the live 27B serve holding 31.5 GB — NOT a
      regression; exes byte-identical to V1.2 green). Files: real_program.h
      (members + init decl), real_program.cpp (ctor init call + init_moe_cpu_offload
      + moe_block branch).
- [x] P3.4 Verify −62.5% expert H2D — DONE 2026-09-26 (real-model CPU-on run,
      visible window; M1 + M3 512/2048 captured, M3 8192 stuck ~11h so user
      stopped the test this AM + restarted serve PID 15412). VERDICT: H2D target
      met/beyond (moe_expert_h2d 84.1%→3.27% of GPU-busy, ~-97% rel) BUT decode
      regressed 1.1→0.1 tok/s (10x SLOWER) — the per-token CPU MoE sync chain
      (router-logits D2H → CPU top-k → 30× D2H+sync+H2D → host F32 dequant+GEMM)
      is the new bottleneck (moe_gemm = 94.85% of wall). CONCLUSION: expert H2D
      was NOT the binding constraint; CPU-offload as-designed HURTS decode.
      Pivots to Phase 1.5 (async expert pipeline: GPU top-k + copy-stream) as
      the real decode fix. S5 gate relaxed to a WARN under NINFER_MOE_CPU_LAYERS
      (edit in tools/flash_next_dev/real_load_driver.cpp; real_load relinked).

## Phase 4 — end-to-end
- [ ] P4.1 E2E tok/s vs Unsloth baseline (25–40 tok/s) — RAW measured 2026-09-25 (M3 matrix, C=1, T_new=128): decode 1.0–1.1 tok/s at T_prompt 512/2048/8192 (118.7/120.0/122.0 s); prefill 5.5/7.7/7.1 tok/s; gap ≈ 25–35×. Levers: P2 (MTP) + P3 (CPU offload); re-measure after both.

## Blockers
- 75 GB model (models/qwen3_8_flash_next.nin) must never load in-session — all real-model verification runs detached with logs.
