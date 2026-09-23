# Flash-Next acceleration — checkpoint

Status:
- Phase 1 (decode expert residency) code is DONE and consistent:
  - M4 (GPU LRU) slot budget: headroom 2 GiB -> 512 MiB, so a 32 GB card with
    ~1.9 GiB free gets a ~530-slot pool (>= the 480-blob decode working set).
  - M4 miss fast-path gated to T==1 (pin_ok at real_program.cpp:1546/1586).
  - M2 (pinned) gated to T==1 (1690).
- S5 analysis done: the gpu_fits (M4) divergence is PRE-EXISTING (the M1
  baseline stages the host top-10 union; the op re-runs routing on device and
  re-reads device top-10; a boundary flip -> device reads an un-staged slot).
  It is LATENT in the M1 baseline too, not a new class. VERIFY self-check
  (NINFER_EXPERT_GPU_VERIFY) is wired to localize it when the model runs.
- Build env in flux: the core ninfer-win tree moved from
  C:\Users\Micke\Documents\Ninfer\ninfer-win to
  C:\Users\Micke\Documents\Kodprojekt\Ninfer\ninfer-win. The old .bat build
  scripts reference the stale path.

Next action:
- Build the target against the relocated core tree
  (C:\Users\Micke\Documents\Kodprojekt\Ninfer\ninfer-win, NOT the old
  Documents\Ninfer path the .bat scripts hard-code) and run `ctest -L
  flash_next` (fixtures only). Then verify decode residency on the real model.
  Blocker: real-model verification (S5 + speed) requires the 75 GB artifact —
  NOT allowed this session (would crash the session).

Key facts:
- kExpertBytes = 2,764,800 B; 4 planes (gu_codes/gu_scales/dn_codes/dn_scales).
- Decode working set = 480 blobs (10 experts/layer x 48 layers) = 1.33 GiB/round.
- Pinned hit ~105 us (~26 GB/s) vs pageable ~0.92 ms/expert (cold, page faults).
- moe_expert_h2d = 86.65% of wall (P13); GPU-idle 0.16%.
- Core ninfer-win tree: C:\Users\Micke\Documents\Kodprojekt\Ninfer\ninfer-win
  (NOT the old Documents\Ninfer path the .bat scripts hard-code).
- NEVER load the 75 GB .ninfer model (crashes the session).

Pointers:
- docs/superpowers/specs/2026-09-23-flash-next-moe-residency-design.md
- out/flash_next_dev/P13_timing_report.md
- src/targets/qwen3_8_flash_next/impl/runtime/real_program.cpp (moe_block 1467-1820;
  expert_gpu_slot_count 464-499)
- src/targets/qwen3_8_flash_next/impl/runtime/real_program.h (ExpertGpuLru 161-247)
- C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\serve.ps1 (Unsloth reference)
