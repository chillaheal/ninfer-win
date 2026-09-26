Status:
P1.5 (async expert pipeline) acceleration arc COMPLETE + verified (2026-09-26).
  - P1.5.3 device-side top-k (kills router D2H stall): done, in the tree.
  - Expert residency 3-tier (GPU dynamic LRU / pinned RAM / mmap RAM-warm) +
    usage-file seeding (ninfer-expert-usage.txt): done.
  - NEW this turn: periodic usage flush — RealProgram::run_round now re-flushes
    expert_usage_ to NINFER_EXPERT_USAGE_FILE (default ninfer-expert-usage.txt)
    at most every 30 s (time-gated steady_clock; armed to now() in ctor so the
    first periodic flush never clobbers the seed file with an empty histogram).
    Closes the hard-kill gap: a Stop-Process'd serve previously skipped the dtor
    flush, so usage only persisted on a clean shutdown. Now "remember hot
    experts across restarts" is robust. Edits: real_program.h (last_usage_flush_
    member), real_program.cpp (ctor arm + run_round flush). VERIFIED: overlay
    synced project-root -> core/src (COPY attach — that was the earlier "ninja:
    no work to do" root cause), p1_rebuild_tests GREEN (real_program.cpp.obj
    compiled, 6 targets linked, 0 errors), full ctest -L flash_next GREEN 14/14
    (13 passed + s1_real_bind skip; 528 s; p8 260 s, p10 214 s).
  - P1.5.2 copy-stream: SKIPPED by data (per-layer GEMM is 1.7% of GPU; decode
    GPU already 99.5% busy -> no overlap to win).
Perf ceiling (real 177 GB MoE on 32 GB card, no MTP head):
  - decode ~1.0 tok/s = GPU-compute-bound (GPU-idle 0.07%): hardware ceiling.
  - prefill ~4.2 tok/s = warm-H2D-miss-bound; residency tiers + usage-file
    seeding address the COLD first request (benchmark self-warms via S5 16
    rounds, so a seeded benchmark shows no delta — p153 vs p154 identical).
Plan position:
Current step: Full build into project-root build/ DONE + verified (09:42-09:47, [580/580], exit 0); P1.5 arc closed; usage persistence robust to hard-kill.
Completed: P1 (all), P1V.1-4, V1.1, V1.2 (ctest green), P3.1-3.4, P1.5.3,
residency tiers + seeding, periodic usage flush (build + ctest green 2026-09-26).
Next step: ENDGAME — project sits at its documented hardware ceiling on this
box. Remaining open items are low-value or blocked: #4 V1.3
NINFER_EXPERT_GPU_VERIFY (diagnostic only, needs serve stop + real model),
#5 Phase 2 MTP (BLOCKED: no MTP head for this model). No further heavy runs
needed; acceleration work is done and verified.

Key facts:
- [2026-09-26] FULL BUILD to project-root build/ DONE + verified (09:42 -> 09:47, exit 0, [580/580], ~5 min; ninja reused prior objs + recompiled changed TUs, all links fresh 09:47). Reusable recipe: configure the RESOLVED core-tree root as the CMake source (-S), NOT the `core` junction — CMAKE_SOURCE_DIR must be the real core root because top-level CMake derives `${CMAKE_SOURCE_DIR}/../deps/*` (FFmpeg, CURL); the `core` junction's parent has no deps/. So -S = the dir `core` points at (the one with a sibling `deps/`). Flags: -G Ninja, Release, -DBUILD_TESTING=ON (gates all 18 flash_next targets in core/tests/CMakeLists.txt; default OFF -> "ninja: unknown target"), pinned toolchain (MSVC 14.44.35207 Hostx64/x64 cl, CUDA v13.3 nvcc, VS-bundled ninja), -DNINFER_FFMPEG_ROOT=<core-parent>/deps/ffmpeg. Artifacts (FN_PROJ_ROOT=C:/fn): build/tests/{flash_next_real_load,flash_next_placement}.exe + 14 flash_next test exes, build/apps/ninfer-cli.exe, build/src/ninfer_engine.lib. Driver: .agent/full_build.bat; visible-window wrapper .agent/full_build_run.ps1; transcript .agent/full_build_run.log. build/ now in .gitignore.
- [2026-09-25] REBUILD ROOT CAUSE + FIX: core/tests/CMakeLists.txt:180
  FN_PROJ_ROOT literal held the PRE-RENAME project name (real bytes
  "Ninfer Flash"); after the 2026-09-25 root rename to "nVidia Infer Flash"
  the if(EXISTS) gate went false -> all flash_next targets dropped from
  build.ninja ("ninja: unknown target"). Literal fixed (od-verified real f's,
  matches disk dir name from Kodprojekt listing od). Ninja auto-reconfigured;
  build.ninja now has 29 flash_next_real_load refs; 13 ctest test targets
  (LABELS flash_next) also registered.
- [2026-09-25] Launch gotchas confirmed: `cmd //c "tools\...\x.bat"` MUST use
  backslashes (forward-slash path -> "'tools' is not recognized"); the
  "'vswhere.exe' is not recognized" line from VsDevCmd.bat is a benign probe —
  build proceeds.
- [2026-09-25] MSYS one-liner quoting: inline `cmd //c 'call "...\VsDevCmd.bat"...'`
  one-liners get their quotes MANGLED by Git Bash ("'...\VsDevCmd.bat' is not
  recognized"). For compound cmd commands with embedded quotes, write a scratch
  .bat and run `cmd //c ".agent\x.bat"` (od-verify f-bytes clean first).
  .agent/p10_rebuild.bat = scratch p10-only rebuild (ninfer_engine + p10 test).
- [2026-09-25] CLI target name is `ninfer` (ninja), OUTPUT_NAME ninfer-cli — NOT
  "ninfer-cli" as a ninja target ("ninja: unknown target 'ninfer-cli'").
- [2026-09-25] e4m3_lut codegen gotcha: a function-local magic static that holds
  a raw pointer returned from a `make_unique<vector<float>>(256)` lambda
  materialized as an EMPTY vector (size=0, data=NULL) at runtime under MSVC /O2
  + /arch:AVX2 on that TU (p10 segfault, NULL deref in dequant_plane). Use a
  plain `static float table[256]` + `std::call_once` instead (thread-safe, no
  magic-static pointer). See cpu_moe.cpp.
- [2026-09-25] DEPLOYED SERVE (NOT my build): the GPU-holding process is
  `C:\Users\Micke\Desktop\Ninfer AI\ninfer-serve.exe` (PID was 24000), the
  user's PRIMARY daily-work serve, launched from cwd
  `C:\Users\Micke\Desktop\Ninfer AI`. Exact command line (argv oracle for any
  stop->restart; verified via Win32_Process):
  "C:\Users\Micke\Desktop\Ninfer AI\ninfer-serve.exe" "C:\Users\Micke\Desktop\Ninfer AI\models\qwen3_8_27b_nvfp4 23,7gb.ninfer" --host 127.0.0.1 --port 8888 --default-max-tokens 8192 --preserve-thinking --system-prompt-file "C:\Users\Micke\Desktop\Ninfer AI\system-prompt.md" --kv-capacity 262144 --max-context 262144 --kv-dtype fp8 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --spec mtp --draft-tokens 3 --lm-head-draft --default-thinking-budget 6144
  It serves the 27B qwen3_8_27b_nvfp4 model (NOT the flash_next 75GB MoE) and
  holds ~31 GB of the 32 GB VRAM. Do NOT stop it without explicit user ask;
  p10 prod does NOT need it stopped (fits in ~1.4 GB free).
- [2026-09-25] MAX_PATH fix: MSVC -scanDependencies .ddi output for
  ninfer_qwen3_8_flash_next_test_flash_next_mini_reader was 261 chars > 260
  (long C:/ path + long target dir). Created machine junction C:\fn -> project
  root (PowerShell New-Item -ItemType Junction; cmd mklink mangles the target)
  and set FN_PROJ_ROOT="C:/fn" in core/tests/CMakeLists.txt (comment added).
  All flash_next obj/dep paths now ~47 chars shorter. C:/fn is load-bearing
  for builds — do not delete. NOTE: full-tree build also hits a PRE-EXISTING
  core-tree bug (tests/test_context_cost.cpp includes unistd.h, fails on
  Windows) — build flash_next targets by name, not the whole tree.

Key facts:
- [2026-09-25] Project CLAUDE.md created at root: CLAUDE.md (paths+hex oracle+build+gotchas). Byte-verified (core name f=0x66).
- [2026-09-25] Project root renamed to "nVidia Infer Flash".
  New root: C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash.
  Core-tree sibling (via core/ junction) is UNCHANGED.
- flash_next_real_load CMake target added to core/tests/CMakeLists.txt
  (mirrors flash_next_placement; include src + include + flash_next export;
  CUDA::cudart; OUTPUT build/tests/flash_next_real_load.exe).
- p12_s1b_build.bat build path fixed to coreuild (relative).
- Model files: models/qwen3_8_flash_next.ninfer (75,397,717,504 B) and
  models/qwen3_8_flash_next.ngram (51,840,558,936 B).
- S5 expected stream: 3241 7 10435 72181 18532 224070 1347 153926 198 28958
  211173 381 1317 17598 183870 220.
- S5 stream so far all match: 3241 7 10435 72181 18532 224070 1347 153926
  198 28958 211173 381 1317 17598 183870 (14/16 so far).
- P2 (MTP) blocked: no MTP head in model.
- Build: coreuild (junction to Ninner
inner-winuild), Ninja, Release.
- f/n mangling: f(0x66) mangles to n in some tool inputs + display; use
  od -c for byte truth; use bash globs / relative paths to avoid retyping.

Pointers:
- Design: docs/superpowers/specs/2026-09-23-flash-next-moe-residency-design.md
- TODO: .agent/TODO.md
- P1V.3 log: .agent/p1v3_run.log
- Build bat: tools/flash_next_dev/p12_s1b_build.bat (fixed path)

[2026-09-24 23:24] P1V.4 M3 timed T=1 decode launched (detached stop->test->restart cycle). Check ~00:15 via scheduled task 'check-p1v4-m3-decode'. Logs: .agent/p1v4_m3_run.log + .agent/p1v4_cycle.log

[2026-09-24 23:37] P1V.4 M3 cycle LAUNCHED detached (stop serve -> M3 T=1 decode T_prompt 512/2048/8192 x T_new=128, NINFER_P13_M3=1, NINFER_P13_TIMING=1) -> restart serve.
Logs: .agent/p1v4_cycle.log (orchestrator), .agent/p1v4_m3_run.log (test output).
Next step: read p1v4_cycle.log + p1v4_m3_run.log at 00:45 check; extract tok/s per T_prompt, compare to Unsloth 25-40 tok/s; then P4.1 (or P3.2 if run failed/then P4.1 (or P3.2 if run failed/OOM/timeout).

[2026-09-25] Project CLAUDE.md created at project root (authoritative
paths + hex oracle + build + gotchas). Byte-verified via hex: Ninner/ninner
and nVidia Infer Flash all carry 0x66. Core tree path unchanged.
