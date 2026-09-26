import io

state = """Status:
P1V verification nearly complete on the real 75 GB model.
P1V.1 + P1V.2 DONE (ctest -L flash_next bit-exact PASS).
P1V.3 (NINFER_EXPERT_GPU_VERIFY on 75 GB model): RUNNING (2026-09-24).
Gate driver flash_next_real_load built + running detached
(log .agent/p1v3_run.log); S1(b)/S3/S4 PASS confirmed; S5 in progress.

Plan position:
Current step: P1V.3 - NINFER_EXPERT_GPU_VERIFY self-check on 75 GB model
Completed: P1 (all), P1V.1, P1V.2
Next step: P1V.4 - timed T=1 decode (detached, NINFER_P13_M3=1)

Key facts:
- Root cause of CLI bind failure: CLI/registry load path uses ONLY the
  mini loader (package.cpp construct_loaded_model -> mini LoadedModel::open).
  Real loader (real_loader.cpp) is complete/correct but only reachable via
  gate driver create_real_program (program.cpp:1569) -> flash_next_real_load.
- flash_next_real_load CMake target added to core/tests/CMakeLists.txt
  (after flash_next_placement block), mirroring it (links ninfer_engine +
  ninfer_core, includes include/src/flash_next export, CUDA::cudart,
  RUNTIME_OUTPUT_DIRECTORY FN_PROJ_ROOT/build/tests).
- Build cmd that works: cmake --build core\\build --target ninfer_engine
  flash_next_real_load -j8 (via p12_s1b_build.bat, which cd's to project
  root; the bat now uses core\\build relative).
- exe output location: project-root build/tests/ (NOT core/build/tests).
- Model files (cwd-relative): models/qwen3_8_flash_next.ninner (75.4 GB),
  models/qwen3_8_flash_next.ngram (51.8 GB).
- S5 expected stream (token 0..): 3241 7 10435 72181 18532 224070 1347
  153926 198 28958 211173 381 1317 17598 183870 220
- Root cause of slow: moe_expert_h2d 86.65% of wall; H2D bench: SSD-mmap
  cold 0.77-2.86 GB/s vs warm-RAM pageable 26.6 GB/s vs pinned 32.4 GB/s.
- Phase 2 (MTP) blocked: no MTP head in model. P3 (CPU MoE offload) next.
- core/ is a junction to C:\\Users\\Micke\\Documents\\Kodprojekt\\Ninner\\ninner-win.
  Build tree = core\\build (junction target build).
- f/n mangling: lowercase f (0x46 typed) mangles to n in some tool inputs
  (Grep patterns, some bash args) and in display. Write CONTENT is
  byte-exact. od -c is ground truth. Use relative paths from project root.

Pointers:
- Design: docs/superpowers/specs/2026-09-23-flash-next-moe-residency-design.md
- TODO: .agent/TODO.md
- Driver source: tools/flash_next_dev/real_load_driver.cpp
- CMake target added via .agent/add_real_load_target.py (throwaway).
- P1V.3 run log: .agent/p1v3_run.log; build log .agent/p1v3_build.log.
"""

with io.open(".agent/STATE.md", "w", encoding="utf-8") as fh:
    fh.write(content)
print("WROTE STATE.md")
