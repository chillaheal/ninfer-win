import io
content = """Status:
P1V in progress (2026-09-24): P1V.3 NINFER_EXPERT_GPU_VERIFY on the 75 GB
model RUNNING detached (log .agent/p1v3_run.log). S1(b)/S3/S4 PASS so far;
S5 decode in progress (token 14 = 183870, matches stream so far).

Plan position:
Current step: P1V.3 - NINFER_EXPERT_GPU_VERIFY self-check on 75 GB model
Completed: P1 (all), P1V.1 (#44), P1V.2 (#45)
Next step: P1V.3 finish (S5 16-token bit-exact stream) -> P1V.4 (#47) timed T=1 decode

Key facts:
- flash_next_real_load CMake target added to core/tests/CMakeLists.txt
  (mirrors flash_next_placement; include src + include + flash_next export;
  CUDA::cudart; OUTPUT build/tests/flash_next_real_load.exe).
- p12_s1b_build.bat build path fixed to core\build (relative).
- Model files: models/qwen3_8_flash_next.ninfer (75,397,717,504 B) and
  models/qwen3_8_flash_next.ngram (51,840,558,936 B).
- S5 expected stream: 3241 7 10435 72181 18532 224070 1347 153926 198 28958
  211173 381 1317 17598 183870 220.
- S5 stream so far all match: 3241 7 10435 72181 18532 224070 1347 153926
  198 28958 211173 381 1317 17598 183870 (14/16 so far).
- P2 (MTP) blocked: no MTP head in model.
- Build: core\build (junction to Ninner\ninner-win\build), Ninja, Release.
- f/n mangling: f(0x66) mangles to n in some tool inputs + display; use
  od -c for byte truth; use bash globs / relative paths to avoid retyping.

Pointers:
- Design: docs/superpowers/specs/2026-09-23-flash-next-moe-residency-design.md
- TODO: .agent/TODO.md
- P1V.3 log: .agent/p1v3_run.log
- Build bat: tools/flash_next_dev/p12_s1b_build.bat (fixed path)
"""
with io.open(".agent/STATE.md", "w", encoding="utf-8", newline="\n") as fh:
    fh.write(content)
print("WROTE")
