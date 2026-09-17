#!/usr/bin/env bash
# Flash-Next phase runner (spec section 0: one phase at a time, green before next).
# Usage: run_phase.sh P0
#
# - Prerequisite gate: Pn refuses to run unless P(n-1) is in state.json "green".
# - Runs the phase's tests from the repo root (ninfer-win).
# - Records the result in out/flash_next_dev/state.json (green list + last_result).
# P0 is python-only; CMake/ctest wiring (labels flash_next / flash_next_Pn,
# option NINFER_ENABLE_FLASH_NEXT) lands at P1 with the first C++ target.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
STATE="$REPO_ROOT/out/flash_next_dev/state.json"
PY=${PYTHON:-python}
PHASE="${1:?usage: run_phase.sh <P0|P1|...>}"

case "$PHASE" in
  P[0-9]*) ;;
  *) echo "phase must look like P0..P13, got: $PHASE" >&2; exit 2 ;;
esac

cd "$REPO_ROOT"

# --- prerequisite gate --------------------------------------------------------
"$PY" - "$PHASE" "$STATE" <<'PYEOF'
import json, sys
phase, state_path = sys.argv[1], sys.argv[2]
num = int(phase[1:])
if num == 0:
    raise SystemExit(0)
state = json.load(open(state_path, encoding="utf-8"))
prev = f"P{num - 1}"
if prev not in state.get("green", []):
    sys.exit(f"refusing {phase}: {prev} is not green (see {state_path})")
PYEOF

# --- run the phase -------------------------------------------------------------
status=0
case "$PHASE" in
  P0)
    "$PY" -m pytest -v \
      tests/targets/qwen3_8_flash_next/test_p0_inspect_fake_fixture.py \
      tests/targets/qwen3_8_flash_next/test_p0_inventory_contract.py \
      tests/targets/qwen3_8_flash_next/test_p0_real_artifact.py || status=$?
    ;;
  P1)
    # Pin the interpreter: FindPython picks the newest (3.14, no torch), but the
    # fixture generator and pytest entries need the torch-bearing Python 3.13.
    PY_EXE="$("$PY" -c 'import sys; print(sys.executable)')"
    cmake -S . -B build -DBUILD_TESTING=ON -DNINFER_ENABLE_FLASH_NEXT=ON \
      -DPython3_EXECUTABLE="$PY_EXE" || status=$?
    if [ "$status" -eq 0 ]; then
      # MSVC std headers need the VsDevCmd env (INCLUDE/LIB/PATH); p1_build.bat
      # holds it (pattern: .logs/build_serve_fix.bat). Absolute, unquoted path:
      # Git Bash mangles double-quoted args to cmd.
      cmd //c "$REPO_ROOT/tools/flash_next_dev/p1_build.bat" || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ctest --test-dir build -L flash_next_P1 --output-on-failure || status=$?
    fi
    ;;
  P2)
    # Same environment pattern as P1 (pinned Python, VsDevCmd env via the bat,
    # absolute unquoted cmd path). P2 builds the engine + the 3 C++ tests and
    # runs them via the flash_next_P2 label.
    PY_EXE="$("$PY" -c 'import sys; print(sys.executable)')"
    cmake -S . -B build -DBUILD_TESTING=ON -DNINFER_ENABLE_FLASH_NEXT=ON \
      -DPython3_EXECUTABLE="$PY_EXE" || status=$?
    if [ "$status" -eq 0 ]; then
      cmd //c "$REPO_ROOT/tools/flash_next_dev/p2_build.bat" || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ctest --test-dir build -L flash_next_P2 --output-on-failure || status=$?
    fi
    ;;
  P3)
    # Same environment pattern as P2 (pinned Python, VsDevCmd env via the bat,
    # absolute unquoted cmd path). P3 adds paging/pager.cpp to the engine and
    # runs the self-contained pager test (label flash_next_P3).
    PY_EXE="$("$PY" -c 'import sys; print(sys.executable)')"
    cmake -S . -B build -DBUILD_TESTING=ON -DNINFER_ENABLE_FLASH_NEXT=ON \
      -DPython3_EXECUTABLE="$PY_EXE" || status=$?
    if [ "$status" -eq 0 ]; then
      cmd //c "$REPO_ROOT/tools/flash_next_dev/p3_build.bat" || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ctest --test-dir build -L flash_next_P3 --output-on-failure || status=$?
    fi
    ;;
  P4)
    # Same environment pattern as P3. P4 adds ple/ngram_embedding.cpp (the
    # PLE mmap gather Op) to the engine and runs the ngram test (label
    # flash_next_P4) against the mini fixture + a self-written ~68 MB store.
    PY_EXE="$("$PY" -c 'import sys; print(sys.executable)')"
    cmake -S . -B build -DBUILD_TESTING=ON -DNINFER_ENABLE_FLASH_NEXT=ON \
      -DPython3_EXECUTABLE="$PY_EXE" || status=$?
    if [ "$status" -eq 0 ]; then
      cmd //c "$REPO_ROOT/tools/flash_next_dev/p4_build.bat" || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ctest --test-dir build -L flash_next_P4 --output-on-failure || status=$?
    fi
    ;;
  P5)
    # Same environment pattern as P4. P5 adds hc/gated_residual.cu (the
    # HyperConnection / gated_residual v1 contract Op) to the engine and runs
    # the self-contained math test (label flash_next_P5, no fixture needed).
    PY_EXE="$("$PY" -c 'import sys; print(sys.executable)')"
    cmake -S . -B build -DBUILD_TESTING=ON -DNINFER_ENABLE_FLASH_NEXT=ON \
      -DPython3_EXECUTABLE="$PY_EXE" || status=$?
    if [ "$status" -eq 0 ]; then
      cmd //c "$REPO_ROOT/tools/flash_next_dev/p5_build.bat" || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ctest --test-dir build -L flash_next_P5 --output-on-failure || status=$?
    fi
    ;;
  P6)
    # Same environment pattern as P5. P6 extends the gated_delta_net domain at
    # the Flash Next geometry (16x128 keys / 48x128 values, FP32 state, hidden
    # 2560) -- no kernel forks. It publishes the GDN state byte size to the
    # planner (real formula 48 x 128 x 128 x 4 per layer) and updates the P2
    # planner test with it; runs the self-contained math test (label
    # flash_next_P6; the updated P2 planner test is built alongside).
    PY_EXE="$("$PY" -c 'import sys; print(sys.executable)')"
    cmake -S . -B build -DBUILD_TESTING=ON -DNINFER_ENABLE_FLASH_NEXT=ON \
      -DPython3_EXECUTABLE="$PY_EXE" || status=$?
    if [ "$status" -eq 0 ]; then
      cmd //c "$REPO_ROOT/tools/flash_next_dev/p6_build.bat" || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ctest --test-dir build -L flash_next_P6 --output-on-failure || status=$?
    fi
    ;;
  P7)
    # Same environment pattern as P6. P7 adds the QSA v1 ops (qsa/qsa_indexer.cu
    # 7a indexer+topk, qsa/qsa_sparse_gqa.cu 7b sparse GQA) to the engine; 7c
    # reuses kv_cache_append + the existing paged store types. Self-contained
    # math test with FP64 oracles (label flash_next_P7, no fixture needed).
    PY_EXE="$("$PY" -c 'import sys; print(sys.executable)')"
    cmake -S . -B build -DBUILD_TESTING=ON -DNINFER_ENABLE_FLASH_NEXT=ON \
      -DPython3_EXECUTABLE="$PY_EXE" || status=$?
    if [ "$status" -eq 0 ]; then
      cmd //c "$REPO_ROOT/tools/flash_next_dev/p7_build.bat" || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ctest --test-dir build -L flash_next_P7 --output-on-failure || status=$?
    fi
    ;;
  P8)
    # Same environment pattern as P7. P8 adds the NVFP4 sparse MoE kernels
    # (moe/moe_nvfp4.cu) + the winner-only dispatch
    # (moe/moe_nvfp4_dispatch.cpp) to the engine; self-contained math test with
    # FP64 oracles over both the W4A4 and W4A16 routes + a W4A4-vs-W4A16 decode
    # microbench (tools/.../p8_bench.cu -> out/.../P8.log). Label flash_next_P8.
    PY_EXE="$("$PY" -c 'import sys; print(sys.executable)')"
    cmake -S . -B build -DBUILD_TESTING=ON -DNINFER_ENABLE_FLASH_NEXT=ON \
      -DPython3_EXECUTABLE="$PY_EXE" || status=$?
    if [ "$status" -eq 0 ]; then
      cmd //c "$REPO_ROOT/tools/flash_next_dev/p8_build.bat" || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ctest --test-dir build -L flash_next_P8 --output-on-failure || status=$?
    fi
    ;;
  P9)
    # Same environment pattern as P8. P9 adds the flash_next mini Program
    # (impl/runtime/program.cpp + family.h + frontend.cpp + sequence_planner.cpp
    # + target-private leaves/) to the engine and runs the first-forward test
    # (label flash_next_P9): (a) exarch+pager staging, (b) greedy golden,
    # (c) bit-identical 2nd run, (d) prefix reuse, (e) peak GPU cap, (f) CLI
    # determinism. (f) invokes build/apps/ninfer-cli.exe, which p9_build.bat also
    # builds (target ninfer -> OUTPUT_NAME ninfer-cli).
    PY_EXE="$("$PY" -c 'import sys; print(sys.executable)')"
    cmake -S . -B build -DBUILD_TESTING=ON -DNINFER_ENABLE_FLASH_NEXT=ON \
      -DPython3_EXECUTABLE="$PY_EXE" || status=$?
    if [ "$status" -eq 0 ]; then
      cmd //c "$REPO_ROOT/tools/flash_next_dev/p9_build.bat" || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ctest --test-dir build -L flash_next_P9 --output-on-failure || status=$?
    fi
    ;;
  P10)
    # Same environment pattern as P9. P10 wires the mini flash_next target into the
    # EXISTING serve (same Engine route as 27B -- no new HTTP stack) and runs the
    # adapted mini serve smoke (label flash_next_P10): launch
    # build/apps/ninfer-serve.exe on the mini fixture, check /health + /v1/models +
    # /v1/chat/completions (OpenAI envelope). No Vision, no thinking.
    # p10_build.bat builds ninfer_engine + ninfer-serve (the serve the smoke hits).
    PY_EXE="$("$PY" -c 'import sys; print(sys.executable)')"
    cmake -S . -B build -DBUILD_TESTING=ON -DNINFER_ENABLE_FLASH_NEXT=ON \
      -DPython3_EXECUTABLE="$PY_EXE" || status=$?
    if [ "$status" -eq 0 ]; then
      cmd //c "$REPO_ROOT/tools/flash_next_dev/p10_build.bat" || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ctest --test-dir build -L flash_next_P10 --output-on-failure || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      # P10 also requires the product schema/protocol tests to stay green. They are
      # not flash_next-labeled, so the label gate above does not run them.
      ctest --test-dir build -R "ninfer_(openai|anthropic|responses)_schema_test|ninfer_tool_call_parser_test|ninfer_request_log_test|ninfer_response_store_test" --output-on-failure || status=$?
    fi
    ;;
  P11)
    # Same environment pattern as P10. P11 drives the P2 planner with the REAL
    # P0-audit byte sums + this machine's real budgets (32 GB card / 68.4 GB RAM)
    # and snapshots the placement dry-run tool (flash_next_placement.exe). The
    # binding finding: the ~69.4 GB NVFP4 expert payload exceeds the 68.4 GB RAM,
    # so --expert-host-all does not fit on THIS machine (host_pinned overflow);
    # the GPU dimension fits comfortably. p11_build.bat builds the engine +
    # flash_next_placement + the P11 test.
    PY_EXE="$("$PY" -c 'import sys; print(sys.executable)')"
    cmake -S . -B build -DBUILD_TESTING=ON -DNINFER_ENABLE_FLASH_NEXT=ON \
      -DPython3_EXECUTABLE="$PY_EXE" || status=$?
    if [ "$status" -eq 0 ]; then
      cmd //c "$REPO_ROOT/tools/flash_next_dev/p11_build.bat" || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ctest --test-dir build -L flash_next_P11 --output-on-failure || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      # P11 also requires the whole flash_next phase suite (P1-P11) to stay green.
      ctest --test-dir build -L flash_next --output-on-failure || status=$?
    fi
    ;;
  *)
    echo "phase $PHASE not implemented yet (its runner lands with its code)" >&2
    exit 2
    ;;
esac

# --- record the result in state.json -------------------------------------------
"$PY" - "$PHASE" "$STATE" "$status" <<'PYEOF'
import json, sys, datetime
phase, state_path, status = sys.argv[1], sys.argv[2], int(sys.argv[3])
state = json.load(open(state_path, encoding="utf-8"))
now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
state["last_result"] = {"phase": phase, "status": "pass" if status == 0 else "fail", "ts": now}
if status == 0:
    if phase not in state["green"]:
        state["green"].append(phase)
    state["status"] = "green"
else:
    state["status"] = "red"
state["phase"] = phase
json.dump(state, open(state_path, "w", encoding="utf-8"), indent=2)
sys.exit(status)
PYEOF
