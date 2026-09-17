# qwen3_8_flash_next target tests

Tests for the new `qwen3_8_flash_next` target package
(`src/targets/qwen3_8_flash_next/` once code lands).

## Conventions (spec section 6)

- ctest labels: `flash_next` + per-phase `flash_next_P?`; real-weight tests
  additionally `flash_next_real` and SKIP unless
  `NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS` is set.
- CMake option `NINFER_ENABLE_FLASH_NEXT=ON` gates the target's test wiring.
- Runner per phase: `tools/flash_next_dev/run_phase.sh P?` (prerequisite
  gate + state recording in `out/flash_next_dev/state.json`).

## Status

- **P0 (python-only, no CMake yet):** the three `test_p0_*.py` files run
  under pytest from the repo root:
  `python -m pytest tests/targets/qwen3_8_flash_next/test_p0_*.py -v`
  (or `bash tools/flash_next_dev/run_phase.sh P0`). CMake/ctest wiring is
  added at P1 together with the first C++ target.
- `fixtures/fake_header_4t.ninfer`: checked-in 4-tensor fake `.ninfer` v2
  header (deterministic zeroed payloads); regenerate with
  `python -m tools.flash_next_dev.make_fake_header_fixture`.
