# nVidia Infer Flash (flash_next) — Project Rules

Project-specific facts only. Global operating rules live in `~/.claude/CLAUDE.md`.

`flash_next` (`qwen3_8_flash_next`) is a **MoE target attached as an overlay into the
Ninfer core tree**. This repo holds the overlay source + tools + tests; the build happens
in the core tree's build dir, not a local one.

## Paths (authoritative — hex is the oracle)

- **Project root (this repo):** `C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash`
  - hex: `433a55736572735c4d69636b655c446f63756d656e74735c4b6f6470726f6a656b745c6e566964696120496e66657220466c617368`
- **Core tree (sibling, DO NOT rename/move):** `C:\Users\Micke\Documents\Kodprojekt\Ninfer\ninfer-win`
  - hex: `433a55736572735c4d69636b655c446f63756d656e74735c4b6f6470726f6a656b745c4e696e6665725c6e696e6665722d77696e`
  - `core` (repo root) is a symlink/junction to this; the build happens in `core\build`.

Disambiguation: the **project** is `name` + **space** + `Flash`; the **core tree** is
`name` + **backslash** + `ninfer-win`. Never conflate them.

## Layout

- `src/targets/qwen3_8_flash_next` — overlay target source (attached into the core tree)
- `tools/flash_next_dev/` — build/verify scripts (`p1`–`p12`) + run scripts (`pn_*`)
- `tests/` — tests
- `models/` — model artifacts (gitignored): `qwen3_8_flash_next.ninfer` (~70 GB), `qwen3_8_flash_next.ngram`
- `core` — symlink/junction to the core tree (build happens here)

## Build & run

- Toolchain: VS 2022 BuildTools (`VsDevCmd x64`) + CMake 4.4.3 + Ninja (both WinGet)
- Attach overlay (one-time, idempotent): `powershell -NoProfile -File tools\flash_next_dev\attach_overlay.ps1`
- Rebuild + relink tests: `cmd //c "tools\flash_next_dev\p1_rebuild_tests.bat"`
- Model-load test: `cmd //c "tools\flash_next_dev\p1_build.bat"`
- ctest: `cd core\build && ctest -L flash_next --timeout 420 --output-on-failure`
  (p8's prod 512-expert case takes ~4.5 min; 420 s is a per-test TIMEOUT in
  `core\tests\CMakeLists.txt`. `--timeout` overrides that property, so keep the
  flag ≥ 420 or drop it. `NINFER_P8_SKIP_PROD=1` skips p8's prod case when the
  GPU is busy — run ctest with that env only in that situation.)
- Build targets: `ninfer_engine`, `ninfer_qwen3_8_flash_next_test_p2_identity`,
  `ninfer_qwen3_8_flash_next_test_p7_qsa`, `ninfer_qwen3_8_flash_next_test_p9_program`,
  `ninfer-cli`, `flash_next_mini_fixture`, `flash_next_mini_reader_test`

## Gotchas

- **f/n mangle defect:** `f` (0x66) is transmitted/displayed as `n` (0x6e) in some tool I/O.
  The core-tree name `Ninfer`/`ninfer` is mangle-prone (f is the 4th char). Verify paths by
  hex / `od -c`, never by eye. Prefer relative paths (e.g. `core\build`) over retyping the
  absolute core path. (See `.agent/STATE.md`.)
- `core` is a symlink/junction to the core tree — never delete it.
- `models/` are gitignored and huge; never `git add` them.

## Pointers

- `.agent/STATE.md` — status/current step (re-read before resuming)
- `.agent/TODO.md` — numbered plan
- `docs/superpowers/specs/2026-09-23-flash-next-moe-residency-design.md` — MoE residency design
- `tools/flash_next_dev/` — all build/verify bats
