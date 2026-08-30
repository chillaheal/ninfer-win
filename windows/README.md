# Ninfer — Windows Port, GUI & Claude Code Integration

This directory contains the complete Windows (MSVC / x64) port of [NInfer](https://github.com/Neroued/ninfer),
plus a native Win32 desktop GUI and the serve-side integrations that let
[Claude Code](https://claude.ai/code) run against a local model.

> Upstream engine: `Neroued/ninfer` (this port is built from the POSIX source tree;
> no upstream files are modified — everything here is additive).

## What is in here

| Area | Status |
|---|---|
| Core engine port (C++/MSVC x64) | Complete — full e2e verified |
| `ninfer-cli` (generate/eval CLI) | Complete |
| `ninfer-serve` (OpenAI + Anthropic + Responses API) | Complete — Claude Code compatible |
| Win32 GUI (`Ninfer.exe`) | Complete — serve launcher, system-prompt / chat-template editors |
| Vision + speculative decoding (MTP) | Verified on a single serve |
| Windows build pipeline (CMake + vcpkg, `build_windows.ps1`) | Complete |

## The GUI

`Ninfer.exe` (Win32, no dependencies beyond the bundled DLLs):

- **Serve launcher** — model artifact picker (opens in the model's folder),
  context sizing (`Auto` probes VRAM and fills the window with the fitting KV
  capacity), KV quantization (fp8/fp16), spec decoding (mtp), vision, greedy,
  temperature/sampling defaults for Qwen3.8-27B.
- **System-prompt editor** — loads/saves the `--system-prompt-file` content with
  live byte accounting (the serve trims trailing whitespace; the editor shows
  both the on-disk and the canonical active byte count).
- **Chat-template editor** — loads a Jinja template, computes its SHA-256 and
  compares it against the template the serve actually runs (`/health`
  `chat_template.sha256`), with a "differs / matches" indicator.
- **Stop** — terminates only the GUI's own child serve; **Exit** kills all
  `ninfer-cli`/`ninfer-serve` engines (posted `WM_CLOSE`, never a raw
  `DestroyWindow`).

Prefill: `Ninfer.exe <model.ninfer> [prompt words...] [--max-new N] [--max-ctx N]`

## The chat-template system (important design note)

Templates are **not executed** at runtime. `resolve()` maps
`sha256(template text) -> allowlist digest -> semantics` and all prompt text is
hardcoded C++ (`src/serve/chat_template.cpp`). This is deliberate: only
pre-verified templates with known semantics are accepted.

Swapping a template therefore means: (1) a **new model artifact** carrying the
new template bytes (tools in `tools/artifact/`), (2) a new allowlist digest in
`chat_template.cpp`, (3) a rebuild, (4) starting the serve against the new
artifact. The running serve mmaps its artifact and is never mutated in place.

`/health` reports the active template (`sha256`, `semantics`,
`reasoning_effort.default/supported`) and the active system prompt
(`active`, trimmed `bytes`, `file`).

Effort handling: requested effort `high` rounds up to the closest supported
level (e.g. `xhigh`) — required because Claude Code's internal fast-model calls
(WebFetch summarization etc.) send `high`.

## Serving & connecting Claude Code

```bat
:: serve (from the dist bundle, or ninfer-serve.exe directly)
ninfer-serve.exe --model models\qwen3_8_27b.ninfer --max-context 32768 --spec mtp --vision

:: then, in Claude Code:
set ANTHROPIC_BASE_URL=http://127.0.0.1:8080
set ANTHROPIC_AUTH_TOKEN=sk-ninfer-local
claude
```

- `--max-context 32768` is required (the default 8192 is too small for Claude
  Code's system prompt). The GUI's Serve prefills it.
- No `/v1` suffix on the base URL.
- `ANTHROPIC_AUTH_TOKEN` (bearer), **not** `ANTHROPIC_API_KEY` — a non-Anthropic
  base URL with `ANTHROPIC_API_KEY` drops Claude Code into the OAuth login flow.
  `start-claude-ninfer.ps1` wires both up and health-checks the port first.

## Building on Windows

Prereqs: Visual Studio 2022 Build Tools (C++ x64), CMake >= 3.25, vcpkg
(deps under `deps/` on the reference machine: curl, ffmpeg, vcpkg tree).

```powershell
# full build (a few minutes)
powershell -NoProfile -ExecutionPolicy Bypass -File build_windows.ps1
# targeted rebuild inside the VsDevCmd env (fast):
cmake --build ninfer-win\build --config Release --target ninfer-gui
```

Targets: `ninfer` (CLI), `ninfer-serve`, `ninfer-gui`, plus the test suites.
Note: three test targets are POSIX-only (`test_context_cost`,
`test_gdn_replay_records`, `test_frontend.cpp`) — with `BUILD_TESTING=ON`
persisted in the cache, build the targets you need individually instead of the
full script.

Run from the dist bundle directory (DLLs resolve next to the exes).

## Verified configuration (reference machine)

- Model: Qwen3.8-27B NVFP4 (`.ninfer` artifact, ~21.5 GB) + community chat
  template (allowlist digest `7f0e5290...`) + 8.6 KB system prompt.
- Vision + `--spec mtp` on a single serve; KV pool sized by the GUI's `Auto`
  VRAM probe (`--kv-capacity`); weights load in ~5 s.
- E2E drivers (PowerShell 5.1, Win32 oracles): serve launch + `/health`
  contract + Anthropic Messages round-trip, GUI defaults, exit-kill semantics.
