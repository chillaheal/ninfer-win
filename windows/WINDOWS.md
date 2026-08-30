# NInfer — Windows Port

Windows port of the NInfer C++20/CUDA inference engine. The working copy is
`ninfer-win/`; `ninfer-master/` is the pristine upstream reference and must
never be edited.

```
Ninfer/
├── ninfer-master/   pristine upstream (reference only)
├── ninfer-win/      Windows port (CMake + source), build tree at ninfer-win/build
├── deps/            prebuilt FFmpeg (shared libs) and curl (Schannel TLS)
├── models/          .ninfer model artifacts
├── dist/            assembled product output (build_windows.ps1 step 4)
└── build_windows.ps1  one-shot build + dist assembly + verification
```

## Prerequisites

| Requirement | Notes |
|---|---|
| Windows 10/11 x64 | |
| NVIDIA GPU + driver | `nvcuda.dll` comes from the installed driver, not from this repo. Tested on RTX 5090. |
| CUDA Toolkit v13.x | Installed under `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.*`. The script auto-detects the newest v13.x directory (winget installs leave no registry key). |
| VS 2022 Build Tools (or full VS) | With the "Desktop development with C++" workload. Provides `cl.exe`, `VsDevCmd.bat` (looked up in both `Common7\Tools` and `VC\Auxiliary\Build`), and the VC CRT redist DLLs. |
| CMake + Ninja | On PATH, or installed via winget (`Kitware.CMake`, `Ninja-build.Ninja`) — the script also searches the winget package directories. |
| `deps/ffmpeg`, `deps/curl` | Prebuilt binaries, already in the repo. |
| A `.ninfer` model artifact | Must have a registered target for your device (see Models). |

## Build

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File build_windows.ps1          # incremental
powershell -NoProfile -ExecutionPolicy Bypass -File build_windows.ps1 -Clean   # from scratch
```

The script runs five steps: tool check → CMake configure → Release build →
assemble `dist/` → verify (CLI `--help` + GUI window launch from the dist
location) and prints `BUILD_WINDOWS PASS` on success.

Build internals:

- Configure and build run inside a `VsDevCmd` environment via a temporary bat
  file (`nvcc` uses `cl.exe` as host compiler), so the script works from any shell.
- The CUDA runtime is linked **statically** into the executables — no CUDA DLLs
  are shipped in `dist/`.
- The apps use the **dynamic** CRT, so `vcruntime140.dll`, `msvcp140.dll` and
  `vcruntime140_1.dll` are copied next to the exes.

## Run

Everything in `dist/` must stay together — all third-party DLLs resolve from
the exe's own directory. Do not copy a single exe out of `dist/`.

```
dist/
├── Ninfer.exe            GUI (Win32)
├── ninfer-cli.exe        CLI
├── ninfer-serve.exe      HTTP server
├── vcruntime140.dll, msvcp140.dll, vcruntime140_1.dll   VC runtime
├── libcurl.dll, z.dll    HTTP + TLS (Schannel)
└── avcodec/avformat/avutil/swscale/swresample DLLs      FFmpeg (media input)
```

### GUI — `Ninfer.exe`

Win32 form: model path, sampling fields (Max context / Max new / KV cache / temperature /
top-p / top-k / spec decoding / draft tokens / seed), Thinking and LM-head-draft
checkboxes, Host/Port, and the **Serve** / **Stop** / **Exit** buttons. Plus a **System prompt…**
and a **Chat template…** button, each opening its own editor window. The old one-shot Run
flow (prompt/messages panes, Run / Cancel, answer + log panes) was removed — `Ninfer.exe`
is a serve launcher.

Status strings: `Idle`, `Serving — watch the terminal window for status and stats…`
(a clamping warning is appended before the launch when Max context exceeds the VRAM fit),
`A serve is already running on <host:port> — Stop or Exit it first, or pick another
port.` (pre-launch port check; the launch is aborted), `Stopping serve…`, `No serve
started from this GUI.` (Stop with no child), and `Server stopped (exit 0)` /
`Server exited with code N` once the server process ends.
The Thinking checkbox is checked by default (unchecking passes `--no-thinking`).

**Serve mode**: clicking **Serve** starts `ninfer-serve.exe` in its own terminal window
(CREATE_NEW_CONSOLE — unsloth-style), passing the model row, the sampling fields,
Host/Port, `--default-max-tokens` (from "Max new") and `--preserve-thinking` (Qwen's
recommended agentic server config), and — unconditionally —
`--system-prompt-file <exe-dir>\system-prompt.md` (the serve reads the file at startup, so
a save from the editor takes effect from the next serve start). The terminal shows the load
progress, `listening on http://host:port`, a line per request, and a 5 s interval line with
`prefill=…tok/s decode=…tok/s`. The server is stopped with **Stop** — it terminates only
the serve this GUI started and the GUI stays open (the button re-enables when the server
exits) — or with **Exit** / the window X, which additionally kills every engine and
closes the GUI (see below). The old one-shot Cancel button is not back.

The form is prefilled with the Qwen3.8-27B recommended coding defaults: max context 32768,
max new 8192, KV cache fp8, temperature 1.0, top-p 0.95, top-k 20, spec decoding mtp with
3 draft tokens, Thinking and LM-head-draft checked, Host 127.0.0.1, Port 8080. The model
row starts EMPTY — select the artifact explicitly (rollout default:
`models\qwen3_8_27b_nvfp4_ct.ninfer`).

**Auto context size**: the **Auto** button (next to Max context) runs
`ninfer-cli.exe --probe` on the selected model — no weights are loaded, so it finishes in
seconds — and fills Max context with the largest KV window that fits the current free VRAM
(free-after-weights minus a 3% headroom). The status line reports `Auto: ~N tokens fit
(free after weights: X.XX GiB)`. Every Serve pins the KV pool to the last probed fit for
the model (`--kv-capacity <fit>`); if the requested Max context exceeds it, the launch is
clamped to the fit and the status line shows `Warning: context <requested> exceeds VRAM fit
~<fit>; clamped to fit.` before the serve starts. A failed probe aborts the launch with an
error in the status line.

**System prompt editor** (**System prompt…** button): opens `<exe-dir>\system-prompt.md` in
a monospace edit window (8646 B on disk for the rollout default). Line breaks are shown as
CRLF so the text renders with its paragraphs and bullets (LF-only text renders as one
continuous line in an EDIT control on this machine); the on-disk file stays canonical
LF-only. The edit is length-capped to the display-form size (8811 = 8646 + 165 line
breaks); the status line shows `<path> — N bytes. Active from the next serve restart
(read at startup).`; **Save** writes the file and reports the new byte count. A missing
file shows `File does not exist yet — it is created on Save.`, and Save creates it.

**Chat template editor** (**Chat template…** button): a template-source window with
**Open…** (Jinja filter, initial dir `<exe-dir>\..\models`), **Refresh** and **Save…**.
**Refresh** queries the running serve's `/health` and shows the loaded template (`Loaded:
sha256 <hex>  ·  semantics <name>`, plus default effort / supported levels for
reasoning-effort templates) and whether the editor content `Matches the template loaded by
the server.` or `Differs from the loaded template — active only after artifact swap +
server rebuild.` (a template takes effect only once baked into an artifact that the serve
loads — e.g. `models\qwen3_8_27b_nvfp4_ct.ninfer`). An unreachable serve degrades to
`Serve not reachable at <host:port> — editor only.`; the editor remains fully usable.
Closing either editor window only closes the window — it never touches the engines.

**Exit kills engines**: closing the window (X) or clicking **Exit** terminates every
running `ninfer-cli.exe` and `ninfer-serve.exe` process on the machine — including servers
started outside the GUI — so no orphaned engine keeps holding GPU memory after the GUI
closes. Note: that also stops a serve that a Claude Code session may be running on, so
restart it deliberately (see the project plan's final-restart runbook).

Command-line prefill fills the form before the window is shown:

```
Ninfer.exe <model.ninfer> [--max-new N] [--max-ctx N]
```

Example:

```powershell
.\dist\Ninfer.exe .\models\qwen3_8_27b_nvfp4_ct.ninfer
```

### CLI — `ninfer-cli.exe`

Streams answer content to **stdout**; reasoning and diagnostics go to
**stderr**.

```
usage: ninfer-cli.exe <model.ninfer> (--prompt <text>|--messages <messages.json>|--probe)
       [--max-context N] [--kv-capacity N|auto] [--prefill-chunk N] [--max-new N]
       [--device N]
       [--kv-dtype bf16|int8|fp8] [--spec mtp|dflash --draft-tokens N]
       [--lm-head-draft]
       [--temperature F] [--top-p F] [--top-k N] [--min-p F]
       [--presence-penalty F] [--frequency-penalty F] [--seed N] [--greedy]
       [--stop-token-id N]... [--stop <text>]... [--reasoning-stop <text>]...
       [--raw-output] [--print-token-ids] [--no-thinking] [--thinking-budget N]
       [--reasoning-effort low|medium|xhigh] [--vision]
       [--no-cuda-graph]
```

Structured message content accepts text, image/image_url, and video/video_url
parts; media sources may be local paths, HTTP(S) URLs, or base64 data URIs.
`--vision` enables image/video input. Sampling defaults come from the loaded
model and thinking mode; flags override individual fields.

`--probe` runs the VRAM probe instead of a generation run: it plans the load
(without materializing weights), measures free-after-weights VRAM, and reports
the largest KV window that fits (3% headroom) as key=value lines on stdout —
`ninfer-probe model=…`, `vram_total_bytes=…`, `weights_bytes=…`,
`vram_free_after_weights_bytes=…`, `kv_headroom_percent=3`, `kv_fit_tokens=…`.
The GUI **Auto** button parses `kv_fit_tokens`; a timing note goes to stderr.

### Server — `ninfer-serve.exe`

Serves OpenAI Responses / Chat Completions and Anthropic Messages endpoints:

```
usage: ninfer-serve.exe <model.ninfer> [--host H] [--port N] [--api-key KEY]
       [--model-id ID] [--max-context N] [--kv-capacity N|auto] [--max-concurrency N]
       ... (see --help for the full list)
```

### Connecting Claude Code to ninfer-serve

ninfer-serve speaks the Anthropic Messages protocol natively, so Claude Code
works against it with no proxy or adapter. Verified end-to-end with a real
`claude -p` run (`.logs/claude_code_e2e.ps1`).

1. Start the server — GUI **Serve** button (prefills max context 32768), or from `dist/`:

   ```powershell
   .\dist\ninfer-serve.exe .\models\qwen3_8_27b_nvfp4.ninfer --port 8080 --max-context 32768
   ```

   Wait for `listening on http://127.0.0.1:8080` in the terminal. The server's bare
   default max context is **8192 tokens**, which is too small for Claude Code (its
   system prompt alone exceeds that and requests are rejected with a 400) — always
   pass a larger `--max-context` when starting it manually.

2. Point Claude Code at it (session env vars, or your shell profile):

   ```powershell
   $env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8080"   # no /v1 suffix — Claude Code appends /v1/messages itself
   $env:ANTHROPIC_API_KEY  = "anything"                # auth is disabled unless you pass --api-key to the server
   ```

3. Run it:

   ```powershell
   claude -p "Reply with exactly: OK"
   ```

Shortcut: `start-claude-ninfer.ps1` at the project root does steps 2–3 for you. It
checks that the server answers on port 8080 (warns and stops if not), sets the two env
vars, and launches `claude` — any extra arguments pass through:

```powershell
.\start-claude-ninfer.ps1                 # interactive session
.\start-claude-ninfer.ps1 -p "Say OK"     # one-shot
```

It never starts or stops the server itself (edit `$Port` in the script for a
non-default port).

Notes:

- The `model` field is just a label — the Anthropic endpoint accepts **any**
  model string (Claude Code sends its real Claude model names) and echoes it
  back; it never 404s on the id. So an existing config with
  `ANTHROPIC_MODEL=unsloth/Qwen3.8-27B-GGUF` or similar works unchanged.
- If you already run Claude Code against another local server (e.g. unsloth on
  port 8888), switching to ninfer means changing **only** `ANTHROPIC_BASE_URL`.
- A harmless client-side diagnostic `[claude-code:unrecognized_model]` may
  appear in stderr for non-Claude model labels — it is not an error.

## Models

`models/` currently contains:

- `qwen3_8_27b_nvfp4.ninfer` (~20 GiB) — **works** on this build.
- `qwen3_5_9b.ninfer` (~6.5 GiB) — **not supported**: the artifact identity
  `qwen3.5-9b/groupwise-int` has no registered target for this device, and the
  CLI exits with:

  ```
  error: artifact identity 'qwen3.5-9b/groupwise-int' has no registered target for this device
  ```

A model is usable only if its artifact identity was compiled with a target for
the local device; this is a property of the `.ninfer` artifact, not a flag.

## Development notes

- Build tree: `ninfer-win/build` (Ninja, Release). Re-run `build_windows.ps1`
  after source changes — it configures incrementally and rebuilds only what's
  stale.
- Port shim: on Windows the MSVC CRT encodes narrow `argv` with the ANSI code
  page, which corrupts non-ASCII text; the apps re-encode the wide command line
  as UTF-8 (`apps/common/utf8_argv.h`).
- GUI e2e drivers: `.logs/gui_e2e.ps1` (Run mode), `.logs/gui_serve_e2e.ps1`
  (serve controls + defaults + live API round-trip from the GUI),
  `.logs/gui_auto_ctx_e2e.ps1` (Auto button fills Max context with the probed
  fit; oversized `--max-ctx` triggers the VRAM warning on Run),
  `.logs/gui_exit_kill_e2e.ps1` (Exit reaps an orphaned ninfer-serve the GUI
  never started), and `.logs/claude_code_e2e.ps1` (real `claude -p` against
  ninfer-serve). Control IDs come from `apps/gui/main.cpp`; see
  `manual-e2e/RESULT.md` for the last verified run.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Exit code `-1073741515` (0xC0000135) when launching an exe | A required DLL is missing next to the exe. Run from `dist/`, or re-run `build_windows.ps1`. |
| `artifact identity ... has no registered target for this device` | The `.ninfer` artifact has no compiled target for your GPU/device — use a supported model. |
| Script throws "VsDevCmd.bat not found" | Install VS 2022 Build Tools with the C++ workload. |
| Script throws "No CUDA v13.x toolkit found" | Install a CUDA v13.x toolkit (the script only auto-detects `v13.*` directories). |
| Non-ASCII arguments mangled when passed from `cmd.exe` | The console code page may corrupt å/ä/ö; use PowerShell, or pass the prompt via the GUI. |
