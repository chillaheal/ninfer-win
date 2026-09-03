# Changelog — NInfer for Windows

All changes to the Windows port (`windows/` tree). Semver:
patch = bugfix, minor = new feature, major = breaking change.

## v1.0.1 (2026-09-03)

- GUI: settings persistence — last-used model + max context, KV dtype,
  vision mode, and speculation settings saved to `gui-settings.ini` next
  to the exe (atomic write; restored at startup; CLI args still win).
  Vision mode now defaults to CPU. Fixed a latent dangling `string_view`
  UB that made settings-key matching unreliable.
- Serve: Anthropic server-tools emulation — `web_search_20250305` and
  `{type:"url"}` sub-requests from Claude Code execute server-side
  (keyless DuckDuckGo-lite backend, multi-phase tool loop with per-tool
  usage caps, rate-limit/anomaly detection, 10-min result cache).
  New runtime deps: libcurl/z.
- Context sizing: `--probe` now sizes the actual launch regime (KV dtype
  + MTP speculation forwarded; search-ceiling fix); `/health` reports
  `context {max_context, kv_capacity}`; the launcher derives
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS` / `CLAUDE_CODE_AUTO_COMPACT_WINDOW`.
- CPU vision: fixed 3 latent layout/scale bugs (fp16 subnormal scaling,
  pos-embed index order, RoPE position-id split) and optimized the
  encode path (~40x on the 220-token HELLO reference: 22.0 s -> 0.54 s).
- docs: Playwright MCP browser-automation setup for Claude Code
  (WINDOWS.md section + project gotcha) — pinned
  `@playwright/mcp@0.0.80`, headless msedge, user scope, portable Node
  24; no code changes to ninfer.

(next planned: v1.0.2)

## v1.0.0 (2026-08-30)

First public release.

- Windows (MSVC/x64) port of the engine, `ninfer-cli`, `ninfer-serve`
  (OpenAI + Anthropic + Responses schemas, `/health` contract).
- Win32 GUI (`Ninfer.exe`): serve launcher with Auto VRAM context sizing,
  system-prompt editor, chat-template editor with live SHA-256 comparison
  against the running serve.
- Claude Code hookup: `WINDOWS.md`, `start-claude-ninfer.ps1`.
- Release bundle zip (GUI + serve + CLI + DLLs + guides + model downloader).
- Verified: RTX 5090 (32 GB), Qwen3.8-27B NVFP4 — `/health` contract,
  Anthropic Messages round-trip, vision + MTP speculative decoding, GUI e2e.
