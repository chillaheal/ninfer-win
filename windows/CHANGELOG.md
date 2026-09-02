# Changelog — NInfer for Windows

All changes to the Windows port (`windows/` tree). Semver:
patch = bugfix, minor = new feature, major = breaking change.

## Unreleased

- docs: Playwright MCP browser-automation setup for Claude Code (WINDOWS.md
  section + project gotcha) — pinned `@playwright/mcp@0.0.80`, headless
  msedge, user scope, portable Node 24; no code changes to ninfer.

(next planned: v1.0.1)

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
