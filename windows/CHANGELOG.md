# Changelog — NInfer for Windows

All changes to the Windows port (`windows/` tree). Semver:
patch = bugfix, minor = new feature, major = breaking change.

## v1.0.2 (2026-09-19)

- v3 artifact format: full v3 artifact support (`NINFER\0\0\x03` magic) —
  registry routes by magic, load side reads fine-grained named sub-range
  views from fused parent tensors, and NVFP4 block scales from
  `block_scale_k16_m128x4_v1`. New `src/core/weight_view.{h,cpp}` +
  `src/artifact/v3` library; v2 artifacts load exactly as before.
- v3 CPU vision: CPU vision (ViT in host RAM, zero VRAM) works for v3
  artifacts — the 27B target now binds vision to the host backend when
  `--vision cpu` is set, replacing the old "CPU vision is not supported"
  throw on v3 models. Verified end-to-end on qwen3.8-27b NVFP4 (image ->
  coherent caption, ViT fully off GPU).
- Speculative decoding on v3 27B: MTP and dflash2 (+ ngram self-
  speculation) now work on v3 artifacts.
- Fix: bf16 KV cache crash at warmup ("causal_softmax_attention: invalid
  KV cache data dtype"). Root cause: `plan_cache` built K and V planes
  both from the raw KV dtype, but the BF16 paged layout is asymmetric
  (keys BF16, values FP16); planes are now built from the per-plane
  layout dtypes. All three KV dtypes (fp8/int8/bf16) verified on the
  27B model.
- Serve: invalid generated tokens (undecodable UTF-8 from stochastic
  sampling tails) are now per-request 500 `corrupt_generated_token`
  (mid-stream SSE error when streaming) instead of an engine-wide
  failure latch that turned every later request into a 503 until
  restart.
- Serve: request logging — JSONL request log
  (`ninfer-serve-request.jsonl`, next to the serve binary) records each
  request with sampling params and errors. `--kv-capacity` is validated
  strictly against `--max-context x --max-concurrency` at parse time,
  before model load.
- Build: MSVC 19.44 /std:c++20 compatibility fixes in the test tree
  (constexpr `std::sqrt`, `_aligned_malloc`).

(next planned: v1.0.3)

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
