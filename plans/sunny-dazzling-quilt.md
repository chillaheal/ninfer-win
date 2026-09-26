# flash_next MoE — request-hang root cause + fix

## Context

flash_next (177B MoE, 6B active). Serve **loads and reaches "listening on"** — it is
NOT a warmup stall (earlier misdiagnosis). A "hej" request via Claude Desktop, with
**100K context set in the GUI**, shows **"prompt processing 0 tok"** for ~58 min, then
`QueueTimeout` "expired while waiting for admission". GPU idle, CPU ~7%, SSD idle, RAM
unchanged. The 27B dense model serves fine — this is flash_next-specific.

Reference the user wants compared against:
`C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next` — a **prebuilt llama.cpp**
(`llama-server.exe` + GGUF `UD-IQ3_XXS`), same model, serves in seconds. No source tree,
only binaries + a design doc (`docs/superpowers/specs/2026-09-23-qwen38-flash-next-llamacpp-design.md`).

## Findings (verified)

- **Phase 3 (CPU MoE offload) is in-tree** — `CpuMoe` NVFP4 GEMM (`cpu_moe.{h,cpp}`),
  env gate `NINFER_MOE_CPU_LAYERS` (real_program.cpp:1510), off by default.
- **Model loads** (47.8 s; GPU LRU 3609 slots = 9.29 GiB; KV resolved=8192 tokens;
  free-after-weights=23.27 GiB, free-after-startup=5.91 GiB).
- **LRU is dynamic** (`expert_gpu_slot_count`, real_program.cpp:486): sizes to leftover
  VRAM (free − 512 MiB) at ctor time, *before* KV sizing. It grabs 9.29 GiB of ~9.8 GiB
  free, leaving KV at 8192 tokens. It does **not** shrink when `max_context` grows.
- **`isolated_request_feasible` (program.cpp:1363) is a stub — always `true`.**
  Over-large requests are not rejected early; they enter the queue and time out
  (`QueueTimeout` "expired while waiting for admission").
- **`barrier_begin`/`barrier_end` (1188/1208) are timing markers only** (no-op when
  timing off) — not a real barrier; not a deadlock source. Ruled out.
- **Two concrete gaps vs the reference:**
  - **KV dtype:** ours is **BF16 (2 B/token)**; reference uses **Q8_0 (1 B/token)** →
    our KV is 2x heavier per token, so our KV capacity is ~half the reference's.
  - **LRU eats VRAM:** a 9.29 GiB LRU is allocated before KV sizing, so KV is capped.
    The reference targets "KV-cache VRAM, Q8_0, ~2GB @100K" with `-ncmoe 30` (30 layers'
    experts on RAM).

## Root cause (leading)

The dynamic LRU (`expert_gpu_slot_count`) consumes all leftover VRAM at ctor time, so KV
is capped at 8192 tokens. A real request via Claude Desktop (system prompt ~10K+ tokens)
exceeds 8192 → cannot be admitted → `QueueTimeout`. Because `isolated_request_feasible`
is a stub, it is never rejected early; it just queues and times out.

## Plan

### Phase 1 — Reproduce + localize (no behavior change)
- Rebuild: `cmd //c "tools\flash_next_dev\p1_build.bat"`.
- Launch flash_next serve on a **free port** (not 8888 — the 27B is there, PID 24984,
  do not touch) in a **visible terminal**, `NINFER_P13_TIMING=1`.
- Send a "hej" request; capture the serve log + P13 timing dump (which segment never
  finishes: B_MOE_ROUTER / B_MOE_SCATTER / B_MOE_GEMM).
- **Bisect** `NINFER_MOE_CPU_LAYERS` (unset vs `=1`) to isolate the CPU-offload path.
- Thread stack dump at the stall (`procdump -ma -pm <pid>`) to name the blocked thread.
- Compare against the reference: read
  `C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\docs\superpowers\specs\
  2026-09-23-qwen38-flash-next-llamacpp-design.md` and `serve.ps1` to see how the
  reference admits/processes a request and where ours diverges.

### Phase 2 — Root-cause (test one at a time, smallest first)
- **H1 — KV too small (leading):** request context (~10K+ tokens) > KV capacity (8192)
  → TemporarilyBlocked → QueueTimeout. Fix: yield LRU VRAM to KV.
- **H2 — MoE-dispatch stall:** a `cudaStreamSynchronize` (real_program.cpp:1660/1713)
  blocks on a stream op that never completes. T-dependent (warmup T=4 passes; prefill
  T=thousands hangs). Ruled out if the P13 timing dump shows all segments completing.
- **H3 — LRU-acquire stall:** 3609/7218 slots; a miss storm wedges `promote`.

### Phase 3 — Fix + verify
- Smallest fix that removes the stall; no unrelated refactors.
- Rebuild: `cmd //c "tools\flash_next_dev\p1_rebuild_tests.bat"`.
- Re-run serve; "hej" returns tokens in seconds.

### Step 4 — VRAM/KV tradeoff ("higher context → fewer VRAM experts")
- Make the LRU slot count yield VRAM to KV as `max_context` grows: reserve VRAM for KV
  first (from `max_context`), then size the LRU to the remainder. Mirrors llama.cpp
  `-ncmoe`. Optionally switch KV to FP8 to halve per-token cost (parity with reference).

## Verification
- `cd core\build && ctest -L flash_next --timeout 420 --output-on-failure` green.
- Serve serves "hej" in seconds at the configured context.
- 27B serve (PID 24984, port 8888) left untouched.

## Constraints (in force)
- Long-running/kill scripts in a **visible terminal** (user watches live).
- Do **not** stop the deployed 27B serve (PID 24984) without an explicit ask.
- `core` is a symlink — never delete. `models/` gitignored — never `git add`.
- Verify paths by hex/`od -c` (f/n mangle).
