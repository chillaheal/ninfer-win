#pragma once
// P5 v1 contract — `gated_residual` (HyperConnection). A NEW Op family: do
// not reuse the qwen3_6 residual helpers. The production modeling code is
// unrecoverable (the artifact ships only the object shapes), so this is a
// documented v1 math contract anchored on those shapes:
//   input_mix_weight_down  [HC_LOWRANK, HC_STREAM]  (320 x 10240 prod)
//   input_mix_weight_up    [HC_STREAM, HC_LOWRANK]  (10240 x 320)
//   block_inject_weight    [HC_COUNT, HC_STREAM]    (4 x 10240)
//   hc_norm                [HC_STREAM]
// where HC_STREAM = HC_COUNT * HIDDEN (4 x 2560 = 10240): the residual stream
// is hc_count copies of the hidden state.

#include <cuda_runtime.h>

namespace ninfer::ops {

// Math (per token t; all tensors fp32 on device, row-major):
//   stream    [T, S]   S = B*H — the residual stream
//   block_out [T, H]   the block's output
//   w_down    [R, S], w_up [S, R], w_inj [B, S], norm_w [S]
//
// read (block input x [T, H]):
//   h[t,r] = sum_c w_down[r,c] * stream[t,c]         (low-rank down)
//   a[t,r] = gelu_exact(h[t,r])
//   g[t,s] = sigmoid(sum_r w_up[s,r] * a[t,r])       (per-dimension gate)
//   rms[t,b] = sqrt( (1/H) * sum_c stream[t,b*H+c]^2 )
//   x[t,c] = sum_b g[t,b*H+c] * stream[t,b*H+c] / max(rms[t,b], 1e-8)
//                 * norm_w[b*H+c]
//   (norm_branches=false: x[t,c] = sum_b g[t,b*H+c] * stream[t,b*H+c])
//
// write (updated stream [T, S]):
//   stream'[t,s] = stream[t,s] + sigmoid(w_inj[s/H, s]) * block_out[t, s%H]
//
// v1 choices (BLOCKER: exact production function is unrecoverable):
//   - activation = exact GELU: gelu(x) = 0.5*x*(1+erf(x/sqrt(2)))
//   - read gate  = sigmoid over the low-rank mixer output (per stream dim)
//   - write gate = sigmoid of the per-(branch, dim) block_inject weight
//   - per-branch RMSNorm gamma = hc_norm (stream-wide vector, branch b
//     normalizes its own H-wide segment); norm_branches mirrors the artifact
//     config (hc_norm always present -> target config true)
//
// Non-finite policy: the Op computes the documented formulas exactly per
// IEEE-754 and never traps (no denormal flush, no traps). The only guarded
// op is the branch RMS division (1e-8 floor: an all-zero branch yields a
// zero normed contribution instead of 0/0). Consequences (tested against
// the FP64 oracle's per-element pattern — finite / NaN / +/-inf):
//   - all-zero stream -> finite outputs with x == 0;
//   - stream[t,s] = +inf -> the inf also enters the low-rank down GEMV, so
//     NaN can appear in several x[t,c] (0*inf = NaN, gelu(-inf) =
//     -inf*(1-1) = -inf*0 = NaN), branch RMS = inf makes other branch-0
//     normed terms finite/inf = 0, and stream'[t,s] = +inf. Exact positions
//     are weight-dependent; the test asserts pattern agreement with the
//     oracle, not exact positions.
//
// fp32 throughout (bf16 artifact weights are converted by the P9 binder);
// prefill T>1 and decode T=1 share one code path. v1 geometry limits:
// B == 4, S = 4*H with S % 1024 == 0, S <= 10240, H <= 2560, R <= 320.
// The Op performs no device allocation; it only launches on `stream`.

struct GatedResidualParams {
  int tokens = 0;       // T >= 1
  int hidden = 0;       // H
  int hc_count = 4;     // B (v1: must be 4)
  int lowrank = 0;      // R
  const float* stream_dev = nullptr;     // [T, S]
  const float* block_out_dev = nullptr;  // [T, H]
  const float* w_down_dev = nullptr;     // [R, S]
  const float* w_up_dev = nullptr;       // [S, R]
  const float* w_inj_dev = nullptr;      // [B, S]
  const float* norm_w_dev = nullptr;     // [S]
  float* read_out_dev = nullptr;         // [T, H]
  float* stream_out_dev = nullptr;       // [T, S], distinct from stream_dev
  bool norm_branches = true;
};

[[noreturn]] void gr_fail(const char* what);
void gr_launch(const GatedResidualParams& p, cudaStream_t stream);

void gated_residual(const GatedResidualParams& p, cudaStream_t stream);

}  // namespace ninfer::ops
