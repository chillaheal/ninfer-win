// Flash-Next P9 — layout + hyper-connection glue leaves.
//
// Transposes, the 4-branch stream reduction (v1), the final collapse, the
// block-output builder and the PLE add. All element-wise with fixed FP32
// intermediate math; no cross-thread accumulation except none at all.

#include "targets/qwen3_8_flash_next/impl/runtime/leaves/mini_leaves.h"

#include "mini_leaves_common.cuh"

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace {

constexpr int kMiniHidden = 256;
constexpr int kHcCount = 4;
constexpr int kStreamWidth = kHcCount * kMiniHidden;  // 1024
constexpr int kPleEmbedDim = 128;

__global__ void mini_transpose_tm_cm_kernel(const __nv_bfloat16* __restrict__ x, int T, int K,
                                            __nv_bfloat16* __restrict__ out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K * T) {
    return;
  }
  const int k = idx / T;
  const int t = idx - k * T;
  out[idx] = x[static_cast<std::size_t>(t) * K + k];
}

__global__ void mini_transpose_cm_tm_kernel(const __nv_bfloat16* __restrict__ x, int T, int K,
                                            __nv_bfloat16* __restrict__ out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K * T) {
    return;
  }
  const int t = idx / K;
  const int k = idx - t * K;
  out[idx] = x[static_cast<std::size_t>(k) * T + t];
}

// One thread per (t, b, c): stream[t, b*256 + c] = f32(h[t, c]).
__global__ void mini_stream_init_kernel(const __nv_bfloat16* __restrict__ h, int T,
                                        float* __restrict__ stream) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = T * kStreamWidth;
  if (idx >= total) {
    return;
  }
  const int t = idx / kStreamWidth;
  const int c = idx - t * kStreamWidth;
  const int b = c / kMiniHidden;
  const int cc = c - b * kMiniHidden;
  stream[idx] = __bfloat162float(h[static_cast<std::size_t>(t) * kMiniHidden + cc]);
}

// One thread per (t, c): mean over the 4 branches, fixed order b = 0..3.
__global__ void mini_final_collapse_kernel(const float* __restrict__ stream, int T,
                                           __nv_bfloat16* __restrict__ h_final) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * kMiniHidden) {
    return;
  }
  const int t = idx / kMiniHidden;
  const int c = idx - t * kMiniHidden;
  const std::size_t base = static_cast<std::size_t>(t) * kStreamWidth;
  float acc = 0.0F;
#pragma unroll
  for (int b = 0; b < kHcCount; ++b) {
    acc += stream[base + static_cast<std::size_t>(b) * kMiniHidden + c];
  }
  h_final[idx] = __float2bfloat16(acc * 0.25F);
}

// One thread per (t, c): block_out[t, c] = f32(y[t, c])  ->  [T, 256] FP32
// (the gated_residual write call reads block_out[t, s%H] with H=256).
__global__ void mini_block_out_kernel(const __nv_bfloat16* __restrict__ y, int T,
                                      float* __restrict__ block_out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * kMiniHidden) {
    return;
  }
  block_out[idx] = __bfloat162float(y[idx]);
}

// One thread per element: out[t, c] = bf16(x[t, c])  ->  [T, H] BF16.
__global__ void mini_f32_to_bf16_kernel(const float* __restrict__ x, int T, int H,
                                        __nv_bfloat16* __restrict__ out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * H) {
    return;
  }
  out[idx] = __float2bfloat16(x[idx]);
}

// One thread per (t, c): x[t,c] = bf16(f32(x[t,c]) + mem[t, c % 128]).
__global__ void mini_ple_add_kernel(__nv_bfloat16* __restrict__ x,
                                    const float* __restrict__ mem, int T) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * kMiniHidden) {
    return;
  }
  const int t = idx / kMiniHidden;
  const int c = idx - t * kMiniHidden;
  x[idx] = __float2bfloat16(
      __bfloat162float(x[idx]) + mem[static_cast<std::size_t>(t) * kPleEmbedDim + c % kPleEmbedDim]);
}

// GDN zero-pad pack: one thread per (d, h, t) of the 128-wide d extent; the
// mini channels occupy d in [0,32).
__global__ void mini_gdn_pack_kernel(const __nv_bfloat16* __restrict__ conv_out, int T,
                                     __nv_bfloat16* __restrict__ q_op,
                                     __nv_bfloat16* __restrict__ k_op,
                                     __nv_bfloat16* __restrict__ v_op) {
  // q/k: 128 x 2 x T ; v: 128 x 4 x T.
  const int qk_total = 128 * 2 * T;
  const int v_total = 128 * 4 * T;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < qk_total) {
    const int t = idx % T;
    const int rem = idx / T;
    const int h = rem % 2;
    const int d = rem / 2;
    const __nv_bfloat16 zero{};
    const int q_src = 32 * h + d;
    const int k_src = 64 + 32 * h + d;
    q_op[idx] = (d < 32) ? conv_out[static_cast<std::size_t>(q_src) * T + t] : zero;
    k_op[idx] = (d < 32) ? conv_out[static_cast<std::size_t>(k_src) * T + t] : zero;
    return;
  }
  const int v_idx = idx - qk_total;
  if (v_idx >= v_total) {
    return;
  }
  const int t = v_idx % T;
  const int rem = v_idx / T;
  const int h = rem % 4;
  const int d = rem / 4;
  const __nv_bfloat16 zero{};
  const int v_src = 128 + 32 * h + d;
  v_op[v_idx] = (d < 32) ? conv_out[static_cast<std::size_t>(v_src) * T + t] : zero;
}

// One thread per (t, d): z[t, d] = qkvz[t, 256 + d] (qkvz row stride 384).
__global__ void mini_gdn_zslice_kernel(const __nv_bfloat16* __restrict__ qkvz, int T,
                                       __nv_bfloat16* __restrict__ z) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * 128) {
    return;
  }
  const int t = idx / 128;
  const int d = idx - t * 128;
  z[idx] = qkvz[static_cast<std::size_t>(t) * 384 + 256 + d];
}

}  // namespace

void mini_transpose_tm_cm(const __nv_bfloat16* x, int T, int K, __nv_bfloat16* out,
                          cudaStream_t stream) {
  dim3 grid((K * T + 255) / 256);
  mini_transpose_tm_cm_kernel<<<grid, 256, 0, stream>>>(x, T, K, out);
}

void mini_transpose_cm_tm(const __nv_bfloat16* x, int T, int K, __nv_bfloat16* out,
                          cudaStream_t stream) {
  dim3 grid((K * T + 255) / 256);
  mini_transpose_cm_tm_kernel<<<grid, 256, 0, stream>>>(x, T, K, out);
}

void mini_stream_init(const __nv_bfloat16* h, int T, float* stream, cudaStream_t stream_) {
  dim3 grid((T * kStreamWidth + 255) / 256);
  mini_stream_init_kernel<<<grid, 256, 0, stream_>>>(h, T, stream);
}

void mini_final_collapse(const float* stream, int T, __nv_bfloat16* h_final,
                         cudaStream_t stream_) {
  dim3 grid((T * kMiniHidden + 255) / 256);
  mini_final_collapse_kernel<<<grid, 256, 0, stream_>>>(stream, T, h_final);
}

void mini_block_out(const __nv_bfloat16* y, int T, float* block_out, cudaStream_t stream_) {
  dim3 grid((T * kMiniHidden + 255) / 256);
  mini_block_out_kernel<<<grid, 256, 0, stream_>>>(y, T, block_out);
}

void mini_f32_to_bf16(const float* x, int T, int H, __nv_bfloat16* out, cudaStream_t stream) {
  dim3 grid((T * H + 255) / 256);
  mini_f32_to_bf16_kernel<<<grid, 256, 0, stream>>>(x, T, H, out);
}

void mini_ple_add(__nv_bfloat16* x, const float* mem, int T, cudaStream_t stream_) {
  dim3 grid((T * kMiniHidden + 255) / 256);
  mini_ple_add_kernel<<<grid, 256, 0, stream_>>>(x, mem, T);
}

void mini_gdn_pack(const __nv_bfloat16* conv_out, int T, __nv_bfloat16* q_op,
                   __nv_bfloat16* k_op, __nv_bfloat16* v_op, cudaStream_t stream) {
  const int total = 128 * 2 * T + 128 * 4 * T;
  dim3 grid((total + 255) / 256);
  mini_gdn_pack_kernel<<<grid, 256, 0, stream>>>(conv_out, T, q_op, k_op, v_op);
}

void mini_gdn_zslice(const __nv_bfloat16* qkvz, int T, __nv_bfloat16* z,
                     cudaStream_t stream) {
  dim3 grid((T * 128 + 255) / 256);
  mini_gdn_zslice_kernel<<<grid, 256, 0, stream>>>(qkvz, T, z);
}

}  // namespace ninfer::targets::qwen3_8_flash_next::detail
