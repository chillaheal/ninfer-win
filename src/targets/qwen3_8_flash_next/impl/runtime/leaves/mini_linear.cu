// Flash-Next P9 — mini linear + embedding leaves.
//
// One thread per output element; FP32 accumulation over k in ascending
// order — the fixed bit-deterministic contract.

#include "targets/qwen3_8_flash_next/impl/runtime/leaves/mini_leaves.h"

#include "mini_leaves_common.cuh"

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace {

constexpr int kMiniVocab = 256;
constexpr int kMiniHidden = 256;

__global__ void mini_embedding_gather_kernel(const std::int32_t* __restrict__ ids,
                                             const std::uint8_t* __restrict__ code,
                                             const std::uint16_t* __restrict__ scale,
                                             int T, __nv_bfloat16* __restrict__ h) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * kMiniHidden) {
    return;
  }
  const int t = idx / kMiniHidden;
  const int c = idx - t * kMiniHidden;
  const int id = ids[t];
  const float w = e4m3_decode(code[static_cast<std::size_t>(id) * kMiniHidden + c]) *
                  bf16_decode(scale[id]);
  h[idx] = __float2bfloat16(w);
}

__global__ void mini_fp8_linear_kernel(const __nv_bfloat16* __restrict__ x, int T, int K,
                                       const std::uint8_t* __restrict__ code,
                                       const std::uint16_t* __restrict__ scale, int N,
                                       __nv_bfloat16* __restrict__ out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * T) {
    return;
  }
  const int n = idx / T;
  const int t = idx - n * T;
  const float row_scale = bf16_decode(scale[n]);
  const __nv_bfloat16* x_row = x + static_cast<std::size_t>(t) * K;
  const std::uint8_t* w_row = code + static_cast<std::size_t>(n) * K;
  float acc = 0.0F;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    acc += __bfloat162float(x_row[k]) * (e4m3_decode(w_row[k]) * row_scale);
  }
  out[idx] = __float2bfloat16(acc);
}

// One thread per (t, e): logits[t, e] = sum_k bf16(router[e, k]) * bf16(x[t, k]).
__global__ void mini_router_kernel(const __nv_bfloat16* __restrict__ x, int T,
                                   const std::uint16_t* __restrict__ router,
                                   float* __restrict__ logits) {
  constexpr int kExperts = 8;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * kExperts) {
    return;
  }
  const int t = idx / kExperts;
  const int e = idx - t * kExperts;
  const std::uint16_t* w = router + static_cast<std::size_t>(e) * kMiniHidden;
  const __nv_bfloat16* xr = x + static_cast<std::size_t>(t) * kMiniHidden;
  float acc = 0.0F;
#pragma unroll
  for (int k = 0; k < kMiniHidden; ++k) {
    acc += bf16_decode(w[k]) * __bfloat162float(xr[k]);
  }
  logits[idx] = acc;
}

}  // namespace

void mini_embedding_gather(const std::int32_t* ids, const std::uint8_t* code,
                           const std::uint16_t* scale, int T, __nv_bfloat16* h,
                           cudaStream_t stream) {
  dim3 grid((T * kMiniHidden + 255) / 256);
  mini_embedding_gather_kernel<<<grid, 256, 0, stream>>>(ids, code, scale, T, h);
}

void mini_fp8_linear(const __nv_bfloat16* x, int T, int K, const std::uint8_t* code,
                     const std::uint16_t* scale, int N, __nv_bfloat16* out,
                     cudaStream_t stream) {
  dim3 grid((N * T + 255) / 256);
  mini_fp8_linear_kernel<<<grid, 256, 0, stream>>>(x, T, K, code, scale, N, out);
}

void mini_router(const __nv_bfloat16* x, int T, const __nv_bfloat16* router,
                 float* logits, cudaStream_t stream) {
  dim3 grid((T * 8 + 255) / 256);
  mini_router_kernel<<<grid, 256, 0, stream>>>(x, T, reinterpret_cast<const std::uint16_t*>(router),
                                               logits);
}

}  // namespace ninfer::targets::qwen3_8_flash_next::detail
