// Flash-Next P12 S2 — real embedding + router glue leaves (target-private).
//
// Real-geometry generalizations of the P9 mini embedding_gather / router
// leaves: the hidden width K and the expert count E are runtime parameters
// (2560 / 512 real). Reuses the shared e4m3/BF16 decoders from
// mini_leaves_common.cuh so device and host reference agree bit-for-bit.
// Every accumulation is fixed-order; no atomics.

#include "targets/qwen3_8_flash_next/impl/runtime/leaves/real_leaves.h"

#include "mini_leaves_common.cuh"

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace {

// One thread per (t, c): h[t, c] = bf16( e4m3(code[id*K + c]) * bf16(scale[id]) ).
__global__ void real_embedding_gather_kernel(const std::int32_t* __restrict__ ids,
                                             const std::uint8_t* __restrict__ code,
                                             const std::uint16_t* __restrict__ scale,
                                             int T, int K, __nv_bfloat16* __restrict__ h) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * K) {
    return;
  }
  const int t = idx / K;
  const int c = idx - t * K;
  const std::int32_t id = ids[t];
  h[idx] = __float2bfloat16(e4m3_decode(code[static_cast<std::size_t>(id) * K + c]) *
                            bf16_decode(scale[id]));
}

// One thread per (t, e): logits[t, e] = Sum_k bf16(router[e*K + k]) *
// bf16(x[t, k]), k ascending.
__global__ void real_router_kernel(const __nv_bfloat16* __restrict__ x, int T, int K, int E,
                                   const std::uint16_t* __restrict__ router,
                                   float* __restrict__ logits) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * E) {
    return;
  }
  const int t = idx / E;
  const int e = idx - t * E;
  const __nv_bfloat16* xr = x + static_cast<std::size_t>(t) * K;
  const std::uint16_t* wr = router + static_cast<std::size_t>(e) * K;
  float acc = 0.0F;
  for (int k = 0; k < K; ++k) {
    acc += bf16_decode(wr[k]) * __bfloat162float(xr[k]);
  }
  logits[idx] = acc;
}

}  // namespace

void real_embedding_gather(const std::int32_t* ids, const std::uint8_t* code,
                           const std::uint16_t* scale, int T, int K, __nv_bfloat16* h,
                           cudaStream_t stream) {
  const int total = T * K;
  const int blocks = (total + 255) / 256;
  real_embedding_gather_kernel<<<blocks, 256, 0, stream>>>(
      ids, code, reinterpret_cast<const std::uint16_t*>(scale), T, K, h);
}

void real_router(const __nv_bfloat16* x, int T, int K, int E, const __nv_bfloat16* router,
                 float* logits, cudaStream_t stream) {
  const int total = T * E;
  const int blocks = (total + 255) / 256;
  real_router_kernel<<<blocks, 256, 0, stream>>>(
      x, T, K, E, reinterpret_cast<const std::uint16_t*>(router), logits);
}

}  // namespace ninfer::targets::qwen3_8_flash_next::detail
