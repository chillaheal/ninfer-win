// Flash-Next P12 S2 — real GDN gating + QSA attention glue leaves
// (target-private).
//
// Real-geometry generalizations of the P9 mini gdn_gating / gated_rmsnorm /
// head_rmsnorm / rope / kv_append leaves. The GDN glue matches the P6
// gated_delta_net contract (out [128,Hv,T] d-fastest, i.e. memory
// (t*Hv+h)*128 + d) and the QSA glue matches the P7 contract: per-head RMSNorm
// at head dim 256, split-half NeoX RoPE (D=256, R=64, theta=10000 — the shared
// rope op's D256/R64 Text domain), and the paged KV append into SEPARATE
// K/V page pools via the paged_kv_element_offset<256,KV> formula (head extent
// at runtime). Every reduction is fixed-order; no atomics. The two
// in-place RMSNorm leaves read a lane from memory immediately before its own
// store (and no other lane reads it), so the store is safe without holding the
// whole row in registers (the real head dim 256 exceeds the per-thread
// register budget).

#include "targets/qwen3_8_flash_next/impl/runtime/leaves/real_leaves.h"

#include "mini_leaves_common.cuh"

#include <ops/kernel/paged_kv_address.cuh>

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace {

// Real QSA head dim (the paged-KV leading extent and the RoPE head width).
constexpr int kRealQsaD = 256;

// One thread per (t, h): a_h = A_h . x[t], b_h = B_h . x[t]. g/beta are
// [H][T] (the gated_delta_net contract).
__global__ void real_gdn_gating_kernel(const __nv_bfloat16* __restrict__ x, int T, int K,
                                       int H, const std::uint16_t* __restrict__ a_weight,
                                       const std::uint16_t* __restrict__ b_weight,
                                       const float* __restrict__ a_log,
                                       const float* __restrict__ dt_bias,
                                       float* __restrict__ g, float* __restrict__ beta) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * H) {
    return;
  }
  const int t = idx / H;
  const int h = idx - t * H;
  const __nv_bfloat16* xr = x + static_cast<std::size_t>(t) * K;
  const std::uint16_t* aw = a_weight + static_cast<std::size_t>(h) * K;
  const std::uint16_t* bw = b_weight + static_cast<std::size_t>(h) * K;
  float a = 0.0F;
  float b = 0.0F;
  for (int k = 0; k < K; ++k) {
    const float xv = __bfloat162float(xr[k]);
    a += bf16_decode(aw[k]) * xv;
    b += bf16_decode(bw[k]) * xv;
  }
  const std::size_t out = static_cast<std::size_t>(h) * T + t;
  g[out] = -expf(a_log[h]) * mini_softplus(a + dt_bias[h]);
  beta[out] = mini_sigmoid(b);
}

// Fused gated RMSNorm: one thread per (t, h). x = o[(t*H+h)*S + d] (d in
// [0,D)); onx[t, h*D + d] = bf16( x * inv_r * bf16(w[d]) * silu(f32(z[...])) ).
// o and onx are disjoint, so o is re-read from memory (no register row).
__global__ void real_gated_rmsnorm_kernel(const __nv_bfloat16* __restrict__ o,
                                          const __nv_bfloat16* __restrict__ z,
                                          const std::uint16_t* __restrict__ w, int T, int H,
                                          int D, int S, __nv_bfloat16* __restrict__ onx) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * H) {
    return;
  }
  const int t = idx / H;
  const int h = idx - t * H;
  const __nv_bfloat16* orow = o + (static_cast<std::size_t>(t) * H + h) * S;
  const __nv_bfloat16* zrow =
      z + (static_cast<std::size_t>(t) * H + h) * D;
  __nv_bfloat16* orow_out = onx + (static_cast<std::size_t>(t) * H + h) * D;
  float ss = 0.0F;
  for (int d = 0; d < D; ++d) {
    const float xv = __bfloat162float(orow[d]);
    ss += xv * xv;
  }
  const float inv_r = 1.0F / sqrtf(ss * (1.0F / static_cast<float>(D)) + 1e-6F);
  for (int d = 0; d < D; ++d) {
    const float xv = __bfloat162float(orow[d]);
    const float zv = __bfloat162float(zrow[d]);
    orow_out[d] =
        __float2bfloat16(xv * inv_r * bf16_decode(w[d]) * mini_silu(zv));
  }
}

// Per-head RMSNorm (in place): read each lane before its own store.
__global__ void real_head_rmsnorm_kernel(__nv_bfloat16* __restrict__ x, int T, int H, int D,
                                         const std::uint16_t* __restrict__ g) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * H) {
    return;
  }
  __nv_bfloat16* row = x + static_cast<std::size_t>(idx) * D;
  float ss = 0.0F;
  for (int d = 0; d < D; ++d) {
    const float xv = __bfloat162float(row[d]);
    ss += xv * xv;
  }
  const float inv = 1.0F / sqrtf(ss * (1.0F / static_cast<float>(D)) + 1e-6F);
  for (int d = 0; d < D; ++d) {
    const float xv = __bfloat162float(row[d]);
    row[d] = __float2bfloat16(xv * bf16_decode(g[d]) * inv);
  }
}

// Split-half NeoX RoPE, D=256, R=64. One thread per (t, h, i), i in [0,32):
// pair (i, i+32), angle = (pos0 + t) * 10000^(-i/32); dims [64,256) pass.
__global__ void real_rope_kernel(__nv_bfloat16* __restrict__ x, int T, int H, int pos0) {
  constexpr int R = 64;
  constexpr int D = kRealQsaD;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = T * H * (R / 2);
  if (idx >= total) {
    return;
  }
  const int i = idx % (R / 2);
  const int th = idx / (R / 2);
  const int h = th % H;
  const int t = th / H;
  const float angle =
      static_cast<float>(pos0 + t) * powf(10000.0F, -(2.0F * static_cast<float>(i)) / R);
  const float cs = cosf(angle);
  const float sn = sinf(angle);
  __nv_bfloat16* row = x + (static_cast<std::size_t>(t) * H + h) * D;
  const float x0 = __bfloat162float(row[i]);
  const float x1 = __bfloat162float(row[i + R / 2]);
  row[i]       = __float2bfloat16(x0 * cs - x1 * sn);
  row[i + R / 2] = __float2bfloat16(x1 * cs + x0 * sn);
}

// Paged KV append (real, D=256): one thread per (t, h, d); writes k and v at
// absolute position pos + t into SEPARATE page pools using the
// paged_kv_element_offset<256,KV> formula (head extent at runtime).
__global__ void real_kv_append_kernel(const __nv_bfloat16* __restrict__ k,
                                      const __nv_bfloat16* __restrict__ v, int T, int pos,
                                      int KV, const std::int32_t* __restrict__ block_table,
                                      __nv_bfloat16* __restrict__ k_pages,
                                      __nv_bfloat16* __restrict__ v_pages) {
  constexpr int D = kRealQsaD;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * KV * D) {
    return;
  }
  const int d = idx % D;
  const int rem = idx / D;
  const int h = rem % KV;
  const int t = rem / KV;
  const int p = pos + t;
  const int page = block_table[p >> ninfer::ops::kPagedKVPageShift];
  const int off = p & ninfer::ops::kPagedKVPageMask;
  const std::int64_t base =
      static_cast<std::int64_t>(D) * ninfer::kPagedKVPageSize *
          (static_cast<std::int64_t>(h) + static_cast<std::int64_t>(KV) * page) +
      static_cast<std::int64_t>(D) * off + d;
  k_pages[base] = k[idx];
  v_pages[base] = v[idx];
}

// QSA output-side head reduction 48 -> 24 (SwiGLU split-half gate, the
// S4-confirmed v1). One thread per (t, h, d) for the Q/2 reduced heads:
// reduced[t, h*D + d] = bf16( f32(qsa[t, h*D + d]) *
// silu( f32(qsa[t, (h+Q/2)*D + d]) ) ). qsa [T][Q][D] token-major, reduced
// [T][Q/2][D] token-major. Q is the (even) query-head count (real 48).
__global__ void real_qsa_gate_kernel(const __nv_bfloat16* __restrict__ qsa, int T, int Q, int D,
                                     __nv_bfloat16* __restrict__ reduced) {
  const int Qr = Q / 2;  // split-half: 48 -> 24
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * Qr * D) {
    return;
  }
  const int d = idx % D;
  const int rem = idx / D;
  const int h = rem % Qr;
  const int t = rem / Qr;
  const __nv_bfloat16* arow = qsa + (static_cast<std::size_t>(t) * Q + h) * D;
  const __nv_bfloat16* brow = qsa + (static_cast<std::size_t>(t) * Q + (h + Qr)) * D;
  const float av = __bfloat162float(arow[d]);
  const float bv = __bfloat162float(brow[d]);
  reduced[static_cast<std::size_t>(t) * Qr * D + static_cast<std::size_t>(h) * D + d] =
      __float2bfloat16(av * mini_silu(bv));
}

}  // namespace

void real_gdn_gating(const __nv_bfloat16* x, int T, int K, int H,
                     const __nv_bfloat16* a_weight, const __nv_bfloat16* b_weight,
                     const float* a_log, const float* dt_bias, float* g, float* beta,
                     cudaStream_t stream) {
  const int total = T * H;
  const int blocks = (total + 255) / 256;
  real_gdn_gating_kernel<<<blocks, 256, 0, stream>>>(
      x, T, K, H, reinterpret_cast<const std::uint16_t*>(a_weight),
      reinterpret_cast<const std::uint16_t*>(b_weight), a_log, dt_bias, g, beta);
}

void real_gated_rmsnorm(const __nv_bfloat16* o, const __nv_bfloat16* z,
                        const __nv_bfloat16* w, int T, int H, int D, int S,
                        __nv_bfloat16* onx, cudaStream_t stream) {
  const int total = T * H;
  const int blocks = (total + 255) / 256;
  real_gated_rmsnorm_kernel<<<blocks, 256, 0, stream>>>(
      o, z, reinterpret_cast<const std::uint16_t*>(w), T, H, D, S, onx);
}

void real_head_rmsnorm(__nv_bfloat16* x, int T, int H, int D, const __nv_bfloat16* g,
                       cudaStream_t stream) {
  const int total = T * H;
  const int blocks = (total + 255) / 256;
  real_head_rmsnorm_kernel<<<blocks, 256, 0, stream>>>(x, T, H, D,
                                                       reinterpret_cast<const std::uint16_t*>(g));
}

void real_rope(__nv_bfloat16* q, int T, int Hq, int pos0, __nv_bfloat16* k, int Hk,
               cudaStream_t stream) {
  {
    const int total = T * Hq * 32;
    const int blocks = (total + 255) / 256;
    real_rope_kernel<<<blocks, 256, 0, stream>>>(q, T, Hq, pos0);
  }
  {
    const int total = T * Hk * 32;
    const int blocks = (total + 255) / 256;
    real_rope_kernel<<<blocks, 256, 0, stream>>>(k, T, Hk, pos0);
  }
}

void real_kv_append(const __nv_bfloat16* k, const __nv_bfloat16* v, int T, int pos, int KV,
                    const std::int32_t* block_table, __nv_bfloat16* k_pages,
                    __nv_bfloat16* v_pages, cudaStream_t stream) {
  const int total = T * KV * kRealQsaD;
  const int blocks = (total + 255) / 256;
  real_kv_append_kernel<<<blocks, 256, 0, stream>>>(k, v, T, pos, KV, block_table, k_pages,
                                                    v_pages);
}

void real_qsa_gate(const __nv_bfloat16* qsa, int T, int Q, int D, __nv_bfloat16* reduced,
                   cudaStream_t stream) {
  const int Qr = Q / 2;
  const int total = T * Qr * D;
  const int blocks = (total + 255) / 256;
  real_qsa_gate_kernel<<<blocks, 256, 0, stream>>>(qsa, T, Q, D, reduced);
}

}  // namespace ninfer::targets::qwen3_8_flash_next::detail
