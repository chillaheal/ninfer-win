// Flash-Next P9 — mini GDN gating + QSA attention leaves (target-private).
//
// GDN gating, the fused gated RMSNorm (design step 6), per-head RMSNorm, the
// v1 RoPE (D=32, R=8), the paged KV append, the round-local mini indexer and
// the mini sparse GQA. Every reduction is fixed-order; no atomics.
//
// Mini KV layout (one 16,384 B physical page per lane slot): the K plane
// occupies the first 8,192 B, the V plane the second 8,192 B. Each plane is
// addressed by the paged_kv_element_offset<32,2> formula on its own base
// (P7 separate-plane semantics): plane element (h, po, d) =
// 32*64*h + 32*po + d.

#include "targets/qwen3_8_flash_next/impl/runtime/leaves/mini_leaves.h"

#include "mini_leaves_common.cuh"

#include <algorithm>
#include <cfloat>

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace {

constexpr int kGdnHeads = 4;
constexpr int kMiniHidden = 256;
constexpr int kQsaQ = 4;
constexpr int kQsaKV = 2;
constexpr int kQsaD = 32;
constexpr int kIdxChunks = 4;
constexpr int kIdxDim = 32;
constexpr int kIdxZ = 160;             // q (32) + 4 k chunks (32 each)
constexpr int kBudget = 64;
constexpr std::size_t kPageElements = 8192;  // 16,384 B / 2
constexpr std::size_t kVPlaneOffset = 4096;  // V plane start (elements)

// ---------------------------------------------------------------------------
// GDN gating
// ---------------------------------------------------------------------------

// One thread per (t, h): a_h = A_h . x[t], b_h = B_h . x[t].
// g/beta are [4][T] (the gated_delta_net contract).
__global__ void mini_gdn_gating_kernel(const __nv_bfloat16* __restrict__ x, int T,
                                       const std::uint16_t* __restrict__ a_weight,
                                       const std::uint16_t* __restrict__ b_weight,
                                       const float* __restrict__ a_log,
                                       const float* __restrict__ dt_bias,
                                       float* __restrict__ g, float* __restrict__ beta) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * kGdnHeads) {
    return;
  }
  const int t = idx / kGdnHeads;
  const int h = idx - t * kGdnHeads;
  const __nv_bfloat16* xr = x + static_cast<std::size_t>(t) * kMiniHidden;
  const std::uint16_t* aw = a_weight + static_cast<std::size_t>(h) * kMiniHidden;
  const std::uint16_t* bw = b_weight + static_cast<std::size_t>(h) * kMiniHidden;
  float a = 0.0F;
  float b = 0.0F;
#pragma unroll
  for (int k = 0; k < kMiniHidden; ++k) {
    const float xv = __bfloat162float(xr[k]);
    a += bf16_decode(aw[k]) * xv;
    b += bf16_decode(bw[k]) * xv;
  }
  const std::size_t out = static_cast<std::size_t>(h) * T + t;
  g[out] = -expf(a_log[h]) * mini_softplus(a + dt_bias[h]);
  beta[out] = mini_sigmoid(b);
}

// ---------------------------------------------------------------------------
// fused gated RMSNorm (design step 6): one thread per (t, h)
// ---------------------------------------------------------------------------

// o_op is [128,4,T] d-fastest; head h's 32-dim output occupies d in [0,32),
// so element (t, h, d) = t*512 + h*128 + d.
__global__ void mini_gated_rmsnorm_kernel(const __nv_bfloat16* __restrict__ o,
                                          const __nv_bfloat16* __restrict__ z,
                                          const std::uint16_t* __restrict__ w, int T,
                                          __nv_bfloat16* __restrict__ onx) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * kQsaQ) {
    return;
  }
  const int t = idx / kQsaQ;
  const int h = idx - t * kQsaQ;
  const __nv_bfloat16* orow = o + (static_cast<std::size_t>(t) * kQsaQ + h) * 128;
  float xreg[kIdxDim];
  float ss = 0.0F;
#pragma unroll
  for (int d = 0; d < kIdxDim; ++d) {
    xreg[d] = __bfloat162float(orow[d]);
    ss += xreg[d] * xreg[d];
  }
  const float inv_r = 1.0F / sqrtf(ss * (1.0F / 32.0F) + 1e-6F);
  const __nv_bfloat16* zrow =
      z + static_cast<std::size_t>(t) * (kQsaQ * kIdxDim) + h * kIdxDim;
  __nv_bfloat16* orow_out =
      onx + static_cast<std::size_t>(t) * (kQsaQ * kIdxDim) + h * kIdxDim;
#pragma unroll
  for (int d = 0; d < kIdxDim; ++d) {
    const float zv = __bfloat162float(zrow[d]);
    orow_out[d] = __float2bfloat16(xreg[d] * inv_r * bf16_decode(w[d]) * mini_silu(zv));
  }
}

// ---------------------------------------------------------------------------
// per-head RMSNorm (in place) + v1 RoPE
// ---------------------------------------------------------------------------

// x[t, h, d] = bf16( f32(x) * g[d] / sqrt( (1/32) sum_d f32(x)^2 + 1e-6 ) ).
// Registers hold the row before the in-place write (same-thread safety).
__global__ void mini_head_rmsnorm_kernel(__nv_bfloat16* __restrict__ x, int T, int H,
                                         const std::uint16_t* __restrict__ g) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * H) {
    return;
  }
  __nv_bfloat16* row = x + static_cast<std::size_t>(idx) * kIdxDim;
  float reg[kIdxDim];
  float ss = 0.0F;
#pragma unroll
  for (int d = 0; d < kIdxDim; ++d) {
    reg[d] = __bfloat162float(row[d]);
    ss += reg[d] * reg[d];
  }
  const float inv = 1.0F / sqrtf(ss * (1.0F / 32.0F) + 1e-6F);
#pragma unroll
  for (int d = 0; d < kIdxDim; ++d) {
    row[d] = __float2bfloat16(reg[d] * bf16_decode(g[d]) * inv);
  }
}

// Pairs (m, m+4), m in [0,4); angle = (pos0 + t) * 10000^(-m/4); dims 8..31
// pass through. One thread per (t, h, m); both register values are read
// before the in-place writes.
__global__ void mini_rope_kernel(__nv_bfloat16* __restrict__ x, int T, int H, int pos0) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = T * H * 4;
  if (idx >= total) {
    return;
  }
  const int m = idx % 4;
  const int th = idx / 4;
  const int h = th % H;
  const int t = th / H;
  const float angle =
      static_cast<float>(pos0 + t) * powf(10000.0F, -0.25F * static_cast<float>(m));
  const float cs = cosf(angle);
  const float sn = sinf(angle);
  __nv_bfloat16* row = x + (static_cast<std::size_t>(t) * H + h) * kIdxDim;
  const float q0 = __bfloat162float(row[m]);
  const float q1 = __bfloat162float(row[m + 4]);
  row[m]     = __float2bfloat16(q0 * cs - q1 * sn);
  row[m + 4] = __float2bfloat16(q0 * sn + q1 * cs);
}

// ---------------------------------------------------------------------------
// paged KV append
// ---------------------------------------------------------------------------

// One thread per (t, h, d): write k and v at absolute position pos + t.
__global__ void mini_kv_append_kernel(const __nv_bfloat16* __restrict__ k,
                                      const __nv_bfloat16* __restrict__ v, int T, int pos,
                                      const std::int32_t* __restrict__ block_table,
                                      __nv_bfloat16* __restrict__ kv) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * kQsaKV * kIdxDim) {
    return;
  }
  const int d = idx % kIdxDim;
  const int rem = idx / kIdxDim;
  const int h = rem % kQsaKV;
  const int t = rem / kQsaKV;
  const std::int32_t p = pos + t;
  const std::int32_t page = block_table[p >> 6];
  const std::int32_t po = p & 63;
  const std::size_t base = static_cast<std::size_t>(page) * kPageElements +
                           static_cast<std::size_t>(kIdxDim) * 64 * h + 32 * po + d;
  kv[base]               = k[idx];
  kv[base + kVPlaneOffset] = v[idx];
}

// ---------------------------------------------------------------------------
// mini indexer (round-local)
// ---------------------------------------------------------------------------

// z[t, c] = sum_k bf16(qk_proj[c, k]) * bf16(x[t, k]), c in [0, 160).
__global__ void mini_indexer_z_kernel(const __nv_bfloat16* __restrict__ x, int T,
                                      const std::uint16_t* __restrict__ qk_proj,
                                      float* __restrict__ z) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * kIdxZ) {
    return;
  }
  const int t = idx / kIdxZ;
  const int c = idx - t * kIdxZ;
  const std::uint16_t* w = qk_proj + static_cast<std::size_t>(c) * kMiniHidden;
  const __nv_bfloat16* xr = x + static_cast<std::size_t>(t) * kMiniHidden;
  float acc = 0.0F;
#pragma unroll
  for (int k = 0; k < kMiniHidden; ++k) {
    acc += bf16_decode(w[k]) * __bfloat162float(xr[k]);
  }
  z[idx] = acc;
}

// One block (256 threads) per query token t: rms the query + the 4 key chunks
// of every row, score the row, select ids. The normalized key rows live in
// shared memory (sh_kn[i][c][d]); every reduction is a serial fixed-order sum
// inside one thread, so the result is bit-deterministic.
__global__ void mini_indexer_select_kernel(const float* __restrict__ z, int T,
                                           const std::uint16_t* __restrict__ q_norm,
                                           const std::uint16_t* __restrict__ k_norm,
                                           float* __restrict__ scores,
                                           std::int32_t* __restrict__ ids,
                                           std::int32_t* __restrict__ counts) {
  __shared__ float sh_qn[kIdxDim];
  __shared__ float sh_kn[kBudget][kIdxChunks][kIdxDim];
  __shared__ float sh_s[kBudget];

  const int t = blockIdx.x;
  const float* zt = z + static_cast<std::size_t>(t) * kIdxZ;

  // Phase A: normalized query of row t (thread 0, serial).
  if (threadIdx.x == 0) {
    float ss = 0.0F;
#pragma unroll
    for (int d = 0; d < kIdxDim; ++d) {
      ss += zt[d] * zt[d];
    }
    const float qinv = 1.0F / sqrtf(ss * (1.0F / 32.0F) + 1e-6F);
#pragma unroll
    for (int d = 0; d < kIdxDim; ++d) {
      sh_qn[d] = zt[d] * bf16_decode(q_norm[d]) * qinv;
    }
  }
  __syncthreads();

  // Phase B: normalized key chunks of every row i, one (i, c) per thread.
  {
    const int i = threadIdx.x % kBudget;
    const int c = threadIdx.x / kBudget;
    const float* zr = z + (static_cast<std::size_t>(i) * kIdxZ + kIdxDim + c * kIdxDim);
    float ss = 0.0F;
#pragma unroll
    for (int d = 0; d < kIdxDim; ++d) {
      ss += zr[d] * zr[d];
    }
    const float kinv = 1.0F / sqrtf(ss * (1.0F / 32.0F) + 1e-6F);
    float* kn = &sh_kn[i][c][0];
#pragma unroll
    for (int d = 0; d < kIdxDim; ++d) {
      kn[d] = zr[d] * bf16_decode(k_norm[d]) * kinv;
    }
  }
  __syncthreads();

  // Phase C: score row t — score(t, i) = i <= t ?
  // (1/sqrt(32)) * sum_c sum_d qn[d] * kn[i][c][d] : -inf.
  if (threadIdx.x < kBudget) {
    if (threadIdx.x > t) {
      sh_s[threadIdx.x] = -INFINITY;
    } else {
      float acc = 0.0F;
#pragma unroll
      for (int c = 0; c < kIdxChunks; ++c) {
        const float* kr = &sh_kn[threadIdx.x][c][0];
#pragma unroll
        for (int d = 0; d < kIdxDim; ++d) {
          acc += sh_qn[d] * kr[d];
        }
      }
      sh_s[threadIdx.x] = acc * (1.0F / sqrtf(32.0F));
    }
  }
  __syncthreads();

  // Phase D: selection + global writes (thread 0, fixed order).
  if (threadIdx.x == 0) {
    std::int32_t* idrow = ids + static_cast<std::size_t>(t) * kBudget;
    for (int i = 0; i < kBudget; ++i) {
      scores[static_cast<std::size_t>(t) * kBudget + i] = sh_s[i];
    }
    const int n = (kBudget < (t + 1)) ? kBudget : (t + 1);
    if (kBudget >= T) {
      // Dense shortcut: n = t+1 for every row, ids = 0..t ascending.
      for (int j = 0; j <= t; ++j) {
        idrow[j] = j;
      }
      for (int j = t + 1; j < kBudget; ++j) {
        idrow[j] = -1;
      }
    } else {
      // Topk: n highest scores from the pool {0..t}; (score DESC, position
      // DESC) — the descending scan keeps the larger position on exact ties.
      bool used[kBudget] = {};
      std::int32_t sel[kBudget];
      for (int jj = 0; jj < n; ++jj) {
        int best = -1;
        for (int i = t; i >= 0; --i) {
          if (used[i]) {
            continue;
          }
          if (best < 0 || sh_s[i] > sh_s[best]) {
            best = i;
          }
        }
        used[best] = true;
        sel[jj] = best;
      }
      for (int jj = 1; jj < n; ++jj) {
        const std::int32_t v = sel[jj];
        int s = jj - 1;
        while (s >= 0 && sel[s] > v) {
          sel[s + 1] = sel[s];
          --s;
        }
        sel[s + 1] = v;
      }
      for (int jj = 0; jj < n; ++jj) {
        idrow[jj] = sel[jj];
      }
      for (int jj = n; jj < kBudget; ++jj) {
        idrow[jj] = -1;
      }
    }
    counts[t] = n;
  }
}

// ---------------------------------------------------------------------------
// mini sparse GQA (D=32, Q=4, KV=2, g = h/2)
// ---------------------------------------------------------------------------

// One thread per (t, h): online softmax over the selected positions, fixed
// order j = 0..n-1. expf(-inf) = 0 keeps the first step exact.
__global__ void mini_sparse_gqa_kernel(const __nv_bfloat16* __restrict__ q, int T,
                                       const std::int32_t* __restrict__ ids,
                                       const std::int32_t* __restrict__ counts,
                                       const std::int32_t* __restrict__ block_table,
                                       const __nv_bfloat16* __restrict__ k_pages,
                                       const __nv_bfloat16* __restrict__ v_pages,
                                       __nv_bfloat16* __restrict__ out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * kQsaQ) {
    return;
  }
  const int t = idx / kQsaQ;
  const int h = idx - t * kQsaQ;
  const int g = h / 2;
  const int n = counts[t];
  const __nv_bfloat16* qrow = q + static_cast<std::size_t>(idx) * kQsaD;
  float qreg[kQsaD];
#pragma unroll
  for (int d = 0; d < kQsaD; ++d) {
    qreg[d] = __bfloat162float(qrow[d]);
  }
  const std::int32_t* idrow = ids + static_cast<std::size_t>(t) * kBudget;
  float m = -INFINITY;
  float zsum = 0.0F;
  float acc[kQsaD] = {};
  for (int j = 0; j < n; ++j) {
    const std::int32_t p = idrow[j];
    const std::int32_t page = block_table[p >> 6];
    const std::int32_t po = p & 63;
    const std::size_t base = static_cast<std::size_t>(page) * kPageElements +
                             static_cast<std::size_t>(kQsaD) * 64 * g + 32 * po;
    float s = 0.0F;
#pragma unroll
    for (int d = 0; d < kQsaD; ++d) {
      s += qreg[d] * __bfloat162float(k_pages[base + d]);
    }
    s *= 1.0F / sqrtf(32.0F);
    const float nm = fmaxf(m, s);
    const float rescale = expf(m - nm);
    const float wj = expf(s - nm);
    m = nm;
    zsum = zsum * rescale + wj;
#pragma unroll
    for (int d = 0; d < kQsaD; ++d) {
      acc[d] = acc[d] * rescale +
               wj * __bfloat162float(v_pages[base + kVPlaneOffset + d]);
    }
  }
  const float inv = 1.0F / zsum;
#pragma unroll
  for (int d = 0; d < kQsaD; ++d) {
    out[static_cast<std::size_t>(idx) * kQsaD + d] = __float2bfloat16(acc[d] * inv);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// host wrappers
// ---------------------------------------------------------------------------

void mini_gdn_gating(const __nv_bfloat16* x, int T, const __nv_bfloat16* a_weight,
                     const __nv_bfloat16* b_weight, const float* a_log,
                     const float* dt_bias, float* g, float* beta, cudaStream_t stream) {
  dim3 grid((T * kGdnHeads + 255) / 256);
  mini_gdn_gating_kernel<<<grid, 256, 0, stream>>>(
      x, T, reinterpret_cast<const std::uint16_t*>(a_weight),
      reinterpret_cast<const std::uint16_t*>(b_weight), a_log, dt_bias, g, beta);
}

void mini_gated_rmsnorm(const __nv_bfloat16* o, const __nv_bfloat16* z,
                        const __nv_bfloat16* w, int T, __nv_bfloat16* onx,
                        cudaStream_t stream) {
  dim3 grid((T * kQsaQ + 255) / 256);
  mini_gated_rmsnorm_kernel<<<grid, 256, 0, stream>>>(
      o, z, reinterpret_cast<const std::uint16_t*>(w), T, onx);
}

void mini_head_rmsnorm(__nv_bfloat16* x, int T, int H, const __nv_bfloat16* g,
                       cudaStream_t stream) {
  dim3 grid((T * H + 255) / 256);
  mini_head_rmsnorm_kernel<<<grid, 256, 0, stream>>>(
      x, T, H, reinterpret_cast<const std::uint16_t*>(g));
}

void mini_rope(__nv_bfloat16* q, int T, int Hq, int pos0, __nv_bfloat16* k, int Hk,
               cudaStream_t stream) {
  {
    dim3 grid((T * Hq * 4 + 255) / 256);
    mini_rope_kernel<<<grid, 256, 0, stream>>>(q, T, Hq, pos0);
  }
  {
    dim3 grid((T * Hk * 4 + 255) / 256);
    mini_rope_kernel<<<grid, 256, 0, stream>>>(k, T, Hk, pos0);
  }
}

void mini_kv_append(const __nv_bfloat16* k, const __nv_bfloat16* v, int T, int pos,
                    const std::int32_t* block_table, __nv_bfloat16* kv,
                    cudaStream_t stream) {
  dim3 grid((T * kQsaKV * kIdxDim + 255) / 256);
  mini_kv_append_kernel<<<grid, 256, 0, stream>>>(k, v, T, pos, block_table, kv);
}

void mini_indexer(const __nv_bfloat16* x, int T, const std::uint16_t* qk_proj,
                  const std::uint16_t* q_norm, const std::uint16_t* k_norm,
                  float* z, float* scores, std::int32_t* ids, std::int32_t* counts,
                  cudaStream_t stream) {
  {
    dim3 grid((T * kIdxZ + 255) / 256);
    mini_indexer_z_kernel<<<grid, 256, 0, stream>>>(x, T, qk_proj, z);
  }
  mini_indexer_select_kernel<<<T, 256, 0, stream>>>(z, T, q_norm, k_norm, scores, ids,
                                                    counts);
}

void mini_sparse_gqa(const __nv_bfloat16* q, int T, const std::int32_t* ids,
                     const std::int32_t* counts, const std::int32_t* block_table,
                     const __nv_bfloat16* k_pages, const __nv_bfloat16* v_pages,
                     __nv_bfloat16* out, cudaStream_t stream) {
  dim3 grid((T * kQsaQ + 255) / 256);
  mini_sparse_gqa_kernel<<<grid, 256, 0, stream>>>(q, T, ids, counts, block_table,
                                                   k_pages, v_pages, out);
}

}  // namespace ninfer::targets::qwen3_8_flash_next::detail
