// P7 7b v1 qsa_sparse_gqa kernel — contract in
// include/ninfer/ops/qsa_sparse_gqa.h. One block (256 threads) per
// (token, q-head); thread d owns head-dim lane d. The GQA group
// g = h / (Q/KV) selects the kv head; the K/V rows are GATHERED from the
// paged store through the block table, so unselected positions are never
// read and decode T=1 is O(counts) work, never a T x T dense pass.
//
// Flash-style online softmax over 16-position chunks: per-position score
// dots (every warp reduces its 32-dim slice for every position with a
// fixed-order shfl_down tree and publishes to sh_s[warp][j]; lanes 0..15
// sum the 8 slices), an inclusive warp max scan, and an (m, z, output)
// rescale at every chunk boundary (the denominator partials partition over
// the 16 position-lanes by k&7). The running output is a PER-THREAD
// REGISTER accumulator (thread d owns dim d end to end — the rescaled
// factor and every chunk's V rows accumulate in that register); shared
// memory is never used for the output, so there is no cross-warp output
// merge and no dependence on shared-memory initialization. Every shuffle
// runs under a block-uniform condition, so full-mask participation always
// holds. sh_m / sh_z / sh_cbest each have exactly one writer per chunk
// (d == 0, after a barrier), so there is no concurrent shared RMW — no
// float atomics anywhere, and reruns are bit-equal.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <cstdint>

#include "ops/kernel/paged_kv_address.cuh"

#include <ninfer/ops/qsa_sparse_gqa.h>

namespace ninfer::ops {

namespace {

constexpr int kGqaHeadDim = 256;
constexpr int kGqaChunk   = 16;
constexpr int kGqaWarps   = 8;  // 256 threads / 32

__device__ __forceinline__ float bf16f(const std::uint16_t h) {
  return __uint_as_float(static_cast<std::uint32_t>(h) << 16);
}

// RNE bf16 rounding of an fp32 value (matches the op_tester reference).
__device__ __forceinline__ std::uint16_t to_bf16u(float f) {
  const std::uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    return std::uint16_t((u >> 16) | 0x0040u);
  }
  const std::uint32_t lsb = (u >> 16) & 1u;
  return static_cast<std::uint16_t>((u + 0x7fffu + lsb) >> 16);
}

// Element offset of (kv head, position) in the page-major [256, 64, KV, P]
// bf16 plane: the paged_kv_element_offset<256, KVHeads>(block_table, head,
// position, leading=0) formula with the head extent taken at runtime (the
// Op contract pins the page-major plane order; KV is a runtime 1..8).
__device__ __forceinline__ std::int64_t qsa_kv_row_offset(
    const std::int32_t* __restrict__ bt, int kv_heads, int head, int position) {
  const std::int32_t page = bt[position >> kPagedKVPageShift];
  const std::int32_t off  = position & kPagedKVPageMask;
  return static_cast<std::int64_t>(kGqaHeadDim) * kPagedKVPageSize *
             (static_cast<std::int64_t>(head) +
              static_cast<std::int64_t>(kv_heads) * page) +
         static_cast<std::int64_t>(kGqaHeadDim) * off;
}

__global__ void qsa_gqa_kernel(const std::uint16_t* __restrict__ q,
                               const std::int32_t* __restrict__ ids,
                               const std::int32_t* __restrict__ counts,
                               const std::int32_t* __restrict__ bt,
                               const std::uint16_t* __restrict__ kpg,
                               const std::uint16_t* __restrict__ vpg,
                               std::uint16_t* __restrict__ out,
                               int Q, int KV, int B, float scale) {
  __shared__ float sh_cbest;
  __shared__ float sh_m;
  __shared__ float sh_z;
  __shared__ float sh_s[kGqaWarps][kGqaChunk];
  __shared__ float sh_zw[kGqaWarps];

  const int t    = blockIdx.x / Q;
  const int h    = blockIdx.x % Q;
  const int d    = threadIdx.x;
  const int warp = d >> 5, lane = d & 31;
  const int g    = h / (Q / KV);
  const int n    = counts[t];
  const std::size_t ids_row = static_cast<std::size_t>(t) * B;
  const int qbase = (static_cast<std::size_t>(t) * Q + h) * kGqaHeadDim;
  const float qd  = bf16f(q[qbase + d]);

  if (d == 0) {
    sh_m = -INFINITY;
    sh_z = 0.f;
  }
  __syncthreads();
  // Per-thread running output for dim d (rescaled at every chunk boundary).
  float col = 0.f;

  for (int j0 = 0; j0 < n; j0 += kGqaChunk) {
    const int m = (j0 + kGqaChunk <= n) ? kGqaChunk : n - j0;

    // Chunk score s_j = scale * dot(q, K[p_j]) for j < m. The 256-dim dot
    // spans all 8 warps, so an intra-warp shfl of the q register (srcLane
    // wraps mod 32) CANNOT express it: every warp computes its own 32-dim
    // slice of the dot for every position (each thread contributes its own
    // dim d, the warp reduces with a fixed-order shfl_down tree, lane 0
    // publishes to sh_s[warp][j] — one writer per element, and all 8
    // slices are written before any is read). Lanes 0..15 then sum the 8
    // warp partials (position j is owned by lane j). j/m are block-uniform,
    // so every shfl below runs with the full warp participating.
    #pragma unroll
    for (int j = 0; j < kGqaChunk; ++j) {
      if (j < m) {
        const std::int32_t pos = ids[ids_row + j0 + j];
        const std::int64_t base = qsa_kv_row_offset(bt, KV, g, pos);
        float partial = qd * bf16f(kpg[base + d]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          partial += __shfl_down_sync(0xffffffffu, partial, off);
        }
        if (lane == 0) sh_s[warp][j] = partial;
      }
    }
    __syncthreads();
    float s_local = -INFINITY;
    if (lane < m) {
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < kGqaWarps; ++w) {
        s += sh_s[w][lane];
      }
      s_local = scale * s;
    }
    // Inclusive warp max starting from the own value (shfl_down): lane 0
    // ends with max over lanes 0..16, which covers every real score
    // (scores live in lanes 0..m-1 with m <= 16). Warp 0 alone publishes it
    // (single writer).
    float best = s_local;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      best = fmaxf(best, __shfl_down_sync(0xffffffffu, best, off));
    }
    if (warp == 0 && lane == 0) {
      sh_cbest = best;
    }
    __syncthreads();
    const float mm = fmaxf(sh_m, sh_cbest);
    const float factor = expf(sh_m - mm);
    const float u =
        (lane < m) ? expf(s_local - mm) : 0.f;

    // Rescale the running output for the new max, then accumulate this
    // chunk's u-weighted V rows. Thread d owns dim d end to end: the
    // accumulator is a register, every thread accumulates EVERY position
    // (a per-position partition by warp would leave each dim missing the
    // 224 positions its warp does not own), and nothing touches shared
    // memory here.
    col *= factor;
    {
      float acc = col;
#pragma unroll
      for (int k = 0; k < kGqaChunk; ++k) {
        const int j = j0 + k;
        if (j < n) {
          const std::int32_t pos = ids[ids_row + j];
          const std::int64_t base = qsa_kv_row_offset(bt, KV, g, pos);
          acc = fmaf(__shfl_sync(0xffffffffu, u, k), bf16f(vpg[base + d]), acc);
        }
      }
      col = acc;
    }
    // Chunk denominator contribution: warp (k & 7) sums the u of the
    // positions it owns (fixed-order shfl_down tree, single writer per
    // sh_zw slot — no cross-thread RMW on sh_z itself).
    {
      float zu = ((lane < m) && ((lane & 7) == warp)) ? u : 0.f;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        zu += __shfl_down_sync(0xffffffffu, zu, off);
      }
      if (lane == 0) sh_zw[warp] = zu;
    }
    __syncthreads();
    // Single-thread state update: the only writer of sh_z / sh_m is d == 0,
    // after a barrier — concurrent RMW on shared float is UB and was the
    // source of lost rescale factors / chunk contributions.
    if (d == 0) {
      float zz = sh_z * factor;
#pragma unroll
      for (int w = 0; w < kGqaWarps; ++w) {
        zz += sh_zw[w];
      }
      sh_z = zz;
      sh_m = mm;
    }
    __syncthreads();
  }

  // Final output: each thread holds its dim's complete rescaled sum in a
  // register; sh_z is shared (single writer, barrier-synced).
  __syncthreads();
  // Empty causal set (counts[t] == 0): the loop never ran, so col == 0 and
  // sh_z == 0. Guard the 0/0 -> NaN (token 0 attends to nothing before it).
  out[qbase + d] = (sh_z == 0.f) ? 0u : to_bf16u(col / sh_z);
}

}  // namespace

[[noreturn]] void qg_fail(const char* what) {
  std::fprintf(stderr, "qsa_sparse_gqa: %s\n", what);
  std::abort();
}

void qg_launch(const QsaSparseGqaParams& p, cudaStream_t stream) {
  dim3 grid(static_cast<unsigned>(p.tokens) * p.q_heads), block(kGqaHeadDim);
  qsa_gqa_kernel<<<grid, block, 0, stream>>>(p.q_dev, p.ids_dev, p.counts_dev,
                                             p.block_table_dev, p.k_pages_dev,
                                             p.v_pages_dev, p.out_dev, p.q_heads,
                                             p.kv_heads, p.budget, p.scale);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    qg_fail(cudaGetErrorString(err));
  }
}

void qsa_sparse_gqa(const QsaSparseGqaParams& p, cudaStream_t stream) {
  if (p.tokens <= 0 || p.tokens > 2048) {
    qg_fail("tokens out of v1 range (1..2048)");
  }
  if (p.q_heads <= 0 || p.kv_heads <= 0 || p.kv_heads > 8 ||
      p.q_heads % p.kv_heads != 0) {
    qg_fail("heads out of v1 range (1 <= KV <= 8, Q % KV == 0)");
  }
  if (p.budget <= 0 || !(p.scale > 0.f)) {
    qg_fail("budget must be >= 1 and scale > 0");
  }
  if (!p.q_dev || !p.ids_dev || !p.counts_dev || !p.block_table_dev ||
      !p.k_pages_dev || !p.v_pages_dev || !p.out_dev) {
    qg_fail("missing buffer");
  }
  qg_launch(p, stream);
}

}  // namespace ninfer::ops
