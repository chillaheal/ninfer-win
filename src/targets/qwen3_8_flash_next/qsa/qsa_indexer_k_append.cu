// qsa_indexer_k_append kernels — contract in include/ninfer/ops/qsa_indexer_k_append.h.
// One block (128 threads) per token of the round. Each thread owns dim `tid` of
// each of the four 128-dim key chunks (qk_proj rows 128..640). Per token:
//   1. GEMV the four key chunks (fixed sequential accumulation -> bit-equal),
//   2. rms128-normalize each chunk with the shared k_norm,
//   3. reduce the per-token absmax over the 512 normalized dims -> FP8 scale,
//   4. quantize the 512-dim key to FP8 (E4M3, per-token scale) and scatter it
//      into the paged pool at the token's GLOBAL position (pos0 + t) through
//      the shared block table,
//   5. optionally write the pre-quant BF16 row to work_dev (test-only).
// All reductions are fixed-order trees (warp shuffle + shared-memory tree), so
// reruns are bit-equal for a given input.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"

#include <ninfer/ops/qsa_indexer_k_append.h>

namespace ninfer::ops {

namespace {

constexpr int kQueryDims = 128;  // qk_proj rows [0..128) are the query chunk
constexpr int kKeyChunks = 4;    // qk_proj rows [128..640) are the four key chunks
constexpr int kKeyDim    = 128;
constexpr int kKeyWidth  = 512;  // kKeyChunks * kKeyDim
constexpr int kThreads   = 128;
constexpr int kMaxTokens = 2048;
constexpr float kRmsEps  = 1e-6f;

__device__ __forceinline__ float bf16f(const std::uint16_t h) {
  return __uint_as_float(static_cast<std::uint32_t>(h) << 16);
}

__global__ void qsa_idxk_append_kernel(
    const std::uint16_t* __restrict__ hidden, const std::uint16_t* __restrict__ w,
    const std::uint16_t* __restrict__ knorm, const std::int32_t* __restrict__ bt,
    std::uint8_t* __restrict__ pages, std::uint16_t* __restrict__ scale,
    std::uint16_t* __restrict__ work, int H, int pos0) {
  __shared__ float sh_part[kKeyChunks][4];
  __shared__ float sh_inv[kKeyChunks];
  __shared__ float sh_abs_part[4];
  __shared__ float sh_absmax;

  const int t    = blockIdx.x;
  const int tid  = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const std::int32_t pos  = pos0 + t;
  const std::int32_t pp   = paged_kv_physical_page(bt, pos);
  const std::int32_t slot = pos & kPagedKVPageMask;
  const std::uint16_t* hrow = hidden + static_cast<std::size_t>(t) * H;

  // 1. GEMV the four key chunks. qk_proj row for key chunk kc is
  //    (kQueryDims + kc*128 + tid); fixed sequential accumulation order.
  float zv[kKeyChunks];
#pragma unroll
  for (int kc = 0; kc < kKeyChunks; ++kc) {
    const std::uint16_t* wrow =
        w + static_cast<std::size_t>(kQueryDims + kc * kKeyDim + tid) * H;
    float acc = 0.f;
    for (int i = 0; i < H; ++i) {
      acc = fmaf(bf16f(wrow[i]), bf16f(hrow[i]), acc);
    }
    zv[kc] = acc;
  }

  // 2. Per key-chunk sum-of-squares over the 128 dims: warp tree + 4 partials.
#pragma unroll
  for (int kc = 0; kc < kKeyChunks; ++kc) {
    float v = zv[kc] * zv[kc];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      sh_part[kc][warp] = v;
    }
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int kc = 0; kc < kKeyChunks; ++kc) {
      float s = sh_part[kc][0] + sh_part[kc][1];
      s += sh_part[kc][2];
      s += sh_part[kc][3];
      sh_inv[kc] = 1.0f / sqrtf(s * (1.0f / 128.0f) + kRmsEps);
    }
  }
  __syncthreads();

  // 3. rms128-normalize with k_norm; track the per-token absmax over the 512 dims.
  float kv[kKeyChunks];
  float m = 0.f;
#pragma unroll
  for (int kc = 0; kc < kKeyChunks; ++kc) {
    kv[kc] = zv[kc] * bf16f(knorm[tid]) * sh_inv[kc];
    m = fmaxf(m, fabsf(kv[kc]));
  }
  // Per-token absmax: warp tree, then the 4 warp partials in fixed order.
  {
    float v = m;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    if (lane == 0) {
      sh_abs_part[warp] = v;
    }
  }
  __syncthreads();
  if (tid == 0) {
    float s = fmaxf(sh_abs_part[0], sh_abs_part[1]);
    s = fmaxf(s, fmaxf(sh_abs_part[2], sh_abs_part[3]));
    sh_absmax = s;
  }
  __syncthreads();
  const KVCacheFp8QuantParams q = kv_cache_fp8_quant_params(sh_absmax);

  // 4/5. Quantize to FP8 + scatter at global pos; optional BF16 pre-quant write.
#pragma unroll
  for (int kc = 0; kc < kKeyChunks; ++kc) {
    const int d = kc * kKeyDim + tid;
    const std::int64_t off = paged_kv_element_offset<kKeyWidth, 1>(pp, 0, slot, d);
    pages[off] = kv_cache_fp8_quant_code(kv[kc], q.inverse_scale);
    if (work != nullptr) {
      work[static_cast<std::size_t>(t) * kKeyWidth + d] =
          __bfloat16_as_ushort(__float2bfloat16_rn(kv[kc]));
    }
  }
  if (tid == 0) {
    scale[static_cast<std::size_t>(pp) * kPagedKVPageSize + slot] = __half_as_ushort(q.scale);
  }
}

}  // namespace

[[noreturn]] void qia_fail(const char* what) {
  std::fprintf(stderr, "qsa_indexer_k_append: %s\n", what);
  std::abort();
}

void qsa_indexer_k_append(const QsaIndexerKAppendParams& p, cudaStream_t stream) {
  if (p.tokens <= 0 || p.tokens > kMaxTokens) {
    qia_fail("tokens out of range (1..2048)");
  }
  if (p.hidden < 128 || p.hidden > 2560 || p.hidden % 128 != 0) {
    qia_fail("hidden out of range (128..2560, multiple of 128)");
  }
  if (p.pos0 < 0) {
    qia_fail("pos0 must be >= 0");
  }
  if (!p.hidden_dev || !p.qk_proj_dev || !p.k_norm_dev || !p.block_table_dev ||
      !p.idx_k_pages_dev || !p.idx_k_scale_dev) {
    qia_fail("missing buffer");
  }
  dim3 grid(p.tokens), block(kThreads);
  qsa_idxk_append_kernel<<<grid, block, 0, stream>>>(
      p.hidden_dev, p.qk_proj_dev, p.k_norm_dev, p.block_table_dev,
      p.idx_k_pages_dev, p.idx_k_scale_dev, p.work_dev, p.hidden, p.pos0);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    qia_fail(cudaGetErrorString(err));
  }
}

}  // namespace ninfer::ops
