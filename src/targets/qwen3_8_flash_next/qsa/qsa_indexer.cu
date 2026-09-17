// P7 7a qsa_indexer — contract in include/ninfer/ops/qsa_indexer.h.
//
// The query selects B key positions from the FULL context, not just the current
// prefill round. Two stream-ordered kernels share the caller-provided q̃ scratch
// (work [T, 512] fp32), plus a per-chunk logit scratch (logits [chunk, n_full]
// fp32) that is reused across chunks:
//
//  Q stage: one block (128 threads) per token — GEMV the 128-dim query chunk
//           (qk_proj rows [0..128)), rms128-normalize with q_norm, tile the 128
//           result 4x into the [T, 512] q̃ row (the four 128-dim key chunks
//           share the same query).
//  logit GEMM: one block (256 threads) per query row of the chunk — score every
//           key position j in [0, n_full) of the paged indexer-K pool (built by
//           qsa_indexer_k_append) by the fixed-order 512-term dot
//           logit[m, j] = (1/sqrt(128)) * sum_d q̃[m][d] * k̃[j][d]. k̃[j] is the
//           dequantized pool row at GLOBAL j: FP8 (E4M3 code * per-token scale,
//           via the row codec) or BF16 (the stored value). A per-chunk qsa_topk
//           then takes the top-B valid positions per query (valid =
//           j < context_len[m]).
//
// Chunking over the query (M) dimension keeps the [chunk, n_full] FP32 scratch
// within max_logits_bytes; each (m, j) logit is a fixed-order dot independent of
// the chunk boundary, so a chunked run is BIT-EQUAL to an unchunked run of the
// same inputs (test 7f a). All reductions are fixed-order trees / fixed-order
// dots, so reruns are bit-equal.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"

#include <ninfer/ops/qsa_indexer.h>
#include <ninfer/ops/qsa_topk.h>

namespace ninfer::ops {

namespace {

constexpr int kQueryWidth  = 512;  // tiled q̃ width (4 x 128)
constexpr int kQueryDims   = 128;  // native query chunk (qk_proj rows [0..128))
constexpr int kMaxTokens   = 2048;
constexpr float kRmsEps    = 1e-6f;
// 1 / sqrt(128)
constexpr float kScoreScale = 0.088388347648318427f;

__device__ __forceinline__ float bf16f(const std::uint16_t h) {
  return __uint_as_float(static_cast<std::uint32_t>(h) << 16);
}

// Q stage: one block (128 threads) per token. Thread tid owns query dim tid.
__global__ void qsa_idx_q_kernel(const std::uint16_t* __restrict__ hidden,
                                 const std::uint16_t* __restrict__ w,
                                 const std::uint16_t* __restrict__ qnorm,
                                 float* __restrict__ work, int H) {
  __shared__ float sh_part[4];
  __shared__ float sh_inv;
  const int t   = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const std::uint16_t* hrow = hidden + static_cast<std::size_t>(t) * H;
  const std::uint16_t* wrow = w + static_cast<std::size_t>(tid) * H;

  // GEMV the 128-dim query chunk (fixed sequential accumulation order).
  float acc = 0.f;
  for (int i = 0; i < H; ++i) {
    acc = fmaf(bf16f(wrow[i]), bf16f(hrow[i]), acc);
  }

  // rms128: sum-of-squares over the 128 query dims (warp tree + 4 partials).
  float v = acc * acc;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, off);
  }
  if (lane == 0) {
    sh_part[warp] = v;
  }
  __syncthreads();
  if (tid == 0) {
    float s = sh_part[0] + sh_part[1];
    s += sh_part[2];
    s += sh_part[3];
    sh_inv = 1.0f / sqrtf(s * (1.0f / 128.0f) + kRmsEps);
  }
  __syncthreads();
  const float qtilde = acc * bf16f(qnorm[tid]) * sh_inv;

  // Tile the 128-dim q̃ 4x into the [T, 512] row.
  const std::size_t wbase = static_cast<std::size_t>(t) * kQueryWidth;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    work[wbase + c * kQueryDims + tid] = qtilde;
  }
}

// Logit GEMM: one block (256 threads) per query row in [base, base+chunk).
// Grid-stride over the key positions j in [0, n_full); each thread owns one j
// and does the 512-term dot. The pool row at GLOBAL j is addressed through the
// shared block table (the same table the main QSA K/V pool uses), so j is a
// GLOBAL key position.
__global__ void qsa_idx_logit_kernel(
    const float* __restrict__ work, const void* __restrict__ pool,
    const std::uint16_t* __restrict__ scale, const std::int32_t* __restrict__ bt,
    float* __restrict__ logits, int base, int chunk, int n_full, bool is_fp8) {
  __shared__ float sh_q[kQueryWidth];
  const int m_abs = base + blockIdx.x;
  const float* qrow = work + static_cast<std::size_t>(m_abs) * kQueryWidth;
  for (int d = threadIdx.x; d < kQueryWidth; d += blockDim.x) {
    sh_q[d] = qrow[d];
  }
  __syncthreads();
  // Logit scratch is CHUNK-LOCAL [chunk, n_full] (qsa_topk reads row blockIdx.x
  // with no base offset; max_logits_bytes bounds this buffer). The work row is
  // absolute m_abs (work is the full [T, 512]).
  float* lrow = logits + static_cast<std::size_t>(blockIdx.x) * n_full;

  if (is_fp8) {
    const std::uint8_t* fp8 = static_cast<const std::uint8_t*>(pool);
    for (int j = threadIdx.x; j < n_full; j += blockDim.x) {
      const int pp   = paged_kv_physical_page(bt, j);
      const int slot = j & kPagedKVPageMask;
      const std::int64_t off = paged_kv_element_offset<kQueryWidth, 1>(pp, 0, slot, 0);
      const __half sc =
          __ushort_as_half(scale[static_cast<std::size_t>(pp) * kPagedKVPageSize + slot]);
      float acc = 0.f;
      for (int d = 0; d < kQueryWidth; d += 2) {
        const std::uint16_t storage =
            static_cast<std::uint16_t>(fp8[off + d]) |
            (static_cast<std::uint16_t>(fp8[off + d + 1]) << 8);
        const float2 h2 = __half22float2(kv_cache_fp8_dequant_code2_to_half2(storage, sc));
        acc = fmaf(sh_q[d], h2.x, acc);
        acc = fmaf(sh_q[d + 1], h2.y, acc);
      }
      lrow[j] = kScoreScale * acc;
    }
  } else {
    const std::uint16_t* bf16 = static_cast<const std::uint16_t*>(pool);
    for (int j = threadIdx.x; j < n_full; j += blockDim.x) {
      const int pp   = paged_kv_physical_page(bt, j);
      const int slot = j & kPagedKVPageMask;
      const std::int64_t off = paged_kv_element_offset<kQueryWidth, 1>(pp, 0, slot, 0);
      float acc = 0.f;
      for (int d = 0; d < kQueryWidth; ++d) {
        acc = fmaf(sh_q[d], bf16f(bf16[off + d]), acc);
      }
      lrow[j] = kScoreScale * acc;
    }
  }
}

}  // namespace

[[noreturn]] void qi_fail(const char* what) {
  std::fprintf(stderr, "qsa_indexer: %s\n", what);
  std::abort();
}

void qi_launch(const QsaIndexerParams& p, cudaStream_t stream) {
  const int T = p.tokens;
  const int n_full = p.n_full;
  const int B = p.budget;

  // Q stage: the full [T, 512] tiled q̃ (written once).
  qsa_idx_q_kernel<<<dim3(T), 128, 0, stream>>>(
      p.hidden_dev, p.qk_proj_dev, p.q_norm_dev, p.work_dev, p.hidden);

  const bool is_fp8 = (p.dtype == QsaIndexerKvDtype::Fp8);
  // chunk keeps [chunk, n_full] fp32 within max_logits_bytes; below the 8e6
  // fp32 threshold a single chunk of T suffices.
  const long long work_elems = static_cast<long long>(n_full) * T;
  const int chunk_full = (work_elems < 8000000LL)
      ? T
      : std::min(T, std::max(1, p.max_logits_bytes / (n_full * 4)));

  for (int base = 0; base < T; base += chunk_full) {
    const int chunk = std::min(chunk_full, T - base);
    qsa_idx_logit_kernel<<<dim3(chunk), 256, 0, stream>>>(
        p.work_dev, p.idx_k_pages_dev, p.idx_k_scale_dev, p.block_table_dev,
        p.logits_dev, base, chunk, n_full, is_fp8);

    QsaTopkParams tp;
    tp.rows            = chunk;
    tp.n_full          = n_full;
    tp.budget          = B;
    tp.logits_dev      = p.logits_dev;
    tp.context_len_dev = p.context_len_dev + base;
    tp.ids_dev         = p.ids_dev + static_cast<std::size_t>(base) * B;
    tp.counts_dev      = p.counts_dev + base;
    qsa_topk(tp, stream);
  }

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    qi_fail(cudaGetErrorString(err));
  }
}

void qsa_indexer(const QsaIndexerParams& p, cudaStream_t stream) {
  if (p.tokens <= 0 || p.tokens > kMaxTokens) {
    qi_fail("tokens out of range (1..2048)");
  }
  if (p.hidden < 128 || p.hidden > 2560 || p.hidden % 128 != 0) {
    qi_fail("hidden out of range (128..2560, multiple of 128)");
  }
  if (p.budget <= 0) {
    qi_fail("budget must be >= 1");
  }
  if (p.pos0 < 0) {
    qi_fail("pos0 must be >= 0");
  }
  if (p.n_full <= 0) {
    qi_fail("n_full must be >= 1");
  }
  if (!p.hidden_dev || !p.qk_proj_dev || !p.q_norm_dev ||
      !p.block_table_dev || !p.idx_k_pages_dev || !p.work_dev ||
      !p.logits_dev || !p.ids_dev || !p.counts_dev || !p.context_len_dev) {
    qi_fail("missing buffer");
  }
  if (p.dtype == QsaIndexerKvDtype::Fp8 && !p.idx_k_scale_dev) {
    qi_fail("missing idx_k_scale_dev (Fp8)");
  }
  qi_launch(p, stream);
}

}  // namespace ninfer::ops
