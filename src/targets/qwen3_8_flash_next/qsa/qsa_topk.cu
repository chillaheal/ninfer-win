// qsa_topk kernels — contract in include/ninfer/ops/qsa_topk.h. One block per query
// row. For row r the valid logits are columns [0, n_t) with n_t = context_len[r];
// columns [n_t, n_full) are IGNORED (not -inf-prefilled; the GEMM may leave garbage).
// The op emits the top-`budget` column POSITIONS ranked by (logit DESC, position DESC):
// a larger logit wins, and on an exact logit tie a LARGER position wins (matching v1
// `index_topk_row`, test_p7_qsa.cpp). ids are ascending by position, -1 fill for
// j >= counts_dev[r]; counts_dev[r] = min(budget, n_t). When n_t <= budget the row is
// dense: ids = 0..n_t-1.
//
// Method: encode each valid column into a 64-bit ORDERING KEY
//   key = (value_key << 32) | position        (unsigned)
// where value_key is a monotone float->u32 map (x < y  <=>  value_key(x) < value_key(y))
// and position is the column index. Comparing keys as UNSIGNED 64-bit then orders by
// (value DESC, position DESC), and because position is unique per column there are NO
// ties. A 8-pass 8-bit RADIX SELECT (MSB->LSB) finds the budget-th largest key T; the
// top-`budget` set = (budget-1) strictly-above keys + the single key == T. All
// reductions are fixed-order, so a rerun over the same inputs is bit-equal.

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include <ninfer/ops/qsa_topk.h>

namespace ninfer::ops {

namespace {

constexpr int kThreads = 256;
constexpr int kMaxRows = 2048;
constexpr int kMaxN    = 1 << 20;  // n_full cap (production ~262144)
constexpr int kMaxB    = 1024;     // shared-memory sized (QSA budget is 64)

// Monotone non-decreasing in x for finite floats: x < y  <=>  key(x) < key(y).
__device__ __forceinline__ std::uint32_t value_key(float x) {
  const std::uint32_t u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// 64-bit ordering key: (value DESC, position DESC). Larger == better; unique per column.
__device__ __forceinline__ std::uint64_t order_key(float x, int pos) {
  return (static_cast<std::uint64_t>(value_key(x)) << 32) |
         static_cast<std::uint64_t>(pos & 0x7FFFFFFF);
}

__global__ void qsa_topk_kernel(
    const float* __restrict__ logits, const std::int32_t* __restrict__ ctx,
    std::int32_t* __restrict__ ids, std::int32_t* __restrict__ counts,
    int n_full, int B) {
  __shared__ std::uint32_t sh_hist[256];
  __shared__ std::uint64_t sh_thr;     // committed radix prefix (grows to T)
  __shared__ int sh_above_;            // running count of keys above the prefix
  __shared__ int sh_gt;                // count of keys strictly > T  (== B-1 when n>B)
  __shared__ std::int32_t sh_winner;   // the single position whose key == T
  __shared__ std::int32_t sh_pos_gt[kMaxB];

  const int r   = blockIdx.x;
  const int t   = threadIdx.x;
  const int n_t = ctx[r];

  if (t == 0) {
    counts[r] = (n_t < B) ? n_t : B;
  }
  __syncthreads();
  const int n    = n_t;
  const float* row = logits + static_cast<std::size_t>(r) * n_full;

  // Dense-skip: the whole valid prefix is selected (budget covers it).
  if (n <= B) {
    if (t == 0) {
      std::int32_t* out = ids + static_cast<std::size_t>(r) * B;
      for (int i = 0; i < n; ++i) out[i] = i;
      for (int i = n; i < B; ++i) out[i] = -1;
    }
    return;
  }

  if (t == 0) {
    sh_thr    = 0ull;
    sh_above_ = 0;
    sh_gt     = 0;
    sh_winner = -1;
  }
  __syncthreads();

  // ---- 8-pass radix select (64-bit unsigned, MSB->LSB) -> threshold key T. ----
  for (int pass = 0; pass < 8; ++pass) {
    const int shift = 56 - 8 * pass;
    if (t < 256) {
      sh_hist[t] = 0u;
    }
    __syncthreads();
    // Filter on the bytes STRICTLY ABOVE the current byte (bits [shift+8, 63]);
    // at the top byte (shift+8 == 64) the mask is 0 so every key is counted.
    // (Masking the current byte instead -- the old bug -- matched only keys whose
    //  already-decided high bytes were 0, i.e. no real key, stalling the select
    //  at T=0 and making Phase 3 index sh_pos_gt out of range.)
    const int upper = shift + 8;
    const std::uint64_t prefix_mask = (upper >= 64) ? 0ull : ((~0ull) << upper);
    const std::uint64_t prefix_val  = sh_thr & prefix_mask;
    for (int j = t; j < n; j += kThreads) {
      const std::uint64_t k = order_key(row[j], j);
      if ((k & prefix_mask) == prefix_val) {
        atomicAdd(&sh_hist[(k >> shift) & 0xFFu], 1u);
      }
    }
    __syncthreads();
    if (t == 0) {
      const int target = B - sh_above_;
      int cum  = 0;
      int star = 0;
      int above = 0;
      for (int d = 255; d >= 0; --d) {
        const int c = sh_hist[d];
        if (cum + c >= target) {
          star  = d;
          above = cum;
          break;
        }
        cum += c;
      }
      sh_thr    |= (static_cast<std::uint64_t>(star) << shift);
      sh_above_ += above;
    }
    __syncthreads();
  }
  const std::uint64_t T = sh_thr;

  // ---- Phase 2: collect the (B-1) strictly-above positions and the one == T. ---
  if (t == 0) {
    for (int i = 0; i < B; ++i) {
      sh_pos_gt[i] = -1;
    }
  }
  __syncthreads();
  for (int j = t; j < n; j += kThreads) {
    const std::uint64_t k = order_key(row[j], j);
    if (k > T) {
      const int slot = atomicAdd(&sh_gt, 1);  // # above T == B-1, so slot < B
      if (slot < B) {
        sh_pos_gt[slot] = j;
      }
    } else if (k == T) {
      sh_winner = j;  // exactly one (unique keys)
    }
  }
  __syncthreads();

  // ---- Phase 3: emit the B winners ascending by position (t==0; B small: 64). ---
  if (t == 0) {
    const int gt = sh_gt;  // == B-1 (keys unique, n > B)
    // Gather the B selected positions into sh_pos_gt[0..B).
    sh_pos_gt[gt] = sh_winner;
    // Insertion-sort ascending (B is small in practice: QSA budget is 64).
    for (int i = 1; i < B; ++i) {
      const std::int32_t v = sh_pos_gt[i];
      int m = i - 1;
      while (m >= 0 && sh_pos_gt[m] > v) {
        sh_pos_gt[m + 1] = sh_pos_gt[m];
        --m;
      }
      sh_pos_gt[m + 1] = v;
    }
    std::int32_t* out = ids + static_cast<std::size_t>(r) * B;
    for (int i = 0; i < B; ++i) {
      out[i] = sh_pos_gt[i];
    }
  }
}

}  // namespace

[[noreturn]] void qtk_fail(const char* what) {
  std::fprintf(stderr, "qsa_topk: %s\n", what);
  std::abort();
}

void qsa_topk(const QsaTopkParams& p, cudaStream_t stream) {
  if (p.rows <= 0 || p.rows > kMaxRows) {
    qtk_fail("rows out of range (1..2048)");
  }
  if (p.n_full <= 0 || p.n_full > kMaxN) {
    qtk_fail("n_full out of range (1..1048576)");
  }
  if (p.budget <= 0 || p.budget > kMaxB) {
    qtk_fail("budget out of range (1..1024)");
  }
  if (!p.logits_dev || !p.context_len_dev || !p.ids_dev || !p.counts_dev) {
    qtk_fail("missing buffer");
  }
  const dim3 grid(p.rows), block(kThreads);
  qsa_topk_kernel<<<grid, block, 0, stream>>>(
      p.logits_dev, p.context_len_dev, p.ids_dev, p.counts_dev, p.n_full, p.budget);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    qtk_fail(cudaGetErrorString(err));
  }
}

}  // namespace ninfer::ops
