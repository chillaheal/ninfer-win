#pragma once
// P7 7a contract — `qsa_indexer`: Qwen Sparse Attention indexer logit GEMM +
// topk. A NEW Op family (do not reuse the qwen3_6 attention helpers). The
// production modeling code is unrecoverable (the artifact ships only the object
// shapes), so this is a documented v1 math contract anchored on those shapes:
//   attention/indexer/qk_proj  [640, H]   BF16   (rows [0..128) = the query)
//   attention/indexer/q_norm   [128]      BF16
// (the shared k_norm / four 128-dim key chunks are derived by the SEPARATE op
//  qsa_indexer_k_append (T1) and cached in the paged indexer-K pool — see
//  qsa_indexer_k_append.h — so this op no longer takes a k_norm.)
//
// Long-context structure: the query selects B key positions from the FULL
// context, not just the current prefill round. This op therefore scores each of
// the round's T queries against every key position [0, n_full) (n_full = pos0 +
// T) by a GEMM against the paged indexer-K pool, then takes the top-B valid
// positions per query (valid = j < context_len[t] = pos0 + t) via qsa_topk.
//
// Math (per round token t; H = hidden width, runtime; all bf16 on device):
//   z[t,d]    = sum_i qk_proj[d,i] * hidden[t,i]          (d in [0,128), FP32)
//   q_tilde   = rms128(z, q_norm)  = z * q_norm / sqrt(mean(z^2) + 1e-6)
//   (q_tilde, 128-dim, is tiled 4x to 512-dim [q,q,q,q] to align with the
//    four 128-dim key chunks; work_dev holds the tiled [T, 512] fp32 rows)
//   logit[t, j] = (1/sqrt(128)) * sum_d q_tilde_tiled[t][d] * k_tilde[j][d]
//                 for j in [0, n_full); k_tilde[j] is the DEQUANTIZED indexer-K
//                 pool row at GLOBAL position j (FP8 E4M3 + per-token half
//                 scale, or BF16 for the fallback dtype). The pool is addressed
//                 through the shared block table (the same table the main QSA
//                 K/V pool uses), so j is a GLOBAL key position and ids are
//                 emitted already global (no +pos0 shift).
//
// Topk (per query row t): the budget B positions of the VALID prefix
// [0, context_len[t]) ranked by (logit DESC, position DESC); ids[t, j<counts]
// are ASCENDING, ids[t, j>=counts] = -1 (the whole [T,B] region is written);
// counts[t] = min(B, context_len[t]). When context_len[t] <= B the row is dense
// (ids = 0..context_len[t]-1). This is delegated to ninfer::ops::qsa_topk (T2),
// which masks on context_len (the GEMM leaves garbage at j >= context_len[t]).
//
// Chunking: the logit scratch [chunk, n_full] fp32 must fit max_logits_bytes,
// so qi_launch partitions the M (query) dimension into chunks
//   chunk = (n_full*T < 8e6) ? T : min(T, max(1, max_logits_bytes / (n_full*4)))
// and launches (GEMM over the chunk -> qsa_topk over the chunk) per chunk.
// Each (t, j) logit is computed by a fixed-order dot independent of the chunk
// boundary, so a chunked run is BIT-EQUAL to an unchunked run of the same
// inputs (verified by test 7f a). work_dev is the full [T, 512] tiled q
// (written once by the Q stage); logits_dev is reused per chunk (it holds only
// the LAST chunk after qi_launch returns).
//
// Determinism: every reduction is a fixed-order tree; the FP8 dequant and the
// FP32 dot are fixed-order, so reruns over the same inputs are bit-equal.
//
// Geometry limits: 1 <= T <= 2048, 128 <= H <= 2560 with H % 128 == 0
// (production H = 2560), B >= 1, pos0 >= 0, n_full >= 1. The Op performs no
// device allocation; it only launches on `stream`. Inputs and outputs are
// pairwise non-aliasing (work_dev and logits_dev are caller scratch).

#include "qsa_indexer_k_append.h"  // QsaIndexerKvDtype

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

struct QsaIndexerParams {
  int tokens = 0;            // T (1..2048)
  int hidden = 0;            // H (2560)
  int budget = 0;            // B (64)
  std::int32_t pos0 = 0;     // global start position of this round
  std::int32_t n_full = 0;   // = pos0 + T (max context this round)
  const std::uint16_t* hidden_dev  = nullptr;  // BF16 [T, H]
  const std::uint16_t* qk_proj_dev = nullptr;  // BF16 [640, H]; rows [0..128) are the query
  const std::uint16_t* q_norm_dev  = nullptr;  // BF16 [128]
  const std::int32_t*  block_table_dev = nullptr;  // I32 [page_capacity]
  const void*          idx_k_pages_dev = nullptr;  // FP8 [512,64,1,P] OR BF16
  const std::uint16_t* idx_k_scale_dev = nullptr;  // half per token slot [64,1,1,P] (Fp8)
  float* work_dev = nullptr;        // FP32 scratch: tiled q_tilde [T, 512]
  float* logits_dev = nullptr;      // FP32 scratch: [chunk, n_full] (<= max_logits_bytes)
  std::int32_t* ids_dev = nullptr;     // I32 [T, B]
  std::int32_t* counts_dev = nullptr;  // I32 [T]
  std::int32_t* context_len_dev = nullptr;  // I32 [T]; context_len[t] = pos0 + t
  std::int32_t max_logits_bytes = (1 << 30);  // default 1 GiB logit scratch budget
  QsaIndexerKvDtype dtype = QsaIndexerKvDtype::Fp8;
};

[[noreturn]] void qi_fail(const char* what);
void qi_launch(const QsaIndexerParams& p, cudaStream_t stream);

void qsa_indexer(const QsaIndexerParams& p, cudaStream_t stream);

}  // namespace ninfer::ops
