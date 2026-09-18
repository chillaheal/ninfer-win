// qsa_topk contract — global row-wise top-B over the QSA indexer logits.
//
// For each of `rows` query rows the logits row is `logits_dev[r] = [n_full]` FP32, but
// only the first `context_len_dev[r]` (= n_t) columns are valid; the columns
// [n_t, n_full) must be IGNORED (they are not -inf-prefilled — the GEMM may leave them
// garbage). The op emits, per row, the top-`budget` column POSITIONS ranked by
// (logit DESC, position DESC): a larger logit wins, and on an exact logit tie a LARGER
// position wins (matching v1 `index_topk_row`, test_p7_qsa.cpp). `ids_dev[r]` holds the
// winners in ASCENDING position order, with -1 fill for j >= counts_dev[r];
// `counts_dev[r] = min(budget, n_t)`. When n_t <= budget the row is dense: ids = 0..n_t-1.
//
// All reductions are fixed-order, so a rerun over the same inputs is bit-equal.

#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace ninfer::ops {

struct QsaTopkParams {
  int rows = 0;                            // q rows in this chunk (1..2048)
  int n_full = 0;                          // columns (full context this round, >= 1)
  int budget = 0;                          // B (64 in production; 1..1024)
  const float* logits_dev = nullptr;       // FP32 [rows, n_full]
  const std::int32_t* context_len_dev = nullptr;  // I32 [rows]; valid columns [0, n_t)
  std::int32_t* ids_dev = nullptr;         // I32 [rows, budget] (global, ascending, -1 fill)
  std::int32_t* counts_dev = nullptr;      // I32 [rows] = min(budget, n_t)
};

[[noreturn]] void qtk_fail(const char* what);
void qsa_topk(const QsaTopkParams&, cudaStream_t);

}  // namespace ninfer::ops
