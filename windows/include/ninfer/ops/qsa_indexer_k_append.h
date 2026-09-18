#pragma once
// qsa_indexer_k_append — Qwen Sparse Attention indexer KEY append (paged, FP8).
//
// Complements the main QSA sparse GQA: the QSA indexer selects the B key
// positions a query attends to. For long context the selection must look at the
// FULL context, not just the current prefill round, so the indexer keeps its own
// paged cache of the normalized 512-dim key per token (SGLang's long-context
// structure), separate from the main K/V pool.
//
// This op derives, for each token t of the round, the four 128-dim key chunks
// (dims 128..639 of the qk_proj projection), rms128-normalizes each chunk with
// the shared k_norm, quantizes the 512-dim key to FP8 (E4M3) with a per-token
// scale, and scatters it into the paged pool at the token's GLOBAL position
// (pos0 + t) through the shared block table — the same table the main QSA K/V
// pool uses, so the pages stay aligned with the main KV.
//
// Layout (page-major, 64-token pages; mirrors the main QSA K/V pools):
//   value:  idx_k_pages_dev[ physical_page * 64 * 512 + (pos & 63) * 512 + d ]
//           == paged_kv_element_offset<512,1>(block_table, 0, pos, d)   (FP8, 1 B)
//   scale:  idx_k_scale_dev[ physical_page * 64 + (pos & 63) ]          (half)
// The per-token scale is kscale[t] = clamp(absmax(key_row)/448, 2^-24, 65504);
// the stored code is E4M3(key_row[t] / kscale[t]), so the dequantized key is
// kscale[t] * fp8. In the logit GEMM kscale[t] factors out of the dot product.
//
// work_dev (optional, test-only): the pre-quantization BF16 512-dim key row per
// token, so a CPU reference can verify the GEMV+rms128 math bit-exactly.
//
// Determinism: the GEMV and rms128 reductions are fixed-order trees (warp
// shuffle + shared-memory tree), so reruns are bit-equal; the FP8 codes and the
// BF16 pre-quant row are exact for a given input.
//
// Geometry: 1 <= T <= 2048, 128 <= H <= 2560 with H % 128 == 0. The op performs
// no device allocation; it only launches on `stream`.
//
// Note: this is the v1 math anchored on the artifact shapes (qk_proj [640,H]
// split into one 128-dim query chunk + four 128-dim key chunks, a single shared
// [128] k_norm) — the same documented approximation qsa_indexer uses.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Indexer-K pool storage dtype. Fp8 (E4M3 codes + per-token half scale) is the
// default; Bf16 is the fallback. Consumed by qsa_indexer (the logit GEMM reads
// the pool in this dtype); qsa_indexer_k_append is Fp8-only today.
enum class QsaIndexerKvDtype : std::uint8_t { Fp8, Bf16 };

struct QsaIndexerKAppendParams {
  int tokens = 0;            // T (1..2048)
  int hidden = 0;            // H (2560)
  std::int32_t pos0 = 0;     // global start position of this round
  const std::uint16_t* hidden_dev  = nullptr;  // BF16 [T, H]
  const std::uint16_t* qk_proj_dev = nullptr;  // BF16 [640, H]; rows [128..640) are the 4 key chunks
  const std::uint16_t* k_norm_dev  = nullptr;  // BF16 [128]
  const std::int32_t*  block_table_dev = nullptr;  // I32 [page_capacity]
  std::uint8_t*        idx_k_pages_dev = nullptr;  // FP8 [512, 64, 1, P]
  std::uint16_t*       idx_k_scale_dev = nullptr;  // half, one per token slot [64, 1, 1, P]
  std::uint16_t*       work_dev = nullptr;         // BF16 [T, 512] pre-quant (optional, test-only)
};

[[noreturn]] void qia_fail(const char* what);
void qsa_indexer_k_append(const QsaIndexerKAppendParams& p, cudaStream_t stream);

}  // namespace ninfer::ops
