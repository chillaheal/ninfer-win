#pragma once
// P7 7b v1 contract — `qsa_sparse_gqa`: Qwen Sparse Attention grouped-query
// attention over READY selected ids (from `qsa_indexer`, 7a). A NEW Op
// family — not `softmax_attention` with a mask hack: the K/V rows are
// GATHERED from paged storage at the selected positions; no unselected
// position is ever read.
//
// The production modeling code is unrecoverable (the artifact ships only
// the object shapes), so this is a documented v1 math contract anchored on
// those shapes:
//   attention/query  [12288, 2560]  -> 48 q heads x 256 head dim
//   attention/key    [512, 2560]    -> 2 kv heads x 256 head dim
//   attention/value  [512, 2560]    -> 2 kv heads x 256 head dim
// GQA ratio 24 (48/2); q/k norm and rope are the CALLER's job — the Op
// consumes q exactly as supplied (the P9 target applies them).
//
// Math (per (t, h); head dim D = 256 fixed, kv head g = h / (Q/KV)):
//   n   = counts[t]  (>= 1 always: budget >= 1 and the indexer emits
//         counts[t] = min(budget, t+1) >= 1)
//   p_j = ids[t, j], j < n                       (positions, any order;
//                                                 the indexer emits ascending)
//   s_j = scale * sum_d q[t,h,d] * K[p_j, g, d]  (FP32 over bf16 source)
//   m   = max_j s_j
//   u_j = exp(s_j - m);  z = sum_j u_j
//   out[t,h,d] = bf16( (1/z) * sum_j u_j * V[p_j, g, d] )
// scale = 1/sqrt(256) = 1/16 at the QSA geometry; the Op takes it as a
// parameter so the same kernel serves any documented scale.
//
// K/V rows are read from paged storage: the plane is [256, 64, KV, P]
// page-major BF16 (KVPageGeometry page_tokens=64, the default PageMajor
// order, plane {BF16, 256, KV, 256}) and the element for (position p, kv
// head g, dim d) is paged_kv_element_offset<256, KV>(block_table, g, p, d)
// — the same addressing as the write path (7c). K/V may live in ANY
// physical pages (the gather follows the block table per position), so the
// Op is paging-transparent.
//
// Decode guarantee: work per (t, h) is O(n) gather rows — decode T=1 runs
// O(budget) work and NEVER a T x T dense pass (there is no dense path at
// all). Dense equivalence: with ids[t] = 0..t (the 7a B >= T case) the Op
// equals plain causal attention over the same KV rows; the test checks it
// against the FP64 oracle within BF16 tolerance.
//
// v1 geometry limits: 1 <= T <= 2048; 1 <= KV <= 8 with Q % KV == 0;
// D = 256 fixed; B >= 1; positions in [0, 64 * block_table capacity);
// counts[t] >= 1. Non-finite policy: finite inputs give finite outputs
// (z >= 1 after the max shift; no 0/0). NaN inputs are not contracted.
//
// Determinism: the softmax runs in 16-position chunks with a fixed-order
// online (max, z, per-dim output) rescale; every accumulation is a fixed
// tree — no float atomics, so reruns are bit-equal.
//
// 7c note (paged KV write/read, QSA layers only — 12 in production, 1 in
// the mini artifact): the WRITE path is the existing shared Op
// ninfer::ops::kv_cache_append (bf16 [256, 2|4, T] copied bit-for-bit at
// sequential positions through the block table — no new kernel). The
// store geometry is KVPageGeometry{page_tokens=64 (the default page size),
// the default PageMajor order, planes {BF16, 256, KV, 256} x 2} — the
// existing KV store types fit the QSA 64-token pages exactly, so there is
// no deviation to document in the Op contract. The host KV replica path works on the same
// types: DeviceKVPagePool::copy_to_host / zero_pages / copy_from_host with
// a one-page HostKVArena allocation (tested at the unit level, 7c).
//
// The Op performs no device allocation; it only launches on `stream`.
// Inputs and the cache planes/tables are pairwise non-aliasing.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

struct QsaSparseGqaParams {
  int tokens = 0;      // T
  int q_heads = 0;     // Q
  int kv_heads = 0;    // KV
  int budget = 0;      // B (ids width)
  float scale = 0.f;   // attention scale (1/16 at the QSA geometry)
  const std::uint16_t* q_dev = nullptr;   // BF16 [T, Q, 256]
  const std::int32_t* ids_dev = nullptr;  // I32 [T, B]
  const std::int32_t* counts_dev = nullptr; // I32 [T], counts[t] >= 1
  const std::int32_t* block_table_dev = nullptr; // I32 [capacity]
  const std::uint16_t* k_pages_dev = nullptr;    // BF16 [256, 64, KV, P]
  const std::uint16_t* v_pages_dev = nullptr;    // BF16 [256, 64, KV, P]
  std::uint16_t* out_dev = nullptr;             // BF16 [T, Q, 256]
};

[[noreturn]] void qg_fail(const char* what);
void qg_launch(const QsaSparseGqaParams& p, cudaStream_t stream);

void qsa_sparse_gqa(const QsaSparseGqaParams& p, cudaStream_t stream);

}  // namespace ninfer::ops
