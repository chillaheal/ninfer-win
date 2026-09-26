// P7 gate — QSA (Qwen Sparse Attention) v1 ops:
//  7a qsa_indexer    (chunked logit GEMM over the paged FP8 indexer-K pool +
//                     per-chunk qsa_topk; causal logit FP64 oracle; exact
//                     ids/counts; tie-break)
//  7f qsa_indexer    (chunking bit-equal, global pos0 shift, decode T=1,
//                     dense-skip, T=2048)
//  7b qsa_sparse_gqa (FP64 oracle: gather-KV attention; dense equivalence;
//                     decode T=1 exact single-V-row proof)
//  7c paged KV       (paged BF16 K/V scatter + one-page host demote/restore)
// Contracts: include/ninfer/ops/qsa_indexer.h, qsa_indexer_k_append.h,
//            qsa_sparse_gqa.h.

#include <ninfer/ops/qsa_indexer.h>
#include <ninfer/ops/qsa_indexer_k_append.h>
#include <ninfer/ops/qsa_sparse_gqa.h>
#include <ninfer/ops/qsa_topk.h>

#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;

namespace {

constexpr int kIdxWidth    = 640;
constexpr int kIdxChunk    = 128;
constexpr double kScoreScale = 0.088388347648318427;  // 1 / sqrt(128)
constexpr double kAttnScale  = 0.0625;               // 1 / sqrt(256)

int failures = 0;

std::vector<double> read_f32(const void* device, std::size_t n) {
  const std::vector<float> f = from_device<float>(device, n);
  std::vector<double> o(f.size());
  for (std::size_t i = 0; i < f.size(); ++i) {
    o[i] = static_cast<double>(f[i]);
  }
  return o;
}

// --- 7a oracle (FP64, over the EXACT bf16-grid inputs) ----------------------

// z = W h, then per-chunk RMSNorm x * g / sqrt(mean(x^2) + 1e-6).
// out[t][c*128+d]: c=0 uses q_norm, c=1..4 use k_norm (shared, per contract).
std::vector<double> index_normed_z(const std::vector<float>& hidden, int T, int H,
                                   const std::vector<float>& w,
                                   const std::vector<float>& qnorm,
                                   const std::vector<float>& knorm) {
  std::vector<double> z(static_cast<std::size_t>(T) * kIdxWidth);
  for (int t = 0; t < T; ++t) {
    for (int o = 0; o < kIdxWidth; ++o) {
      double acc = 0.0;
      const float* wrow = w.data() + static_cast<std::size_t>(o) * H;
      const float* hrow = hidden.data() + static_cast<std::size_t>(t) * H;
      for (int i = 0; i < H; ++i) {
        acc += static_cast<double>(wrow[i]) * static_cast<double>(hrow[i]);
      }
      z[static_cast<std::size_t>(t) * kIdxWidth + o] = acc;
    }
  }
  std::vector<double> out(z.size());
  for (int t = 0; t < T; ++t) {
    const double* zrow = z.data() + static_cast<std::size_t>(t) * kIdxWidth;
    double* orow       = out.data() + static_cast<std::size_t>(t) * kIdxWidth;
    for (int c = 0; c < 5; ++c) {
      const double* chunk = zrow + c * kIdxChunk;
      double sumsq        = 0.0;
      for (int d = 0; d < kIdxChunk; ++d) {
        sumsq += chunk[d] * chunk[d];
      }
      const double inv = 1.0 / std::sqrt(sumsq / kIdxChunk + 1e-6);
      const float* g   = (c == 0 ? qnorm : knorm).data();
      for (int d = 0; d < kIdxChunk; ++d) {
        orow[c * kIdxChunk + d] = chunk[d] * static_cast<double>(g[d]) * inv;
      }
    }
  }
  return out;
}

// score(t, i) = i <= t ? (1/sqrt(128)) * sum_c sum_d q_tilde[d] * k_c_tilde[i][d]
//              : -inf
std::vector<double> index_scores(const std::vector<double>& nt, int T) {
  std::vector<double> s(static_cast<std::size_t>(T) * T, -INFINITY);
  for (int t = 0; t < T; ++t) {
    const double* q = nt.data() + static_cast<std::size_t>(t) * kIdxWidth;
    double* row     = s.data() + static_cast<std::size_t>(t) * T;
    for (int i = 0; i <= t; ++i) {
      const double* k = nt.data() + static_cast<std::size_t>(i) * kIdxWidth;
      double acc      = 0.0;
      for (int c = 0; c < 4; ++c) {
        const double* kc = k + (1 + c) * kIdxChunk;
        for (int d = 0; d < kIdxChunk; ++d) {
          acc += q[d] * kc[d];
        }
      }
      row[i] = kScoreScale * acc;
    }
  }
  return s;
}

// topk with the (score DESC, position DESC) key; returns the winning
// positions of row t in ASCENDING order, padded with -1 to width B.
std::vector<int> index_topk_row(const std::vector<double>& scores, int T, int t, int B) {
  const double* row = scores.data() + static_cast<std::size_t>(t) * T;
  const int n       = std::min(B, t + 1);
  std::vector<std::pair<double, int>> cands;
  cands.reserve(static_cast<std::size_t>(t) + 1);
  for (int i = 0; i <= t; ++i) {
    cands.emplace_back(row[i], i);
  }
  std::sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) {
      return a.first > b.first;
    }
    return a.second > b.second;
  });
  std::vector<int> win;
  win.reserve(static_cast<std::size_t>(n));
  for (int r = 0; r < n; ++r) {
    win.push_back(cands[static_cast<std::size_t>(r)].second);
  }
  std::sort(win.begin(), win.end());
  std::vector<int> out(static_cast<std::size_t>(B), -1);
  for (int j = 0; j < n; ++j) {
    out[static_cast<std::size_t>(j)] = win[static_cast<std::size_t>(j)];
  }
  return out;
}

// Forward decls: the host FP8/FP16 decoders and the 7e top-B oracle are defined
// below; the 7a/7f helpers below use them.
float fp8_e4m3_to_f32(std::uint8_t x);
float fp16_to_f32_h(std::uint16_t h);
std::vector<int> topk_row(const float* row_f32, int n_t, int B);

// Host copy of the FP8 indexer-K pool read back from device (build_pool).
struct Pool {
  std::vector<std::uint8_t> pages;   // [512, 64, P] E4M3 codes
  std::vector<std::uint16_t> scale;  // [64, P] per-token half scale
};

// Flattened causal-prefix reference logits (t outer, j in [0, ctx[t]) inner),
// FP64, from the dequantized pool + the FP64 query. The dequantized pool is the
// exact k the logit GEMM reads, so the FP8 quantization error is common to ref
// and kernel and cancels; only the FP32-vs-FP64 GEMV/rms + the FP8-dequant
// half-rounding remain (see the 7a tolerance).
std::vector<double> ref7a_causal(int T, int H,
                                 const std::vector<float>& hidden_round,
                                 const std::vector<float>& w,
                                 const std::vector<float>& qnorm,
                                 const Pool& pool, const std::vector<int>& bt,
                                 int n_full, const std::vector<int>& ctx) {
  std::vector<double> qt(static_cast<std::size_t>(T) * 128);
  for (int t = 0; t < T; ++t) {
    std::vector<double> z(128);
    for (int d = 0; d < 128; ++d) {
      double acc = 0.0;
      const float* wrow = w.data() + static_cast<std::size_t>(d) * H;
      const float* hrow = hidden_round.data() + static_cast<std::size_t>(t) * H;
      for (int i = 0; i < H; ++i) {
        acc += static_cast<double>(wrow[i]) * static_cast<double>(hrow[i]);
      }
      z[static_cast<std::size_t>(d)] = acc;
    }
    double sumsq = 0.0;
    for (int d = 0; d < 128; ++d) {
      sumsq += z[static_cast<std::size_t>(d)] * z[static_cast<std::size_t>(d)];
    }
    const double inv = 1.0 / std::sqrt(sumsq / 128.0 + 1e-6);
    for (int d = 0; d < 128; ++d) {
      qt[static_cast<std::size_t>(t) * 128 + d] =
          z[static_cast<std::size_t>(d)] * static_cast<double>(qnorm[d]) * inv;
    }
  }
  std::vector<double> kt(static_cast<std::size_t>(n_full) * 512);
  for (int j = 0; j < n_full; ++j) {
    const int pp   = bt[static_cast<std::size_t>(j >> 6)];
    const int slot = j & 63;
    const double sc =
        fp16_to_f32_h(pool.scale[static_cast<std::size_t>(pp) * 64 + slot]);
    for (int d = 0; d < 512; ++d) {
      const std::int64_t off =
          static_cast<std::int64_t>(512) * 64 * pp + 512 * slot + d;
      kt[static_cast<std::size_t>(j) * 512 + d] =
          static_cast<double>(
              fp8_e4m3_to_f32(pool.pages[static_cast<std::size_t>(off)]) * sc);
    }
  }
  std::vector<double> out;
  for (int t = 0; t < T; ++t) {
    const double* qtr = qt.data() + static_cast<std::size_t>(t) * 128;
    for (int j = 0; j < ctx[static_cast<std::size_t>(t)]; ++j) {
      const double* ktr = kt.data() + static_cast<std::size_t>(j) * 512;
      double acc = 0.0;
      for (int c = 0; c < 4; ++c) {
        for (int d = 0; d < 128; ++d) {
          acc += qtr[d] * ktr[c * 128 + d];
        }
      }
      out.push_back(kScoreScale * acc);
    }
  }
  return out;
}

// Run the NEW qsa_indexer once (single chunk — large max_logits_bytes) and
// verify: (1) the logit GEMM against the FP64 dequant reference over the causal
// prefix [0, context_len[t]); (2) the selected ids/counts EXACTLY against a
// host top-B (topk_row) of the KERNEL'S OWN read-back logits. context_len[t] =
// pos0 + t (exclusive of self); ids are already global (no +pos0 shift).
void run_7a_case(const std::string& label, int T, int H, int B, int pos0,
                 int n_full, const std::vector<float>& hidden_round,
                 const std::vector<float>& w, const std::vector<float>& qnorm,
                 const Pool& pool, const std::vector<int>& bt) {
  DeviceBuffer dh  = to_device_bf16(hidden_round);
  DeviceBuffer dw  = to_device_bf16(w);
  DeviceBuffer dqn = to_device_bf16(qnorm);
  DeviceBuffer dbt = to_device_i32(bt);
  DeviceBuffer dpg = to_device<std::uint8_t>(pool.pages);
  DeviceBuffer dsc = to_device<std::uint16_t>(pool.scale);
  DeviceBuffer dwork(static_cast<std::size_t>(T) * 512 * sizeof(float));

  std::vector<int> ctx(T);
  for (int t = 0; t < T; ++t) {
    ctx[static_cast<std::size_t>(t)] = pos0 + t;
  }
  DeviceBuffer dctx = to_device_i32(ctx);

  GuardedDeviceBuffer glogits(static_cast<std::size_t>(T) * n_full * sizeof(float));
  GuardedDeviceBuffer gids(static_cast<std::size_t>(T) * B * sizeof(std::int32_t));
  GuardedDeviceBuffer gcounts(static_cast<std::size_t>(T) * sizeof(std::int32_t));

  QsaIndexerParams p;
  p.tokens           = T;
  p.hidden           = H;
  p.budget           = B;
  p.pos0             = pos0;
  p.n_full           = n_full;
  p.hidden_dev       = static_cast<const std::uint16_t*>(dh.p);
  p.qk_proj_dev      = static_cast<const std::uint16_t*>(dw.p);
  p.q_norm_dev       = static_cast<const std::uint16_t*>(dqn.p);
  p.block_table_dev  = static_cast<const std::int32_t*>(dbt.p);
  p.idx_k_pages_dev  = dpg.p;
  p.idx_k_scale_dev  = static_cast<const std::uint16_t*>(dsc.p);
  p.work_dev         = static_cast<float*>(dwork.p);
  p.logits_dev       = static_cast<float*>(glogits.data());
  p.ids_dev          = static_cast<std::int32_t*>(gids.data());
  p.counts_dev       = static_cast<std::int32_t*>(gcounts.data());
  p.context_len_dev  = static_cast<std::int32_t*>(dctx.p);
  p.max_logits_bytes = (1 << 29);  // single chunk of T
  p.dtype            = QsaIndexerKvDtype::Fp8;

  qsa_indexer(p, nullptr);
  cuda_synchronize();
  failures += glogits.verify_guards(label.c_str());
  failures += gids.verify_guards(label.c_str());
  failures += gcounts.verify_guards(label.c_str());

  const std::vector<double> got_logits =
      read_f32(glogits.data(), static_cast<std::size_t>(T) * n_full);
  const std::vector<int> got_ids =
      from_device<int>(gids.data(), static_cast<std::size_t>(T) * B);
  const std::vector<int> got_counts = from_device<int>(gcounts.data(), T);

  // (1) causal-prefix logits vs the FP64 dequant reference.
  const std::vector<double> ref =
      ref7a_causal(T, H, hidden_round, w, qnorm, pool, bt, n_full, ctx);
  std::vector<double> got_causal;
  got_causal.reserve(ref.size());
  for (int t = 0; t < T; ++t) {
    for (int j = 0; j < ctx[static_cast<std::size_t>(t)]; ++j) {
      got_causal.push_back(
          got_logits[static_cast<std::size_t>(t) * n_full + j]);
    }
  }
  failures += verify_pointwise(label + " logit (causal)", got_causal, ref,
                               PointwiseCriterion{5e-2, 5e-3});

  // (2) ids/counts EXACTLY vs a host top-B of the kernel's own logits.
  const std::vector<float> got_logits_f =
      from_device<float>(glogits.data(), static_cast<std::size_t>(T) * n_full);
  std::vector<int> exp_ids, exp_counts;
  exp_ids.reserve(static_cast<std::size_t>(T) * B);
  for (int t = 0; t < T; ++t) {
    const int n_t = ctx[static_cast<std::size_t>(t)];
    const float* rw = got_logits_f.data() + static_cast<std::size_t>(t) * n_full;
    const std::vector<int> row = topk_row(rw, n_t, B);
    exp_ids.insert(exp_ids.end(), row.begin(), row.end());
    exp_counts.push_back(std::min(B, n_t));
  }
  failures += verify_exact((label + " ids").c_str(), got_ids, exp_ids);
  failures += verify_exact((label + " counts").c_str(), got_counts, exp_counts);
}

// Slice the round's hidden rows [pos0, pos0+T) out of the full [n_full, H]
// bf16-grid source (the Q-stage input for the last T tokens).
std::vector<float> slice_round(const std::vector<float>& hidden_all, int pos0,
                               int T, int H) {
  std::vector<float> out(static_cast<std::size_t>(T) * H);
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < H; ++i) {
      out[static_cast<std::size_t>(t) * H + i] =
          hidden_all[static_cast<std::size_t>(pos0 + t) * H + i];
    }
  }
  return out;
}

// Build the full paged FP8 indexer-K pool for global positions [0, n_full) by
// calling qsa_indexer_k_append in <=2048-token rounds; return the read-back
// pages + scale planes (host). `hidden_all` is the [n_full, H] bf16-grid source
// for every position; `bt` maps logical -> physical page (size == P).
Pool build_pool(int n_full, int H, const std::vector<float>& hidden_all,
                const std::vector<float>& w, const std::vector<float>& knorm,
                const std::vector<int>& bt) {
  const std::size_t P = bt.size();
  DeviceBuffer dh  = to_device_bf16(hidden_all);
  DeviceBuffer dw  = to_device_bf16(w);
  DeviceBuffer dkn = to_device_bf16(knorm);
  DeviceBuffer dbt = to_device_i32(bt);
  DeviceBuffer dpages(static_cast<std::size_t>(512) * 64 * P);
  DeviceBuffer dscale(static_cast<std::size_t>(64) * P * sizeof(std::uint16_t));

  int done = 0;
  while (done < n_full) {
    const int T = std::min(2048, n_full - done);
    QsaIndexerKAppendParams p;
    p.tokens          = T;
    p.hidden          = H;
    p.pos0            = done;
    p.hidden_dev      = static_cast<const std::uint16_t*>(dh.p) +
                        static_cast<std::size_t>(done) * H;
    p.qk_proj_dev     = static_cast<const std::uint16_t*>(dw.p);
    p.k_norm_dev      = static_cast<const std::uint16_t*>(dkn.p);
    p.block_table_dev = static_cast<const std::int32_t*>(dbt.p);
    p.idx_k_pages_dev = static_cast<std::uint8_t*>(dpages.p);
    p.idx_k_scale_dev = static_cast<std::uint16_t*>(dscale.p);
    qsa_indexer_k_append(p, nullptr);
    cuda_synchronize();
    done += T;
  }

  Pool out;
  out.pages = from_device<std::uint8_t>(dpages.p,
                                        static_cast<std::size_t>(512) * 64 * P);
  out.scale = from_device<std::uint16_t>(dscale.p, static_cast<std::size_t>(64) * P);
  return out;
}

// Raw qsa_indexer output: ids + counts always; logits/pages/scale only when
// single_chunk (the logit scratch then holds the full [T, n_full]).
struct F7a {
  std::vector<int> ids;
  std::vector<int> counts;
  std::vector<float> logits;  // [T, n_full] (empty unless single_chunk)
  std::vector<std::uint8_t> pages;
  std::vector<std::uint16_t> scale;
};

// Run qsa_indexer over the LAST T tokens of a n_full-length pool (pos0 =
// n_full - T). When single_chunk the logit scratch is held at (1<<29) bytes
// (full [T, n_full], single chunk) so the logits are read back; otherwise
// max_logits_bytes is small, forcing multi-chunk (only ids/counts are returned).
F7a run_7f_case(int T, int H, int B, int n_full,
                const std::vector<float>& hidden_all, const std::vector<float>& w,
                const std::vector<float>& qnorm, const std::vector<float>& knorm,
                const std::vector<int>& bt, std::int32_t max_logits_bytes,
                bool single_chunk) {
  const int pos0 = n_full - T;
  const Pool pool = build_pool(n_full, H, hidden_all, w, knorm, bt);
  const std::vector<float> hidden_round = slice_round(hidden_all, pos0, T, H);

  DeviceBuffer dh  = to_device_bf16(hidden_round);
  DeviceBuffer dw  = to_device_bf16(w);
  DeviceBuffer dqn = to_device_bf16(qnorm);
  DeviceBuffer dbt = to_device_i32(bt);
  DeviceBuffer dpg = to_device<std::uint8_t>(pool.pages);
  DeviceBuffer dsc = to_device<std::uint16_t>(pool.scale);
  DeviceBuffer dwork(static_cast<std::size_t>(T) * 512 * sizeof(float));
  std::vector<int> ctx(T);
  for (int t = 0; t < T; ++t) {
    ctx[static_cast<std::size_t>(t)] = pos0 + t;
  }
  DeviceBuffer dctx = to_device_i32(ctx);

  GuardedDeviceBuffer glogits(static_cast<std::size_t>(T) * n_full * sizeof(float));
  GuardedDeviceBuffer gids(static_cast<std::size_t>(T) * B * sizeof(std::int32_t));
  GuardedDeviceBuffer gcounts(static_cast<std::size_t>(T) * sizeof(std::int32_t));

  QsaIndexerParams p;
  p.tokens          = T;
  p.hidden          = H;
  p.budget          = B;
  p.pos0            = pos0;
  p.n_full          = n_full;
  p.hidden_dev      = static_cast<const std::uint16_t*>(dh.p);
  p.qk_proj_dev     = static_cast<const std::uint16_t*>(dw.p);
  p.q_norm_dev      = static_cast<const std::uint16_t*>(dqn.p);
  p.block_table_dev = static_cast<const std::int32_t*>(dbt.p);
  p.idx_k_pages_dev = dpg.p;
  p.idx_k_scale_dev = static_cast<const std::uint16_t*>(dsc.p);
  p.work_dev        = static_cast<float*>(dwork.p);
  p.logits_dev      = static_cast<float*>(glogits.data());
  p.ids_dev         = static_cast<std::int32_t*>(gids.data());
  p.counts_dev      = static_cast<std::int32_t*>(gcounts.data());
  p.context_len_dev = static_cast<std::int32_t*>(dctx.p);
  p.max_logits_bytes = single_chunk ? (1 << 29) : max_logits_bytes;
  p.dtype            = QsaIndexerKvDtype::Fp8;

  qsa_indexer(p, nullptr);
  cuda_synchronize();

  F7a out;
  out.ids    = from_device<int>(gids.data(), static_cast<std::size_t>(T) * B);
  out.counts = from_device<int>(gcounts.data(), T);
  if (single_chunk) {
    out.logits =
        from_device<float>(glogits.data(), static_cast<std::size_t>(T) * n_full);
    out.pages = pool.pages;
    out.scale = pool.scale;
  }
  return out;
}

// --- 7d oracle (FP64) + case: paged FP8 indexer-K append --------------------

// Host E4M3FN decode. The op quantizes with __NV_SATFINITE, so the NaN code
// (E=15, M=7) is never produced; E=15,M=6 is 448 (max finite).
float fp8_e4m3_to_f32(std::uint8_t x) {
  const int s = (x >> 7) & 1;
  const int E = (x >> 3) & 0x0F;
  const int M = x & 0x07;
  const float v = (E == 0) ? std::ldexp(M, -9)
                           : std::ldexp(1.0f + M / 8.0f, E - 7);
  return s ? -v : v;
}

// Host FP16 (half) decode — for the per-token FP8 scale.
float fp16_to_f32_h(std::uint16_t h) {
  const int s = (h >> 15) & 1;
  const int E = (h >> 10) & 0x1F;
  const int M = h & 0x3FF;
  float v;
  if (E == 0) {
    v = (M == 0) ? 0.0f : std::ldexp(M, -24);
  } else if (E == 31) {
    v = std::numeric_limits<float>::infinity();
  } else {
    v = std::ldexp(1.0f + M / 1024.0f, E - 15);
  }
  return s ? -v : v;
}

// FP64 pre-quant 512-dim key row for token t: per-128-chunk GEMV over qk_proj
// rows [128..640) + rms128 with the shared knorm — matches qsa_indexer_k_append.
// Layout out[kc*128+d] (kc = key-chunk index 0..3).
std::vector<double> idxk_normed_row(const std::vector<float>& hidden, int H,
                                    const std::vector<float>& w,
                                    const std::vector<float>& knorm, int t) {
  std::vector<double> z(512);
  for (int kc = 0; kc < 4; ++kc) {
    for (int d = 0; d < 128; ++d) {
      const int o = 128 + kc * 128 + d;  // qk_proj row index
      double acc = 0.0;
      const float* wrow = w.data() + static_cast<std::size_t>(o) * H;
      const float* hrow = hidden.data() + static_cast<std::size_t>(t) * H;
      for (int i = 0; i < H; ++i) {
        acc += static_cast<double>(wrow[i]) * static_cast<double>(hrow[i]);
      }
      z[static_cast<std::size_t>(kc) * 128 + d] = acc;
    }
  }
  std::vector<double> out(512);
  for (int kc = 0; kc < 4; ++kc) {
    const double* chunk = z.data() + kc * 128;
    double sumsq        = 0.0;
    for (int d = 0; d < 128; ++d) {
      sumsq += chunk[d] * chunk[d];
    }
    const double inv = 1.0 / std::sqrt(sumsq / 128.0 + 1e-6);
    for (int d = 0; d < 128; ++d) {
      out[static_cast<std::size_t>(kc) * 128 + d] =
          chunk[d] * static_cast<double>(knorm[d]) * inv;
    }
  }
  return out;
}

// Run qsa_indexer_k_append once and verify, per token: the pre-quant BF16 row
// (isolates the GEMV + rms128) and the dequantized stored FP8 row (isolates
// the per-token scale + the paged block-table addressing).
void run_7d_case(const std::string& label, int T, int H, int pos0,
                 const std::vector<float>& hidden, const std::vector<float>& w,
                 const std::vector<float>& knorm, const std::vector<int>& bt,
                 int P) {
  DeviceBuffer dh  = to_device_bf16(hidden);
  DeviceBuffer dw  = to_device_bf16(w);
  DeviceBuffer dkn = to_device_bf16(knorm);
  DeviceBuffer dbt = to_device_i32(bt);

  // pool [512, 64, P] FP8 bytes; scale [64, P] halves; work [T, 512] bf16.
  GuardedDeviceBuffer gpages(static_cast<std::size_t>(512) * 64 * P);
  GuardedDeviceBuffer gscale(static_cast<std::size_t>(64) * P *
                             sizeof(std::uint16_t));
  GuardedDeviceBuffer gwork(static_cast<std::size_t>(T) * 512 *
                            sizeof(std::uint16_t));

  QsaIndexerKAppendParams p;
  p.tokens          = T;
  p.hidden          = H;
  p.pos0            = pos0;
  p.hidden_dev      = static_cast<const std::uint16_t*>(dh.p);
  p.qk_proj_dev     = static_cast<const std::uint16_t*>(dw.p);
  p.k_norm_dev      = static_cast<const std::uint16_t*>(dkn.p);
  p.block_table_dev = static_cast<const std::int32_t*>(dbt.p);
  p.idx_k_pages_dev = static_cast<std::uint8_t*>(gpages.data());
  p.idx_k_scale_dev = static_cast<std::uint16_t*>(gscale.data());
  p.work_dev        = static_cast<std::uint16_t*>(gwork.data());

  qsa_indexer_k_append(p, nullptr);
  cuda_synchronize();
  failures += gpages.verify_guards(label.c_str());
  failures += gscale.verify_guards(label.c_str());
  failures += gwork.verify_guards(label.c_str());

  const std::vector<std::uint16_t> work =
      from_device<std::uint16_t>(gwork.data(), static_cast<std::size_t>(T) * 512);
  const std::vector<std::uint8_t> pages =
      from_device<std::uint8_t>(gpages.data(), static_cast<std::size_t>(512) * 64 * P);
  const std::vector<std::uint16_t> scale =
      from_device<std::uint16_t>(gscale.data(), static_cast<std::size_t>(64) * P);

  for (int t = 0; t < T; ++t) {
    const int pos  = pos0 + t;
    const int pp   = bt[pos >> 6];
    const int slot = pos & 63;
    const std::vector<double> ref = idxk_normed_row(hidden, H, w, knorm, t);

    std::vector<double> got_pre(512);
    for (int d = 0; d < 512; ++d) {
      got_pre[d] = static_cast<double>(
          bf16_to_f32(work[static_cast<std::size_t>(t) * 512 + d]));
    }
    failures += verify_pointwise(
        (label + " pre-quant t=" + std::to_string(t)).c_str(), got_pre, ref,
        PointwiseCriterion{3e-3, 5e-3});

    const double sc = fp16_to_f32_h(scale[static_cast<std::size_t>(pp) * 64 + slot]);
    std::vector<double> got_fp8(512);
    for (int d = 0; d < 512; ++d) {
      const std::int64_t off =
          static_cast<std::int64_t>(512) * 64 * pp + 512 * slot + d;
      got_fp8[d] = static_cast<double>(
          fp8_e4m3_to_f32(pages[static_cast<std::size_t>(off)]) * sc);
    }
    failures += verify_pointwise(
        (label + " fp8 t=" + std::to_string(t)).c_str(), got_fp8, ref,
        PointwiseCriterion{0.05, 0.15});
  }
}

// --- 7e oracle (FP64 over the exact FP32 logits) + case: global row-wise top-B

// top-B of the valid prefix [0, n_t) by (value DESC, position DESC); ascending,
// -1 padded to width B. Mirrors qsa_topk's (logit DESC, position DESC) contract.
std::vector<int> topk_row(const float* row_f32, int n_t, int B) {
  const int n = std::min(B, n_t);
  std::vector<std::pair<double, int>> cands;
  cands.reserve(static_cast<std::size_t>(n_t));
  for (int j = 0; j < n_t; ++j) {
    cands.emplace_back(static_cast<double>(row_f32[static_cast<std::size_t>(j)]), j);
  }
  std::sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) {
      return a.first > b.first;
    }
    return a.second > b.second;
  });
  std::vector<int> win;
  win.reserve(static_cast<std::size_t>(n));
  for (int r = 0; r < n; ++r) {
    win.push_back(cands[static_cast<std::size_t>(r)].second);
  }
  std::sort(win.begin(), win.end());
  std::vector<int> out(static_cast<std::size_t>(B), -1);
  for (int j = 0; j < n; ++j) {
    out[static_cast<std::size_t>(j)] = win[static_cast<std::size_t>(j)];
  }
  return out;
}

// Run qsa_topk once and verify ids + counts EXACTLY against the oracle. The op is
// a fixed-order radix selection over integer keys, so a bit-equal match is expected.
void run_7e_case(const std::string& label, int rows, int n_full, int B,
                 const std::vector<float>& logits, const std::vector<int>& ctx) {
  DeviceBuffer dlogits = to_device_f32(logits);
  DeviceBuffer dctx    = to_device_i32(ctx);
  GuardedDeviceBuffer gids(static_cast<std::size_t>(rows) * B * sizeof(std::int32_t));
  GuardedDeviceBuffer gcounts(static_cast<std::size_t>(rows) * sizeof(std::int32_t));

  QsaTopkParams p;
  p.rows            = rows;
  p.n_full          = n_full;
  p.budget          = B;
  p.logits_dev      = static_cast<const float*>(dlogits.p);
  p.context_len_dev = static_cast<const std::int32_t*>(dctx.p);
  p.ids_dev         = static_cast<std::int32_t*>(gids.data());
  p.counts_dev      = static_cast<std::int32_t*>(gcounts.data());

  qsa_topk(p, nullptr);
  cuda_synchronize();
  failures += gids.verify_guards(label.c_str());
  failures += gcounts.verify_guards(label.c_str());

  const std::vector<int> got_ids =
      from_device<int>(gids.data(), static_cast<std::size_t>(rows) * B);
  const std::vector<int> got_counts = from_device<int>(gcounts.data(), rows);

  std::vector<int> exp_ids, exp_counts;
  exp_ids.reserve(static_cast<std::size_t>(rows) * B);
  for (int r = 0; r < rows; ++r) {
    const int n_t   = ctx[static_cast<std::size_t>(r)];
    const float* rw = logits.data() + static_cast<std::size_t>(r) * n_full;
    const std::vector<int> row = topk_row(rw, n_t, B);
    exp_ids.insert(exp_ids.end(), row.begin(), row.end());
    exp_counts.push_back(std::min(B, n_t));
  }
  failures += verify_exact((label + " ids").c_str(), got_ids, exp_ids);
  failures += verify_exact((label + " counts").c_str(), got_counts, exp_counts);
}

// --- paged store (7c) --------------------------------------------------------

struct StoreBuilder {
  DeviceBuffer backing;
  DeviceKVPagePoolLayout pool_layout;
  KVExecutionTableLayout table_layout;
};

// QSA store geometry: 64-token pages, page-major [256, 64, KV, P] planes,
// K + V (two {BF16, 256, KV, 256} planes). PageMajor is the canonical
// paged_kv_element_offset layout: it is what the production paged append
// (detail::real_kv_append) and the GQA gather both address, and the host
// replica path treats a page as a contiguous first * nb[3] block in that
// order.
StoreBuilder plan_store(std::uint32_t physical_pages, std::uint32_t logical_pages,
                        int kv_heads) {
  KVPageGeometry geom;
  geom.device_plane_order = PagedKVPlaneOrder::PageMajor;
  geom.planes             = {KVPlaneGeometry{DType::BF16, 256, kv_heads, 256},
                   KVPlaneGeometry{DType::BF16, 256, kv_heads, 256}};
  LayoutBuilder builder;
  StoreBuilder out;
  out.pool_layout =
      plan_device_kv_page_pool(builder, DeviceKVPagePoolSpec{physical_pages, geom});
  out.table_layout = plan_kv_execution_tables(
      builder, KVExecutionTableSpec{logical_pages, 1});
  out.backing = DeviceBuffer(builder.finish(256));
  return out;
}

// k/v host layout [256, KV, T] row-major (d fastest within a token):
// linear = d + 256*(g + KV*p).
double kv_row(const std::vector<float>& h, int d, int g, int p, int KV) {
  return static_cast<double>(
      h[static_cast<std::size_t>(d) + 256 * static_cast<std::size_t>(g + KV * p)]);
}

// 7b FP64 oracle: out[t,h,d] = softmax_j(scale * q . K[p_j, g]) . V[p_j, g, d]
std::vector<double> gqa_oracle(const std::vector<float>& q, int T, int Q, int KV,
                               const std::vector<int>& ids, int B,
                               const std::vector<int>& counts,
                               const std::vector<float>& k,
                               const std::vector<float>& v, double scale) {
  const int ratio = Q / KV;
  std::vector<double> out(static_cast<std::size_t>(T) * Q * 256);
  for (int t = 0; t < T; ++t) {
    const int n = counts[static_cast<std::size_t>(t)];
    for (int h = 0; h < Q; ++h) {
      const int g = h / ratio;
      const float* qrow = q.data() + (static_cast<std::size_t>(t) * Q + h) * 256;
      std::vector<double> s(static_cast<std::size_t>(n));
      double m = -INFINITY;
      for (int j = 0; j < n; ++j) {
        const int pos = ids[static_cast<std::size_t>(t) * B + j];
        double acc    = 0.0;
        for (int d = 0; d < 256; ++d) {
          acc += static_cast<double>(qrow[d]) * kv_row(k, d, g, pos, KV);
        }
        s[static_cast<std::size_t>(j)] = scale * acc;
        m = std::max(m, s[static_cast<std::size_t>(j)]);
      }
      std::vector<double> u(s.size());
      double z = 0.0;
      for (int j = 0; j < n; ++j) {
        u[static_cast<std::size_t>(j)] = std::exp(s[static_cast<std::size_t>(j)] - m);
        z += u[static_cast<std::size_t>(j)];
      }
      double* orow = out.data() + (static_cast<std::size_t>(t) * Q + h) * 256;
      for (int d = 0; d < 256; ++d) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j) {
          acc += u[static_cast<std::size_t>(j)] *
                 kv_row(v, d, g, ids[static_cast<std::size_t>(t) * B + j], KV);
        }
        orow[d] = acc / z;
      }
    }
  }
  return out;
}

// 7b launch; returns the raw bf16 bits of the output (the exact demote/restore
// and decode checks compare bits) — values are derived via bf16_values.
std::vector<std::uint16_t> run_7b(const std::string& label, int T, int Q, int KV,
                                  int B, const std::vector<float>& q,
                                  const std::vector<int>& ids,
                                  const std::vector<int>& counts, const Tensor& bt,
                                  const Tensor& kpg, const Tensor& vpg) {
  DeviceBuffer dq     = to_device_bf16(q);
  DeviceBuffer dids   = to_device_i32(ids);
  DeviceBuffer dcounts = to_device_i32(counts);
  const std::size_t n = static_cast<std::size_t>(T) * Q * 256;
  GuardedDeviceBuffer gout(n * sizeof(std::uint16_t));

  QsaSparseGqaParams p;
  p.tokens          = T;
  p.q_heads         = Q;
  p.kv_heads        = KV;
  p.budget          = B;
  p.scale           = 0.0625f;
  p.q_dev           = static_cast<const std::uint16_t*>(dq.p);
  p.ids_dev         = static_cast<const std::int32_t*>(dids.p);
  p.counts_dev      = static_cast<const std::int32_t*>(dcounts.p);
  p.block_table_dev = static_cast<const std::int32_t*>(bt.data);
  p.k_pages_dev     = static_cast<const std::uint16_t*>(kpg.data);
  p.v_pages_dev     = static_cast<const std::uint16_t*>(vpg.data);
  p.out_dev         = static_cast<std::uint16_t*>(gout.data());

  qsa_sparse_gqa(p, nullptr);
  cuda_synchronize();
  failures += gout.verify_guards(label.c_str());
  return from_device<std::uint16_t>(gout.data(), n);
}

std::vector<double> bf16_values(const std::vector<std::uint16_t>& b) {
  std::vector<double> o(b.size());
  for (std::size_t i = 0; i < b.size(); ++i) {
    o[i] = static_cast<double>(bf16_to_f32(b[i]));
  }
  return o;
}

void verdict(const std::string& label, const std::vector<double>& got,
             const std::vector<double>& ref) {
  failures += verify_reduction(label.c_str(), got, ref,
                               ReductionCriterion{4.1e-3, 5e-6, 5.5e-3});
}

// Paged QSA pool append: the production write semantics
// (detail::real_kv_append) — the QSA pool holds BF16 K AND V, which the
// shared kv_cache_append op cannot express (its BFloat16 mode requires V
// in FP16). Scatter k/v at positions 0..T-1 into the page-major
// [256, 64, KV, P] planes through the block table (host replica of the
// kernel's offset formula), then H2D the full planes.
void append_store(const std::vector<float>& k, const std::vector<float>& v, int KV,
                  int T, const Tensor& bt, const Tensor& kpg, const Tensor& vpg) {
  const std::int64_t P  = bt.numel();
  const std::int64_t D  = 256, SZ = 64;
  const std::int64_t plane = kpg.numel();
  const std::vector<int> bth = from_device<int>(bt.data, static_cast<std::size_t>(P));
  std::vector<std::uint16_t> kh(static_cast<std::size_t>(plane), 0),
      vh(static_cast<std::size_t>(plane), 0);
  for (int p = 0; p < T; ++p) {
    const int page = bth[static_cast<std::size_t>(p >> 6)];
    const int off  = p & 63;
    for (int g = 0; g < KV; ++g) {
      const std::int64_t base =
          D * SZ * (static_cast<std::int64_t>(g) + static_cast<std::int64_t>(KV) * page) +
          D * off;
      for (int d = 0; d < D; ++d) {
        const std::size_t src = static_cast<std::size_t>(d) + D * (g + KV * p);
        kh[static_cast<std::size_t>(base + d)] = f32_to_bf16(k[src]);
        vh[static_cast<std::size_t>(base + d)] = f32_to_bf16(v[src]);
      }
    }
  }
  cuda_check(cudaMemcpy(kpg.data, kh.data(), static_cast<std::size_t>(plane) * 2,
                        cudaMemcpyHostToDevice),
             "append_store k H2D");
  cuda_check(cudaMemcpy(vpg.data, vh.data(), static_cast<std::size_t>(plane) * 2,
                        cudaMemcpyHostToDevice),
             "append_store v H2D");
  cuda_synchronize();
}

// One QSA store, constructed IN PLACE (the pools are non-movable; there is
// no copy or move of this struct, only direct construction at the use site).
// Pages are published in order to logical 0..pages-1.
struct Store {
  StoreBuilder planned;
  DeviceSpan span;
  DeviceKVPagePool pool;
  KVExecutionTablePool tables;
  std::vector<DeviceKVPageLease> leases;
  KVExecutionRowLease row_lease;
  KVExecutionRowHandle row_handle;
  Tensor bt;
  Tensor k_pages;
  Tensor v_pages;

  Store(std::uint32_t pages, int kv_heads)
      : planned(plan_store(pages, pages, kv_heads)),
        span{planned.backing.p, planned.backing.bytes},
        pool{span, planned.pool_layout},
        tables{span, planned.table_layout, pool},
        row_lease(tables.acquire(0)) {
    std::optional<DeviceKVPageReservation> res = pool.reserve(pages);
    if (!res) {
      std::cerr << "store: reserve failed\n";
      std::abort();
    }
    // materialize validates target_page_count <= destination.capacity()
    // (paged_kv_cache.cpp), so the lease vector must be reserved first —
    // the reference usage pattern (tests/test_kv_cache.cpp:57-59).
    leases.reserve(pages);
    pool.materialize(*res, pages, leases);
    tables.publish(row_lease.handle(), 0, leases);
    row_handle = row_lease.handle();
    bt         = tables.row(row_handle);
    k_pages    = pool.plane(0);
    v_pages    = pool.plane(1);
  }
};

}  // namespace

int main_body();

int main() {
  try {
    return main_body();
  } catch (const std::exception& e) {
    std::cerr << "P7 EXCEPTION: " << e.what() << "\n";
    return 2;
  }
}

int main_body() {
  if (cuda_unavailable()) {
    std::cout << "SKIP: no usable CUDA device\n";
    return 77;
  }

  // --- 7a case 1: mini geometry (real selection, T=64 B=16, pos0=64) ---------
  {
    const int T = 64, n_full = 128, H = 256, B = 16, pos0 = n_full - T;
    std::vector<int> bt(2);
    fill_iota_i32(bt, 0);
    std::vector<float> hidden_all(static_cast<std::size_t>(n_full) * H);
    fill_uniform(hidden_all, 11, -8.f, 8.f);
    round_to_bf16(hidden_all);
    std::vector<float> w(static_cast<std::size_t>(kIdxWidth) * H);
    fill_uniform(w, 12, -0.5f, 0.5f);
    round_to_bf16(w);
    std::vector<float> qnorm(128), knorm(128);
    fill_uniform(qnorm, 13, 0.5f, 1.5f);
    round_to_bf16(qnorm);
    fill_uniform(knorm, 14, 0.5f, 1.5f);
    round_to_bf16(knorm);
    const Pool pool = build_pool(n_full, H, hidden_all, w, knorm, bt);
    const std::vector<float> hidden_round = slice_round(hidden_all, pos0, T, H);
    run_7a_case("7a mini", T, H, B, pos0, n_full, hidden_round, w, qnorm, pool, bt);
  }

  // --- 7a case 2: production shape T=256 H=2560, real selection (n_full > B) -
  {
    const int T = 256, n_full = 1024, H = 2560, B = 128, pos0 = n_full - T;
    std::vector<int> bt(16);
    fill_iota_i32(bt, 0);
    std::vector<float> hidden_all(static_cast<std::size_t>(n_full) * H);
    fill_uniform(hidden_all, 21, -8.f, 8.f);
    round_to_bf16(hidden_all);
    std::vector<float> w(static_cast<std::size_t>(kIdxWidth) * H);
    fill_uniform(w, 22, -0.5f, 0.5f);
    round_to_bf16(w);
    std::vector<float> qnorm(128), knorm(128);
    fill_uniform(qnorm, 23, 0.5f, 1.5f);
    round_to_bf16(qnorm);
    fill_uniform(knorm, 24, 0.5f, 1.5f);
    round_to_bf16(knorm);
    const Pool pool = build_pool(n_full, H, hidden_all, w, knorm, bt);
    const std::vector<float> hidden_round = slice_round(hidden_all, pos0, T, H);
    run_7a_case("7a prod", T, H, B, pos0, n_full, hidden_round, w, qnorm, pool, bt);
  }

  // --- 7a case 3: tie-break (every score exactly +0.0) ------------------------
  {
    const int T = 16, n_full = 16, H = 128, B = 8, pos0 = 0;
    std::vector<int> bt(1);
    fill_iota_i32(bt, 0);
    std::vector<float> hidden_all(static_cast<std::size_t>(n_full) * H, 0.f);
    std::vector<float> w(static_cast<std::size_t>(kIdxWidth) * H, 0.f);
    std::vector<float> qnorm(128, 0.f), knorm(128, 0.f);
    const Pool pool = build_pool(n_full, H, hidden_all, w, knorm, bt);
    const std::vector<float> hidden_round = slice_round(hidden_all, pos0, T, H);
    run_7a_case("7a tie", T, H, B, pos0, n_full, hidden_round, w, qnorm, pool, bt);
  }

  // --- 7f a: chunked run is BIT-EQUAL to the single-chunk run ---------------
  // n_full=4096 (64 pages, two k_append rounds) forces multi-chunk at
  // max_logits_bytes=(1<<23) (chunk=512 -> 4 chunks); the single-chunk run uses
  // (1<<29). Same inputs -> identical ids/counts (the chunking invariant).
  {
    const int T = 2048, n_full = 4096, H = 256, B = 64;
    std::vector<int> bt(64);
    fill_iota_i32(bt, 0);
    std::vector<float> hidden_all(static_cast<std::size_t>(n_full) * H);
    fill_uniform(hidden_all, 51, -8.f, 8.f);
    round_to_bf16(hidden_all);
    std::vector<float> w(static_cast<std::size_t>(kIdxWidth) * H);
    fill_uniform(w, 52, -0.5f, 0.5f);
    round_to_bf16(w);
    std::vector<float> qnorm(128), knorm(128);
    fill_uniform(qnorm, 53, 0.5f, 1.5f);
    round_to_bf16(qnorm);
    fill_uniform(knorm, 54, 0.5f, 1.5f);
    round_to_bf16(knorm);
    const F7a chunked =
        run_7f_case(T, H, B, n_full, hidden_all, w, qnorm, knorm, bt,
                    (1 << 23), false);
    const F7a single =
        run_7f_case(T, H, B, n_full, hidden_all, w, qnorm, knorm, bt, 0, true);
    failures += verify_exact("7f a chunked ids == single ids",
                             chunked.ids, single.ids);
    failures += verify_exact("7f a chunked counts == single counts",
                             chunked.counts, single.counts);
  }

  // --- 7f b: non-zero pos0 / global shift (ids stay within [0, pos0+t)) ------
  {
    const int T = 64, n_full = 256, H = 256, B = 32, pos0 = n_full - T;
    std::vector<int> bt{3, 1, 2, 0};  // logical 0..3 -> physical (permuted)
    std::vector<float> hidden_all(static_cast<std::size_t>(n_full) * H);
    fill_uniform(hidden_all, 61, -8.f, 8.f);
    round_to_bf16(hidden_all);
    std::vector<float> w(static_cast<std::size_t>(kIdxWidth) * H);
    fill_uniform(w, 62, -0.5f, 0.5f);
    round_to_bf16(w);
    std::vector<float> qnorm(128), knorm(128);
    fill_uniform(qnorm, 63, 0.5f, 1.5f);
    round_to_bf16(qnorm);
    fill_uniform(knorm, 64, 0.5f, 1.5f);
    round_to_bf16(knorm);
    const F7a r =
        run_7f_case(T, H, B, n_full, hidden_all, w, qnorm, knorm, bt, 0, true);
    int bad = 0;
    for (int t = 0; t < T; ++t) {
      const int n_t = pos0 + t;
      const int c = r.counts[static_cast<std::size_t>(t)];
      for (int j = 0; j < c; ++j) {
        const int id = r.ids[static_cast<std::size_t>(t) * B + j];
        if (id < 0 || id >= n_t) {
          ++bad;
        }
      }
    }
    failures += (bad ? 1 : 0);
  }

  // --- 7f c: GEMM logit values within FP8 tolerance (single chunk, full grid) -
  {
    const int T = 64, n_full = 256, H = 256, B = 32, pos0 = n_full - T;
    std::vector<int> bt(4);
    fill_iota_i32(bt, 0);
    std::vector<float> hidden_all(static_cast<std::size_t>(n_full) * H);
    fill_uniform(hidden_all, 71, -8.f, 8.f);
    round_to_bf16(hidden_all);
    std::vector<float> w(static_cast<std::size_t>(kIdxWidth) * H);
    fill_uniform(w, 72, -0.5f, 0.5f);
    round_to_bf16(w);
    std::vector<float> qnorm(128), knorm(128);
    fill_uniform(qnorm, 73, 0.5f, 1.5f);
    round_to_bf16(qnorm);
    fill_uniform(knorm, 74, 0.5f, 1.5f);
    round_to_bf16(knorm);
    const F7a r =
        run_7f_case(T, H, B, n_full, hidden_all, w, qnorm, knorm, bt, 0, true);
    const Pool pool{r.pages, r.scale};  // the exact pool the kernel scored
    const std::vector<float> hidden_round = slice_round(hidden_all, pos0, T, H);
    std::vector<int> ctx(T);
    for (int t = 0; t < T; ++t) {
      ctx[static_cast<std::size_t>(t)] = pos0 + t;
    }
    const std::vector<double> ref =
        ref7a_causal(T, H, hidden_round, w, qnorm, pool, bt, n_full, ctx);
    std::vector<double> got_causal;
    got_causal.reserve(ref.size());
    for (int t = 0; t < T; ++t) {
      for (int j = 0; j < ctx[static_cast<std::size_t>(t)]; ++j) {
        got_causal.push_back(
            r.logits[static_cast<std::size_t>(t) * n_full + j]);
      }
    }
    failures += verify_pointwise("7f c logit (causal, FP8)", got_causal, ref,
                                 PointwiseCriterion{5e-2, 5e-3});
  }

  // --- 7f d: decode T=1 (ids == host top-B of the read-back logits) ----------
  {
    const int T = 1, n_full = 128, H = 256, B = 16, pos0 = n_full - T;
    std::vector<int> bt(2);
    fill_iota_i32(bt, 0);
    std::vector<float> hidden_all(static_cast<std::size_t>(n_full) * H);
    fill_uniform(hidden_all, 81, -8.f, 8.f);
    round_to_bf16(hidden_all);
    std::vector<float> w(static_cast<std::size_t>(kIdxWidth) * H);
    fill_uniform(w, 82, -0.5f, 0.5f);
    round_to_bf16(w);
    std::vector<float> qnorm(128), knorm(128);
    fill_uniform(qnorm, 83, 0.5f, 1.5f);
    round_to_bf16(qnorm);
    fill_uniform(knorm, 84, 0.5f, 1.5f);
    round_to_bf16(knorm);
    const F7a r =
        run_7f_case(T, H, B, n_full, hidden_all, w, qnorm, knorm, bt, 0, true);
    const int n_t = pos0 + 0;
    const std::vector<int> exp_row =
        topk_row(r.logits.data() + 0, n_t, B);
    failures += verify_exact("7f d decode ids", r.ids, exp_row);
    failures += verify_exact("7f d decode counts", r.counts,
                             std::vector<int>{std::min(B, n_t)});
  }

  // --- 7f e: dense-skip (n_t <= B -> ids = 0..n_t-1, counts = n_t) -----------
  {
    const int T = 8, n_full = 8, H = 256, B = 64, pos0 = 0;  // n_t = t < 8 <= B
    std::vector<int> bt(1);
    fill_iota_i32(bt, 0);
    std::vector<float> hidden_all(static_cast<std::size_t>(n_full) * H);
    fill_uniform(hidden_all, 91, -8.f, 8.f);
    round_to_bf16(hidden_all);
    std::vector<float> w(static_cast<std::size_t>(kIdxWidth) * H);
    fill_uniform(w, 92, -0.5f, 0.5f);
    round_to_bf16(w);
    std::vector<float> qnorm(128), knorm(128);
    fill_uniform(qnorm, 93, 0.5f, 1.5f);
    round_to_bf16(qnorm);
    fill_uniform(knorm, 94, 0.5f, 1.5f);
    round_to_bf16(knorm);
    const F7a r =
        run_7f_case(T, H, B, n_full, hidden_all, w, qnorm, knorm, bt, 0, true);
    std::vector<int> exp_ids;
    std::vector<int> exp_counts;
    exp_ids.reserve(static_cast<std::size_t>(T) * B);
    for (int t = 0; t < T; ++t) {
      const int n_t = pos0 + t;
      for (int j = 0; j < B; ++j) {
        exp_ids.push_back(j < n_t ? j : -1);
      }
      exp_counts.push_back(n_t);
    }
    failures += verify_exact("7f e dense-skip ids", r.ids, exp_ids);
    failures += verify_exact("7f e dense-skip counts", r.counts, exp_counts);
  }

  // --- 7f f: T=2048 no out-of-range abort, valid ids (single chunk) ----------
  {
    const int T = 2048, n_full = 2048, H = 256, B = 64, pos0 = 0;
    std::vector<int> bt(32);
    fill_iota_i32(bt, 0);
    std::vector<float> hidden_all(static_cast<std::size_t>(n_full) * H);
    fill_uniform(hidden_all, 101, -8.f, 8.f);
    round_to_bf16(hidden_all);
    std::vector<float> w(static_cast<std::size_t>(kIdxWidth) * H);
    fill_uniform(w, 102, -0.5f, 0.5f);
    round_to_bf16(w);
    std::vector<float> qnorm(128), knorm(128);
    fill_uniform(qnorm, 103, 0.5f, 1.5f);
    round_to_bf16(qnorm);
    fill_uniform(knorm, 104, 0.5f, 1.5f);
    round_to_bf16(knorm);
    const F7a r =
        run_7f_case(T, H, B, n_full, hidden_all, w, qnorm, knorm, bt, 0, true);
    int bad = 0;
    for (int t = 0; t < T; ++t) {
      const int n_t = pos0 + t;
      const int c = r.counts[static_cast<std::size_t>(t)];
      if (c != std::min(B, n_t)) {
        ++bad;
      }
      for (int j = 0; j < c; ++j) {
        const int id = r.ids[static_cast<std::size_t>(t) * B + j];
        if (id < 0 || id >= n_t) {
          ++bad;
        }
      }
    }
    failures += (bad ? 1 : 0);
  }

  // --- 7d case 1: mini H=256, permuted block table, pos0=32 ------------------
  // Tokens 32..131 span logical pages 0..2; physical pages are permuted so the
  // paged addressing is exercised (not the identity layout).
  {
    const int T = 100, H = 256, pos0 = 32;
    std::vector<int> bt{5, 2, 7, 0, 1, 3, 4, 6};  // logical 0..7 -> physical
    const int P = 8;
    std::vector<float> hidden(static_cast<std::size_t>(T) * H);
    fill_uniform(hidden, 21, -8.f, 8.f);
    round_to_bf16(hidden);
    std::vector<float> w(static_cast<std::size_t>(kIdxWidth) * H);
    fill_uniform(w, 22, -0.5f, 0.5f);
    round_to_bf16(w);
    std::vector<float> knorm(128);
    fill_uniform(knorm, 23, 0.5f, 1.5f);
    round_to_bf16(knorm);
    run_7d_case("7d mini", T, H, pos0, hidden, w, knorm, bt, P);
  }

  // --- 7d case 2: production H=2560, T=128, pos0=64 --------------------------
  // Spans logical pages 1..2; identity block table (the production default).
  {
    const int T = 128, H = 2560, pos0 = 64;
    std::vector<int> bt(4);
    fill_iota_i32(bt, 0);
    const int P = 4;
    std::vector<float> hidden(static_cast<std::size_t>(T) * H);
    fill_uniform(hidden, 41, -8.f, 8.f);
    round_to_bf16(hidden);
    std::vector<float> w(static_cast<std::size_t>(kIdxWidth) * H);
    fill_uniform(w, 42, -0.5f, 0.5f);
    round_to_bf16(w);
    std::vector<float> knorm(128);
    fill_uniform(knorm, 43, 0.5f, 1.5f);
    round_to_bf16(knorm);
    run_7d_case("7d prod", T, H, pos0, hidden, w, knorm, bt, P);
  }

  // --- 7c + 7b mini: paged write, sparse read, one-page demote/restore -------
  {
    const int T = 64, H = 256, B = 16, KV = 2, Q = 8;

    // Ready ids from the 7a host oracle (same seeds as case 1; the device
    // 7a output was verified against this oracle there).
    std::vector<float> hidden(static_cast<std::size_t>(T) * H);
    fill_uniform(hidden, 11, -8.f, 8.f);
    round_to_bf16(hidden);
    std::vector<float> w(static_cast<std::size_t>(kIdxWidth) * H);
    fill_uniform(w, 12, -0.5f, 0.5f);
    round_to_bf16(w);
    std::vector<float> qnorm(128), knorm(128);
    fill_uniform(qnorm, 13, 0.5f, 1.5f);
    round_to_bf16(qnorm);
    fill_uniform(knorm, 14, 0.5f, 1.5f);
    round_to_bf16(knorm);
    const std::vector<double> nt    = index_normed_z(hidden, T, H, w, qnorm, knorm);
    const std::vector<double> ref_s = index_scores(nt, T);
    std::vector<int> ids, counts;
    ids.reserve(static_cast<std::size_t>(T) * B);
    for (int t = 0; t < T; ++t) {
      const std::vector<int> row = index_topk_row(ref_s, T, t, B);
      ids.insert(ids.end(), row.begin(), row.end());
      counts.push_back(std::min(B, t + 1));
    }

    std::vector<float> k(static_cast<std::size_t>(256) * KV * T);
    fill_uniform(k, 31, -4.f, 4.f);
    round_to_bf16(k);
    std::vector<float> v(static_cast<std::size_t>(256) * KV * T);
    fill_uniform(v, 32, -4.f, 4.f);
    round_to_bf16(v);
    std::vector<float> q(static_cast<std::size_t>(T) * Q * 256);
    fill_uniform(q, 33, -4.f, 4.f);
    round_to_bf16(q);

    // Store: one 64-token page.
    Store store(1, KV);
    append_store(k, v, KV, T, store.bt, store.k_pages, store.v_pages);

    // 7b sparse read over the paged store.
    const std::vector<std::uint16_t> bits1 =
        run_7b("7b mini", T, Q, KV, B, q, ids, counts, store.bt, store.k_pages,
               store.v_pages);
    verdict("7b mini sparse", bf16_values(bits1),
            gqa_oracle(q, T, Q, KV, ids, B, counts, k, v, kAttnScale));

    // 7c demote/restore of the single page; the re-read must be bit-identical.
    {
      const HostKVPageLayout hlayout =
          plan_host_kv_page_layout(store.planned.pool_layout.spec.geometry);
      HostKVArena arena(1024 * 1024, std::span(&hlayout, 1));
      std::optional<HostKVAllocation> alloc = arena.allocate(hlayout, 1);
      if (!alloc) {
        std::cerr << "7c: host arena allocation failed\n";
        return 1;
      }
      const HostKVAllocationView wview = arena.writable_view(*alloc);
      const DeviceKVPageHandle page    = store.leases[0].handle();
      store.pool.copy_to_host(std::span(&page, 1), wview);
      store.pool.zero_pages(std::span(&page, 1));
      store.pool.copy_from_host(arena.view(*alloc), std::span(&page, 1));
      cuda_synchronize();

      const std::vector<std::uint16_t> bits2 =
          run_7b("7b mini (restored)", T, Q, KV, B, q, ids, counts, store.bt,
                 store.k_pages, store.v_pages);
      failures += verify_exact("7c demote/restore (bit-identical re-read)", bits1,
                               bits2);
    }
  }

  // --- 7b production dense-equivalence: T=2048, ids = 0..t (budget 2048) -----
  {
    const int T = 2048, Q = 48, KV = 2, B = 2048;
    std::vector<float> k(static_cast<std::size_t>(256) * KV * T);
    fill_uniform(k, 41, -4.f, 4.f);
    round_to_bf16(k);
    std::vector<float> v(static_cast<std::size_t>(256) * KV * T);
    fill_uniform(v, 42, -4.f, 4.f);
    round_to_bf16(v);
    std::vector<float> q(static_cast<std::size_t>(T) * Q * 256);
    fill_uniform(q, 43, -4.f, 4.f);
    round_to_bf16(q);

    std::vector<int> ids(static_cast<std::size_t>(T) * B, -1), counts(T);
    for (int t = 0; t < T; ++t) {
      for (int j = 0; j <= t; ++j) {
        ids[static_cast<std::size_t>(t) * B + j] = j;
      }
      counts[static_cast<std::size_t>(t)] = t + 1;
    }

    // Store: 32 pages of 64 tokens (T=2048 / 64).
    Store store(32, KV);
    append_store(k, v, KV, T, store.bt, store.k_pages, store.v_pages);

    const std::vector<std::uint16_t> bits =
        run_7b("7b prod", T, Q, KV, B, q, ids, counts, store.bt, store.k_pages,
               store.v_pages);
    verdict("7b prod dense-equivalence", bf16_values(bits),
            gqa_oracle(q, T, Q, KV, ids, B, counts, k, v, kAttnScale));
  }

  // --- 7b decode T=1 exact: q = 0 forces out = the single selected V row -----
  {
    const int Tstore = 64, KV = 2, Q = 8, pos = 41;
    std::vector<float> k(static_cast<std::size_t>(256) * KV * Tstore);
    fill_uniform(k, 31, -4.f, 4.f);
    round_to_bf16(k);
    std::vector<float> v(static_cast<std::size_t>(256) * KV * Tstore);
    fill_uniform(v, 32, -4.f, 4.f);
    round_to_bf16(v);

    Store store(1, KV);
    append_store(k, v, KV, Tstore, store.bt, store.k_pages, store.v_pages);

    std::vector<float> q(256 * Q, 0.f);  // T=1, Q heads
    std::vector<int> ids{pos}, counts{1};
    const std::vector<std::uint16_t> bits =
        run_7b("7b decode T=1", 1, Q, KV, 1, q, ids, counts, store.bt,
               store.k_pages, store.v_pages);

    // u = exp(0) = 1, z = 1  =>  out[h, d] = V[pos, g, d] bit-exact.
    const int ratio = Q / KV;
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(Q) * 256);
    for (int h = 0; h < Q; ++h) {
      const int g = h / ratio;
      for (int d = 0; d < 256; ++d) {
        expected[static_cast<std::size_t>(h) * 256 + d] =
            f32_to_bf16(v[d + 256 * (g + KV * pos)]);
      }
    }
    failures += verify_exact("7b decode T=1 exact (single V row)", bits, expected);
  }

  // --- 7e case 1: varied context lengths incl. dense-skip (n_t <= B) ---------
  {
    const int rows = 8, n_full = 2048, B = 64;
    std::vector<float> logits(static_cast<std::size_t>(rows) * n_full);
    fill_uniform(logits, 61, -4.f, 4.f);
    std::vector<int> ctx = {1000, 512, 63, 64, 65, 2048, 3, 200};
    run_7e_case("7e varied", rows, n_full, B, logits, ctx);
  }

  // --- 7e case 2: all-ties row (validates the position-DESC tie-break) -------
  {
    const int rows = 4, n_full = 2048, B = 64;
    std::vector<float> logits(static_cast<std::size_t>(rows) * n_full, 0.f);
    std::vector<int> ctx = {200, 64, 5, 1024};
    run_7e_case("7e ties", rows, n_full, B, logits, ctx);
  }

  // --- 7e case 3: masked high-score column at j >= n_t (proves the mask) -----
  {
    const int rows = 3, n_full = 2048, B = 64;
    std::vector<float> logits(static_cast<std::size_t>(rows) * n_full);
    fill_uniform(logits, 81, -4.f, 4.f);
    std::vector<int> ctx = {128, 500, 2048};
    // plant huge values OUTSIDE the valid prefix; the op must never select them
    logits[0 * n_full + 128]  = 1e30f;
    logits[0 * n_full + 2000] = 1e30f;
    logits[1 * n_full + 500]  = 1e30f;
    run_7e_case("7e mask", rows, n_full, B, logits, ctx);
  }

  std::cout << (failures == 0 ? "OK" : "FAIL") << " flash next P7 qsa\n";
  return failures == 0 ? 0 : 1;
}
