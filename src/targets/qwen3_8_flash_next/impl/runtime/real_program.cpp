// Flash-Next S3 — the real-geometry standalone Program implementation.
//
// Mirrors program.cpp's (P9 mini) block structure and copy idioms at REAL
// geometry (detail::Geometry), composing the oracle-qualified real ops (P6 GDN /
// P7 QSA / P8 MoE) with the 13 S2 real glue leaves. Runs a whole sequence in ONE
// round at pos0 = 0; S3 drives it at T = 1. Every buffer is bit-deterministic
// (the ops and leaves are), so a fresh instance is reproducible.
//
// T = 1 layout note: mini_fp8_linear's kernel writes out[n*T + t] (channel-major
// [N][T], token fastest). The GDN/QSA/MoE glue consumes GEMM output as
// token-major [T][N]. The two coincide ONLY at T = 1 (a single row); this driver
// runs T = 1, so mirroring the mini exactly is correct. The T > 1 path is a
// known S5 concern (the z slice + GEMM layout), not exercised here.

#include "targets/qwen3_8_flash_next/impl/runtime/real_program.h"

#include "targets/qwen3_8_flash_next/impl/config.h"
#include "targets/qwen3_8_flash_next/impl/runtime/leaves/mini_leaves.h"
#include "targets/qwen3_8_flash_next/impl/runtime/leaves/real_leaves.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/ngram_embedding.h"
#include "ninfer/ops/qsa_indexer.h"
#include "ninfer/ops/qsa_sparse_gqa.h"
#include "ninfer/ops/sparse_moe_nvfp4.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {
namespace {

using G = detail::Geometry;

// 1/sqrt(128) — the GDN attention scale (the shared op hard-checks
// scale == 1/sqrt(state_dim = 128)). Literal because MSVC /std:c++20 rejects
// constexpr std::sqrt.
constexpr float kGdnScale = 0.08838834764831843F;
// QSA attention scale 1/sqrt(head_dim = 256) = 1/16 (the op takes it as a param).
constexpr float kQsaScale = 0.0625F;

// The QSA indexer/sparse-GQA budget (ids width B). Immaterial at T = 1 (the
// indexer draws n = min(B, t + 1) = 1) but must be >= 1 and fit the ids buffer.
constexpr int kQsaBudget = 64;
// Max tokens in one round. The MoE op's T domain caps a round at 2048 (see
// sparse_moe_nvfp4's T <= 2048 check); the GDN op accepts any T (chunked path
// at T >= 64, recurrent at T = 1) and the QSA ops any T. S3 runs 1; P13 M3
// prefill runs <= 2048-token state-carrying chunks and 1-token decode rounds.
// The arena is sized from arena_capacity_bytes (a scoped per-layer PEAK, not a
// cumulative sum) so T = 2048 fits in a few GB, not the ~60 GB the cumulative
// sizing would need.
constexpr int kMaxRoundTokens = 2048;
static_assert(kMaxRoundTokens == RealProgram::kMaxRoundTokens,
              "real round cap: keep the file-local and the class member in sync");

// GDN conv channels: the conv projection is [10240, T] (qkvz is 16384 = 10240
// qkv + 6144 z; z bypasses the conv). 10240 = 16384 - 6144.
constexpr int kGdnConvChannels = 10240;
// GDN fused a/b projection rows: 48 a-rows [0,48) then 48 b-rows [48,96).
constexpr int kGdnGatingRows = 48;
// GDN qkvz row width (channels): 16384 (q/k/v 10240 + z 6144).
constexpr int kGdnQkvzRow = 16384;
// GDN z slice channels (Hv * D = 48 * 128).
constexpr int kGdnZChannels = 6144;
// GDN state dim (the op's state [128, 128, value_heads]).
constexpr int kGdnStateDim = 128;

// ---------------------------------------------------------------------------
// Per-stage NaN probe (NINFER_NAN_PROBE=1; a no-op when unset). Localizes the
// first non-finite value in the real forward: D2Hs a stage buffer, reports the
// first non-finite element index + max |finite| to stderr (a "NAN! " prefix
// marks the first bad stage). Bounded to small T (the diagnostic runs at
// T<=16) to keep the D2H tiny. Diagnostic-only; the stream sync it adds is
// benign because every op here is bit-deterministic.
struct NanProbeScratch {
  std::vector<std::uint16_t> bf;
  std::vector<std::uint32_t> i32;
};
NanProbeScratch& nan_probe_scratch() {
  static NanProbeScratch sp;
  return sp;
}
bool nan_probe_enabled() {
  static const bool on = std::getenv("NINFER_NAN_PROBE") != nullptr;
  return on;
}
void nan_probe_report(const char* tag, std::size_t n, std::size_t first_bad, double fmx) {
  std::fprintf(stderr, "%s[nanprobe] %s n=%zu first_bad=%s max_finite=%.4g\n",
               first_bad == n ? "" : "NAN! ", tag, n,
               first_bad == n ? "none" : std::to_string(first_bad).c_str(), fmx);
  std::fflush(stderr);
}
void nan_probe_bf16(const char* tag, const __nv_bfloat16* dev, std::size_t n,
                    const cudaStream_t s) {
  if (!nan_probe_enabled() || n > (std::size_t)128 * 48 * 256) return;
  auto& sp = nan_probe_scratch();
  sp.bf.resize(n);
  if (cudaMemcpyAsync(sp.bf.data(), dev, n * 2, cudaMemcpyDeviceToHost, s) !=
          cudaSuccess ||
      cudaStreamSynchronize(s) != cudaSuccess) {
    std::fprintf(stderr, "[nanprobe] %s D2H failed\n", tag);
    return;
  }
  std::size_t first = n;
  double fmx = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const float v = __bfloat162float(sp.bf[i]);
    if (first == n && !std::isfinite(v)) first = i;
    if (std::isfinite(v) && std::fabs(v) > fmx) fmx = std::fabs(v);
  }
  nan_probe_report(tag, n, first, fmx);
}
void nan_probe_f32(const char* tag, const float* dev, std::size_t n,
                   const cudaStream_t s) {
  if (!nan_probe_enabled() || n > (std::size_t)128 * 64 * 2560) return;
  auto& sp = nan_probe_scratch();
  sp.i32.resize(n);
  if (cudaMemcpyAsync(sp.i32.data(), dev, n * 4, cudaMemcpyDeviceToHost, s) !=
          cudaSuccess ||
      cudaStreamSynchronize(s) != cudaSuccess) {
    std::fprintf(stderr, "[nanprobe] %s D2H failed\n", tag);
    return;
  }
  const float* f = reinterpret_cast<const float*>(sp.i32.data());
  std::size_t first = n;
  double fmx = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (first == n && !std::isfinite(f[i])) first = i;
    if (std::isfinite(f[i]) && std::fabs(f[i]) > fmx) fmx = std::fabs(f[i]);
  }
  nan_probe_report(tag, n, first, fmx);
}
void nan_probe_counts(const char* tag, const std::int32_t* dev, std::size_t n,
                      const cudaStream_t s) {
  if (!nan_probe_enabled() || n > (std::size_t)4096) return;
  auto& sp = nan_probe_scratch();
  sp.i32.resize(n);
  if (cudaMemcpyAsync(sp.i32.data(), dev, n * 4, cudaMemcpyDeviceToHost, s) !=
          cudaSuccess ||
      cudaStreamSynchronize(s) != cudaSuccess) {
    std::fprintf(stderr, "[nanprobe] %s D2H failed\n", tag);
    return;
  }
  const std::int32_t* c = reinterpret_cast<const std::int32_t*>(sp.i32.data());
  std::string vals;
  for (std::size_t i = 0; i < n; ++i) {
    vals += std::to_string(c[i]);
    if (i + 1 < n) vals += ",";
  }
  std::fprintf(stderr, "[nanprobe] %s = %s\n", tag, vals.c_str());
  std::fflush(stderr);
}

// Crash-localizing heartbeat (P13 CONT debug; NINFER_P13_HB=<file>; no-op unset).
// Appends a pre-formatted line + flushes so a 0xC0000409 mid-round leaves the
// last-reached phase/layer on disk. Pure fprintf (no D2H, no new allocation) so
// it cannot itself crash. Defined before the probe so the probe's D2H/sync error
// log (below) and run_round's phase markers can both call it.
void cont_hb(const char* line) {
  const char* path = std::getenv("NINFER_P13_HB");
  if (path == nullptr || path[0] == '\0') return;
  static FILE* f = nullptr;
  static std::string last;
  if (f == nullptr || last != path) {
    if (f) std::fclose(f);
    f = std::fopen(path, "a");
    last.assign(path);
  }
  if (!f) return;
  std::fputs(line, f);
  std::fputc('\n', f);
  std::fflush(f);
}

// ---------------------------------------------------------------------------
// Per-layer value-dump probe (NINFER_P13_LAYPROBE=<file>; a no-op when unset).
// P13 CONT root-cause: dumps the attention output `y` (after QSA/GDN) and the
// MoE output `y` at the FINAL position (T-1) after each layer, so the driver can
// compare the fresh vs split per-layer outputs and find the first diverging
// layer. D2Hs only the final position (kHidden bf16 = 5120 B) so it stays tiny.
// The driver sets NINFER_P13_LAYPROBE to a distinct file per path (fresh/split)
// via _putenv_s between the two run_prefill_and_decode calls; the file is
// reopened "w" when the path string changes (so each path truncates its own).
struct LayerProbe {
  std::string last;
  FILE* f = nullptr;
  std::vector<std::uint16_t> scratch;
};
LayerProbe& layer_probe() {
  static LayerProbe p;
  return p;
}
void layer_probe_dump_row(const char* tag, int pos0, int local_idx,
                          const __nv_bfloat16* dev, std::size_t hidden,
                          const cudaStream_t s) {
  const char* path = std::getenv("NINFER_P13_LAYPROBE");
  if (path == nullptr || path[0] == '\0') return;
  if (local_idx < 0) return;
  auto& p = layer_probe();
  if (p.last != path) {
    if (p.f) { std::fclose(p.f); p.f = nullptr; }
    p.f = std::fopen(path, "w");
    p.last.assign(path);
  }
  if (!p.f) return;
  p.scratch.resize(hidden);
  const __nv_bfloat16* row = dev + static_cast<std::size_t>(local_idx) * hidden;
  const cudaError_t d2h =
      cudaMemcpyAsync(p.scratch.data(), row, hidden * 2, cudaMemcpyDeviceToHost, s);
  const cudaError_t syn = cudaStreamSynchronize(s);
  {
    char hb[64];
    std::snprintf(hb, sizeof(hb), "PROBE %s g=%d d2h=%d sync=%d", tag, pos0 + local_idx,
                  static_cast<int>(d2h), static_cast<int>(syn));
    cont_hb(hb);
  }
  if (d2h != cudaSuccess || syn != cudaSuccess) {
    return;
  }
  std::fprintf(p.f, "%s g%d ", tag, pos0 + local_idx);
  for (std::size_t i = 0; i < hidden; ++i) {
    std::fprintf(p.f, "%u ", p.scratch[i]);
  }
  std::fprintf(p.f, "\n");
  std::fflush(p.f);
}
void layer_probe_dump(const char* tag, int pos0, const __nv_bfloat16* dev, int T,
                      std::size_t hidden, const cudaStream_t s) {
  if (T < 2) return;  // skip single-token decode rounds (prefill rounds are T>=256)
  // Primary: the round's final position (local T-1).
  layer_probe_dump_row(tag, pos0, T - 1, dev, hidden, s);
  // Optional extra local index (NINFER_P13_LAYPROBE_EXTRA_LOCAL): lets the driver
  // also dump a mid-round position (e.g. local 255 of a fresh T=512 round) so the
  // split boundary (global 255) can be compared in the same run.
  const char* e = std::getenv("NINFER_P13_LAYPROBE_EXTRA_LOCAL");
  if (e != nullptr) {
    const int eidx = std::atoi(e);
    if (eidx >= 0 && eidx < T) layer_probe_dump_row(tag, pos0, eidx, dev, hidden, s);
  }
}

// Routed-expert window geometry (the P8 512 x top-10 x 640 NVFP4 contract).
constexpr ops::SparseMoeNvfp4Geometry kMoeGeometry{
    G::kNumExperts, G::kExpertsPerToken, G::kHidden, G::kMoeIntermediate};
constexpr std::size_t kExpertBytes =
    1638400u + 204800u + 819200u + 102400u;  // = 2,764,800 per expert
constexpr std::size_t kMoeWindowBytes =
    static_cast<std::size_t>(G::kNumExperts) * kExpertBytes;  // 512 * 2,764,800
// Plane offsets within one expert's kExpertBytes window.
constexpr std::size_t kPlaneGuCodes = 0;
constexpr std::size_t kPlaneGuScales = 1638400u;
constexpr std::size_t kPlaneDnCodes = 1638400u + 204800u;  // 1,843,200
constexpr std::size_t kPlaneDnScales = 1638400u + 204800u + 819200u;  // 2,662,400
// Per-expert source stride within each fused host plane (= the plane size).
constexpr std::size_t kSrcGuCodes = 1638400u;
constexpr std::size_t kSrcGuScales = 204800u;
constexpr std::size_t kSrcDnCodes = 819200u;
constexpr std::size_t kSrcDnScales = 102400u;

// QSA paged KV: one 65,536 B page (page-major [256, 64, KV = 2] BF16); a page
// holds kQsaTokensPerPage = 64 tokens. Each K/V pool holds ceil(max_context / 64)
// pages (sized in the ctor from the max_context the RealProgram was constructed
// with; the P13 default of 8320 -> 130 pages) and the block table is the IDENTITY
// map (logical page i = physical page i) of the same width. A longer context (e.g.
// 100000 -> 1563 pages) only grows the pool; the round-local QSA prefill
// (<=1024-token rounds) and the context-independent GDN recurrent state are
// unchanged.
constexpr std::size_t kQsaKvPageBytes = 256u * 64u * 2u * 1u * sizeof(std::uint16_t);
constexpr std::size_t kQsaTokensPerPage = 64;

// QSA indexer-K pool (long-context block-sparse attention): a per-layer paged
// cache of the normalized 512-dim indexer key per token, SEPARATE from the main
// K/V pool but on the SAME identity block table (pages stay aligned). FP8 E4M3
// (1 B/elem) page = 512 * 64 B; one FP16 scale per 64-token page. The indexer
// logit GEMM (qsa_indexer) chunks the M dim so its [chunk, n_full] FP32 scratch
// stays within kIndexerLogitsBytes (a persistent scratch, not the round arena).
constexpr std::size_t kIndexerKvDim = 512;
constexpr std::size_t kIndexerKvPageBytes = 512u * 64u;        // 32,768 B (FP8)
constexpr std::size_t kIndexerKvScalePageBytes = 64u * 2u;     // 128 B (half/slot)
constexpr std::size_t kIndexerLogitsBytes = 1ULL << 28;        // 256 MB scratch

// GDN/QSA layer ordinals within the 48 text layers (l % 4 == 3 is QSA).
[[nodiscard]] constexpr int gdn_ordinal(std::uint32_t layer) {
  return static_cast<int>(layer) - static_cast<int>(layer / 4);
}
[[nodiscard]] constexpr int qsa_ordinal(std::uint32_t layer) {
  return static_cast<int>((layer - 3) / 4);
}

// Round scratch arena capacity for T tokens. Per-layer transients are freed at
// each block's scope end (gdn/qsa/moe each wrap their body in a scope), so the
// arena peak is NOT the cumulative sum across 48 layers — it is the round-
// shared buffers (held across all layers) plus the LARGEST single block's live
// transient (glue + the op's own workspace), and separately the final-collapse
// / lm_head buffers (allocated after the layer loop, when the block transients
// are freed). The op-workspace terms are the real capacity queries; the ops
// reserve at the round max (see gdn_block / moe_block) so they are
// kMaxRoundTokens-sized and constant across rounds.
std::size_t arena_capacity_bytes(int T) {
  using S = std::size_t;
  const int H = G::kHidden;
  // Round-shared (allocated once per round, held across the 48 layers).
  S shared = S(T) * 4 + S(T) * 2 * H + 2 * S(T) * 4 * G::kHcCount * H + S(T) * 4 * H
           + S(T) * 2 * H + S(T) * 2 * H + S(T) * 4 * H + S(T) * 4 * H + S(T) * 4 * H;
  const S gdn_op = ops::gated_delta_net_workspace_capacity_bytes(
      static_cast<int>(G::kGdnQkHeads), static_cast<int>(G::kGdnValueHeads), true, 1,
      kMaxRoundTokens);
  const S moe_op =
      ops::sparse_moe_nvfp4_workspace_capacity_bytes(kMoeGeometry, 1, kMaxRoundTokens);
  // Per-layer block peaks (only one is live at a time).
  const S gdn = S(T) * 2 * kGdnQkvzRow + S(T) * 2 * kGdnConvChannels * 2
              + S(T) * 2 * kGdnStateDim * (2 * G::kGdnQkHeads + G::kGdnValueHeads)
              + S(T) * 4 * G::kGdnValueHeads * 2
              + S(T) * 2 * kGdnStateDim * G::kGdnValueHeads
              + S(T) * 2 * kGdnZChannels * 2 + gdn_op;
  const S qsa = S(T) * 2 * (48 * 256) + S(T) * 2 * (2 * 256) * 2 + S(T) * 4 * 512
              + S(T) * 4 * kQsaBudget + S(T) * 4 + S(T) * 4
              + S(T) * 2 * (48 * 256) + S(T) * 2 * (24 * 256);
  const S moe = S(T) * 2 * H + S(T) * 4 * G::kNumExperts + S(T) * 2 * H + moe_op;
  // Final collapse + lm_head (h_final, the T-vocab logits GEMM, its transpose,
  // the sample I32); the two T-vocab planes are the large term.
  const S final_ = S(T) * 2 * H + S(T) * 2 * G::kVocab + S(T) * 2 * G::kVocab + S(T) * 4;
  return std::max(shared + std::max({gdn, qsa, moe}), shared + final_)
       + (64u * 1024u * 1024u);
}

// The op's (value DESC, expert id ASC) top-K rule, unioned across T tokens: the
// unique expert ids any token selects (each scattered once into the window).
std::vector<std::uint32_t> moe_topk_union(const float* logits, int T, int E, int K) {
  std::vector<bool> selected(static_cast<std::size_t>(E), false);
  std::vector<std::uint32_t> order(static_cast<std::size_t>(E));
  for (int t = 0; t < T; ++t) {
    const float* row = logits + static_cast<std::size_t>(t) * E;
    for (std::uint32_t e = 0; e < static_cast<std::uint32_t>(E); ++e) {
      order[e] = e;
    }
    std::stable_sort(order.begin(), order.end(), [row](std::uint32_t a, std::uint32_t b) {
      if (row[a] != row[b]) {
        return row[a] > row[b];
      }
      return a < b;
    });
    const int n = (K < E) ? K : E;
    for (int k = 0; k < n; ++k) {
      selected[order[static_cast<std::size_t>(k)]] = true;
    }
  }
  std::vector<std::uint32_t> out;
  for (std::uint32_t e = 0; e < static_cast<std::uint32_t>(E); ++e) {
    if (selected[e]) {
      out.push_back(e);
    }
  }
  return out;
}

Tensor make_tensor(const void* data, DType dtype, std::initializer_list<std::int32_t> shape) {
  return Tensor(const_cast<void*>(data), dtype, shape);
}

// P13 M1 timing bucket ids (RoundTimings::kBuckets = 12 slots, this order).
enum {
  B_STATE_RESET = 0,  // reset_recurrent_state (88 zero memsets)
  B_EMBED,            // zero_block memset + ids H2D + embedding gather + stream init
  B_HC,               // gated_residual (4 per layer: attn read/write, mlp read/write)
  B_CONVERT,          // mini_f32_to_bf16 (2 per layer)
  B_GDN,              // whole gdn_block (36 layers)
  B_QSA,              // whole qsa_block (12 layers)
  B_PLE,              // ngram_embedding + real_ple_add (layer G::kPleLayer only)
  B_BLOCKOUT,         // real_block_out (2 per layer)
  B_MOE_ROUTER,       // real_router GEMM (GPU side) + host top-10 D2H
  B_MOE_SCATTER,      // the per-expert H2D scatter enqueues (GPU-side DMA)
  B_MOE_GEMM,         // sparse_moe_nvfp4 (token-major in/out, no transpose)
  B_LMHEAD,           // final collapse + lm_head GEMM + transpose + argmax
};

// P6-debug: per-segment wall-trace (NINFER_SEG_TRACE=1). When set, seg_begin/seg_end
// each synchronize the stream and print the bucket, so in a HUNG round the last
// printed segment names the non-terminating op: its BEGIN prints, its END never does
// (the seg_end sync blocks on the deadlocked kernel). Off by default -> zero cost.
const char* seg_bucket_name(int b) {
  switch (b) {
    case B_STATE_RESET: return "STATE_RESET";
    case B_EMBED:       return "EMBED";
    case B_HC:          return "HC";
    case B_CONVERT:     return "CONVERT";
    case B_GDN:         return "GDN";
    case B_QSA:         return "QSA";
    case B_PLE:         return "PLE";
    case B_BLOCKOUT:    return "BLOCKOUT";
    case B_MOE_ROUTER:  return "MOE_ROUTER";
    case B_MOE_SCATTER: return "MOE_SCATTER";
    case B_MOE_GEMM:    return "MOE_GEMM";
    case B_LMHEAD:      return "LMHEAD";
    default:            return "OTHER";
  }
}
bool seg_trace_enabled() {
  static const bool e = [] {
    const char* v = std::getenv("NINFER_SEG_TRACE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return e;
}

}  // namespace

// P13 M2: the pinned expert LRU slot count (env override; 0 = the legacy M1
// pageable scatter path, which is the DEFAULT). All measured pinned variants
// regressed on the P13 driver's re-prefill worst case (M1 122.7 s < 2400-slot
// 133.4 s < hybrid 6000-slot 162.8 s < serial-miss 6000-slot 168.1 s):
// (a) per-round top-10 MEMBERSHIP churns ~65% (union only grows, membership
// flips with T because the router GEMM tiling changes FMA ordering) — hits
// are capped at ~35% retention regardless of pool size; (b) below the max
// per-round union (5928 keys at T=16) the LRU evicts the upcoming access order
// ("ghost" effect — 2400 slots: 2.6% hits); (c) every synchronous miss path
// pays ~700 soft page faults per expert (~2.4-2.7 ms) that the driver's async
// pageable staging never puts on the critical path (0.92 ms/expert effective).
// Set NINFER_EXPERT_PINNED_SLOTS (e.g. 6000) to opt in: production decode
// (T=1 incremental, stable routing) retains ~95%+ of the ~480-key union,
// which is the workload where a pinned hit (one contiguous DMA, ~26 GB/s
// warm) pays for the pool.
std::size_t expert_pinned_slot_count() {
  constexpr std::size_t kDefaultSlots = 0;
  const char* env = std::getenv("NINFER_EXPERT_PINNED_SLOTS");
  if (env == nullptr) {
    return kDefaultSlots;
  }
  const unsigned long v = std::strtoul(env, nullptr, 10);
  return (v > 0U) ? static_cast<std::size_t>(v) : 0;
}

// P9 (dynamic expert-VRAM sizing): the GPU-resident expert tier (tier-1) sizes
// itself to the LEFTOVER VRAM once every other allocation (GDN state, QSA K/V +
// indexer pools, the 256 MB scratch) is resident. Each slot is one 2,764,800 B
// expert blob. NINFER_EXPERT_GPU_SLOTS: unset = DYNAMIC (fit as many whole
// expert blobs as the free VRAM allows, minus a headroom so the next allocation
// never OOMs); "0" = OFF (the legacy window scatter, the P8-oracle-safe path);
// "N" = exactly N slots (exact override, e.g. a probe or a known-good count).
// Called from the RealProgram ctor body AFTER all other VRAM is resident, so
// cudaMemGetInfo reflects the true free VRAM. Adapts to any max_context.
std::size_t expert_gpu_slot_count() {
  const char* env = std::getenv("NINFER_EXPERT_GPU_SLOTS");
  if (env != nullptr) {
    const unsigned long v = std::strtoul(env, nullptr, 10);
    return static_cast<std::size_t>(v);  // "0" -> OFF; "N" -> exactly N.
  }
  // Dynamic: leftover VRAM minus a 2 GiB headroom, in whole expert-blob units,
  // capped at the full (layer, expert) set.
  constexpr std::size_t kExpertGpuVramHeadroom = 2ULL * 1024ULL * 1024ULL * 1024ULL;
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    return 0;  // cannot measure the free VRAM -> stay disabled (safe fallback)
  }
  const std::size_t headroom =
      (free_bytes > kExpertGpuVramHeadroom) ? (free_bytes - kExpertGpuVramHeadroom) : 0;
  const std::size_t slots = headroom / kExpertBytes;
  const std::size_t max_slots = static_cast<std::size_t>(G::kNumLayers) * G::kNumExperts;
  return (slots < max_slots) ? slots : max_slots;
}

// P13 M4: how many of the top-used (layer, expert) blobs to fault into RAM
// (the mmap page cache) at startup so a later H2D scatter runs at RAM speed
// instead of SSD. DEFAULT 0 = no pre-warm. Ordered by usage count descending.
std::size_t expert_ram_warm_count() {
  constexpr std::size_t kDefaultWarm = 0;
  const char* env = std::getenv("NINFER_EXPERT_RAM_WARM");
  if (env == nullptr) {
    return kDefaultWarm;
  }
  const unsigned long v = std::strtoul(env, nullptr, 10);
  return (v > 0U) ? static_cast<std::size_t>(v) : 0;
}

ExpertPinnedLru::ExpertPinnedLru(std::size_t slot_count, std::size_t bytes_per_expert)
    : slots_(slot_count), slot_bytes_(bytes_per_expert) {
  if (slot_count == 0) {
    return;
  }
  void* base = nullptr;
  const cudaError_t e =
      cudaHostAlloc(&base, slot_count * bytes_per_expert, cudaHostAllocDefault);
  if (e != cudaSuccess) {
    // Graceful degrade: a too-large pool must never crash a model load. Log the
    // failure and stay disabled (base_ == nullptr -> enabled() false), so the
    // scatter falls back to the legacy pageable path and the run still works.
    std::fprintf(stderr,
                 "[pinned-lru] cudaHostAlloc %zu slots * 2,764,800 B failed (%s); "
                 "expert pinned LRU DISABLED (falling back to pageable scatter)\n",
                 slot_count, cudaGetErrorString(e));
    return;
  }
  base_ = static_cast<std::uint8_t*>(base);
}

ExpertPinnedLru::~ExpertPinnedLru() {
  if (base_ != nullptr) {
    cudaFreeHost(base_);
  }
}

// P13 M2 loop-4 (hybrid scatter): the lookup and the promotion are split.
// find() = hit test only (no copy); promote() = evict the LRU victim and
// copy the four planes into the expert's slot. The scatter calls find()
// first; on a hit the H2D is ONE contiguous pinned DMA; on a miss it runs
// the M1 4-plane pageable DMAs (the driver's staging faults the mmap pages
// in) and THEN promotes — so the promotion memcpys re-read pages the
// staging just resident, at RAM speed instead of faulting them at disk rate.
std::uint8_t* ExpertPinnedLru::find(std::uint32_t layer, std::uint32_t expert) {
  const std::uint64_t key =
      (static_cast<std::uint64_t>(layer) << 32) | static_cast<std::uint32_t>(expert);
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    Slot& slot = slots_[i];
    if (slot.in_use && slot.key == key) {
      slot.tick = ++tick_;
      ++hits_;
      return base_ + i * slot_bytes_;
    }
  }
  return nullptr;
}

void ExpertPinnedLru::promote(std::uint32_t layer, std::uint32_t expert,
                              const std::uint8_t* gu_codes,
                              std::size_t gu_codes_bytes,
                              const std::uint8_t* gu_scales,
                              std::size_t gu_scales_bytes,
                              const std::uint8_t* dn_codes,
                              std::size_t dn_codes_bytes,
                              const std::uint8_t* dn_scales,
                              std::size_t dn_scales_bytes) {
  if (!enabled()) {
    throw std::runtime_error("flash_next P13 expert pinned LRU: promote on a disabled LRU");
  }
  const std::uint64_t key =
      (static_cast<std::uint64_t>(layer) << 32) | static_cast<std::uint32_t>(expert);
  // First free slot, else the least-recently-used one (pager.cpp
  // acquire_slot semantics). Safe to overwrite: the router gate's
  // cudaStreamSynchronize drained every H2D the round stream still owed (see
  // the class doc), so no in-flight DMA reads the victim slot.
  std::size_t victim = 0;
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    if (!slots_[i].in_use) {
      victim = i;
      break;
    }
    if (slots_[i].tick < slots_[victim].tick) {
      victim = i;
    }
  }
  Slot& slot = slots_[victim];
  slot.key = key;
  slot.in_use = true;
  slot.tick = ++tick_;
  ++misses_;
  std::uint8_t* dst = base_ + victim * slot_bytes_;
  std::memcpy(dst + kPlaneGuCodes, gu_codes, gu_codes_bytes);
  std::memcpy(dst + kPlaneGuScales, gu_scales, gu_scales_bytes);
  std::memcpy(dst + kPlaneDnCodes, dn_codes, dn_codes_bytes);
  std::memcpy(dst + kPlaneDnScales, dn_scales, dn_scales_bytes);
}

ExpertGpuLru::ExpertGpuLru(std::size_t slot_count, std::size_t bytes_per_expert) {
  init(slot_count, bytes_per_expert);
}

void ExpertGpuLru::init(std::size_t slot_count, std::size_t bytes_per_expert) {
  if (!slots_.empty()) {
    throw std::runtime_error(
        "flash_next P13 expert GPU LRU: init() on an already-allocated instance");
  }
  slot_bytes_ = bytes_per_expert;
  if (slot_count == 0) {
    return;  // explicitly OFF: stay disabled (the legacy pageable window scatter)
  }
  // Back off by halving on a cudaMalloc miss; on total failure stay disabled
  // (base_ == nullptr -> enabled() false) so the scatter falls back to the legacy
  // pageable window. The dynamic count (expert_gpu_slot_count) already sizes to
  // free VRAM minus a headroom, so a miss should be rare; the back-off is the
  // safety net if other allocations grew between the measurement and this point.
  std::size_t n = slot_count;
  void* base = nullptr;
  cudaError_t e = cudaSuccess;
  while (n > 0) {
    e = cudaMalloc(&base, n * bytes_per_expert);
    if (e == cudaSuccess) {
      break;
    }
    cudaGetLastError();  // clear the sticky error before the next attempt
    n /= 2;
  }
  if (e != cudaSuccess || base == nullptr) {
    std::fprintf(stderr,
                 "[gpu-lru] cudaMalloc %zu slots * %zu B failed (%s); expert GPU LRU "
                 "DISABLED (falling back to pageable window scatter)\n",
                 slot_count, bytes_per_expert, cudaGetErrorString(e));
    return;
  }
  slots_.assign(n, Slot{});
  base_ = static_cast<std::uint8_t*>(base);
  idx_.reserve(n);
  std::fprintf(stderr,
               "[gpu-lru] expert GPU LRU allocated: %zu slots x %zu B = %.2f GiB "
               "(requested %zu slots)\n",
               n, bytes_per_expert, (double)(n * bytes_per_expert) / 1073741824.0,
               slot_count);
}

ExpertGpuLru::~ExpertGpuLru() {
  if (base_ != nullptr) {
    cudaFree(base_);
  }
}

const std::uint8_t* ExpertGpuLru::peek(std::uint32_t layer, std::uint32_t expert) {
  const std::uint64_t key =
      (static_cast<std::uint64_t>(layer) << 32) | static_cast<std::uint32_t>(expert);
  const auto it = idx_.find(key);
  if (it == idx_.end()) {
    return nullptr;
  }
  ++hits_;
  // Re-tick to the LATEST tick: the scatter peeks every selected expert before
  // promoting, and promote evicts the min-tick victim, so a peeked hit (now the
  // newest) is never a same-scatter victim. A read is a use, so the recency bump
  // is the LRU's correct behavior.
  slots_[it->second].tick = ++tick_;
  return base_ + it->second * slot_bytes_;
}

const std::uint8_t* ExpertGpuLru::promote(std::uint32_t layer, std::uint32_t expert,
                                          const std::uint8_t* gu_codes,
                                          const std::uint8_t* gu_scales,
                                          const std::uint8_t* dn_codes,
                                          const std::uint8_t* dn_scales,
                                          cudaStream_t stream) {
  if (!enabled()) {
    throw std::runtime_error("flash_next P13 expert GPU LRU: promote on a disabled LRU");
  }
  const std::uint64_t key =
      (static_cast<std::uint64_t>(layer) << 32) | static_cast<std::uint32_t>(expert);
  // First free slot, else the least-recently-used one. The scatter calls peek()
  // for every selected expert first, bumping each hit's tick to the newest, so
  // this min-tick victim is a COLD slot, never a same-scatter recorded hit — the
  // GEMM's pending read of a hit's slot is safe (see the class doc). Across
  // layers the router gate's cudaStreamSynchronize drained every prior GEMM, so
  // the victim's last read has completed.
  std::size_t victim = 0;
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    if (!slots_[i].in_use) {
      victim = i;
      break;
    }
    if (slots_[i].tick < slots_[victim].tick) {
      victim = i;
    }
  }
  const bool had_resident = slots_[victim].in_use;
  const std::uint64_t old_key = slots_[victim].key;
  Slot& slot = slots_[victim];
  slot.key = key;
  slot.in_use = true;
  slot.tick = ++tick_;
  ++misses_;
  if (had_resident) {
    idx_.erase(old_key);
  }
  std::uint8_t* dst = base_ + victim * slot_bytes_;
  const cudaError_t e1 =
      cudaMemcpyAsync(dst + kPlaneGuCodes, gu_codes, kSrcGuCodes, cudaMemcpyHostToDevice, stream);
  const cudaError_t e2 =
      cudaMemcpyAsync(dst + kPlaneGuScales, gu_scales, kSrcGuScales, cudaMemcpyHostToDevice, stream);
  const cudaError_t e3 =
      cudaMemcpyAsync(dst + kPlaneDnCodes, dn_codes, kSrcDnCodes, cudaMemcpyHostToDevice, stream);
  const cudaError_t e4 =
      cudaMemcpyAsync(dst + kPlaneDnScales, dn_scales, kSrcDnScales, cudaMemcpyHostToDevice, stream);
  if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess || e4 != cudaSuccess) {
    throw std::runtime_error(std::string("flash_next P13 expert GPU LRU: H2D miss DMA failed: ") +
                             cudaGetErrorString(e1 != cudaSuccess ? e1 :
                                                e2 != cudaSuccess ? e2 :
                                                e3 != cudaSuccess ? e3 : e4));
  }
  idx_[key] = victim;
  return dst;
}

const std::uint8_t* ExpertGpuLru::promote_contiguous(std::uint32_t layer, std::uint32_t expert,
                                                     const std::uint8_t* pinned_src,
                                                     cudaStream_t stream) {
  if (!enabled()) {
    throw std::runtime_error("flash_next P13 expert GPU LRU: promote_contiguous on a disabled LRU");
  }
  const std::uint64_t key =
      (static_cast<std::uint64_t>(layer) << 32) | static_cast<std::uint32_t>(expert);
  // Same victim-selection and same-stream safety as promote(): the scatter peeks
  // every selected expert first, so this min-tick victim is a cold slot. pinned_src
  // is a pinned (cudaHostAlloc) blob laid out at the kPlane* offsets, identical to a
  // device window slot, so ONE contiguous DMA populates the whole slot (no staging,
  // no per-plane page faults).
  std::size_t victim = 0;
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    if (!slots_[i].in_use) {
      victim = i;
      break;
    }
    if (slots_[i].tick < slots_[victim].tick) {
      victim = i;
    }
  }
  const bool had_resident = slots_[victim].in_use;
  const std::uint64_t old_key = slots_[victim].key;
  Slot& slot = slots_[victim];
  slot.key = key;
  slot.in_use = true;
  slot.tick = ++tick_;
  ++misses_;
  if (had_resident) {
    idx_.erase(old_key);
  }
  std::uint8_t* dst = base_ + victim * slot_bytes_;
  const cudaError_t e =
      cudaMemcpyAsync(dst, pinned_src, slot_bytes_, cudaMemcpyHostToDevice, stream);
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("flash_next P13 expert GPU LRU: contiguous H2D miss DMA failed: ") +
                             cudaGetErrorString(e));
  }
  idx_[key] = victim;
  return dst;
}

const std::uint8_t* ExpertGpuLru::upsert(std::uint32_t layer, std::uint32_t expert,
                                         const std::uint8_t* gu_codes,
                                         const std::uint8_t* gu_scales,
                                         const std::uint8_t* dn_codes,
                                         const std::uint8_t* dn_scales,
                                         cudaStream_t stream) {
  const std::uint8_t* hit = peek(layer, expert);
  if (hit != nullptr) {
    return hit;
  }
  return promote(layer, expert, gu_codes, gu_scales, dn_codes, dn_scales, stream);
}

RealProgram::RealProgram(const RealModelView& view, DeviceContext& device,
                         std::uint32_t max_context, ops::QsaIndexerKvDtype idx_dtype)
    : view_(view),
      moe_window_(kMoeWindowBytes),
      expert_lru_(expert_pinned_slot_count(), kExpertBytes),
      expert_gpu_lru_(),
      moe_expert_ptrs_dev_(G::kNumExperts * sizeof(const std::uint8_t*)),
      moe_expert_ptrs_host_(G::kNumExperts * sizeof(const std::uint8_t*)),
      workspace_(arena_capacity_bytes(kMaxRoundTokens)),
      ids_host_(static_cast<std::size_t>(kMaxRoundTokens) * kQsaBudget * sizeof(std::int32_t)),
      router_host_(static_cast<std::size_t>(kMaxRoundTokens) * G::kNumExperts * sizeof(float)),
      sample_host_(static_cast<std::size_t>(kMaxRoundTokens) * sizeof(std::int32_t)),
      logits_host_(static_cast<std::size_t>(G::kVocab) * 2),
      stream_(device.stream) {
  if (view_.layers.size() != G::kNumLayers) {
    throw std::runtime_error("flash_next real Program: layer count != 48");
  }

  // Indexer-K pool storage dtype (Fp8 default; Bf16 fallback). The qsa_indexer logit
  // GEMM reads the pool in this dtype and the pool below is sized to match it.
  indexer_kv_dtype_ = idx_dtype;

  // Expert-usage histogram: per-(layer, expert) routed-expert selection counts.
  expert_usage_.assign(static_cast<std::size_t>(G::kNumLayers) * G::kNumExperts, 0);

  // Recurrent GDN state: one [128, 128, 48] FP32 per GDN layer, zeroed.
  const std::size_t gdn_state_bytes =
      static_cast<std::size_t>(G::kGdnStateDim) * G::kGdnStateDim * G::kGdnValueHeads *
      sizeof(float);
  gdn_states_.resize(G::kGdnLayers);
  for (auto& b : gdn_states_) {
    b = DeviceBuffer(gdn_state_bytes);
    b.fill(0);
  }
  // GDN conv state: one [10240, 3] BF16 per GDN layer, zeroed.
  const std::size_t conv_state_bytes =
      static_cast<std::size_t>(kGdnConvChannels) * G::kGdnConvWidth * sizeof(std::uint16_t);
  conv_states_.resize(G::kGdnLayers);
  for (auto& b : conv_states_) {
    b = DeviceBuffer(conv_state_bytes);
    b.fill(0);
  }

  // QSA paged KV: one SEPARATE K pool and V pool per QSA layer, each
  // qsa_page_count_ pages (zeroed); one IDENTITY block table per QSA layer (logical
  // page i -> physical page i), set once in the ctor and never mutated (the
  // absolute-position write at real_kv_append maps position p through
  // block_table[p >> 6] == p >> 6, i.e. its own page). The identity table makes
  // every page reachable, so a forward of up to qsa_page_count_ * 64 positions
  // (the constructed max_context) is fully covered. qsa_page_count_ =
  // ceil(max_context / 64); the default 8320 -> 130 pages (byte-identical to P13).
  qsa_page_count_ = (static_cast<std::size_t>(max_context) + kQsaTokensPerPage - 1) /
                   kQsaTokensPerPage;
  if (qsa_page_count_ == 0) qsa_page_count_ = 1;
  const std::size_t block_table_bytes = qsa_page_count_ * sizeof(std::int32_t);
  const std::size_t pool_bytes = qsa_page_count_ * kQsaKvPageBytes;
  std::vector<std::int32_t> identity(qsa_page_count_);
  for (std::size_t i = 0; i < qsa_page_count_; ++i) identity[i] = static_cast<std::int32_t>(i);
  k_pools_.resize(G::kFullAttentionLayers);
  v_pools_.resize(G::kFullAttentionLayers);
  block_tables_.resize(G::kFullAttentionLayers);
  for (std::uint32_t i = 0; i < G::kFullAttentionLayers; ++i) {
    k_pools_[i] = DeviceBuffer(pool_bytes);
    v_pools_[i] = DeviceBuffer(pool_bytes);
    block_tables_[i] = DeviceBuffer(block_table_bytes);
    k_pools_[i].fill(0);
    v_pools_[i].fill(0);
    const cudaError_t t = cudaMemcpy(block_tables_[i].p, identity.data(), block_table_bytes,
                                     cudaMemcpyHostToDevice);
    if (t != cudaSuccess) {
      throw std::runtime_error("flash_next real Program: identity block table H2D failed");
    }
  }

  // QSA indexer-K pool: one paged 512-dim FP8 key pool + its per-token scale per
  // QSA layer, on the SAME identity block table (idx_k_pools_[i] reuses
  // block_tables_[i]). Sized to qsa_page_count_ like the main K/V pool; zeroed.
  // Indexer-K value pool is 1 B/elem (Fp8) or 2 B/elem (Bf16); the per-token half
  // scale plane is Fp8-only (allocated regardless, negligible cost).
  const std::size_t idx_elem_bytes =
      (indexer_kv_dtype_ == ops::QsaIndexerKvDtype::Bf16) ? 2u : 1u;
  const std::size_t idx_pool_bytes =
      qsa_page_count_ * kIndexerKvPageBytes * idx_elem_bytes;
  const std::size_t idx_scale_bytes = qsa_page_count_ * kIndexerKvScalePageBytes;
  idx_k_pools_.resize(G::kFullAttentionLayers);
  idx_k_scale_.resize(G::kFullAttentionLayers);
  for (std::uint32_t i = 0; i < G::kFullAttentionLayers; ++i) {
    idx_k_pools_[i] = DeviceBuffer(idx_pool_bytes);
    idx_k_scale_[i] = DeviceBuffer(idx_scale_bytes);
    idx_k_pools_[i].fill(0);
    idx_k_scale_[i].fill(0);
  }
  // Persistent indexer logit scratch (the qsa_indexer M-chunked [chunk, n_full]
  // FP32 region; bounded by kIndexerLogitsBytes). Allocated once, reused.
  idx_logits_scratch_ = DeviceBuffer(kIndexerLogitsBytes);

  // Shared routed-expert window, the round scratch arena, and the pinned staging
  // are sized in the member initializer list (move-only, no default ctor).

  // P9 (dynamic expert-VRAM sizing): size the GPU-resident expert tier to the
  // leftover VRAM now that every other allocation (GDN state, QSA K/V + indexer
  // pools, the 256 MB scratch) is resident. Default (NINFER_EXPERT_GPU_SLOTS unset)
  // fits as many whole expert blobs as free VRAM allows minus a headroom; "0" = off.
  // A too-large pool degrades to disabled (init backs off / falls back), never crashes.
  expert_gpu_lru_.init(expert_gpu_slot_count(), kExpertBytes);

  // P13 M4: seed the expert residency hierarchy from a prior run's usage file
  // (top-N -> VRAM via the GPU LRU, top-R -> RAM via the mmap page cache).
  seed_expert_residency();
}

RealProgram::~RealProgram() {
  // Flush the per-(layer, expert) routed-expert usage histogram to a file so the
  // most-active experts are inspectable after the run (NINFER_EXPERT_USAGE_FILE).
  dump_expert_usage();
}

void RealProgram::dump_expert_usage() const {
  if (expert_usage_.empty()) { return; }
  const char* env = std::getenv("NINFER_EXPERT_USAGE_FILE");
  const std::string path =
      (env && *env) ? std::string(env) : std::string("ninfer-expert-usage.txt");

  // Global per-expert totals (summed over all 48 layers) for the top-N ranking.
  std::vector<std::uint64_t> global(G::kNumExperts, 0);
  std::uint64_t total = 0;
  for (std::uint32_t l = 0; l < G::kNumLayers; ++l) {
    for (std::uint32_t e = 0; e < G::kNumExperts; ++e) {
      const std::uint64_t c = expert_usage_[static_cast<std::size_t>(l) * G::kNumExperts + e];
      global[e] += c;
      total += c;
    }
  }
  std::vector<std::uint32_t> order(G::kNumExperts);
  for (std::uint32_t e = 0; e < G::kNumExperts; ++e) { order[e] = e; }
  std::sort(order.begin(), order.end(),
            [&global](std::uint32_t a, std::uint32_t b) {
              if (global[a] != global[b]) { return global[a] > global[b]; }
              return a < b;  // tie-break: lower expert id first (stable, deterministic)
            });

  std::ofstream out(path, std::ios::trunc);
  if (!out) { return; }  // cannot open: drop the dump (dev diagnostic, non-fatal)
  char ts[32];
  const std::time_t now = std::time(nullptr);
  std::tm* tmb = std::localtime(&now);
  if (tmb != nullptr) { std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", tmb); }
  out << "# ninfer flash_next expert-usage histogram\n";
  out << "# layers=" << G::kNumLayers << " experts=" << G::kNumExperts
      << " experts_per_token=" << G::kExpertsPerToken << "\n";
  out << "# timestamp=" << ts << "\n";
  out << "# total_selections=" << total
      << "  (1 = that expert was active in that layer for this round)\n";
  out << "# top 64 experts (global, summed over all 48 layers, most-active first)\n";
  for (std::uint32_t i = 0; i < order.size() && i < 64; ++i) {
    const std::uint32_t e = order[i];
    out << std::setw(4) << i << ": expert=" << std::setw(4) << e << " count=" << global[e]
        << "\n";
  }
  out << "# per-(layer, expert) matrix (each row = one layer, 512 columns = expert id)\n";
  for (std::uint32_t l = 0; l < G::kNumLayers; ++l) {
    for (std::uint32_t e = 0; e < G::kNumExperts; ++e) {
      if (e > 0) { out << ' '; }
      out << expert_usage_[static_cast<std::size_t>(l) * G::kNumExperts + e];
    }
    out << "\n";
  }
}

void RealProgram::seed_expert_residency() {
  // Nothing to seed unless a VRAM, RAM, or pinned-RAM budget is set.
  const std::size_t gpu_n = expert_gpu_lru_.slot_count();
  const std::size_t ram_r = expert_ram_warm_count();
  const std::size_t pin_n = expert_pinned_slot_count();
  if (gpu_n == 0 && ram_r == 0 && pin_n == 0) { return; }

  // Parse the per-(layer, expert) matrix from a prior run's usage file. No-op if
  // the file is absent (first run) — the residency hierarchy then fills in
  // live during the first forward, exactly like the legacy path.
  const char* env = std::getenv("NINFER_EXPERT_USAGE_FILE");
  const std::string path =
      (env && *env) ? std::string(env) : std::string("ninfer-expert-usage.txt");
  std::ifstream in(path, std::ios::in);
  if (!in) { return; }

  // (count, layer, expert) for every non-zero histogram cell.
  struct Cell {
    std::uint64_t count = 0;
    std::uint32_t layer = 0;
    std::uint32_t expert = 0;
  };
  std::vector<Cell> cells;
  std::string line;
  bool at_matrix = false;
  while (std::getline(in, line)) {
    if (!at_matrix) {
      if (line.rfind("# per-(layer, expert) matrix", 0) == 0) {
        at_matrix = true;
      }
      continue;
    }
    if (line.empty()) { continue; }
    std::istringstream ls(line);
    std::uint32_t layer = 0;
    std::uint32_t col = 0;
    std::uint64_t v = 0;
    while (ls >> v) {
      if (v > 0 && col < G::kNumExperts) {
        cells.push_back(Cell{v, layer, col});
      }
      ++col;
    }
    if (col == 0) { break; }  // malformed / short final line: stop
    ++layer;
    if (layer >= G::kNumLayers) { break; }
  }
  if (cells.empty()) { return; }

  // Rank: most-used first; tie -> lower layer, then lower expert (deterministic).
  std::sort(cells.begin(), cells.end(), [](const Cell& a, const Cell& b) {
    if (a.count != b.count) { return a.count > b.count; }
    if (a.layer != b.layer) { return a.layer < b.layer; }
    return a.expert < b.expert;
  });

  // (A) Fault the broader hot set's mmap pages into RAM so a later H2D scatter
  // runs at RAM speed instead of SSD. The read target is a throwaway buffer —
  // the point is to page-fault the source. The GPU and pinned tiers are now
  // DISJOINT by usage rank (GPU = top-gpu_n, pinned = the NEXT pin_n), so the
  // union of what seeding reads is the top (gpu_n + pin_n) ranks; cover that
  // plus the RAM-warm count.
  std::size_t warm_r = gpu_n + pin_n;
  if (ram_r > warm_r) { warm_r = ram_r; }
  const std::size_t warm_count = (warm_r < cells.size()) ? warm_r : cells.size();
  std::vector<std::uint8_t> scratch(kExpertBytes);
  for (std::size_t i = 0; i < warm_count; ++i) {
    const std::uint32_t l = cells[i].layer;
    const std::uint32_t e = cells[i].expert;
    const RoutedHostBases& routed = view_.routed_host[l];
    std::memcpy(scratch.data(), routed.gu_codes + static_cast<std::size_t>(e) * kSrcGuCodes,
                kSrcGuCodes);
    std::memcpy(scratch.data(), routed.gu_scales + static_cast<std::size_t>(e) * kSrcGuScales,
                kSrcGuScales);
    std::memcpy(scratch.data(), routed.dn_codes + static_cast<std::size_t>(e) * kSrcDnCodes,
                kSrcDnCodes);
    std::memcpy(scratch.data(), routed.dn_scales + static_cast<std::size_t>(e) * kSrcDnScales,
                kSrcDnScales);
  }

  // (A') STATIC pinned-RAM pre-warm (the PREFILL fix): promote the NEXT pin_n
  // usage-ranked blobs — ranks [gpu_n, gpu_n + pin_n), DISJOINT from the GPU
  // tier's top-gpu_n — into the pinned pool ONCE, before any round, so the
  // hybrid scatter's find() hit path (one contiguous pinned->device DMA, ~26
  // GB/s) serves them instead of the reactive M1 4-plane pageable miss path
  // (~1.5 GB/s). Keeping the tiers disjoint (the user's 2026-09-14 design)
  // means the pinned pool holds NEW experts rather than duplicating the
  // GPU-resident ones: the whole hot working set spans gpu_n + pin_n experts
  // across the two tiers with zero overlap. promote() memcpys read the
  // (now-warm) mmap pages.
  if (expert_lru_.enabled() && pin_n > 0) {
    const std::size_t start = (gpu_n < cells.size()) ? gpu_n : cells.size();
    const std::size_t remaining = cells.size() - start;
    const std::size_t pin_count = (pin_n < remaining) ? pin_n : remaining;
    for (std::size_t i = 0; i < pin_count; ++i) {
      const std::uint32_t l = cells[start + i].layer;
      const std::uint32_t e = cells[start + i].expert;
      const RoutedHostBases& routed = view_.routed_host[l];
      expert_lru_.promote(
          l, e,
          routed.gu_codes + static_cast<std::size_t>(e) * kSrcGuCodes, kSrcGuCodes,
          routed.gu_scales + static_cast<std::size_t>(e) * kSrcGuScales, kSrcGuScales,
          routed.dn_codes + static_cast<std::size_t>(e) * kSrcDnCodes, kSrcDnCodes,
          routed.dn_scales + static_cast<std::size_t>(e) * kSrcDnScales, kSrcDnScales);
    }
  }

  // (B) Promote the hottest blobs into VRAM via the GPU LRU (zero-DMA resident
  // reads). The upsert H2D's the four planes from the (now-warm) mmap onto the
  // round stream; drain before returning so the seeded slots are readable by the
  // first forward.
  if (gpu_n > 0) {
    const std::size_t gpu_count = (gpu_n < cells.size()) ? gpu_n : cells.size();
    for (std::size_t i = 0; i < gpu_count; ++i) {
      const std::uint32_t l = cells[i].layer;
      const std::uint32_t e = cells[i].expert;
      const RoutedHostBases& routed = view_.routed_host[l];
      expert_gpu_lru_.upsert(
          l, e,
          routed.gu_codes + static_cast<std::size_t>(e) * kSrcGuCodes,
          routed.gu_scales + static_cast<std::size_t>(e) * kSrcGuScales,
          routed.dn_codes + static_cast<std::size_t>(e) * kSrcDnCodes,
          routed.dn_scales + static_cast<std::size_t>(e) * kSrcDnScales, stream_);
    }
    const cudaError_t e = cudaStreamSynchronize(stream_);
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("flash_next P13 expert residency seed: drain failed: ") +
                               cudaGetErrorString(e));
    }
  }
}

const char* RoundTimings::name(int bucket) {
  static const char* kNames[kBuckets] = {"state_reset", "embed", "hc_gated_residual",
                                         "f32_to_bf16", "gdn", "qsa", "ple_gather",
                                         "block_out", "moe_router", "moe_expert_h2d",
                                         "moe_gemm", "lm_head"};
  return (bucket >= 0 && bucket < kBuckets) ? kNames[bucket] : "?";
}

void RealProgram::set_timings_enabled(bool on) {
  if (!on) {
    timings_enabled_ = false;
    return;
  }
  if (timer_.ev.empty()) {
    timer_.ev.resize(static_cast<std::size_t>(2) * Timer::kMaxPairs);
    for (cudaEvent_t& e : timer_.ev) {
      const cudaError_t err = cudaEventCreate(&e);
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("flash_next P13 timer event create failed: ") +
                                 cudaGetErrorString(err));
      }
    }
  }
  timings_enabled_ = true;
}

void RealProgram::seg_begin(int bucket) {
  if (seg_trace_enabled()) {
    cudaStreamSynchronize(stream_);
    fprintf(stderr, "[seg] L%d BEGIN %-11s\n", seg_trace_layer_, seg_bucket_name(bucket));
    fflush(stderr);
  }
  if (!timings_enabled_) {
    return;
  }
  if (seg_pair_ >= 0) {
    throw std::runtime_error("flash_next P13 timer: nested seg_begin");
  }
  if (timer_.next_pair >= Timer::kMaxPairs) {
    throw std::runtime_error("flash_next P13 timer: pair pool exhausted");
  }
  seg_pair_ = timer_.next_pair++;
  seg_bucket_ = bucket;
  const cudaError_t e = cudaEventRecord(timer_.ev[static_cast<std::size_t>(2) * seg_pair_],
                                        stream_);
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("flash_next P13 timer event record failed: ") +
                             cudaGetErrorString(e));
  }
}

void RealProgram::seg_end() {
  if (seg_trace_enabled()) {
    const cudaError_t ste = cudaStreamSynchronize(stream_);
    fprintf(stderr, "[seg] L%d END   %-11s %s\n", seg_trace_layer_, seg_bucket_name(seg_bucket_),
            ste == cudaSuccess ? "ok" : cudaGetErrorString(ste));
    fflush(stderr);
  }
  if (!timings_enabled_) {
    return;
  }
  if (seg_pair_ < 0) {
    throw std::runtime_error("flash_next P13 timer: seg_end without seg_begin");
  }
  const cudaError_t e =
      cudaEventRecord(timer_.ev[static_cast<std::size_t>(2) * seg_pair_ + 1], stream_);
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("flash_next P13 timer event record failed: ") +
                             cudaGetErrorString(e));
  }
  timer_.segs.push_back(Timer::Seg{seg_bucket_, seg_pair_});
  seg_pair_ = -1;
  seg_bucket_ = -1;
}

void RealProgram::barrier_begin() {
  if (!timings_enabled_) {
    return;
  }
  if (timer_.barrier_pair >= 0) {
    throw std::runtime_error("flash_next P13 timer: nested barrier window");
  }
  if (timer_.next_pair >= Timer::kMaxPairs) {
    throw std::runtime_error("flash_next P13 timer: pair pool exhausted");
  }
  timer_.barrier_pair = timer_.next_pair++;
  const cudaError_t e = cudaEventRecord(
      timer_.ev[static_cast<std::size_t>(2) * timer_.barrier_pair], stream_);
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("flash_next P13 timer event record failed: ") +
                             cudaGetErrorString(e));
  }
  timer_.barrier_t0 = std::chrono::steady_clock::now();
}

void RealProgram::barrier_end() {
  if (!timings_enabled_) {
    return;
  }
  if (timer_.barrier_pair < 0) {
    throw std::runtime_error("flash_next P13 timer: barrier_end without barrier_begin");
  }
  const cudaError_t e =
      cudaEventRecord(timer_.ev[static_cast<std::size_t>(2) * timer_.barrier_pair + 1],
                      stream_);
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("flash_next P13 timer event record failed: ") +
                             cudaGetErrorString(e));
  }
  // Bucket id kBuckets is the barrier sentinel (finalize_round routes it to
  // barrier_gpu_ms instead of the gpu_ms array).
  timer_.segs.push_back(Timer::Seg{RoundTimings::kBuckets, timer_.barrier_pair});
  timer_.barrier_wall_ms +=
      std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() -
                                               timer_.barrier_t0)
          .count();
  timer_.barrier_pair = -1;
}

void RealProgram::finalize_round() {
  if (!timings_enabled_) {
    return;
  }
  RoundTimings rt;
  for (const Timer::Seg& seg : timer_.segs) {
    float ms = 0.0F;
    const cudaError_t e = cudaEventElapsedTime(&ms, timer_.ev[static_cast<std::size_t>(2) *
                                                                 seg.pair],
                                               timer_.ev[static_cast<std::size_t>(2) *
                                                            seg.pair + 1]);
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("flash_next P13 timer event query failed: ") +
                               cudaGetErrorString(e));
    }
    if (seg.bucket == RoundTimings::kBuckets) {
      rt.barrier_gpu_ms += ms;
    } else {
      rt.gpu_ms[seg.bucket] += ms;
    }
  }
  rt.host_barrier_ms = timer_.barrier_wall_ms;
  rt.moe_union_experts = timer_.union_experts;
  // The per-round lru_hits/lru_misses pair is the tier-1 GPU LRU's (ExpertGpuLru):
  // it is the tier the moe_expert_h2d bucket measures (a hit = expert already
  // GPU-resident, no H2D; a miss = a promote H2D). expert_lru_ is the tier-2
  // PINNED LRU — reading it here (the prior bug) reported 0/0 whenever the pinned
  // tier was off, hiding the GPU tier's real hit rate. Drain BOTH so neither
  // tier's counters accumulate across rounds.
  rt.lru_hits = expert_gpu_lru_.hits();
  rt.lru_misses = expert_gpu_lru_.misses();
  expert_gpu_lru_.reset_counters();
  expert_lru_.reset_counters();
  timings_ = rt;
  timer_.segs.clear();
  timer_.next_pair = 0;
  timer_.barrier_wall_ms = 0.0F;
  timer_.union_experts = 0;
}

void RealProgram::gdn_block(std::uint32_t layer, int T, __nv_bfloat16* xhat,
                            __nv_bfloat16* y) {
  const GdnLayerWeights& g = *view_.layers[layer].gdn;
  const int oi = gdn_ordinal(layer);
  const cudaStream_t s = stream_;
  const __nv_bfloat16* gdn_state =
      static_cast<const __nv_bfloat16*>(gdn_states_[static_cast<std::size_t>(oi)].p);
  float* gdn_state_f32 =
      reinterpret_cast<float*>(const_cast<__nv_bfloat16*>(gdn_state));
  const __nv_bfloat16* conv_state =
      static_cast<const __nv_bfloat16*>(conv_states_[static_cast<std::size_t>(oi)].p);

  // Per-layer transients free at block end, so the round's peak is the largest
  // single block (glue + op workspace), not a sum over the 48 layers (see
  // arena_capacity_bytes).
  auto block_scope = workspace_.scope();
  __nv_bfloat16* qkvz =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * kGdnQkvzRow).data);
  detail::mini_fp8_linear(xhat, T, G::kHidden, g.qkvz.codes, g.qkvz.scale, kGdnQkvzRow, qkvz, s);

  // Causal conv over the qkv channels (z bypasses). Channel-major staging.
  __nv_bfloat16* conv_in =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * kGdnConvChannels).data);
  detail::mini_transpose_tm_cm(qkvz, T, kGdnConvChannels, conv_in, s);
  __nv_bfloat16* conv_out =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * kGdnConvChannels).data);
  // The op allows conv_state_in == conv_state_out; the layer's state buffer is
  // used for both (named lvalues: conv_state_out / out are non-const Tensor&).
  Tensor conv_x_t = make_tensor(conv_in, DType::BF16, {kGdnConvChannels, T});
  Tensor conv_w_t =
      make_tensor(g.conv, DType::BF16,
                  {kGdnConvChannels, static_cast<int>(G::kGdnConvWidth)});
  Tensor conv_state_t = make_tensor(conv_state, DType::BF16, {kGdnConvChannels, 3});
  Tensor conv_out_t = make_tensor(conv_out, DType::BF16, {kGdnConvChannels, T});
  ops::causal_conv1d_silu(conv_x_t, conv_w_t, conv_state_t, conv_state_t, conv_out_t, s);

  // Zero-pad pack into the op's [128, H, T] d-fastest buffers.
  __nv_bfloat16* q_op = static_cast<__nv_bfloat16*>(
      workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * kGdnStateDim * G::kGdnQkHeads).data);
  __nv_bfloat16* k_op = static_cast<__nv_bfloat16*>(
      workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * kGdnStateDim * G::kGdnQkHeads).data);
  __nv_bfloat16* v_op = static_cast<__nv_bfloat16*>(
      workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * kGdnStateDim * G::kGdnValueHeads)
          .data);
  detail::real_gdn_pack(conv_out, T, kGdnStateDim, kGdnStateDim,
                        static_cast<int>(G::kGdnQkHeads), static_cast<int>(G::kGdnValueHeads),
                        q_op, k_op, v_op, s);

  // Gating from the layer input: a rows [0,48), b rows [48,96) of a_b_projection.
  float* gbuf =
      static_cast<float*>(workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4 *
                                                 G::kGdnValueHeads).data);
  float* betabuf =
      static_cast<float*>(workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4 *
                                                 G::kGdnValueHeads).data);
  const __nv_bfloat16* a_b = reinterpret_cast<const __nv_bfloat16*>(g.a_b_projection.data);
  detail::real_gdn_gating(xhat, T, G::kHidden, kGdnGatingRows, a_b,
                          a_b + static_cast<std::size_t>(kGdnGatingRows) * G::kHidden, g.a_log,
                          g.dt_bias, gbuf, betabuf, s);

  // Distinct-state recurrent step (in == out state storage).
  __nv_bfloat16* o_op = static_cast<__nv_bfloat16*>(
      workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * kGdnStateDim * G::kGdnValueHeads)
          .data);
  {
    auto scope = workspace_.scope();
    // The op's transient capacity is reserved at the round max (kMaxRoundTokens,
    // chunked path for T >= 64; recurrent for T = 1) so one allocation covers
    // every round. The op still requires a non-null arena, so floor a zero
    // capacity to a small sentinel to keep alloc_bytes and the WorkspaceArena
    // borrow valid.
    const std::size_t gdn_cap =
        ops::gated_delta_net_workspace_capacity_bytes(
            static_cast<int>(G::kGdnQkHeads), static_cast<int>(G::kGdnValueHeads), true, 1,
            kMaxRoundTokens);
    const DeviceSpan gws = workspace_.alloc_bytes(gdn_cap > 0 ? gdn_cap : std::size_t(256));
    WorkspaceArena leaf(gws);
    Tensor gdn_state_t =
        make_tensor(gdn_state_f32, DType::FP32, {kGdnStateDim, kGdnStateDim,
                                                 static_cast<int>(G::kGdnValueHeads)});
    Tensor gdn_out_t =
        make_tensor(o_op, DType::BF16, {kGdnStateDim, static_cast<int>(G::kGdnValueHeads), T});
    ops::gated_delta_net(make_tensor(q_op, DType::BF16,
                                     {kGdnStateDim, static_cast<int>(G::kGdnQkHeads), T}),
                         make_tensor(k_op, DType::BF16,
                                     {kGdnStateDim, static_cast<int>(G::kGdnQkHeads), T}),
                         make_tensor(v_op, DType::BF16,
                                     {kGdnStateDim, static_cast<int>(G::kGdnValueHeads), T}),
                         make_tensor(gbuf, DType::FP32, {static_cast<int>(G::kGdnValueHeads), T}),
                         make_tensor(betabuf, DType::FP32,
                                     {static_cast<int>(G::kGdnValueHeads), T}),
                         kGdnScale, true, leaf, gdn_state_t, gdn_state_t, gdn_out_t, s);
  }

  // Fused gated RMSNorm; z = qkvz[t, ROW - ZC + d] (the last 6144 of 16384).
  __nv_bfloat16* z =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * kGdnZChannels).data);
  detail::real_gdn_zslice(qkvz, T, kGdnQkvzRow, kGdnZChannels, z, s);
  __nv_bfloat16* onx =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * kGdnZChannels).data);
  detail::real_gated_rmsnorm(o_op, z, reinterpret_cast<const __nv_bfloat16*>(g.norm), T,
                             static_cast<int>(G::kGdnValueHeads), kGdnStateDim, kGdnStateDim,
                             onx, s);

  detail::mini_fp8_linear(onx, T, kGdnZChannels, g.output.codes, g.output.scale, G::kHidden, y, s);
}

void RealProgram::qsa_block(std::uint32_t layer, int T, int pos0, __nv_bfloat16* xhat,
                            __nv_bfloat16* y) {
  const QsaLayerWeights& q = *view_.layers[layer].qsa;
  const int oi = qsa_ordinal(layer);
  const cudaStream_t s = stream_;

  // Per-layer transients free at block end (see gdn_block / arena_capacity_bytes).
  auto block_scope = workspace_.scope();

  // q_full [T][48][256] token-major.
  __nv_bfloat16* q_full =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * 48 * 256).data);
  detail::mini_fp8_linear(xhat, T, G::kHidden, q.query.codes, q.query.scale, 48 * 256, q_full, s);
  __nv_bfloat16* k =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * 2 * 256).data);
  detail::mini_fp8_linear(xhat, T, G::kHidden, q.key.codes, q.key.scale, 2 * 256, k, s);
  __nv_bfloat16* v =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * 2 * 256).data);
  detail::mini_fp8_linear(xhat, T, G::kHidden, q.value.codes, q.value.scale, 2 * 256, v, s);

  detail::real_head_rmsnorm(q_full, T, 48, 256,
                            reinterpret_cast<const __nv_bfloat16*>(q.query_norm), s);
  detail::real_head_rmsnorm(k, T, 2, 256,
                            reinterpret_cast<const __nv_bfloat16*>(q.key_norm), s);
  detail::real_rope(q_full, T, 48, pos0, k, 2, s);

  // Paged KV append (SEPARATE K/V pools per QSA layer; the pools are device
  // memory the leaf writes, so non-const device pointers).
  __nv_bfloat16* k_pages = static_cast<__nv_bfloat16*>(k_pools_[static_cast<std::size_t>(oi)].p);
  __nv_bfloat16* v_pages = static_cast<__nv_bfloat16*>(v_pools_[static_cast<std::size_t>(oi)].p);
  const std::int32_t* block_table =
      static_cast<const std::int32_t*>(block_tables_[static_cast<std::size_t>(oi)].p);
  detail::real_kv_append(k, v, T, pos0, 2, block_table, k_pages, v_pages, s);

  // Paged indexer-K append: derive this round's 512-dim keys and scatter them
  // to GLOBAL positions pos0 + t through the shared block table. work_dev is
  // test-only (null in production); the op quantizes to FP8 + a per-token half
  // scale into idx_k_pools_[oi] / idx_k_scale_[oi].
  ops::QsaIndexerKAppendParams kapp{};
  kapp.tokens = T;
  kapp.hidden = G::kHidden;
  kapp.pos0 = pos0;
  kapp.hidden_dev = reinterpret_cast<const std::uint16_t*>(xhat);
  kapp.qk_proj_dev = q.indexer_qk_proj.data;
  kapp.k_norm_dev = q.indexer_k_norm;
  kapp.block_table_dev = block_table;
  kapp.idx_k_pages_dev = static_cast<std::uint8_t*>(idx_k_pools_[static_cast<std::size_t>(oi)].p);
  kapp.idx_k_scale_dev = static_cast<std::uint16_t*>(idx_k_scale_[static_cast<std::size_t>(oi)].p);
  ops::qsa_indexer_k_append(kapp, s);

  // Global block-sparse indexer: score the round's T queries against the FULL
  // context [0, n_full) (n_full = pos0 + T) by a chunked GEMM over the paged
  // indexer-K pool, then top-B per query. ids are emitted ALREADY global (the
  // logit rows read the pool through the block table), so qsa_sparse_gqa
  // resolves them directly and no host +pos0 shift is needed. context_len[t] =
  // pos0 + t (exclusive of self) drives the causal mask. The logit scratch is
  // the persistent [chunk, n_full] fp32 region (the op chunks the M dim).
  float* work =
      static_cast<float*>(workspace_.alloc_bytes(static_cast<std::size_t>(T) * 512 * 4).data);
  std::int32_t* ids =
      static_cast<std::int32_t*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4 * kQsaBudget).data);
  std::int32_t* counts =
      static_cast<std::int32_t*>(workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4).data);
  std::int32_t* context_len =
      static_cast<std::int32_t*>(workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4).data);
  std::vector<std::int32_t> ctx(static_cast<std::size_t>(T));
  for (std::size_t t = 0; t < static_cast<std::size_t>(T); ++t) {
    ctx[t] = pos0 + static_cast<std::int32_t>(t);
  }
  const cudaError_t ctx_err =
      cudaMemcpyAsync(context_len, ctx.data(), static_cast<std::size_t>(T) * 4,
                      cudaMemcpyHostToDevice, s);
  if (ctx_err != cudaSuccess) {
    throw std::runtime_error(std::string("flash_next real QSA context_len H2D failed: ") +
                             cudaGetErrorString(ctx_err));
  }
  ops::QsaIndexerParams ip{};
  ip.tokens = T;
  ip.hidden = G::kHidden;
  ip.budget = kQsaBudget;
  ip.pos0 = pos0;
  ip.n_full = pos0 + T;
  ip.hidden_dev = reinterpret_cast<const std::uint16_t*>(xhat);
  ip.qk_proj_dev = q.indexer_qk_proj.data;
  ip.q_norm_dev = q.indexer_q_norm;
  ip.block_table_dev = block_table;
  ip.idx_k_pages_dev = idx_k_pools_[static_cast<std::size_t>(oi)].p;
  ip.idx_k_scale_dev = static_cast<const std::uint16_t*>(idx_k_scale_[static_cast<std::size_t>(oi)].p);
  ip.work_dev = work;
  ip.logits_dev = static_cast<float*>(idx_logits_scratch_.p);
  ip.ids_dev = ids;
  ip.counts_dev = counts;
  ip.context_len_dev = context_len;
  ip.max_logits_bytes = static_cast<std::int32_t>(kIndexerLogitsBytes);
  ip.dtype = indexer_kv_dtype_;
  ops::qsa_indexer(ip, s);
  { char tag[40]; std::snprintf(tag, sizeof(tag), "L%u.qsa.counts", layer);
    nan_probe_counts(tag, counts, (std::size_t)T, s); }

  __nv_bfloat16* qsa_out =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * 48 * 256).data);
  ops::QsaSparseGqaParams gp{};
  gp.tokens = T;
  gp.q_heads = 48;
  gp.kv_heads = 2;
  gp.budget = kQsaBudget;
  gp.scale = kQsaScale;
  gp.q_dev = reinterpret_cast<const std::uint16_t*>(q_full);
  gp.ids_dev = ids;
  gp.counts_dev = counts;
  gp.block_table_dev = block_table;
  gp.k_pages_dev = reinterpret_cast<const std::uint16_t*>(k_pages);
  gp.v_pages_dev = reinterpret_cast<const std::uint16_t*>(v_pages);
  gp.out_dev = reinterpret_cast<std::uint16_t*>(qsa_out);
  ops::qsa_sparse_gqa(gp, s);
  { char tag[40]; std::snprintf(tag, sizeof(tag), "L%u.qsa.gqa", layer);
    nan_probe_bf16(tag, qsa_out, (std::size_t)T * 48 * 256, s); }

  // Output-side head reduction 48 -> 24 (SwiGLU split-half), then output GEMM.
  __nv_bfloat16* reduced =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * 24 * 256).data);
  detail::real_qsa_gate(qsa_out, T, 48, 256, reduced, s);
  detail::mini_fp8_linear(reduced, T, 24 * 256, q.output.codes, q.output.scale, G::kHidden, y, s);
}

void RealProgram::moe_block(std::uint32_t layer, int T, __nv_bfloat16* xhat,
                            __nv_bfloat16* y) {
  const MoeLayerWeights& m = view_.layers[layer].mlp;
  const cudaStream_t s = stream_;

  // Per-layer transients (logits + the op workspace) free at block
  // end (see gdn_block / arena_capacity_bytes).
  auto block_scope = workspace_.scope();

  seg_begin(B_MOE_ROUTER);
  // Host top-10 (real_router + D2H) drives the per-token expert scatter into the
  // shared moe_window; the op re-runs routing deterministically and reads
  // expert_window + e * kExpertBytes.
  float* logits =
      static_cast<float*>(workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4 *
                                                 G::kNumExperts).data);
  detail::real_router(xhat, T, G::kHidden, static_cast<int>(G::kNumExperts),
                      reinterpret_cast<const __nv_bfloat16*>(m.router_bf16), logits, s);
  seg_end();
  // The router D2H gate: from here the CPU blocks (sync + D2H + host top-10
  // union + the pageable-H2D staging copies). barrier_begin/end bracket the
  // whole window (P13 M1).
  barrier_begin();
  const std::size_t logits_count =
      static_cast<std::size_t>(T) * G::kNumExperts * sizeof(float);
  const cudaError_t se = cudaStreamSynchronize(s);
  if (se != cudaSuccess) {
    throw std::runtime_error(std::string("flash_next real MoE D2H sync failed: ") +
                             cudaGetErrorString(se));
  }
  const cudaError_t ce =
      cudaMemcpy(router_host_.data(), logits, logits_count, cudaMemcpyDeviceToHost);
  if (ce != cudaSuccess) {
    throw std::runtime_error(std::string("flash_next real MoE D2H failed: ") +
                             cudaGetErrorString(ce));
  }
  const std::vector<std::uint32_t> selected = moe_topk_union(
      static_cast<const float*>(router_host_.data()), T,
      static_cast<int>(G::kNumExperts), static_cast<int>(G::kExpertsPerToken));
  if (timings_enabled_) {
    timer_.union_experts += static_cast<long long>(selected.size());
  }
  // Per-(layer, expert) routed-expert usage: count every distinct expert the host
  // top-k selected for this layer (each selected expert is exactly the one the
  // scatter stages into the window, so this is the real set the GPU GEMM consumes).
  for (const std::uint32_t e : selected) {
    ++expert_usage_[static_cast<std::size_t>(layer) * G::kNumExperts + e];
  }

  const RoutedHostBases& routed = view_.routed_host[layer];
  __nv_bfloat16* window = static_cast<__nv_bfloat16*>(moe_window_.p);
  // M4 tier-1: route every selected expert through the GPU LRU when it fits
  // (distinct count <= slot count). Otherwise fall through to the pinned/legacy
  // window scatter and steer the op at the window (w.expert_ptrs = null).
  const bool gpu_fits =
      expert_gpu_lru_.enabled() && selected.size() <= expert_gpu_lru_.slot_count();
  seg_begin(B_MOE_SCATTER);
  if (gpu_fits) {
    // Two-phase (see the ExpertGpuLru class doc): peek every selected expert
    // FIRST (bumping each hit's tick to the newest), THEN promote the misses
    // (each evicts a min-tick COLD slot, never a recorded hit). The op reads each
    // routed expert from its LRU slot via the per-expert device pointer table
    // (w.expert_ptrs): a hit is a zero-DMA in-place read; a miss is one
    // 2,764,800 B H2D issued on the round stream (the GEMM, same stream, waits
    // for it). NO window scatter for the selected experts — their slots are the
    // source of truth. Unselected experts (never routed, never dereferenced by
    // the kernel) point at their window slot.
    const std::uint8_t* const window_bytes =
        static_cast<const std::uint8_t*>(moe_window_.p);
    const std::uint8_t** table =
        static_cast<const std::uint8_t**>(moe_expert_ptrs_host_.data());
    for (std::uint32_t e = 0; e < G::kNumExperts; ++e) {
      table[e] = window_bytes + static_cast<std::size_t>(e) * kExpertBytes;
    }
    const std::uint8_t* where[G::kNumExperts] = {};
    for (const std::uint32_t e : selected) {
      where[e] = expert_gpu_lru_.peek(layer, e);  // slot base, or nullptr (miss)
    }
    for (const std::uint32_t e : selected) {
      const std::uint8_t* hit = where[e];
      const std::uint8_t* slot;
      if (hit != nullptr) {
        slot = hit;  // GPU-resident: zero-DMA in-place read
      } else {
        // GPU miss: consult the disjoint pinned tier first. A hit there is one
        // contiguous pinned->device DMA (~26 GB/s, ~105 us/expert) instead of
        // the pageable 4-plane H2D (~0.92 ms/expert) — the prefill win. The
        // pinned blob is laid out at the kPlane* offsets (identical slot layout),
        // so promote_contiguous copies it verbatim into the GPU slot.
        const std::uint8_t* pin =
            (expert_lru_.enabled()) ? expert_lru_.find(layer, e) : nullptr;
        if (pin != nullptr) {
          slot = expert_gpu_lru_.promote_contiguous(layer, e, pin, s);
        } else {
          slot = expert_gpu_lru_.promote(
              layer, e,
              routed.gu_codes + static_cast<std::size_t>(e) * kSrcGuCodes,
              routed.gu_scales + static_cast<std::size_t>(e) * kSrcGuScales,
              routed.dn_codes + static_cast<std::size_t>(e) * kSrcDnCodes,
              routed.dn_scales + static_cast<std::size_t>(e) * kSrcDnScales, s);
        }
      }
      where[e] = slot;
      table[e] = slot;
    }
    const cudaError_t te = cudaMemcpy(moe_expert_ptrs_dev_.p, table,
                                      G::kNumExperts * sizeof(const std::uint8_t*),
                                      cudaMemcpyHostToDevice);
    if (te != cudaSuccess) {
      throw std::runtime_error(std::string("flash_next real MoE expert-table H2D failed: ") +
                               cudaGetErrorString(te));
    }
    // NINFER_EXPERT_GPU_VERIFY: diagnostic-only self-check to localize an S5
    // divergence in the gpu_fits path into (1) device expert-table != host table,
    // (2) slot bytes != host mmap source, or (3) both match (bug is in the op's
    // read). Fires on every gpu_fits moe_block for the cheap table compare and on
    // layer 0 (once) for the expensive slot-data compare. The same-stream GEMM
    // already waits on these H2D ops, so the drain below masks no ordering.
    if (std::getenv("NINFER_EXPERT_GPU_VERIFY") != nullptr) {
      static const bool verify_armed = [] {
        const char* v = std::getenv("NINFER_EXPERT_GPU_VERIFY");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
      }();
      if (verify_armed) {
        static std::vector<const std::uint8_t*> dev_table(G::kNumExperts);
        const cudaError_t vs = cudaStreamSynchronize(s);
        if (vs == cudaSuccess &&
            cudaMemcpy(dev_table.data(), moe_expert_ptrs_dev_.p,
                       G::kNumExperts * sizeof(const std::uint8_t*),
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
          for (const std::uint32_t e : selected) {
            if (dev_table[e] != table[e]) {
              std::fprintf(stderr,
                  "  VERIFY TABLE-MISMATCH layer=%u expert=%u host=%p dev=%p\n",
                  layer, e, static_cast<const void*>(table[e]),
                  static_cast<const void*>(dev_table[e]));
            }
          }
        }
        // Per-layer slot-data check (EVERY layer, not just 0). The layer-0-only
        // check cannot split "slot data wrong at layer >= 1 (a promote/eviction
        // bug)" from "data correct everywhere but the op's expert_ptrs read path."
        // Copy a small slice of each selected expert's slot (all four planes) and
        // compare against the CURRENT layer's host mmap source. A wrong-expert or
        // wrong-layer slot shows fully-different bytes at once; the tiny slices
        // (not a full 2,764,800 B D2H) keep the cost trivial across 48 layers.
        static std::uint32_t total_data_mismatch = 0;
        static std::size_t total_data_checked = 0;
        {
          const std::size_t CSCODE = 4096;   // code slice length
          const std::size_t CSCALE = 512;    // scale slice length
          struct PlaneRef {
            std::uint32_t off; std::size_t len; const std::uint8_t* src; const char* name;
          };
          for (const std::uint32_t e : selected) {
            if (total_data_checked >= 2048) break;   // cap total experts checked
            const PlaneRef planes[4] = {
                {kPlaneGuCodes,  CSCODE, routed.gu_codes  + e * kSrcGuCodes,  "gu_codes"},
                {kPlaneGuScales, CSCALE, routed.gu_scales + e * kSrcGuScales, "gu_scales"},
                {kPlaneDnCodes,  CSCODE, routed.dn_codes  + e * kSrcDnCodes,  "dn_codes"},
                {kPlaneDnScales, CSCALE, routed.dn_scales + e * kSrcDnScales, "dn_scales"},
            };
            bool mismatched = false;
            for (const PlaneRef& pc : planes) {
              if (mismatched) break;
              std::vector<std::uint8_t> slice(pc.len);
              if (cudaMemcpy(slice.data(), table[e] + pc.off, pc.len,
                             cudaMemcpyDeviceToHost) != cudaSuccess) break;
              for (std::size_t i = 0; i < pc.len; ++i) {
                if (slice[i] != pc.src[i]) {
                  if (total_data_mismatch < 64) {
                    std::fprintf(stderr,
                        "  VERIFY DATA-MISMATCH layer=%u expert=%u plane=%s off=%zu dev=0x%02x host=0x%02x\n",
                        layer, e, pc.name, i, static_cast<unsigned>(slice[i]),
                        static_cast<unsigned>(pc.src[i]));
                  }
                  ++total_data_mismatch;
                  mismatched = true;
                  break;
                }
              }
            }
            ++total_data_checked;
          }
          // Log a per-layer marker only on a mismatch (keeps a clean run's log
          // small) so the first bad layer/expert is obvious.
          if (total_data_mismatch > 0) {
            std::fprintf(stderr,
                "  VERIFY layer=%u T=%d selected=%zu data-mismatch-so-far=%u total-checked=%zu\n",
                layer, T, selected.size(), total_data_mismatch, total_data_checked);
          }
        }
      }
    }
  } else if (expert_lru_.enabled()) {
    // M2 loop-4: hybrid scatter. Hit = ONE contiguous 2,764,800 B pinned->device
    // DMA (no page faults, ~26 GB/s measured warm). Miss = the M1 4-plane
    // pageable DMAs (the driver's staging faults the mmap pages in at the M1
    // rate) followed by the pinned promotion — POST-staging, so the promotion
    // memcpys read pages the staging just made resident (RAM speed) instead of
    // faulting them at disk rate as the pre-miss staging of loops 2-3 did.
    for (const std::uint32_t e : selected) {
      const std::size_t dst_base = static_cast<std::size_t>(e) * kExpertBytes;
      const std::uint8_t* src = expert_lru_.find(layer, e);
      if (src != nullptr) {
        const cudaError_t ec =
            cudaMemcpyAsync(window + dst_base / 2, src, kExpertBytes,
                            cudaMemcpyHostToDevice, s);
        if (ec != cudaSuccess) {
          throw std::runtime_error(std::string("flash_next real MoE scatter H2D failed: ") +
                                   cudaGetErrorString(ec));
        }
        continue;
      }
      const cudaError_t e1 = cudaMemcpyAsync(
          window + (dst_base + kPlaneGuCodes) / 2, routed.gu_codes + static_cast<std::size_t>(e) * kSrcGuCodes,
          kSrcGuCodes, cudaMemcpyHostToDevice, s);
      const cudaError_t e2 = cudaMemcpyAsync(
          window + (dst_base + kPlaneGuScales) / 2, routed.gu_scales + static_cast<std::size_t>(e) * kSrcGuScales,
          kSrcGuScales, cudaMemcpyHostToDevice, s);
      const cudaError_t e3 = cudaMemcpyAsync(
          window + (dst_base + kPlaneDnCodes) / 2, routed.dn_codes + static_cast<std::size_t>(e) * kSrcDnCodes,
          kSrcDnCodes, cudaMemcpyHostToDevice, s);
      const cudaError_t e4 = cudaMemcpyAsync(
          window + (dst_base + kPlaneDnScales) / 2, routed.dn_scales + static_cast<std::size_t>(e) * kSrcDnScales,
          kSrcDnScales, cudaMemcpyHostToDevice, s);
      if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess || e4 != cudaSuccess) {
        throw std::runtime_error(std::string("flash_next real MoE scatter H2D failed: ") +
                                 cudaGetErrorString(e1 != cudaSuccess ? e1 :
                                                     e2 != cudaSuccess ? e2 :
                                                     e3 != cudaSuccess ? e3 : e4));
      }
      expert_lru_.promote(
          layer, e,
          routed.gu_codes + static_cast<std::size_t>(e) * kSrcGuCodes, kSrcGuCodes,
          routed.gu_scales + static_cast<std::size_t>(e) * kSrcGuScales, kSrcGuScales,
          routed.dn_codes + static_cast<std::size_t>(e) * kSrcDnCodes, kSrcDnCodes,
          routed.dn_scales + static_cast<std::size_t>(e) * kSrcDnScales, kSrcDnScales);
    }
  } else {
    // Legacy pageable path (NINFER_EXPERT_PINNED_SLOTS=0): the M1 behavior.
    for (const std::uint32_t e : selected) {
      const std::size_t dst_base =
          static_cast<std::size_t>(e) * kExpertBytes;
      const cudaError_t e1 = cudaMemcpyAsync(
          window + (dst_base + kPlaneGuCodes) / 2, routed.gu_codes + static_cast<std::size_t>(e) * kSrcGuCodes,
          kSrcGuCodes, cudaMemcpyHostToDevice, s);
      const cudaError_t e2 = cudaMemcpyAsync(
          window + (dst_base + kPlaneGuScales) / 2, routed.gu_scales + static_cast<std::size_t>(e) * kSrcGuScales,
          kSrcGuScales, cudaMemcpyHostToDevice, s);
      const cudaError_t e3 = cudaMemcpyAsync(
          window + (dst_base + kPlaneDnCodes) / 2, routed.dn_codes + static_cast<std::size_t>(e) * kSrcDnCodes,
          kSrcDnCodes, cudaMemcpyHostToDevice, s);
      const cudaError_t e4 = cudaMemcpyAsync(
          window + (dst_base + kPlaneDnScales) / 2, routed.dn_scales + static_cast<std::size_t>(e) * kSrcDnScales,
          kSrcDnScales, cudaMemcpyHostToDevice, s);
      if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess || e4 != cudaSuccess) {
        throw std::runtime_error(std::string("flash_next real MoE scatter H2D failed: ") +
                                 cudaGetErrorString(e1 != cudaSuccess ? e1 :
                                                     e2 != cudaSuccess ? e2 :
                                                     e3 != cudaSuccess ? e3 : e4));
      }
    }
  }
  seg_end();
  barrier_end();

  seg_begin(B_MOE_GEMM);
  ops::SparseMoeNvfp4Weights w;
  w.router_bf16 = m.router_bf16;
  w.shared_gate_bf16 = m.shared_gate_bf16;
  w.shared_gu_codes = m.shared_gu_codes;
  w.shared_gu_scales = m.shared_gu_scales;
  w.shared_dn_codes = m.shared_dn_codes;
  w.shared_dn_scales = m.shared_dn_scales;
  w.shared_gu_divisor = m.shared_gu_divisor;
  w.shared_dn_divisor = m.shared_dn_divisor;
  w.expert_window = reinterpret_cast<const std::uint8_t*>(window);  // the shared device window (m.expert_window is null for S1)
  // Tier-1 (M4): the op reads each routed expert from the per-expert device pointer
  // table when the GPU LRU scatter ran this round (gpu_fits); null otherwise = the
  // pure window path (byte-identical to the pre-M4 behavior, the P8-oracle path).
  w.expert_ptrs = gpu_fits ? static_cast<const std::uint8_t* const*>(moe_expert_ptrs_dev_.p)
                           : nullptr;
  w.routed_gu_divisor = m.routed_gu_divisor;
  w.routed_dn_divisor = m.routed_dn_divisor;

  // VERIFY diagnostic: expose the device top-K ids + router logits so we can compare
  // them against the host top-K (selected) and host router (router_host_). This
  // localizes the S5 gpu_fits divergence: the op's GEMM reads the DEVICE dids, so a
  // device-selected expert that is NOT in host-selected reads an un-populated window
  // slot (garbage) in the gpu_fits path.
  std::vector<std::int32_t> diag_ids;
  std::vector<float> diag_logits;
  const bool gpu_verify_armed = [] {
    const char* v = std::getenv("NINFER_EXPERT_GPU_VERIFY");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  if (gpu_verify_armed) {
    diag_ids.resize(static_cast<std::size_t>(T) * G::kExpertsPerToken);
    diag_logits.resize(static_cast<std::size_t>(T) * G::kNumExperts);
    w.diag_ids_out = diag_ids.data();
    w.diag_logits_out = diag_logits.data();
  }

  auto scope = workspace_.scope();
  const DeviceSpan mws = workspace_.alloc_bytes(
      ops::sparse_moe_nvfp4_workspace_capacity_bytes(kMoeGeometry, 1, kMaxRoundTokens));
  WorkspaceArena leaf(mws);
  // The op is token-major in AND out (x[t*K+k] / dest[t*K+k]): feed it xhat directly
  // and take the output straight into the residual. (A cm sandwich here scrambled a
  // token-major op and made the op's dids diverge from the host `selected`.)
  Tensor moe_x = make_tensor(xhat, DType::BF16, {static_cast<int>(G::kHidden), T});
  Tensor moe_dst = make_tensor(y, DType::BF16, {static_cast<int>(G::kHidden), T});
  ops::sparse_moe_nvfp4(moe_x, kMoeGeometry, w, ops::SparseMoeNvfp4Route::W4A16,
                        ops::SparseMoeNvfp4Epilogue::Identity, nullptr, moe_dst, leaf, s);

  if (gpu_verify_armed) {
    // Compare the device top-K (diag_ids) against the host top-K (selected). Any
    // device-selected expert not in host-selected reads an un-populated slot in the
    // gpu_fits path. Also report max |device-logit - host-logit| over the [T*E]
    // overlap (both row-major t*E+r) -- the FMA divergence that flips a near-tie.
    const float* host_logits = static_cast<const float*>(router_host_.data());
    const std::size_t idn = diag_ids.size();
    std::vector<std::uint8_t> sel_mark(G::kNumExperts, 0);
    for (const std::uint32_t e : selected) sel_mark[e] = 1;
    std::size_t device_unique = 0;
    for (std::size_t i = 0; i < idn; ++i) {
      const std::uint32_t e = static_cast<std::uint32_t>(diag_ids[i]);
      if (e < G::kNumExperts && sel_mark[e] == 0) {
        if (device_unique < 32) {
          std::fprintf(stderr,
              "  VERIFY ID-UNIQUE layer=%u T=%d idx=%zu expert=%u (tok=%zu slot=%zu) "
              "sel=%zu gpu_fits=%d\n",
              layer, T, i, e, i / G::kExpertsPerToken, i % G::kExpertsPerToken,
              selected.size(), gpu_fits ? 1 : 0);
        }
        ++device_unique;
      }
    }
    float max_logit_diff = 0.0F;
    const std::size_t ln = diag_logits.size();
    for (std::size_t i = 0; i < ln; ++i) {
      const float d = diag_logits[i] - host_logits[i];
      const float a = d < 0 ? -d : d;
      if (a > max_logit_diff) max_logit_diff = a;
    }
    std::fprintf(stderr,
        "  VERIFY IDS layer=%u T=%d sel=%zu dev_uniq=%zu max|dlogit|=%.9g gpu_fits=%d\n",
        layer, T, selected.size(), device_unique,
        static_cast<double>(max_logit_diff), gpu_fits ? 1 : 0);
  }

  seg_end();
}

void RealProgram::reset_recurrent_state(cudaStream_t s) {
  auto zero = [&](const std::vector<DeviceBuffer>& buffers, const char* what) {
    for (const auto& b : buffers) {
      const cudaError_t e = cudaMemsetAsync(b.p, 0, b.bytes, s);
      if (e != cudaSuccess) {
        throw std::runtime_error(std::string("flash_next real ") + what +
                                 " reset failed: " + cudaGetErrorString(e));
      }
    }
  };
  zero(gdn_states_, "GDN state");
  zero(conv_states_, "GDN conv state");
  zero(k_pools_, "QSA K pool");
  zero(v_pools_, "QSA V pool");
  zero(idx_k_pools_, "QSA indexer-K pool");
  zero(idx_k_scale_, "QSA indexer-K scale");
}

std::uint32_t RealProgram::run_round(const std::vector<std::uint32_t>& tokens, int pos0,
                                     std::vector<float>* final_logits) {
  const int T = static_cast<int>(tokens.size());
  if (T < 1) {
    throw std::runtime_error("flash_next real Program: empty round");
  }
  if (T > kMaxRoundTokens) {
    throw std::runtime_error("flash_next real Program: round exceeds the arena size");
  }
  {
    char hb[64];
    std::snprintf(hb, sizeof(hb), "RR in  T=%d pos0=%d", T, pos0);
    cont_hb(hb);
  }
  const cudaStream_t s = stream_;
  // State-carrying round: the GDN / conv / QSA-KV / PLE state accumulates across
  // calls (no reset here; run_sequence and begin_sequence reset before a fresh
  // sequence). pos0 is the absolute start position of this round's tokens.
  auto round_scope = workspace_.scope();

  // Working-set allocations (shared by every layer).
  std::int32_t* ids_dev =
      static_cast<std::int32_t*>(workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4).data);
  __nv_bfloat16* h =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * G::kHidden).data);
  float* a0 =
      static_cast<float*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4 * G::kHcCount * G::kHidden).data);
  float* a1 =
      static_cast<float*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4 * G::kHcCount * G::kHidden).data);
  float* xhat_f32 =
      static_cast<float*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4 * G::kHidden).data);
  __nv_bfloat16* xhat_bf16 =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * G::kHidden).data);
  __nv_bfloat16* y =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * G::kHidden).data);
  float* block_out =
      static_cast<float*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4 * G::kHidden).data);
  float* ple_mem =
      static_cast<float*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4 * G::kHidden).data);
  const float* zero_block_f32 = static_cast<const float*>(
      workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4 * G::kHidden).data);
  seg_begin(B_EMBED);
  cudaMemsetAsync(const_cast<float*>(zero_block_f32), 0,
                  static_cast<std::size_t>(T) * 4 * G::kHidden, s);

  // Embedding.
  const std::int32_t* ids_i32 = reinterpret_cast<const std::int32_t*>(tokens.data());
  std::memcpy(ids_host_.data(), ids_i32, static_cast<std::size_t>(T) * sizeof(std::int32_t));
  const cudaError_t h2d =
      cudaMemcpyAsync(ids_dev, ids_host_.data(),
                      static_cast<std::size_t>(T) * sizeof(std::int32_t),
                      cudaMemcpyHostToDevice, s);
  if (h2d != cudaSuccess) {
    throw std::runtime_error(std::string("flash_next real embedding H2D failed: ") +
                             cudaGetErrorString(h2d));
  }
  detail::real_embedding_gather(ids_dev, view_.embedding.codes, view_.embedding.scale, T,
                                G::kHidden, h, s);

  // Stream init: the embedding is replicated into all 4 branches.
  detail::real_stream_init(h, T, G::kHidden, static_cast<int>(G::kHcCount), a0, s);
  seg_end();
  { nan_probe_bf16("embed.h", h, (std::size_t)T * G::kHidden, s); }
  { nan_probe_f32("init.a0", a0, (std::size_t)T * G::kHcCount * G::kHidden, s); }

  {
    char hb[64];
    std::snprintf(hb, sizeof(hb), "RR preloop T=%d pos0=%d", T, pos0);
    cont_hb(hb);
  }
  for (std::uint32_t layer = 0; layer < G::kNumLayers; ++layer) {
    const LayerWeights& lw = view_.layers[layer];
    seg_trace_layer_ = (int)layer;
    {
      char hb[64];
      std::snprintf(hb, sizeof(hb), "RR L%u T=%d pos0=%d", layer, T, pos0);
      cont_hb(hb);
    }

    // HC read (attention): xhat from A0; A1 <- A0 (+ sigma * 0).
    seg_begin(B_HC);
    {
      ops::GatedResidualParams p{};
      p.tokens = T;
      p.hidden = G::kHidden;
      p.hc_count = G::kHcCount;
      p.lowrank = G::kHcLowRank;
      p.stream_dev = a0;
      p.block_out_dev = zero_block_f32;
      p.w_down_dev = lw.attn_hc.w_down;
      p.w_up_dev = lw.attn_hc.w_up;
      p.w_inj_dev = lw.attn_hc.w_inj;
      p.norm_w_dev = lw.attn_hc.norm_w;
      p.read_out_dev = xhat_f32;
      p.stream_out_dev = a1;
      p.norm_branches = true;
      ops::gated_residual(p, s);
    }
    seg_end();
    seg_begin(B_CONVERT);
    detail::mini_f32_to_bf16(xhat_f32, T, G::kHidden, xhat_bf16, s);
    seg_end();

    if (layer == G::kPleLayer) {
      seg_begin(B_PLE);
      ops::NgramOpParams np{};
      np.conv_kernel = static_cast<int>(G::kGdnConvWidth);
      np.dilation = 0;
      np.conv_weights = view_.ple_conv_weights.data();
      np.staging_cap_bytes = 0;
      const ops::NgramEmbeddingTable& table = *view_.ple_table;
      ops::ngram_embedding(table,
                           std::span<const std::uint32_t>(tokens.data(),
                                                          static_cast<std::size_t>(T)),
                           np, ple_state_, ple_mem, s);
      detail::real_ple_add(xhat_bf16, ple_mem, T, G::kHidden, G::kHidden, s);
      seg_end();
    }

    if (detail::is_full_attention_layer(layer)) {
      seg_begin(B_QSA);
      qsa_block(layer, T, pos0, xhat_bf16, y);
      seg_end();
    } else {
      seg_begin(B_GDN);
      gdn_block(layer, T, xhat_bf16, y);
      seg_end();
    }
    { char tag[32]; std::snprintf(tag, sizeof(tag), "L%u.attn", layer);
      nan_probe_bf16(tag, y, (std::size_t)T * G::kHidden, s);
      layer_probe_dump(tag, pos0, y, T, G::kHidden, s); }

    seg_begin(B_BLOCKOUT);
    detail::real_block_out(y, T, G::kHidden, block_out, s);
    seg_end();
    // HC write (attention): A1 <- A0 + sigma * y.
    seg_begin(B_HC);
    {
      ops::GatedResidualParams p{};
      p.tokens = T;
      p.hidden = G::kHidden;
      p.hc_count = G::kHcCount;
      p.lowrank = G::kHcLowRank;
      p.stream_dev = a0;
      p.block_out_dev = block_out;
      p.w_down_dev = lw.attn_hc.w_down;
      p.w_up_dev = lw.attn_hc.w_up;
      p.w_inj_dev = lw.attn_hc.w_inj;
      p.norm_w_dev = lw.attn_hc.norm_w;
      p.read_out_dev = xhat_f32;  // overwritten; unused by the write
      p.stream_out_dev = a1;
      p.norm_branches = true;
      ops::gated_residual(p, s);
    }
    seg_end();

    // HC read (mlp): xhat from A1; A0 <- A1.
    seg_begin(B_HC);
    {
      ops::GatedResidualParams p{};
      p.tokens = T;
      p.hidden = G::kHidden;
      p.hc_count = G::kHcCount;
      p.lowrank = G::kHcLowRank;
      p.stream_dev = a1;
      p.block_out_dev = zero_block_f32;
      p.w_down_dev = lw.mlp_hc.w_down;
      p.w_up_dev = lw.mlp_hc.w_up;
      p.w_inj_dev = lw.mlp_hc.w_inj;
      p.norm_w_dev = lw.mlp_hc.norm_w;
      p.read_out_dev = xhat_f32;
      p.stream_out_dev = a0;
      p.norm_branches = true;
      ops::gated_residual(p, s);
    }
    seg_end();
    seg_begin(B_CONVERT);
    detail::mini_f32_to_bf16(xhat_f32, T, G::kHidden, xhat_bf16, s);
    seg_end();

    moe_block(layer, T, xhat_bf16, y);
    { char tag[32]; std::snprintf(tag, sizeof(tag), "L%u.moe", layer);
      nan_probe_bf16(tag, y, (std::size_t)T * G::kHidden, s);
      layer_probe_dump(tag, pos0, y, T, G::kHidden, s); }

    seg_begin(B_BLOCKOUT);
    detail::real_block_out(y, T, G::kHidden, block_out, s);
    seg_end();
    // HC write (mlp): A0 <- A1 + sigma * y.
    seg_begin(B_HC);
    {
      ops::GatedResidualParams p{};
      p.tokens = T;
      p.hidden = G::kHidden;
      p.hc_count = G::kHcCount;
      p.lowrank = G::kHcLowRank;
      p.stream_dev = a1;
      p.block_out_dev = block_out;
      p.w_down_dev = lw.mlp_hc.w_down;
      p.w_up_dev = lw.mlp_hc.w_up;
      p.w_inj_dev = lw.mlp_hc.w_inj;
      p.norm_w_dev = lw.mlp_hc.norm_w;
      p.read_out_dev = xhat_f32;
      p.stream_out_dev = a0;
      p.norm_branches = true;
      ops::gated_residual(p, s);
    }
    seg_end();
    { char tag[32]; std::snprintf(tag, sizeof(tag), "L%u.out", layer);
      nan_probe_f32(tag, a0, (std::size_t)T * G::kHcCount * G::kHidden, s); }
  }

  seg_begin(B_LMHEAD);
  // Final collapse -> lm_head -> argmax -> sampled token.
  __nv_bfloat16* h_final =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * G::kHidden).data);
  detail::real_final_collapse(a0, T, G::kHidden, static_cast<int>(G::kHcCount), h_final, s);
  __nv_bfloat16* logits =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * G::kVocab).data);
  detail::mini_fp8_linear(h_final, T, G::kHidden, view_.output_head.codes,
                          view_.output_head.scale, G::kVocab, logits, s);
  // logits is [248320][T] (channel-major: the fp8 GEMM writes out[n*T + t]); the
  // argmax op reads [T][248320], so transpose first.
  __nv_bfloat16* logits_cm =
      static_cast<__nv_bfloat16*>(
          workspace_.alloc_bytes(static_cast<std::size_t>(T) * 2 * G::kVocab).data);
  detail::mini_transpose_cm_tm(logits, T, G::kVocab, logits_cm, s);
  std::int32_t* sample_dev =
      static_cast<std::int32_t*>(workspace_.alloc_bytes(static_cast<std::size_t>(T) * 4).data);
  Tensor logits_t = make_tensor(logits_cm, DType::BF16, {static_cast<int>(G::kVocab), T});
  Tensor sample_t = make_tensor(sample_dev, DType::I32, {T});
  ops::argmax(logits_t, sample_t, static_cast<int>(G::kVocab), s);
  seg_end();

  const cudaError_t se = cudaStreamSynchronize(s);
  if (se != cudaSuccess) {
    throw std::runtime_error(std::string("flash_next real sample D2H sync failed: ") +
                             cudaGetErrorString(se));
  }
  // P13 M1: every event recorded this round has completed at this sync; read
  // the pair durations into timings_ (no-op unless timing mode is on).
  finalize_round();
  const cudaError_t ce = cudaMemcpy(sample_host_.data(), sample_dev,
                                    static_cast<std::size_t>(T) * sizeof(std::int32_t),
                                    cudaMemcpyDeviceToHost);
  if (ce != cudaSuccess) {
    throw std::runtime_error(std::string("flash_next real sample D2H failed: ") +
                             cudaGetErrorString(ce));
  }
  // S4: D2H the final position's full-vocab lm_head logits (logits_cm row T-1,
  // contiguous BF16) into host floats. The end-to-end logit scale is the MoE-
  // health signal: the S0 ~1e21 MoE epilogue defect (if present at the real 512-
  // expert geometry) propagates through the 48 real MoE layers to O(1e21) here;
  // a healthy real stack reads ~O(10), the P8-oracle scale.
  if (final_logits != nullptr) {
    const std::size_t nbf = static_cast<std::size_t>(G::kVocab) * 2;
    const __nv_bfloat16* row = logits_cm + static_cast<std::size_t>(T - 1) * G::kVocab;
    const cudaError_t le =
        cudaMemcpy(logits_host_.data(), row, nbf, cudaMemcpyDeviceToHost);
    if (le != cudaSuccess) {
      throw std::runtime_error(std::string("flash_next real logits D2H failed: ") +
                               cudaGetErrorString(le));
    }
    final_logits->assign(G::kVocab, 0.0F);
    const __nv_bfloat16* src = static_cast<const __nv_bfloat16*>(logits_host_.data());
    for (std::size_t n = 0; n < G::kVocab; ++n) {
      (*final_logits)[n] = __bfloat162float(src[n]);
    }
  }
  const std::int32_t* samples = static_cast<const std::int32_t*>(sample_host_.data());
  {
    char hb[64];
    std::snprintf(hb, sizeof(hb), "RR out  T=%d pos0=%d tok=%d", T, pos0,
                  static_cast<int>(samples[T - 1]));
    cont_hb(hb);
  }
  return static_cast<std::uint32_t>(samples[T - 1]);
}

std::uint32_t RealProgram::run_sequence(const std::vector<std::uint32_t>& input_ids,
                                        std::vector<float>* final_logits) {
  // Bit-identical to the pre-M3 run_sequence: reset the recurrent state (a fresh
  // pos0 = 0 forward) then run a single round at pos0 = 0. ple_state_ is NOT
  // reset here — it carries across calls (the S5 re-prefill greedy and the P12
  // S5 bit-determinism golden rely on that).
  const cudaStream_t s = stream_;
  seg_begin(B_STATE_RESET);
  reset_recurrent_state(s);
  seg_end();
  return run_round(input_ids, 0, final_logits);
}

void RealProgram::begin_sequence() {
  // M3 benchmark entry: reset the recurrent state AND start a fresh PLE. The
  // PLE's prev1/prev2 + conv_history are per-sequence, so a new sequence zeroes
  // them; run_round then accumulates state across its state-carrying rounds.
  reset_recurrent_state(stream_);
  ple_state_ = ops::NgramRequestState{};
}

}  // namespace ninfer::targets::qwen3_8_flash_next
