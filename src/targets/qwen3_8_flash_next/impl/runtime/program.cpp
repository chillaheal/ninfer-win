// Flash-Next P9 — v1 Program implementation (the family's sole physical
// execution entry). Eager only: no CUDA graph capture in v1.
//
// Every block-math convention here is the documented v1 contract of
// out/flash_next_dev/P9_design.md; the production modeling function is
// unrecoverable and real weights are not loaded until P12. v1-unreachable
// engine seams throw std::logic_error, or return InvariantMismatch where the
// signature is noexcept.

#include "targets/qwen3_8_flash_next/impl/runtime/family.h"
#include "targets/qwen3_8_flash_next/impl/runtime/model_view.h"
#include "targets/qwen3_8_flash_next/impl/runtime/real_program.h"
#include "targets/qwen3_8_flash_next/impl/load/real_loader.h"
#include "targets/qwen3_8_flash_next/impl/runtime/leaves/mini_leaves.h"
#include "targets/qwen3_8_flash_next/paging/pager.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/ngram_embedding.h"
#include "ninfer/ops/sparse_moe_nvfp4.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {

namespace {

// FNV-1a 64 (prefix digests + the v1 identity assessment digest).
constexpr std::uint64_t kFNV64Basis = 1469598103934665603ULL;
constexpr std::uint64_t kFNV64Mult = 1099511628211ULL;

std::uint64_t fnv1a64(std::uint64_t digest, std::uint64_t value) noexcept {
  digest ^= value;
  digest *= kFNV64Mult;
  return digest;
}

// 1/sqrt(128) -- the GDN (gated delta net) attention scale; the shared op
// hard-checks scale == 1/sqrt(state_dim=128) (gated_delta_net.cpp require_scale).
// Literal because MSVC /std:c++20 rejects constexpr std::sqrt.
constexpr float kGdnScale = 0.08838834764831843F;

constexpr std::size_t kGdnStateBytes = 128U * 128U * 4U * sizeof(float);  // 262,144
constexpr std::size_t kConvStateBytes = 256U * 3U * sizeof(std::uint16_t);  // 1,536
constexpr std::size_t kQsaKvPageBytes = 16384;
constexpr std::size_t kQsaKvPageElements = kQsaKvPageBytes / sizeof(std::uint16_t);  // 8,192
constexpr std::size_t kQsaVPlaneOffset = kQsaKvPageElements / 2;  // 4,096
constexpr std::size_t kPersistentBytesPerSeq =
    static_cast<std::size_t>(mini::kGdnLayers) * kGdnStateBytes +
    static_cast<std::size_t>(mini::kGdnLayers) * kConvStateBytes;  // 791,040
constexpr std::size_t kZeroBlockBytes = 64U * 256U * sizeof(float);  // 65,536
constexpr std::uint32_t kPrefixDigestLimit = 8;
constexpr std::uint32_t kIdentityTag = 1;
constexpr std::size_t kPrefixStoreCap = 16;
constexpr int kMaxRoundTokens = 64;

constexpr ops::SparseMoeNvfp4Geometry kMoeGeometry{8, 2, 256, 64};

std::uint64_t prefix_digest(const std::vector<TokenId>& ids, std::uint32_t count) {
  std::uint64_t d = kFNV64Basis;
  for (std::uint32_t i = 0; i < count; ++i) {
    d = fnv1a64(d, static_cast<std::uint32_t>(ids[i]));
  }
  return d;
}

// Largest-round working set (T tokens), every buffer a round carves from the
// scratch region, before alignment. The construction-time dry-run adds the
// GDN and MoE op workspaces on top.
std::size_t round_working_set_bytes(int T) {
  std::size_t b = 0;
  b += static_cast<std::size_t>(T) * 4;                   // token ids I32
  b += static_cast<std::size_t>(T) * 2 * 256;             // h BF16
  b += static_cast<std::size_t>(T) * 4 * 1024 * 2;        // A0 / A1 FP32
  b += static_cast<std::size_t>(T) * 4 * 256;             // xhat_f32
  b += static_cast<std::size_t>(T) * 2 * 256;             // xhat_bf16
  b += static_cast<std::size_t>(T) * 2 * 256;             // y
  b += static_cast<std::size_t>(T) * 4 * 256;             // block_out
  b += static_cast<std::size_t>(T) * 4 * 128;             // PLE mem
  b += static_cast<std::size_t>(T) * 4 * mini::kMoEExperts;  // router logits
  b += static_cast<std::size_t>(T) * 2 * 256 * 2;         // MoE x_cm / out_cm
  b += static_cast<std::size_t>(T) * 2 * 256 * 2;         // logits / logits_cm
  b += static_cast<std::size_t>(T) * 4;                   // sample I32
  b += static_cast<std::size_t>(T) * 2 * 384;             // qkvz
  b += static_cast<std::size_t>(T) * 2 * 256 * 2;         // conv in / out
  b += static_cast<std::size_t>(T) * 4 * mini::kGdnValueHeads * 2;  // g / beta
  b += static_cast<std::size_t>(T) * 2 * (128 * 2 * 2 + 128 * mini::kGdnValueHeads);  // q/k/v op
  b += static_cast<std::size_t>(T) * 2 * 128 * mini::kGdnValueHeads;  // o_op
  b += static_cast<std::size_t>(T) * 2 * 128;             // gdn z slice
  b += static_cast<std::size_t>(T) * 2 * 128;             // onx
  b += static_cast<std::size_t>(T) * 2 * 256;             // q_full
  b += static_cast<std::size_t>(T) * 2 * 64 * 2;          // k / v
  b += static_cast<std::size_t>(T) * 4 * 160;             // indexer z
  b += static_cast<std::size_t>(T) * 4 * 64;              // indexer scores
  b += static_cast<std::size_t>(T) * 4 * 64;              // indexer ids
  b += static_cast<std::size_t>(T) * 4;                   // indexer counts
  b += static_cast<std::size_t>(T) * 2 * 128;             // qsa out
  return b;
}

// sparse_moe_nvfp4's (value DESC, expert id ASC) top-K rule. Exact ties at the
// K-th boundary resolve to the LOWER expert id — the total order guarantees it.
std::vector<std::uint32_t> moe_top2(const float* logits) {
  std::vector<std::uint32_t> order(mini::kMoEExperts);
  for (std::uint32_t e = 0; e < mini::kMoEExperts; ++e) {
    order[e] = e;
  }
  std::stable_sort(order.begin(), order.end(),
                   [logits](std::uint32_t a, std::uint32_t b) {
                     if (logits[a] != logits[b]) {
                       return logits[a] > logits[b];
                     }
                     return a < b;
                   });
  return {order[0], order[1]};
}

Tensor make_tensor(const void* data, DType dtype, std::initializer_list<std::int32_t> shape) {
  return Tensor(const_cast<void*>(data), dtype, shape);
}

// ---------------------------------------------------------------------------
// S0 root-cause instrumentation (env-gated, numerics UNCHANGED — the P9 golden
// still passes). When NINFER_FX_TRACE=<file> is set, each of the first
// NINFER_FX_TRACE_ROUNDS rounds (default 2) logs the max-abs of every device
// boundary buffer to <file>, localizing the carried ~1e21 logit scale.
// ---------------------------------------------------------------------------
std::string g_fx_trace_path;
int g_fx_trace_rounds_left = 2;
bool g_fx_trace_this_round = false;

void fx_trace(const char* label, const void* dev, std::size_t n, bool bf16, cudaStream_t s) {
  if (g_fx_trace_path.empty() || !g_fx_trace_this_round || n == 0) {
    return;
  }
  std::vector<std::uint16_t> u;
  std::vector<float> f;
  if (bf16) {
    u.resize(n);
    const cudaError_t se = cudaStreamSynchronize(s);
    if (se != cudaSuccess) {
      return;
    }
    const cudaError_t ce = cudaMemcpy(u.data(), dev, n * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) {
      return;
    }
    f.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t bits = static_cast<std::uint32_t>(u[i]) << 16;
      float fv;
      std::memcpy(&fv, &bits, sizeof(fv));
      f[i] = fv;
    }
  } else {
    f.resize(n);
    const cudaError_t se = cudaStreamSynchronize(s);
    if (se != cudaSuccess) {
      return;
    }
    const cudaError_t ce =
        cudaMemcpy(f.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) {
      return;
    }
  }
  double mx = 0.0;
  for (float v : f) {
    const double a = std::fabs(static_cast<double>(v));
    if (a > mx) {
      mx = a;
    }
  }
  static std::ofstream os(g_fx_trace_path, std::ios::app);
  os << "[fx] " << label << " n=" << n << " maxabs=" << mx << "\n";
  os.flush();
}

}  // namespace

namespace detail {

struct ProgramImpl {
  struct Lane {
    bool active = false;
    std::uint64_t epoch = 0;
    SequenceHandle handle{};
    std::vector<TokenId> prompt_ids;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    FinishReason effective_limit_reason = FinishReason::None;
    std::uint32_t emitted = 0;
    TokenId last_token = 0;
    std::uint32_t frontier = 0;  // absolute position of the next token
    std::vector<std::int32_t> pages;  // pool page indices, allocation order

    // Persistent device state (workspace arena, stable offsets).
    float* gdn_state[mini::kGdnLayers] = {};
    __nv_bfloat16* conv_state[mini::kGdnLayers] = {};
    std::int32_t* block_table_dev = nullptr;
    ops::NgramRequestState ple_state;
  };

  struct StoredPrefix {
    std::array<std::vector<std::uint8_t>, mini::kGdnLayers> gdn_states;
    std::array<std::vector<std::uint8_t>, mini::kGdnLayers> conv_states;
    std::vector<std::uint8_t> kv_pages;
    TokenId first_token = 0;
    std::uint32_t frontier = 0;
    ops::NgramRequestState ple;
  };

  ModelView model;
  int device_ordinal = 0;
  std::uint32_t max_context = 0;
  std::uint32_t max_concurrency = 0;
  std::uint32_t kv_capacity = 0;
  std::size_t kv_total_pages = 0;
  std::size_t per_lane_pages = 0;

  DeviceArena workspace;            // persistent carve-out + round scratch
  DeviceBuffer kv_pool;             // 16,384 B pages
  std::vector<std::int32_t> free_pages;
  DeviceBuffer block_tables;        // per_lane_pages x 4 B per lane
  DeviceBuffer zero_block;          // [64,256] FP32, zeroed once
  PinnedHostBuffer router_host;     // 64 x 8 floats
  PinnedHostBuffer sample_host;     // 64 ints
  PinnedHostBuffer ids_host;        // 64 ints
  std::vector<TokenId> pending_tokens;  // host source of PendingBatch tokens
  std::vector<Lane> lanes;

  // M2 real mode: the proven 48x512 RealProgram (self-contained GDN/conv/QSA-KV/MoE
  // state) + its RealLoadedModel. In real mode the mini device buffers above are
  // size-1 sentinels (never touched) and advance_prefill / decode / ensure_pages /
  // store_prefix / restore_prefix dispatch to real_. Single-lane only (one recurrent
  // state set). Default is mini mode (real_mode_ false). real_model_ is declared
  // first so it is destroyed LAST: RealProgram holds a const RealModelView& into it.
  bool real_mode_ = false;
  std::unique_ptr<RealLoadedModel> real_model_;
  std::unique_ptr<RealProgram> real_;

  // Per-transaction bookkeeping (v1: a single active transaction).
  bool txn_active = false;
  runtime::LaneId pending_lane_{};
  std::uint32_t txn_lane = 0;
  std::uint32_t txn_reusable = 0;
  bool txn_reuse = false;

  std::unordered_map<std::uint64_t, StoredPrefix> prefix_store;
  std::vector<std::uint64_t> prefix_order;  // oldest first (eviction)

  // Lever B (on-device prefix continuation, real mode only). The RealProgram's
  // recurrent state (GDN / conv / QSA K-V / indexer-K / PLE) is fully positional and
  // accumulates across run_round (real_program.cpp); it is zeroed ONLY by
  // begin_sequence (reset_recurrent_state + ple_state_ = {}). cont_frontier_ is the
  // length of the last processed request's token sequence (prompt + response), and
  // cont_digest_ = the FNV-1a over it (prefix_digest at that length), updated
  // incrementally as decode appends response tokens. A new request whose prompt
  // shares that exact prefix (prefix_digest(prompt, cont_frontier_) == cont_digest_)
  // skips begin_sequence (the shared prefix's recurrent state is retained on-device)
  // and prefills only the delta [cont_frontier_, P) at pos0 = cont_frontier_;
  // fresh begin_sequence + full prefill runs. No host snapshot / new
  // VRAM. Single-lane, append-only (the Anthropic Messages API re-sends the whole
  // conversation each turn, so a continued request's prompt is the prior sequence
  // plus a small delta).
  std::uint32_t cont_frontier_ = 0;
  std::uint64_t cont_digest_ = 0;
  bool cont_valid_ = false;

  cudaStream_t stream = nullptr;

  // P9 test-only logits-capture seam. Gated by the NINFER_FLASH_NEXT_LOGITS
  // environment variable (path); when set, each run_round appends the 256
  // BF16->FP32 logits of the last position, and ~ProgramImpl flushes them to
  // the path. Off by default; production never sets the variable.
  bool capture_logits_ = false;
  std::string logits_capture_path_;
  std::vector<float> logits_capture_;
  // (per-round debug tag moved to a process-global static in run_round)

  void flush_logits_capture() {
    if (logits_capture_.empty()) {
      return;
    }
    const std::uint32_t rows = static_cast<std::uint32_t>(logits_capture_.size() / 256U);
    std::array<std::uint8_t, 12> header{};
    std::memcpy(header.data(), "P9LOGITS", 8);
    std::memcpy(header.data() + 8, &rows, 4);  // x86 little-endian
    std::ofstream os(logits_capture_path_, std::ios::binary | std::ios::trunc);
    if (!os) {
      return;
    }
    os.write(reinterpret_cast<const char*>(header.data()), 12);
    os.write(reinterpret_cast<const char*>(logits_capture_.data()),
             static_cast<std::streamsize>(logits_capture_.size() * sizeof(float)));
  }

  ~ProgramImpl() {
    if (capture_logits_) {
      flush_logits_capture();
    }
  }

  // The device arena/buffer members are non-default-constructible, so the
  // only constructor takes them (sized by create_program) in declaration order.
  explicit ProgramImpl(DeviceArena workspace_, DeviceBuffer kv_pool_,
                       DeviceBuffer block_tables_, DeviceBuffer zero_block_,
                       PinnedHostBuffer router_host_, PinnedHostBuffer sample_host_,
                       PinnedHostBuffer ids_host_)
      : workspace(std::move(workspace_)),
        kv_pool(std::move(kv_pool_)),
        block_tables(std::move(block_tables_)),
        zero_block(std::move(zero_block_)),
        router_host(std::move(router_host_)),
        sample_host(std::move(sample_host_)),
        ids_host(std::move(ids_host_)) {}

  Lane& lane_of(const SequenceHandle& h) {
    return lanes[RuntimeContractAccess::lane(h).value];
  }

  // ---------------------------------------------------------------------
  // Device copies
  // ---------------------------------------------------------------------
  void copy_to_host(const void* dev, void* host, std::size_t bytes, cudaStream_t s) const {
    // The producer runs on a NON-BLOCKING stream (device.stream is created with
    // cudaStreamNonBlocking). A plain cudaMemcpy executes on the legacy default
    // stream, which has NO ordering with a non-blocking stream -- so without this
    // sync the host read races the producing kernel (stale / non-deterministic
    // data). Sync the producer first, mirroring copy_to_device's stream-ordered
    // H2D.
    const cudaError_t se = cudaStreamSynchronize(s);
    if (se != cudaSuccess) {
      throw std::runtime_error(std::string("flash_next D2H sync failed: ") + cudaGetErrorString(se));
    }
    const cudaError_t err = cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("flash_next D2H failed: ") + cudaGetErrorString(err));
    }
  }

  void copy_to_device(const void* host, void* dev, std::size_t bytes, cudaStream_t s) const {
    const cudaError_t err = cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, s);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("flash_next H2D failed: ") + cudaGetErrorString(err));
    }
  }

  // ---------------------------------------------------------------------
  // KV pages
  // ---------------------------------------------------------------------
  void take_page(Lane& l) {
    if (free_pages.empty()) {
      throw std::logic_error("flash_next KV pool exhausted");
    }
    const std::int32_t page = free_pages.back();
    free_pages.pop_back();
    l.pages.push_back(page);
  }

  void release_pages(Lane& l) {
    for (std::int32_t page : l.pages) {
      free_pages.push_back(page);
    }
    l.pages.clear();
  }

  void ensure_pages(Lane& l, std::uint32_t through_exclusive) {
    if (real_mode_) {
      return;  // RealProgram owns its own KV/QSA block tables; the mini page pool is unused.
    }
    const std::size_t need = (static_cast<std::size_t>(through_exclusive) + 63) / 64;
    while (l.pages.size() < need) {
      take_page(l);
      upload_block_table(l);
    }
  }

  void upload_block_table(Lane& l) {
    const std::size_t offset =
        static_cast<std::size_t>(RuntimeContractAccess::lane(l.handle).value) *
        per_lane_pages * sizeof(std::int32_t);
    const std::size_t count = l.pages.size() * sizeof(std::int32_t);
    block_tables.copy_from_host(l.pages.data(), count, offset);
  }

  // ---------------------------------------------------------------------
  // Prefix store (host-side mirror of device lane state)
  // ---------------------------------------------------------------------
  bool prefix_hit(std::uint64_t digest) const { return prefix_store.count(digest) > 0; }

  // D2H the lane's GDN + conv states and the KV lane page (v1 f <= 8 <= 64:
  // exactly page 0). Must run with the compute stream idle (call after the
  // round's final sync).
  void store_prefix(std::uint64_t digest, Lane& l) {
    if (real_mode_) {
      return;  // RealProgram keeps its recurrent/QSA state in the RealProgram itself; a
               // cross-request prefix store is not implemented in real v1 (no-op => prefix_store
               // stays empty => no prefix hit => restore_prefix never fires).
    }
    StoredPrefix sp;
    for (std::uint32_t li = 0; li < mini::kGdnLayers; ++li) {
      sp.gdn_states[li].resize(kGdnStateBytes);
      copy_to_host(l.gdn_state[li], sp.gdn_states[li].data(), kGdnStateBytes, stream);
      sp.conv_states[li].resize(kConvStateBytes);
      copy_to_host(l.conv_state[li], sp.conv_states[li].data(), kConvStateBytes, stream);
    }
    sp.kv_pages.resize(kQsaKvPageBytes);
    const std::int32_t page0 = l.pages.front();
    copy_to_host(static_cast<const std::uint8_t*>(kv_pool.p) + page0 * kQsaKvPageBytes,
                 sp.kv_pages.data(), kQsaKvPageBytes, stream);
    sp.first_token = l.last_token;
    sp.frontier = l.frontier;
    sp.ple = l.ple_state;
    if (prefix_store.size() >= kPrefixStoreCap) {
      const std::uint64_t oldest = prefix_order.front();
      prefix_order.erase(prefix_order.begin());
      prefix_store.erase(oldest);
    }
    prefix_store[digest] = std::move(sp);
    prefix_order.push_back(digest);
  }

  // H2D the stored states into a fresh lane (page 0 already taken).
  void restore_prefix(Lane& l, const StoredPrefix& sp) {
    if (real_mode_) {
      return;  // Defensive: store_prefix is a no-op in real mode, so a hit cannot occur.
    }
    for (std::uint32_t li = 0; li < mini::kGdnLayers; ++li) {
      copy_to_device(sp.gdn_states[li].data(), l.gdn_state[li], kGdnStateBytes, stream);
      copy_to_device(sp.conv_states[li].data(), l.conv_state[li], kConvStateBytes, stream);
    }
    const std::int32_t page0 = l.pages.front();
    copy_to_device(sp.kv_pages.data(),
                   static_cast<std::uint8_t*>(kv_pool.p) + page0 * kQsaKvPageBytes,
                   kQsaKvPageBytes, stream);
    cudaStreamSynchronize(stream);
    l.ple_state = sp.ple;
    l.frontier = sp.frontier;
    l.last_token = sp.first_token;
    l.emitted = 1;
  }

  // ---------------------------------------------------------------------
  // Per-layer blocks
  // ---------------------------------------------------------------------
  void gdn_block(Lane& l, std::uint32_t layer, int T, __nv_bfloat16* xhat,
                 __nv_bfloat16* y, DeviceArena& arena) {
    const GdnLayerWeights& g = *model.layers[layer].gdn;
    const cudaStream_t s = stream;

    __nv_bfloat16* qkvz =
        static_cast<__nv_bfloat16*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 384).data);
    detail::mini_fp8_linear(xhat, T, 256, g.qkvz.codes, g.qkvz.scale, 384, qkvz, s);

    // Causal conv over the qkv channels (z bypasses). Channel-major staging.
    __nv_bfloat16* conv_in =
        static_cast<__nv_bfloat16*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 256).data);
    detail::mini_transpose_tm_cm(qkvz, T, 256, conv_in, s);
    __nv_bfloat16* conv_out =
        static_cast<__nv_bfloat16*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 256).data);
    // The op allows conv_state_in == conv_state_out; the lane's state buffer is
    // used for both.
    Tensor conv_st_out = make_tensor(l.conv_state[layer], DType::BF16, {256, 3});
    Tensor conv_out_t = make_tensor(conv_out, DType::BF16, {256, T});
    ops::causal_conv1d_silu(make_tensor(conv_in, DType::BF16, {256, T}),
                            make_tensor(g.conv, DType::BF16, {256, 4}),
                            make_tensor(l.conv_state[layer], DType::BF16, {256, 3}),
                            conv_st_out, conv_out_t, s);

    // Zero-pad pack into the op's [128, H, T] d-fastest buffers.
    __nv_bfloat16* q_op =
        static_cast<__nv_bfloat16*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 128 * 2).data);
    __nv_bfloat16* k_op =
        static_cast<__nv_bfloat16*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 128 * 2).data);
    __nv_bfloat16* v_op =
        static_cast<__nv_bfloat16*>(
            arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 128 * 4).data);
    detail::mini_gdn_pack(conv_out, T, q_op, k_op, v_op, s);

    // Gating from the layer input: a rows [0,4), b rows [4,8) of a_b_projection.
    float* gbuf = static_cast<float*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 4 * 4).data);
    float* betabuf =
        static_cast<float*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 4 * 4).data);
    detail::mini_gdn_gating(xhat, T,
                            reinterpret_cast<const __nv_bfloat16*>(g.a_b_projection.data),
                            reinterpret_cast<const __nv_bfloat16*>(g.a_b_projection.data) + 4 * 256,
                            g.a_log, g.dt_bias, gbuf, betabuf, s);

    // Distinct-state recurrent step (in == out state storage).
    __nv_bfloat16* o_op =
        static_cast<__nv_bfloat16*>(
            arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 128 * 4).data);
    auto scope = arena.scope();
    const DeviceSpan gws =
        arena.alloc_bytes(ops::gated_delta_net_workspace_capacity_bytes(2, 4, true, 1, 64));
    WorkspaceArena leaf(gws);
    Tensor gdn_state_out = make_tensor(l.gdn_state[layer], DType::FP32, {128, 128, 4});
    Tensor gdn_out_t = make_tensor(o_op, DType::BF16, {128, 4, T});
    ops::gated_delta_net(make_tensor(q_op, DType::BF16, {128, 2, T}),
                         make_tensor(k_op, DType::BF16, {128, 2, T}),
                         make_tensor(v_op, DType::BF16, {128, 4, T}),
                         make_tensor(gbuf, DType::FP32, {4, T}),
                         make_tensor(betabuf, DType::FP32, {4, T}), kGdnScale, true, leaf,
                         make_tensor(l.gdn_state[layer], DType::FP32, {128, 128, 4}),
                         gdn_state_out, gdn_out_t, s);

    // Fused gated RMSNorm; z = qkvz[t, 256 + d] gathered into a contiguous
    // [T, 128] token-major buffer (the qkvz row stride is 384).
    __nv_bfloat16* z =
        static_cast<__nv_bfloat16*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 128).data);
    detail::mini_gdn_zslice(qkvz, T, z, s);
    __nv_bfloat16* onx =
        static_cast<__nv_bfloat16*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 128).data);
    detail::mini_gated_rmsnorm(o_op, z, reinterpret_cast<const __nv_bfloat16*>(g.norm), T, onx, s);

    detail::mini_fp8_linear(onx, T, 128, g.output.codes, g.output.scale, 256, y, s);
  }

  void qsa_block(Lane& l, std::uint32_t layer, int T, int pos0, __nv_bfloat16* xhat,
                 __nv_bfloat16* y, DeviceArena& arena) {
    const QsaLayerWeights& q = *model.layers[layer].qsa;
    const cudaStream_t s = stream;

    __nv_bfloat16* q_full =
        static_cast<__nv_bfloat16*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 256).data);
    detail::mini_fp8_linear(xhat, T, 256, q.query.codes, q.query.scale, 256, q_full, s);
    __nv_bfloat16* k =
        static_cast<__nv_bfloat16*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 64).data);
    detail::mini_fp8_linear(xhat, T, 256, q.key.codes, q.key.scale, 64, k, s);
    __nv_bfloat16* v =
        static_cast<__nv_bfloat16*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 64).data);
    detail::mini_fp8_linear(xhat, T, 256, q.value.codes, q.value.scale, 64, v, s);

    detail::mini_head_rmsnorm(q_full, T, 4,
                              reinterpret_cast<const __nv_bfloat16*>(q.query_norm), s);
    detail::mini_head_rmsnorm(k, T, 2,
                              reinterpret_cast<const __nv_bfloat16*>(q.key_norm), s);
    detail::mini_rope(q_full, T, 4, pos0, k, 2, s);

    ensure_pages(l, static_cast<std::uint32_t>(pos0 + T));
    detail::mini_kv_append(k, v, T, pos0, l.block_table_dev,
                           static_cast<__nv_bfloat16*>(kv_pool.p), s);

    // Round-local indexer; ids come back round-local and GQA wants absolute
    // positions — shift by the round start on the host (v1 eager island).
    float* z = static_cast<float*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 4 * 160).data);
    float* scores = static_cast<float*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 4 * 64).data);
    std::int32_t* ids =
        static_cast<std::int32_t*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 4 * 64).data);
    std::int32_t* counts =
        static_cast<std::int32_t*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 4).data);
    detail::mini_indexer(xhat, T, q.indexer_qk_proj.data, q.indexer_q_norm,
                         q.indexer_k_norm, z, scores, ids, counts, s);
    if (pos0 != 0) {
      const std::size_t count = static_cast<std::size_t>(T) * 64 * sizeof(std::int32_t);
      std::int32_t* host_ids = static_cast<std::int32_t*>(ids_host.data());
      copy_to_host(ids, host_ids, count, s);
      for (std::size_t i = 0; i < static_cast<std::size_t>(T) * 64; ++i) {
        if (host_ids[i] >= 0) {
          host_ids[i] += pos0;
        }
      }
      copy_to_device(host_ids, ids, count, s);
    }

    __nv_bfloat16* qsa_out =
        static_cast<__nv_bfloat16*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 128).data);
    const __nv_bfloat16* pool = static_cast<__nv_bfloat16*>(kv_pool.p);
    detail::mini_sparse_gqa(q_full, T, ids, counts, l.block_table_dev, pool,
                            pool + static_cast<std::ptrdiff_t>(kQsaVPlaneOffset), qsa_out, s);

    detail::mini_fp8_linear(qsa_out, T, 128, q.output.codes, q.output.scale, 256, y, s);
  }

  void moe_block(Lane& l, std::uint32_t layer, int T, __nv_bfloat16* xhat,
                 __nv_bfloat16* y, DeviceArena& arena, bool decode) {
    const MoeLayerWeights& m = model.layers[layer].mlp;
    const cudaStream_t s = stream;

    __nv_bfloat16* x_cm =
        static_cast<__nv_bfloat16*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 256).data);
    detail::mini_transpose_tm_cm(xhat, T, 256, x_cm, s);

    if (decode) {
      float* logits = static_cast<float*>(
          arena.alloc_bytes(static_cast<std::size_t>(T) * 4 * mini::kMoEExperts).data);
      detail::mini_router(xhat, T,
                          reinterpret_cast<const __nv_bfloat16*>(m.router_bf16), logits, s);
      const std::size_t count = static_cast<std::size_t>(T) * mini::kMoEExperts * sizeof(float);
      copy_to_host(logits, router_host.data(), count, s);
      const float* row = static_cast<const float*>(router_host.data());
      const std::vector<std::uint32_t> top2 = moe_top2(row);
      paging::DecodeFetch fetch(*model.pager);
      const std::array<std::uint32_t, 2> sel{top2[0], top2[1]};
      fetch.stage(layer, std::span<const std::uint32_t>(sel));
      model.pager->wait();
    }

    ops::SparseMoeNvfp4Weights w;
    w.router_bf16 = m.router_bf16;
    w.shared_gate_bf16 = m.shared_gate_bf16;
    w.shared_gu_codes = m.shared_gu_codes;
    w.shared_gu_scales = m.shared_gu_scales;
    w.shared_dn_codes = m.shared_dn_codes;
    w.shared_dn_scales = m.shared_dn_scales;
    w.shared_gu_divisor = m.shared_gu_divisor;
    w.shared_dn_divisor = m.shared_dn_divisor;
    w.expert_window = m.expert_window;
    w.routed_gu_divisor = m.routed_gu_divisor;
    w.routed_dn_divisor = m.routed_dn_divisor;

    __nv_bfloat16* y_cm =
        static_cast<__nv_bfloat16*>(arena.alloc_bytes(static_cast<std::size_t>(T) * 2 * 256).data);
    auto scope = arena.scope();
    const DeviceSpan mws =
        arena.alloc_bytes(ops::sparse_moe_nvfp4_workspace_capacity_bytes(kMoeGeometry, 1, 64));
    WorkspaceArena leaf(mws);
    Tensor moe_x = make_tensor(x_cm, DType::BF16, {256, T});
    Tensor moe_dst = make_tensor(y_cm, DType::BF16, {256, T});
    ops::sparse_moe_nvfp4(moe_x, kMoeGeometry, w,
                          ops::SparseMoeNvfp4Route::W4A16, ops::SparseMoeNvfp4Epilogue::Identity,
                          nullptr, moe_dst, leaf, s);

    detail::mini_transpose_cm_tm(y_cm, T, 256, y, s);
  }

  // One hyper-connection call. `read` = block_out is the zero block (xhat =
  // norm(stream)); `write` = stream_out = stream + sigma(w_inj) * block_out.
  // stream_out must differ from stream_dev.
  void hc_call(ops::GatedResidualParams p, cudaStream_t s) {
    ops::gated_residual(p, s);
  }

  // ---------------------------------------------------------------------
  // One round: T tokens at absolute positions [pos0, pos0 + T).
  // Returns the sampled token (host, the row for position pos0 + T - 1).
  // ---------------------------------------------------------------------
  TokenId run_round(Lane& l, int T, int pos0, const std::vector<TokenId>& input_ids,
                    bool decode) {
    const cudaStream_t s = stream;
    auto round_scope = workspace.scope();

    // S0 boundary-trace gate (env-gated; the first N rounds only).
    g_fx_trace_this_round = !g_fx_trace_path.empty() && g_fx_trace_rounds_left > 0;
    if (g_fx_trace_this_round) {
      --g_fx_trace_rounds_left;
    }

    // Working-set allocations (shared by every layer).
    std::int32_t* ids_dev = static_cast<std::int32_t*>(
        workspace.alloc_bytes(static_cast<std::size_t>(T) * 4).data);
    __nv_bfloat16* h = static_cast<__nv_bfloat16*>(
        workspace.alloc_bytes(static_cast<std::size_t>(T) * 2 * 256).data);
    float* a0 = static_cast<float*>(
        workspace.alloc_bytes(static_cast<std::size_t>(T) * 4 * 1024).data);
    float* a1 = static_cast<float*>(
        workspace.alloc_bytes(static_cast<std::size_t>(T) * 4 * 1024).data);
    float* xhat_f32 = static_cast<float*>(
        workspace.alloc_bytes(static_cast<std::size_t>(T) * 4 * 256).data);
    __nv_bfloat16* xhat_bf16 = static_cast<__nv_bfloat16*>(
        workspace.alloc_bytes(static_cast<std::size_t>(T) * 2 * 256).data);
    __nv_bfloat16* y = static_cast<__nv_bfloat16*>(
        workspace.alloc_bytes(static_cast<std::size_t>(T) * 2 * 256).data);
    float* block_out = static_cast<float*>(
        workspace.alloc_bytes(static_cast<std::size_t>(T) * 4 * 256).data);
    float* ple_mem = static_cast<float*>(
        workspace.alloc_bytes(static_cast<std::size_t>(T) * 4 * 128).data);
    const float* zero_block_f32 = static_cast<const float*>(zero_block.p);

    // Embedding.
    std::memcpy(ids_host.data(), input_ids.data(),
                static_cast<std::size_t>(T) * sizeof(TokenId));
    copy_to_device(ids_host.data(), ids_dev, static_cast<std::size_t>(T) * sizeof(TokenId), s);
    detail::mini_embedding_gather(ids_dev, model.embedding.codes, model.embedding.scale, T, h, s);
    fx_trace("00 embed h", h, static_cast<std::size_t>(T) * 256, true, s);

    // Stream init: the embedding is replicated into all 4 branches.
    detail::mini_stream_init(h, T, a0, s);
    fx_trace("01 stream_init a0", a0, static_cast<std::size_t>(T) * 1024, false, s);

    for (std::uint32_t layer = 0; layer < mini::kLayers; ++layer) {
      const LayerWeights& lw = model.layers[layer];

      // HC read (attention): xhat from A0; A1 <- A0 (+ sigma * 0).
      {
        ops::GatedResidualParams p{};
        p.tokens = T;
        p.hidden = 256;
        p.hc_count = 4;
        p.lowrank = 32;
        p.stream_dev = a0;
        p.block_out_dev = zero_block_f32;
        p.w_down_dev = lw.attn_hc.w_down;
        p.w_up_dev = lw.attn_hc.w_up;
        p.w_inj_dev = lw.attn_hc.w_inj;
        p.norm_w_dev = lw.attn_hc.norm_w;
        p.read_out_dev = xhat_f32;
        p.stream_out_dev = a1;
        p.norm_branches = true;
        hc_call(p, s);
      }
      detail::mini_f32_to_bf16(xhat_f32, T, 256, xhat_bf16, s);
      fx_trace((std::string("L") + std::to_string(layer) + ".attn.xhat_in").c_str(),
               xhat_f32, static_cast<std::size_t>(T) * 256, false, s);

      if (layer == mini::kPleLayer) {
        ops::NgramOpParams np{};
        np.conv_kernel = mini::kPleConvKernel;
        np.dilation = 0;
        np.conv_weights = &model.ple_conv_weights[0][0];
        np.staging_cap_bytes = 0;
        const ops::NgramEmbeddingTable& table = *model.ple_table;
        ops::ngram_embedding(
            table,
            std::span<const std::uint32_t>(
                reinterpret_cast<const std::uint32_t*>(input_ids.data()),
                static_cast<std::size_t>(T)),
            np, l.ple_state, ple_mem, s);
        detail::mini_ple_add(xhat_bf16, ple_mem, T, s);
      }

      if (layer < mini::kQsaLayer) {
        gdn_block(l, layer, T, xhat_bf16, y, workspace);
      } else {
        qsa_block(l, layer, T, pos0, xhat_bf16, y, workspace);
      }
      fx_trace((std::string("L") + std::to_string(layer) + ".attn.y_out").c_str(),
               y, static_cast<std::size_t>(T) * 256, true, s);

      detail::mini_block_out(y, T, block_out, s);
      fx_trace((std::string("L") + std::to_string(layer) + ".attn.block_out").c_str(),
               block_out, static_cast<std::size_t>(T) * 256, false, s);
      // HC write (attention): A1 <- A0 + sigma * y.
      {
        ops::GatedResidualParams p{};
        p.tokens = T;
        p.hidden = 256;
        p.hc_count = 4;
        p.lowrank = 32;
        p.stream_dev = a0;
        p.block_out_dev = block_out;
        p.w_down_dev = lw.attn_hc.w_down;
        p.w_up_dev = lw.attn_hc.w_up;
        p.w_inj_dev = lw.attn_hc.w_inj;
        p.norm_w_dev = lw.attn_hc.norm_w;
        p.read_out_dev = xhat_f32;  // overwritten; unused by the write
        p.stream_out_dev = a1;
        p.norm_branches = true;
        hc_call(p, s);
      }
      fx_trace((std::string("L") + std::to_string(layer) + ".attn.a1_stream").c_str(),
               a1, static_cast<std::size_t>(T) * 1024, false, s);

      // HC read (mlp): xhat from A1; A0 <- A1.
      {
        ops::GatedResidualParams p{};
        p.tokens = T;
        p.hidden = 256;
        p.hc_count = 4;
        p.lowrank = 32;
        p.stream_dev = a1;
        p.block_out_dev = zero_block_f32;
        p.w_down_dev = lw.mlp_hc.w_down;
        p.w_up_dev = lw.mlp_hc.w_up;
        p.w_inj_dev = lw.mlp_hc.w_inj;
        p.norm_w_dev = lw.mlp_hc.norm_w;
        p.read_out_dev = xhat_f32;
        p.stream_out_dev = a0;
        p.norm_branches = true;
        hc_call(p, s);
      }
      detail::mini_f32_to_bf16(xhat_f32, T, 256, xhat_bf16, s);
      fx_trace((std::string("L") + std::to_string(layer) + ".mlp.xhat_in").c_str(),
               xhat_f32, static_cast<std::size_t>(T) * 256, false, s);

      moe_block(l, layer, T, xhat_bf16, y, workspace, decode);
      fx_trace((std::string("L") + std::to_string(layer) + ".mlp.y_out_moe").c_str(),
               y, static_cast<std::size_t>(T) * 256, true, s);

      detail::mini_block_out(y, T, block_out, s);
      fx_trace((std::string("L") + std::to_string(layer) + ".mlp.block_out").c_str(),
               block_out, static_cast<std::size_t>(T) * 256, false, s);
      // HC write (mlp): A0 <- A1 + sigma * y.
      {
        ops::GatedResidualParams p{};
        p.tokens = T;
        p.hidden = 256;
        p.hc_count = 4;
        p.lowrank = 32;
        p.stream_dev = a1;
        p.block_out_dev = block_out;
        p.w_down_dev = lw.mlp_hc.w_down;
        p.w_up_dev = lw.mlp_hc.w_up;
        p.w_inj_dev = lw.mlp_hc.w_inj;
        p.norm_w_dev = lw.mlp_hc.norm_w;
        p.read_out_dev = xhat_f32;
        p.stream_out_dev = a0;
        p.norm_branches = true;
        hc_call(p, s);
      }
      fx_trace((std::string("L") + std::to_string(layer) + ".mlp.a0_stream").c_str(),
               a0, static_cast<std::size_t>(T) * 1024, false, s);
    }

    // Final collapse -> lm_head -> argmax -> sampled token.
    __nv_bfloat16* h_final = static_cast<__nv_bfloat16*>(
        workspace.alloc_bytes(static_cast<std::size_t>(T) * 2 * 256).data);
    detail::mini_final_collapse(a0, T, h_final, s);
    fx_trace("99 final h_final", h_final, static_cast<std::size_t>(T) * 256, true, s);
    __nv_bfloat16* logits = static_cast<__nv_bfloat16*>(
        workspace.alloc_bytes(static_cast<std::size_t>(T) * 2 * 256).data);
    detail::mini_fp8_linear(h_final, T, 256, model.output_head.codes,
                            model.output_head.scale, 256, logits, s);
    fx_trace("99 logits", logits, static_cast<std::size_t>(T) * 256, true, s);
    // logits is [256][T] (vocab-major: the fp8 GEMM writes out[n*T + t]). The
    // argmax op reads its buffer as [T][256] (out[t] reduces buffer[t*256 + v]
    // over the vocab), so transpose to [T][256] first; the last position's 256
    // vocab logits then form a contiguous row for capture.
    __nv_bfloat16* logits_cm = static_cast<__nv_bfloat16*>(
        workspace.alloc_bytes(static_cast<std::size_t>(T) * 2 * 256).data);
    detail::mini_transpose_cm_tm(logits, T, 256, logits_cm, s);
    if (capture_logits_) {
      // logits_cm is [T][256]; capture the last position (T-1) row. BF16 -> FP32
      // is the lossless top-16-bit shift.
      std::array<std::uint16_t, 256> h{};
      copy_to_host(logits_cm + static_cast<std::size_t>(T - 1) * 256, h.data(),
                   sizeof(h), s);
      for (int v = 0; v < 256; ++v) {
        const std::uint32_t bits = static_cast<std::uint32_t>(h[v]) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        logits_capture_.push_back(f);
      }
    }
    std::int32_t* sample_dev = static_cast<std::int32_t*>(
        workspace.alloc_bytes(static_cast<std::size_t>(T) * 4).data);
    Tensor logits_t = make_tensor(logits_cm, DType::BF16, {256, T});
    Tensor sample_t = make_tensor(sample_dev, DType::I32, {T});
    ops::argmax(logits_t, sample_t, 256, s);

    copy_to_host(sample_dev, sample_host.data(),
                 static_cast<std::size_t>(T) * sizeof(std::int32_t), s);
    return static_cast<const TokenId*>(sample_host.data())[T - 1];
  }

  PendingBatch make_pending_batch(std::span<const Lane*> rows) {
    // Row-major pending tokens: one sampled token per active lane (the
    // engine commits each lane's last token).
    pending_tokens.clear();
    for (std::size_t i = 0; i < rows.size(); ++i) {
      pending_tokens.push_back(rows[i]->last_token);
    }
    std::vector<SequenceHandle> handles(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
      handles[i] = rows[i]->handle;
    }
    return RuntimeContractAccess::make_pending(
        this, 1 /*txn*/, std::span<const SequenceHandle>(handles),
        std::span<const TokenId>(pending_tokens), std::span<const std::int32_t>{},
        1 /*row_stride*/, runtime::ExecutionTiming{});
  }
};

}  // namespace detail

// ---------------------------------------------------------------------------
// RequestBasePlan
// ---------------------------------------------------------------------------

std::optional<PrefixShortlistKey>
RequestBasePlan::prefix_shortlist_key(std::uint32_t idx) const noexcept {
  const std::uint32_t f = std::min(kPrefixDigestLimit, summary_.prompt_tokens);
  if (idx >= f) {
    return std::nullopt;
  }
  return PrefixShortlistKey{prefix_digests_[idx], idx + 1, prefix_identity_tag_};
}

// ---------------------------------------------------------------------------
// ResourcePlan / family handles (private ctors)
// ---------------------------------------------------------------------------

ResourcePlan::ResourcePlan(AdmissionCandidate&& admission, std::uint64_t revision,
                           bool needs_transfer) noexcept
    : admission_(std::move(admission)), revision_(revision), needs_transfer_(needs_transfer) {}

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

Program::Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept : impl_(std::move(impl)) {}

Program::~Program() noexcept = default;

RequestBasePlan Program::plan_request(const PreparedPrompt& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
  auto* self = impl_.get();
  RequestBasePlan plan;
  const std::vector<TokenId>& ids = prompt.token_ids();
  const std::uint32_t P = static_cast<std::uint32_t>(ids.size());
  const std::uint32_t requested = options.requested_output_tokens;
  const std::uint32_t room =
      (P <= self->max_context) ? self->max_context + 1 - P : 0;
  const std::uint32_t effective = std::min(requested, room);

  plan.summary_.prompt_tokens = P;
  plan.summary_.requested_output_tokens = requested;
  plan.summary_.effective_output_tokens = effective;
  plan.summary_.effective_limit_reason = FinishReason::OutputLimit;  // v1 pinned
  // Total service-work budget: one prefill pass (advance_prefill runs the whole
  // prompt in a single round, emitting the first token) + one decode round per
  // remaining output token. Mirrors qwen3_6's projected_service_work
  // (prefill_units + (effective_output_tokens - 1)): the engine consumes 1 quantum
  // per prefill unit (resolve_prefill_progress) and `accepted` per decode round
  // (run_decode_round), so the budget must equal total consumed or
  // consume_service_work throws "consumed N quanta with 0 remaining".
  const std::uint64_t decode_units =
      effective == 0 ? 0ULL : static_cast<std::uint64_t>(effective - 1);
  plan.summary_.service_work_quanta = 1ULL + decode_units;
  plan.summary_.publish_continuation = false;  // v1: finish never publishes a continuation

  const std::uint32_t f = std::min(kPrefixDigestLimit, P);
  plan.prefix_identity_tag_ = kIdentityTag;
  for (std::uint32_t i = 1; i <= f; ++i) {
    plan.prefix_digests_[i - 1] = prefix_digest(ids, i);
  }
  // v1 reuse is valid only when the whole prompt is the stored prefix
  // (f = P); the round processes all P tokens in one pass.
  if (options.allow_prefix_reuse && P <= kPrefixDigestLimit && f > 0 &&
      self->prefix_hit(plan.prefix_digests_[f - 1])) {
    plan.summary_.reusable_prompt_tokens = f;
    plan.summary_.prefix_reuse_path = PrefixReusePath::PrivateEndpoint;
  }
  return plan;
}

std::optional<AdmissionCandidate>
Program::inspect_admission(const PreparedPrompt& prompt, const RequestBasePlan& base,
                           runtime::LaneId destination, const ContinuationHandle* source,
                           const SharedPrefixHandle* shared_source,
                           std::optional<runtime::CheckpointRef> checkpoint,
                           bool must_retain_private_source,
                           const runtime::ContextMachineCostModel& machine_cost) {
  auto* self = impl_.get();
  (void)prompt;
  (void)machine_cost;
  // v1: no continuation, no shared source, no checkpoint, single transaction.
  if (source != nullptr || shared_source != nullptr || checkpoint.has_value() ||
      must_retain_private_source || self->txn_active) {
    return std::nullopt;
  }
  if (destination.value >= self->lanes.size() || self->lanes[destination.value].active) {
    return std::nullopt;
  }
  self->pending_lane_ = destination;

  AdmissionCandidate candidate;
  candidate.summary_ = base.summary();

  runtime::IdentityMaterializationAssessment identity;
  identity.physical_status = runtime::MaterializationPhysicalStatus::Feasible;
  // Fresh root admission: no source to retain. ConsumedToActive is the qwen3_6
  // default and is REQUIRED here -- the engine's logical_goal rejects a Retained
  // disposition with no source/shared slot (resource_manager.h), which would mark
  // the sole candidate expandable and route into begin_pressure_planning (v1 throw).
  identity.source_disposition = runtime::ClaimDisposition::ConsumedToActive;
  identity.expandable = false;
  identity.projection_work = 1U;
  std::uint64_t d = kFNV64Basis;
  d = fnv1a64(d, 1U /* resource_revision */);
  d = fnv1a64(d, candidate.summary_.reusable_prompt_tokens);
  d = fnv1a64(d, identity.machine.immediate_ns);
  d = fnv1a64(d, static_cast<std::uint8_t>(identity.physical_status));
  identity.assessment_digest = d;
  candidate.identity_ = std::move(identity);
  return candidate;
}

std::optional<ResourcePlan> Program::seal_identity(const AdmissionCandidate& candidate,
                                                   const PreparedPrompt& prompt) {
  (void)prompt;
  AdmissionCandidate fresh;
  fresh.summary_ = candidate.summary();
  fresh.identity_ = candidate.identity_assessment();
  // Construct in the Program context (friend of the private ResourcePlan
  // ctor), then move into the optional (public move ctor).
  ResourcePlan plan_local(std::move(fresh), resource_revision(),
                          false /* needs_transfer */);
  return std::optional<ResourcePlan>(std::move(plan_local));
}

PressurePlanningSession Program::begin_pressure_planning(
    const runtime::ContextMachineCostModel& machine_cost,
    std::span<const AdmissionCandidate* const> candidates,
    std::span<const ContinuationHandle* const> private_owners,
    std::span<const std::uint32_t> private_owner_ordinals,
    std::span<const SharedPrefixHandle* const> shared_owners,
    std::span<const std::uint32_t> shared_owner_ordinals) {
  (void)machine_cost;
  (void)candidates;
  (void)private_owners;
  (void)private_owner_ordinals;
  (void)shared_owners;
  (void)shared_owner_ordinals;
  throw std::logic_error("flash_next v1: pressure planning is out of scope");
}

runtime::ContextTransactionReserveStatus
Program::start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                                    runtime::CancellationFlagView cancellation) {
  auto* self = impl_.get();
  (void)cancellation;
  if (self->txn_active) {
    throw std::logic_error("flash_next v1: a resource transaction is already active");
  }
  if (self->pending_lane_.value >= self->lanes.size()) {
    throw std::logic_error("flash_next v1: admission lane was not reserved");
  }
  detail::ProgramImpl::Lane& lane = self->lanes[self->pending_lane_.value];
  self->txn_lane = self->pending_lane_.value;
  self->txn_reusable = plan.summary().reusable_prompt_tokens;
  self->txn_reuse = plan.summary().prefix_reuse_path == PrefixReusePath::PrivateEndpoint;
  lane.prompt_ids = prompt.token_ids();
  lane.prompt_tokens = plan.summary().prompt_tokens;
  lane.requested_output_tokens = plan.summary().requested_output_tokens;
  lane.effective_output_tokens = plan.summary().effective_output_tokens;
  lane.effective_limit_reason = plan.summary().effective_limit_reason;
  self->txn_active = true;
  return runtime::ContextTransactionReserveStatus::Reserved;
}

std::optional<PersistentBackfillProof>
Program::prove_persistent_backfill(const RequestBasePlan& blocked_head,
                                   const ResourcePlan& candidate,
                                   std::span<const SequenceHandle> persistent_borrowers) const {
  (void)blocked_head;
  (void)candidate;
  (void)persistent_borrowers;
  return std::nullopt;  // v1: nothing to backfill
}

ContextTransactionProgress
Program::progress_context_transaction(runtime::CancellationFlagView cancellation) {
  auto* self = impl_.get();
  (void)cancellation;
  if (!self->txn_active) {
    throw std::logic_error("flash_next v1: no active resource transaction");
  }
  detail::ProgramImpl::Lane& lane = self->lanes[self->txn_lane];
  const std::uint64_t epoch = self->lanes[self->txn_lane].epoch + 1;
  lane.active = true;
  lane.epoch = epoch;
  lane.emitted = 0;
  lane.frontier = 0;
  lane.last_token = 0;
  self->release_pages(lane);
  lane.handle = detail::RuntimeContractAccess::make_sequence(
      self, runtime::LaneId{self->txn_lane}, epoch);

  if (self->txn_reuse) {
    const std::uint32_t P = lane.prompt_tokens;
    const std::uint32_t f = std::min(kPrefixDigestLimit, P);
    const std::uint64_t digest = prefix_digest(lane.prompt_ids, f);
    auto it = self->prefix_store.find(digest);
    if (it == self->prefix_store.end()) {
      throw std::logic_error("flash_next v1: prefix store entry lost between plan and start");
    }
    self->take_page(lane);  // restored KV occupies page 0
    self->upload_block_table(lane);
    self->restore_prefix(lane, it->second);
  }

  MaterializationResult result;
  result.status = runtime::ContextTransactionStatus::Published;
  result.published = StartResult{lane.handle};
  return ContextTransactionProgress{std::move(result)};
}

void Program::finalize_context_transaction() noexcept {
  impl_->txn_active = false;
}

bool Program::has_context_transaction() const noexcept {
  return impl_->txn_active;
}

PrefillProgress Program::advance_prefill(SequenceHandle sequence,
                                         runtime::ExecutionTiming* failed_timing) {
  auto* self = impl_.get();
  (void)failed_timing;
  detail::ProgramImpl::Lane& lane = self->lane_of(sequence);
  if (!lane.active) {
    throw std::logic_error("flash_next v1: prefill on an inactive lane");
  }
  // Single-lane row span for the pending batch (a span of lane pointers).
  const detail::ProgramImpl::Lane* lane_row = &lane;
  std::span<const detail::ProgramImpl::Lane*> lane_rows(&lane_row, 1);

  PrefillProgress progress;
  if (self->real_mode_) {
    // Real v1: chunked state-carrying prefill through RealProgram (the proven
    // 48x512 engine). Lever B (on-device prefix continuation): if this prompt
    // shares the exact token prefix of the last processed request (FNV match over
    // cont_frontier_ ids), skip begin_sequence (the recurrent state already covers
    // [0, L) on-device) and prefill only the delta [L, P) at pos0 = L; otherwise
    // begin_sequence (zero recurrent + PLE) then prefill the whole prompt from 0.
    const std::vector<TokenId>& ids = lane.prompt_ids;
    const std::uint32_t P = static_cast<std::uint32_t>(ids.size());
    std::uint32_t L = 0;
    if (self->cont_valid_ && ids.size() >= self->cont_frontier_) {
      const std::uint32_t c = self->cont_frontier_;
      if (c > 0 && prefix_digest(ids, c) == self->cont_digest_) L = c;
    }
    if (L == 0) {
      self->real_->begin_sequence();
    }
    std::vector<std::uint32_t> chunk;
    std::uint32_t token = 0;
    const std::uint32_t step = static_cast<std::uint32_t>(RealProgram::kMaxRoundTokens);
    for (std::uint32_t pos0 = L; pos0 < P; pos0 += step) {
      const std::uint32_t take = std::min<std::uint32_t>(step, P - pos0);
      chunk.assign(ids.begin() + pos0, ids.begin() + pos0 + take);
      token = self->real_->run_round(chunk, static_cast<int>(pos0));
    }
    // The recurrent state now covers [0, P); refresh the continuation so the next
    // request delta-appends on top of it.
    self->cont_frontier_ = P;
    self->cont_digest_ = prefix_digest(ids, P);
    self->cont_valid_ = true;
    lane.frontier = P;
    lane.emitted = 1;
    lane.last_token = static_cast<TokenId>(token);
    progress.summary.prompt_tokens = lane.prompt_tokens;
    progress.summary.reused_prompt_tokens = static_cast<int>(L);
    progress.processed_prompt_tokens = static_cast<int>(P - L);
    progress.complete = true;
    progress.pending.emplace(std::move(self->make_pending_batch(lane_rows)));
    return progress;
  }
  if (self->txn_reuse && self->txn_reusable == lane.prompt_tokens) {
    // Reuse: the stored state already covers the whole prompt; re-emit the
    // stored first token (set by restore_prefix) as the prefill pending batch.
    progress.summary.prompt_tokens = lane.prompt_tokens;
    progress.summary.reused_prompt_tokens = self->txn_reusable;
    progress.summary.prefix_reuse_path = PrefixReusePath::PrivateEndpoint;
    progress.processed_prompt_tokens = 0;
    progress.complete = true;
    progress.pending.emplace(std::move(self->make_pending_batch(lane_rows)));
    return progress;
  }

  // Fresh prefill: the whole prompt in one round at positions [0, P).
  const int T = static_cast<int>(lane.prompt_tokens);
  if (T > kMaxRoundTokens) {
    throw std::logic_error("flash_next v1: prompt exceeds the 64-token round cap");
  }
  self->ensure_pages(lane, T);
  const TokenId token = self->run_round(lane, T, 0, lane.prompt_ids, false);
  lane.frontier = static_cast<std::uint32_t>(T);
  lane.emitted = 1;
  lane.last_token = token;
  if (T <= kPrefixDigestLimit) {
    self->store_prefix(prefix_digest(lane.prompt_ids, T), lane);
  }
  progress.summary.prompt_tokens = lane.prompt_tokens;
  progress.processed_prompt_tokens = T;
  progress.complete = true;
  progress.pending.emplace(std::move(self->make_pending_batch(lane_rows)));
  return progress;
}

CaptureAssessment Program::inspect_capture(const CaptureOffer& offer,
                                           const SharedPrefixHandle* exact_shared,
                                           const SharedPrefixHandle* replacement,
                                           std::optional<runtime::CheckpointRef> private_replacement)
    const {
  (void)offer;
  (void)exact_shared;
  (void)replacement;
  (void)private_replacement;
  throw std::logic_error("flash_next v1: capture is out of scope (eager only)");
}

bool Program::shared_capture_matches(const CaptureOffer& offer,
                                     const SharedPrefixHandle& shared) const {
  (void)offer;
  (void)shared;
  throw std::logic_error("flash_next v1: capture is out of scope (eager only)");
}

void Program::skip_capture(CaptureOffer&& offer) {
  (void)offer;
  throw std::logic_error("flash_next v1: capture is out of scope (eager only)");
}

runtime::ContextTransactionReserveStatus
Program::reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                                const SharedPrefixHandle* replacement,
                                std::optional<runtime::CheckpointRef> private_replacement,
                                runtime::CancellationFlagView cancellation) {
  (void)offer;
  (void)exact_shared;
  (void)replacement;
  (void)private_replacement;
  (void)cancellation;
  throw std::logic_error("flash_next v1: capture is out of scope (eager only)");
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences,
                             std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* failed_timing) {
  auto* self = impl_.get();
  (void)failed_timing;
  (void)budgets;
  if (self->real_mode_ && sequences.size() != 1) {
    throw std::logic_error("flash_next real v1: decode is single-lane (1 sequence per round)");
  }
  std::vector<const detail::ProgramImpl::Lane*> rows;
  for (const SequenceHandle& handle : sequences) {
    detail::ProgramImpl::Lane& lane = self->lane_of(handle);
    if (!lane.active) {
      throw std::logic_error("flash_next v1: decode on an inactive lane");
    }
    const int pos0 = static_cast<int>(lane.frontier);
    TokenId token;
    if (self->real_mode_) {
      std::vector<std::uint32_t> input(1, static_cast<std::uint32_t>(lane.last_token));
      token = static_cast<TokenId>(self->real_->run_round(input, pos0));
      // Lever B: extend the continuation with the generated token. It sits at global
      // position pos0, so the processed sequence is now [0, pos0+1); fold it into the
      // running digest (the exact per-token step prefix_digest uses) and the frontier.
      self->cont_digest_ = fnv1a64(self->cont_digest_, static_cast<std::uint32_t>(token));
      self->cont_frontier_ = static_cast<std::uint32_t>(pos0 + 1);
    } else {
      std::vector<TokenId> input(1, lane.last_token);
      const int T = 1;
      self->ensure_pages(lane, static_cast<std::uint32_t>(pos0 + T));
      token = self->run_round(lane, T, pos0, input, true);
    }
    lane.frontier = static_cast<std::uint32_t>(pos0 + 1);
    lane.emitted += 1;
    lane.last_token = token;
    rows.push_back(&lane);
  }
  return self->make_pending_batch(std::span<const detail::ProgramImpl::Lane*>(rows));
}

runtime::ExecutionTiming Program::append_forced_tokens(
    std::span<const SequenceHandle> sequences, std::span<const TokenId> row_major_tokens,
    std::uint32_t row_stride, runtime::ExecutionTiming* failed_timing) {
  (void)sequences;
  (void)row_major_tokens;
  (void)row_stride;
  (void)failed_timing;
  throw std::logic_error("flash_next v1: forced tokens are out of scope");
}

CommitResult Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation observation,
                             runtime::ExecutionTiming* failed_timing) {
  auto* self = impl_.get();
  (void)observation;
  (void)failed_timing;
  CommitResult result;
  result.row_count = pending.row_count();
  for (std::size_t i = 0; i < result.row_count; ++i) {
    // Mirror the engine's expected disposition (engine_core.h run_round): a
    // terminal row is Finishable (the request leaves the slot), a cancelled row
    // is CancelledReleased, and everything else stays Active. The mini commits
    // with observation ReleasedRowsOnly, so only the disposition is reported.
    if (decisions[i].cancelled) {
      result.rows[i] = CommitRowResult{};
      result.rows[i].disposition = runtime::CommitDisposition::CancelledReleased;
    } else if (decisions[i].terminal) {
      result.rows[i] = CommitRowResult{};
      result.rows[i].disposition = runtime::CommitDisposition::Finishable;
    } else {
      result.rows[i] = CommitRowResult{};  // default disposition Active
    }
  }
  result.timing = pending.execution_timing();
  return result;
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept {
  return DiscardResult{runtime::ConsumeStatus::Consumed, pending.row_count()};
}

FinishResult Program::finish(SequenceHandle sequence) noexcept {
  auto* self = impl_.get();
  FinishResult result;
  result.status = runtime::ConsumeStatus::Consumed;
  result.disposition = runtime::FinishDisposition::Released;
  detail::ProgramImpl::Lane& lane = self->lane_of(sequence);
  self->release_pages(lane);
  lane.active = false;
  lane.emitted = 0;
  lane.frontier = 0;
  lane.last_token = 0;
  return result;
}

AbortResult Program::abort(SequenceHandle sequence) noexcept {
  auto* self = impl_.get();
  AbortResult result;
  result.status = runtime::ConsumeStatus::Consumed;
  detail::ProgramImpl::Lane& lane = self->lane_of(sequence);
  self->release_pages(lane);
  lane.active = false;
  lane.emitted = 0;
  lane.frontier = 0;
  lane.last_token = 0;
  return result;
}

ReleaseResult Program::release_continuation(ContinuationHandle&& continuation) noexcept {
  (void)continuation;
  return ReleaseResult{runtime::ConsumeStatus::InvariantMismatch};  // v1: none exist
}

ReleaseResult Program::release_shared_prefix(SharedPrefixHandle&& shared) noexcept {
  (void)shared;
  return ReleaseResult{runtime::ConsumeStatus::InvariantMismatch};  // v1: none exist
}

void Program::fail_all_cleanup() noexcept {
  auto* self = impl_.get();
  for (auto& lane : self->lanes) {
    self->release_pages(lane);
    lane.active = false;
    lane.emitted = 0;
    lane.frontier = 0;
    lane.last_token = 0;
  }
  self->prefix_store.clear();
  self->prefix_order.clear();
  self->txn_active = false;
}

bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
  (void)base;
  return true;
}

std::uint64_t Program::resource_revision() const noexcept { return 1; }

PhysicalUsageSnapshot Program::physical_usage() const noexcept {
  const auto* self = impl_.get();
  PhysicalUsageSnapshot snapshot;
  snapshot.resource_revision = 1;
  snapshot.device_state_slots = self->max_concurrency;
  snapshot.device_main_kv_pages = static_cast<std::uint32_t>(self->kv_total_pages -
                                                             self->free_pages.size());
  return snapshot;
}

MemorySummary Program::memory_summary() const noexcept {
  const auto* self = impl_.get();
  MemorySummary summary;
  summary.device = self->device_ordinal;
  summary.max_context = self->max_context;
  summary.kv_capacity_mode = KvCapacityMode::Explicit;
  summary.kv_capacity = self->kv_capacity;
  summary.kv_capacity_page_groups = static_cast<std::uint32_t>(self->kv_total_pages);
  summary.kv_capacity_max_page_groups = static_cast<std::uint32_t>(self->kv_total_pages);
  summary.kv_cache = KvCacheStorage::BFloat16;
  if (self->model.weights_arena != nullptr) {
    summary.weights.capacity_bytes = self->model.weights_arena->capacity();
    summary.weights.used_bytes = self->model.weights_arena->used();
    summary.weights.peak_used_bytes = self->model.weights_arena->peak_used();
  }
  summary.workspace.capacity_bytes = self->workspace.capacity();
  summary.workspace.used_bytes = self->workspace.used();
  summary.workspace.peak_used_bytes = self->workspace.peak_used();
  summary.sequence = ArenaMemorySummary{};
  summary.kv_payload_bytes = self->kv_total_pages * kQsaKvPageBytes;
  return summary;
}

void Program::reset_memory_peaks() noexcept {
  auto* self = impl_.get();
  self->workspace.reset_peak();
  if (self->model.weights_arena != nullptr) {
    self->model.weights_arena->reset_peak();
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<Program> create_program(const ModelView& model, WeightsProfile weights_profile,
                                        SequencePlan&& plan, DeviceContext& device) {
  (void)weights_profile;
  if (model.pager == nullptr || model.archive == nullptr || model.ple_table == nullptr ||
      model.weights_arena == nullptr) {
    throw std::runtime_error("flash_next: ModelView is missing a required component");
  }

  const std::uint32_t max_context = plan.capacity();
  const std::uint32_t max_concurrency = plan.max_concurrency();
  const std::uint32_t kv_capacity = plan.kv_capacity();
  const std::size_t kv_total_pages = (static_cast<std::size_t>(kv_capacity) + 63) / 64;
  const std::size_t per_lane_pages = (static_cast<std::size_t>(max_context) + 63) / 64;

  // Workspace arena: persistent per-lane carve-out + zero block + round scratch.
  auto self = std::make_unique<detail::ProgramImpl>(
      DeviceArena(plan.workspace_capacity_bytes() + kZeroBlockBytes),
      DeviceBuffer(kv_total_pages * kQsaKvPageBytes),
      DeviceBuffer(static_cast<std::size_t>(max_concurrency) * per_lane_pages *
                   sizeof(std::int32_t)),
      DeviceBuffer(kZeroBlockBytes),
      PinnedHostBuffer(static_cast<std::size_t>(kMaxRoundTokens) * mini::kMoEExperts *
                       sizeof(float)),
      PinnedHostBuffer(static_cast<std::size_t>(kMaxRoundTokens) * sizeof(std::int32_t)),
      PinnedHostBuffer(static_cast<std::size_t>(kMaxRoundTokens) * sizeof(std::int32_t)));
  self->model = model;
  self->max_context = max_context;
  self->max_concurrency = max_concurrency;
  self->kv_capacity = kv_capacity;
  self->kv_total_pages = kv_total_pages;
  self->per_lane_pages = per_lane_pages;
  self->stream = device.stream;
  self->device_ordinal = device.device;
  if (const char* logits_path = std::getenv("NINFER_FLASH_NEXT_LOGITS")) {
    if (logits_path[0] != '\0') {
      self->capture_logits_ = true;
      self->logits_capture_path_ = logits_path;
    }
  }
  if (const char* fx_path = std::getenv("NINFER_FX_TRACE")) {
    if (fx_path[0] != '\0') {
      g_fx_trace_path = fx_path;
    }
  }
  if (const char* fx_rounds = std::getenv("NINFER_FX_TRACE_ROUNDS")) {
    if (fx_rounds[0] != '\0') {
      const int r = std::atoi(fx_rounds);
      if (r > 0) {
        g_fx_trace_rounds_left = r;
      }
    }
  }

  self->lanes.resize(self->max_concurrency);
  for (std::uint32_t lane = 0; lane < self->max_concurrency; ++lane) {
    for (std::uint32_t g = 0; g < mini::kGdnLayers; ++g) {
      self->lanes[lane].gdn_state[g] = static_cast<float*>(
          self->workspace.alloc_bytes(kGdnStateBytes).data);
    }
    for (std::uint32_t c = 0; c < mini::kGdnLayers; ++c) {
      self->lanes[lane].conv_state[c] = static_cast<__nv_bfloat16*>(
          self->workspace.alloc_bytes(kConvStateBytes).data);
    }
  }
  // Zero the recurrent lane state (GDN + conv). The GDN reads its initial
  // recurrent state on the first prefill; freshly-allocated device memory is
  // uninitialized and differs between Engine instances, so without this the
  // forward is non-deterministic. (Deterministic empty state.)
  {
    const cudaStream_t z = self->stream;
    for (std::uint32_t lane = 0; lane < self->max_concurrency; ++lane) {
      for (std::uint32_t g = 0; g < mini::kGdnLayers; ++g) {
        const cudaError_t e1 = cudaMemsetAsync(self->lanes[lane].gdn_state[g], 0,
                                              kGdnStateBytes, z);
        const cudaError_t e2 = cudaMemsetAsync(self->lanes[lane].conv_state[g], 0,
                                              kConvStateBytes, z);
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
          throw std::runtime_error(std::string("flash_next lane-state zero-init failed: ") +
                                   cudaGetErrorString(e1 == cudaSuccess ? e2 : e1));
        }
      }
    }
    const cudaError_t zs = cudaStreamSynchronize(z);
    if (zs != cudaSuccess) {
      throw std::runtime_error(std::string("flash_next lane-state zero-init sync failed: ") +
                               cudaGetErrorString(zs));
    }
    // Same for the QSA KV pool. QSA writes prompt KV before attention reads it
    // in the same round, so zeroing cannot change a correct result -- but a
    // freshly-allocated pool holds uninit bytes that differ between Engine
    // instances, and any read-before-write on them would break (c)
    // bit-identity. Zero the whole pool for a deterministic empty state.
    const cudaError_t ekv = cudaMemsetAsync(self->kv_pool.p, 0,
                                            kv_total_pages * kQsaKvPageBytes, z);
    const cudaError_t zk = cudaStreamSynchronize(z);
    if (ekv != cudaSuccess || zk != cudaSuccess) {
      throw std::runtime_error(std::string("flash_next kv-pool zero-init failed: ") +
                               cudaGetErrorString(ekv == cudaSuccess ? zk : ekv));
    }
  }
  self->free_pages.clear();
  self->free_pages.reserve(self->kv_total_pages);
  for (std::size_t page = 0; page < self->kv_total_pages; ++page) {
    self->free_pages.push_back(static_cast<std::int32_t>(page));
  }
  for (std::uint32_t lane = 0; lane < self->max_concurrency; ++lane) {
    self->lanes[lane].block_table_dev =
        reinterpret_cast<std::int32_t*>(
            self->block_tables.p) +
        static_cast<std::ptrdiff_t>(lane) * self->per_lane_pages;
  }
  self->zero_block.fill(0);

  // Dry run: the largest round working set + both op workspaces must fit the
  // round scratch region (arena throws std::bad_alloc past capacity).
  {
    auto scope = self->workspace.scope();
    self->workspace.alloc_bytes(round_working_set_bytes(kMaxRoundTokens));
    self->workspace.alloc_bytes(
        ops::gated_delta_net_workspace_capacity_bytes(2, 4, true, 1, 64));
    self->workspace.alloc_bytes(
        ops::sparse_moe_nvfp4_workspace_capacity_bytes(kMoeGeometry, 1, 64));
    (void)scope;
  }

  // Stage all 32 routed experts (4 layers x 8) into the device window and
  // verify the per-layer windows are contiguous in staging order.
  for (std::uint32_t layer = 0; layer < mini::kLayers; ++layer) {
    paging::PrefillWindow stage(*model.pager);
    stage.stage(layer);
  }
  model.pager->wait();
  for (std::uint32_t layer = 0; layer < mini::kLayers; ++layer) {
    const std::uint8_t* base = model.pager->device_window(layer, 0);
    for (std::uint32_t e = 1; e < mini::kMoEExperts; ++e) {
      const std::uint8_t* expected = base + e * ops::sparse_moe_nvfp4_expert_bytes(kMoeGeometry);
      if (model.pager->device_window(layer, e) != expected) {
        throw std::logic_error("flash_next: pager device window is not contiguous per layer");
      }
    }
    self->model.layers[layer].mlp.expert_window = base;
  }

  // `new` here runs in create_program's context (a friend of the private
  // Program ctor); std::make_unique would not have that access.
  return std::unique_ptr<Program>(new Program(std::move(self)));
}

// M2 real-mode factory. Opens the real 48x512 artifact + ngram via RealLoadedModel,
// builds the proven single-lane RealProgram (self-contained recurrent/KV/MoE state),
// and wraps it in the family's Program facade with the mini device buffers as size-1
// sentinels (never touched in real mode). `max_context` sizes the RealProgram's QSA
// paged-KV pool and the plan_request room; it must be >= the largest prompt the
// serve should accept (e.g. 100000 for the 100k gate).
std::unique_ptr<Program> create_real_program(const std::filesystem::path& artifact,
                                             const std::filesystem::path& ngram,
                                             std::uint32_t max_context, DeviceContext& device,
                                             ops::QsaIndexerKvDtype idx_dtype) {
  if (max_context == 0) {
    throw std::runtime_error("flash_next real: max_context must be > 0");
  }

  // Load the real artifact + ngram (weights arena + RealModelView + artifact reader).
  // `open` throws on a missing/corrupt artifact or a CUDA allocation error.
  auto model = RealLoadedModel::open(artifact, ngram, device);

  // The mini device buffers are non-default-constructible (the ProgramImpl ctor takes
  // them by move) and are NEVER touched in real mode (advance_prefill / decode /
  // ensure_pages / store_prefix / restore_prefix all dispatch to real_), so pass
  // size-1 sentinels rather than risking a 0-size cudaMalloc.
  auto self = std::make_unique<detail::ProgramImpl>(
      DeviceArena(1), DeviceBuffer(1), DeviceBuffer(1), DeviceBuffer(1),
      PinnedHostBuffer(1), PinnedHostBuffer(1), PinnedHostBuffer(1));

  self->real_mode_ = true;
  self->real_model_ = std::move(model);
  self->real_ = std::make_unique<RealProgram>(self->real_model_->view(), device, max_context,
                                              idx_dtype);
  self->max_context = max_context;
  self->max_concurrency = 1;  // real v1: one recurrent state set => single lane
  self->kv_capacity = max_context;  // report-only in real v1 (no mini page pool)
  self->kv_total_pages = 0;
  self->per_lane_pages = 0;
  self->stream = device.stream;
  self->device_ordinal = device.device;
  self->lanes.resize(1);  // one (inactive) lane; activated by plan_request / the engine

  // `new` here runs in create_real_program's context (a friend of the private
  // Program ctor); std::make_unique would not have that access.
  return std::unique_ptr<Program>(new Program(std::move(self)));
}

}  // namespace ninfer::targets::qwen3_8_flash_next
