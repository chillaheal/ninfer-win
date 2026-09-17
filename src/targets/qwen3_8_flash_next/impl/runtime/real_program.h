// Flash-Next S3 — the real-geometry standalone Program.
//
// A one-shot forward driver (NOT the 28-method engine Program in family.h): it
// composes the now-green oracle-qualified real ops (P6 GDN / P7 QSA / P8 MoE /
// P4 PLE / P5 HC) with the 13 S2 real glue leaves into a 48-layer real-geometry
// Program over the real 75.4 GB artifact, and runs a whole sequence in ONE round
// at pos0 = 0 (the QSA indexer is round-local: n = min(B, t+1) drawn from the
// round's own {0..t}, so a length-L sequence processed in a single round at
// pos0 = 0 is the correct L-token forward). S3 runs T = 1.
//
// Lifetime contract: RealProgram is a SHALLOW host-pointer view over
// RealLoadedModel's device bytes (embedding/output-head/GDN/QSA/MoE weights, the
// routed-expert host bases, the PLE table + conv weights). Construct it after
// RealLoadedModel::open and destroy it before the RealLoadedModel (the driver
// owns both on the same scope). The routed-expert scatter reads the artifact
// Reader's mmap via view.routed_host — that mmap is owned by the RealLoadedModel,
// so it must outlive every moe_block call.

#pragma once

#include <cuda_bf16.h>

#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/ngram_embedding.h"
#include "ninfer/ops/qsa_indexer_k_append.h"
#include "targets/qwen3_8_flash_next/impl/runtime/real_model_view.h"

namespace ninfer::targets::qwen3_8_flash_next {

// P13 M1: one round's timing breakdown (dev-only; the default path never
// touches this). GPU stream time per op bucket (CUDA events on the round
// stream, single stream so the buckets partition the stream-busy time), the
// host stall at the MoE router D2H gate, the summed expert-union size (it
// drives the H2D scatter bytes), and the driver-side wall around the whole
// run_sequence call.
struct RoundTimings {
  static constexpr int kBuckets = 12;
  // Bucket order (see the seg_begin/seg_end call sites in real_program.cpp).
  static const char* name(int bucket);
  float gpu_ms[kBuckets] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                            0.0F, 0.0F, 0.0F, 0.0F};
  // CPU-blocked time at the 48 MoE router D2H gates (sync + D2H + host top-10
  // union + the pageable-H2D staging copies of the scatter enqueues).
  float host_barrier_ms = 0.0F;
  // Stream time INSIDE the barrier windows (prior-op drain + scatter DMA).
  // Overlaps the buckets — reported for the pure-stall analysis, not summed.
  float barrier_gpu_ms = 0.0F;
  // Sum of the per-layer selected expert-union sizes (x 48 layers); the H2D
  // scatter bytes per round = moe_union_experts * 2,764,800.
  long long moe_union_experts = 0;
  // P13 M2: pinned expert LRU resolutions this round (a hit re-DMA's a resident
  // pinned blob; a miss evicts an LRU slot + copies the 4 planes from the mmap,
  // then DMA's). 0/0 = legacy pageable path (NINFER_EXPERT_PINNED_SLOTS=0).
  std::uint64_t lru_hits = 0;
  std::uint64_t lru_misses = 0;
  // Wall clock around run_sequence, set by the driver (includes the final
  // sample/logits D2H).
  float wall_ms = 0.0F;
};

// P13 M2: pinned-host LRU over (layer, expert) expert blobs — the P3 Pager's
// host-window pattern (paging/pager.cpp acquire_slot semantics: first free
// slot, else least-recently-used) for the real 4-plane routed layout. A slot
// holds one expert's 2,764,800 B laid out exactly like a device window slot.
// Loop-4 hybrid scatter: a hit is ONE contiguous pinned->device DMA (~26
// GB/s measured warm); a miss uses the M1 4-plane pageable DMAs (driver
// staging faults the mmap pages in at the M1 rate) and THEN promotes the
// expert into a slot. Measured: the promotion did NOT reach RAM speed (the
// per-expert cost is ~700 soft page faults at ~2.4-2.7 ms regardless of
// staging order), so every measured variant lost to the plain M1 pageable
// path on the re-prefill benchmark — the LRU is opt-in for stable-routing
// decode. Window bytes are immutable artifact data, so a
// hit that re-DMA's resident bytes changes no numerics.
//
// Drain-before-reuse (the P9 race fix, specialized): every H2D issued from a
// slot runs on the round stream, and moe_block's router gate starts with
// cudaStreamSynchronize, so a CPU promotion copy only ever overwrites a slot
// whose last DMA has already completed. Within one scatter the miss count is
// <= 512 < the slot count, so a just-written slot cannot be evicted (and
// re-written) again in the same scatter.
class ExpertPinnedLru {
 public:
  // slot_count = resident expert blobs; bytes_per_expert = per-blob size.
  // slot_count == 0 = disabled (legacy pageable scatter). On a cudaHostAlloc
  // failure the LRU logs a warning and degrades to DISABLED (enabled() ==
  // false, the scatter falls back to the legacy pageable path) instead of
  // throwing — a too-large pool must never crash a model load.
  ExpertPinnedLru() = default;
  explicit ExpertPinnedLru(std::size_t slot_count, std::size_t bytes_per_expert);
  ~ExpertPinnedLru();
  ExpertPinnedLru(const ExpertPinnedLru&)            = delete;
  ExpertPinnedLru& operator=(const ExpertPinnedLru&) = delete;

  // Enabled only when the pinned pool is actually allocated (base_ != nullptr),
  // not merely when slots_ is sized — so a failed allocation reads as disabled.
  [[nodiscard]] bool enabled() const { return base_ != nullptr; }
  [[nodiscard]] std::uint64_t hits() const { return hits_; }
  [[nodiscard]] std::uint64_t misses() const { return misses_; }
  void reset_counters() {
    hits_ = 0;
    misses_ = 0;
  }

  // M2 loop-4 (hybrid scatter): find() = hit test only (re-ticks on a hit,
  // nullptr on a miss — no eviction); promote() = evict the LRU victim (or
  // take a free slot) and copy the four source planes into the expert's slot
  // at the device window plane offsets (kPlane*). Host-side only — neither
  // touches the GPU.
  [[nodiscard]] std::uint8_t* find(std::uint32_t layer, std::uint32_t expert);
  void promote(std::uint32_t layer, std::uint32_t expert,
               const std::uint8_t* gu_codes, std::size_t gu_codes_bytes,
               const std::uint8_t* gu_scales, std::size_t gu_scales_bytes,
               const std::uint8_t* dn_codes, std::size_t dn_codes_bytes,
               const std::uint8_t* dn_scales, std::size_t dn_scales_bytes);

 private:
  struct Slot {
    std::uint64_t key = 0;  // (layer << 32) | expert
    bool in_use = false;
    std::uint64_t tick = 0;
  };
  std::vector<Slot> slots_;
  std::uint8_t* base_ = nullptr;  // pinned
  std::size_t slot_bytes_ = 0;
  std::uint64_t tick_ = 0;
  std::uint64_t hits_ = 0;
  std::uint64_t misses_ = 0;
};

// P13 M4 (tier-1): GPU-resident LRU over (layer, expert) expert blobs — the
// VRAM tier of the vram -> ram -> ssd expert residency hierarchy. A slot holds
// one expert's 2,764,800 B in a single DEVICE buffer, laid out exactly like a
// device window slot (kPlane* offsets, kSrc* sizes). A resident expert's routed
// GEMM reads its slot IN PLACE (zero DMA per access) via the op's per-expert
// pointer table (SparseMoeNvfp4Weights::expert_ptrs); a miss evicts the LRU
// victim (or takes a free slot), H2D's the four planes from the mmap on the
// round stream, and the scatter points the table at the fresh slot. The mmap
// (view.routed_host) is the RAM/SSD backing store, so evicting a GPU slot is a
// pure GPU -> RAM demotion with no extra machinery (the OS page cache keeps the
// expert's pages hot). O(1) hit-test via the key -> slot map.
//
// Two-phase safety (the P9 race fix, generalized over ExpertPinnedLru): the
// scatter (a) peeks EVERY selected expert first, recording each hit's slot, and
// (b) then promotes the misses, choosing each victim among FREE slots or the
// cold (non-recorded-hit) slots ONLY. A recorded hit's slot is therefore never
// overwritten by a same-scatter promote, so the GEMM (same stream, after the
// scatter) always reads the hit's intact blob. Across layers the router gate's
// cudaStreamSynchronize drains every prior GEMM before a promote may overwrite a
// slot, so a promote never clobbers a slot whose last read is still in flight.
// The only fallback: if |selected| exceeds the slot count (a big prefill round
// selecting more distinct experts than slots, i.e. slots < 512), the excess
// misses steer the GEMM at the window slot (pageable scatter) and are NOT cached
// — correct, just not accelerated. Production decode (T=1, ~480-key union) and
// a slots >= 512 config avoid the fallback entirely.
class ExpertGpuLru {
 public:
  // slot_count = resident expert blobs; bytes_per_expert = per-blob size.
  // slot_count == 0 = disabled (the legacy window scatter). The 2-arg ctor and
  // init() back off (halving) on a cudaMalloc miss and degrade to disabled on
  // total failure -- a too-large pool must never crash a model load.
  ExpertGpuLru() = default;
  explicit ExpertGpuLru(std::size_t slot_count, std::size_t bytes_per_expert);
  ~ExpertGpuLru();
  // P9 (dynamic expert-VRAM sizing): (re)allocate on a default-constructed or
  // disabled instance. RealProgram calls this AFTER every other VRAM allocation
  // (GDN state, QSA K/V + indexer pools, the 256 MB scratch) is resident, passing
  // a count derived from the true free VRAM (cudaMemGetInfo), so the tier adapts
  // to any max_context. Same back-off/degrade semantics as the ctor. Throws only
  // if called on an already-allocated instance.
  void init(std::size_t slot_count, std::size_t bytes_per_expert);
  ExpertGpuLru(const ExpertGpuLru&)            = delete;
  ExpertGpuLru& operator=(const ExpertGpuLru&) = delete;

  [[nodiscard]] bool enabled() const { return !slots_.empty(); }
  [[nodiscard]] std::size_t slot_count() const { return slots_.size(); }
  [[nodiscard]] std::uint64_t hits() const { return hits_; }
  [[nodiscard]] std::uint64_t misses() const { return misses_; }
  void reset_counters() {
    hits_ = 0;
    misses_ = 0;
  }

  // Hit-test ONLY (no eviction, no DMA). Returns the slot base (a device
  // pointer) if (layer, expert) is resident, else nullptr. On a hit it re-ticks
  // the slot to the LATEST tick. The scatter peeks EVERY selected expert BEFORE
  // any promote; because promote evicts the min-tick victim and a peeked hit now
  // carries the newest ticks, a promote's victim is always a cold slot — never a
  // same-scatter recorded hit — provided |selected| <= slot_count (the scatter's
  // "fits" guard). So the GEMM (same stream, after the scatter) always reads a
  // hit's intact blob.
  [[nodiscard]] const std::uint8_t* peek(std::uint32_t layer, std::uint32_t expert);

  // Evict the LRU victim (or take the first free slot) and enqueue the four-plane
  // H2D into that slot on `stream` (src = the expert's mmap planes, kSrc* sizes).
  // Inserts (layer, expert) -> slot. Returns the slot base. The caller is
  // responsible for steering the GEMM at the returned slot (table[e]); the H2D is
  // ASYNC on `stream` and the GEMM runs on the same stream after the scatter, so
  // the slot is populated by the time the GEMM reads it. Does NOT check for an
  // existing (layer, expert) — the scatter only promotes a peek() MISS.
  const std::uint8_t* promote(std::uint32_t layer, std::uint32_t expert,
                              const std::uint8_t* gu_codes,
                              const std::uint8_t* gu_scales,
                              const std::uint8_t* dn_codes,
                              const std::uint8_t* dn_scales,
                              cudaStream_t stream);

  // P10 fast pinned-source promote: like promote(), but `pinned_src` is ONE
  // contiguous pinned (cudaHostAlloc) blob already laid out at the kPlane*
  // offsets (a hit from ExpertPinnedLru::find, blob size == slot_bytes_).
  // Enqueues a SINGLE slot_bytes_ cudaMemcpyAsync into the slot instead of the
  // four-plane pageable H2D (~26 GB/s vs the pageable ~0.92 ms/expert). Same
  // victim-selection and same-stream async contract as promote(); the scatter
  // calls it only for a peek() MISS.
  const std::uint8_t* promote_contiguous(std::uint32_t layer, std::uint32_t expert,
                                         const std::uint8_t* pinned_src,
                                         cudaStream_t stream);

  // Seed helper (used by seed_expert_residency at startup): upsert = peek, or on
  // a miss, promote. Startup is single-threaded and drains after, so no
  // same-scatter eviction interplay.
  const std::uint8_t* upsert(std::uint32_t layer, std::uint32_t expert,
                             const std::uint8_t* gu_codes,
                             const std::uint8_t* gu_scales,
                             const std::uint8_t* dn_codes,
                             const std::uint8_t* dn_scales,
                             cudaStream_t stream);

 private:
  struct Slot {
    std::uint64_t key = 0;  // (layer << 32) | expert
    bool in_use = false;
    std::uint64_t tick = 0;
  };
  std::vector<Slot> slots_;
  std::unordered_map<std::uint64_t, std::size_t> idx_;  // key -> slot index
  std::uint8_t* base_ = nullptr;  // device
  std::size_t slot_bytes_ = 0;
  std::uint64_t tick_ = 0;
  std::uint64_t hits_ = 0;
  std::uint64_t misses_ = 0;
};

// A whole-sequence one-round forward. Returns the sampled (argmax) token for
// the last position. `input_ids` is the full token sequence (length T).
class RealProgram {
 public:
  // Max tokens in one state-carrying round (the MoE op's T-domain cap). A chunked
  // prefill (begin_sequence + repeated run_round) runs a long prompt in
  // <= kMaxRoundTokens-token rounds; the serve's real advance_prefill chunks to this.
  static constexpr std::int32_t kMaxRoundTokens = 2048;

  // Throws on a CUDA allocation error, an arena capacity overflow, or a D2H/H2D
  // copy failure. Zero-initializes all recurrent state (GDN + conv) and the QSA
  // KV pools so a fresh instance is bit-deterministic.
  // `max_context` (default 8320 = the P13 QSA paged-KV cap, 130 x 64) sizes the
  // QSA paged-KV pool to ceil(max_context / 64) pages per K/V pool + the identity
  // block table. The GDN recurrent state (context-independent, no position limit)
  // and the round-local QSA prefill (<=2048-token rounds) are ALREADY
  // context-independent, so a larger max_context (e.g. 100000 -> 1563 pages) only
  // grows the K/V pool (~2.5 GiB at 100k across the 12 QSA layers) and enables the
  // longer context. `idx_dtype` (default Fp8; Bf16 fallback) selects the indexer-K
  // pool storage dtype: the pool is sized to 1 B/elem (Fp8) or 2 B/elem (Bf16) and
  // the qsa_indexer logit GEMM reads it in that dtype. Throws on a CUDA allocation
  // error.
  explicit RealProgram(const RealModelView& view, DeviceContext& device,
                       std::uint32_t max_context = 8320,
                       ops::QsaIndexerKvDtype idx_dtype = ops::QsaIndexerKvDtype::Fp8);
  ~RealProgram();

  // P13 M1 timing mode: on, each run_sequence records per-bucket stream events
  // plus the host barrier windows and publishes the round's RoundTimings via
  // last_timings(). Off (default) the code path is untouched. Enabling creates
  // the event pool (throws on a CUDA failure).
  void set_timings_enabled(bool on);
  [[nodiscard]] const RoundTimings* last_timings() const {
    return timings_enabled_ ? &timings_ : nullptr;
  }

  RealProgram(const RealProgram&)            = delete;
  RealProgram& operator=(const RealProgram&) = delete;
  RealProgram(RealProgram&&)                 = delete;
  RealProgram& operator=(RealProgram&&)      = delete;

  // `final_logits` (optional, dev-only): when non-null, receives the final
  // position's full-vocab lm_head logit distribution as host floats (one per
  // vocab slot, converted from BF16) so a driver can confirm the end-to-end
  // logit scale (S4). The S0 ~1e21 MoE epilogue defect (if present at the real
  // geometry) propagates through the 48 real MoE layers to O(1e21) logits; a
  // healthy real stack reads ~O(10), the P8-oracle scale.
  [[nodiscard]] std::uint32_t run_sequence(const std::vector<std::uint32_t>& input_ids,
                                           std::vector<float>* final_logits = nullptr);

  // One state-carrying round of T tokens starting at absolute position pos0 (T in
  // [1, kMaxRoundTokens]). The GDN / conv / QSA-KV / PLE state accumulates across
  // calls (NO reset here); a fresh sequence is started with begin_sequence (or
  // run_sequence, which resets then runs a single pos0 = 0 round). Returns the
  // sampled token at the final position.
  [[nodiscard]] std::uint32_t run_round(const std::vector<std::uint32_t>& tokens, int pos0,
                                        std::vector<float>* final_logits = nullptr);

  // Start a fresh state-carrying sequence: reset the recurrent state (GDN / conv
  // / QSA-KV pools) AND zero the PLE (per-sequence prev1/prev2 + conv_history).
  // Use before a chunked prefill + decode (P13 M3 benchmark).
  void begin_sequence();

 private:
  // One GDN layer (l % 4 != 3): xhat [T][2560] -> y [T][2560].
  void gdn_block(std::uint32_t layer, int T, __nv_bfloat16* xhat, __nv_bfloat16* y);
  // One QSA layer (l % 4 == 3): xhat [T][2560] -> y [T][2560] (round-local, pos0).
  void qsa_block(std::uint32_t layer, int T, int pos0, __nv_bfloat16* xhat, __nv_bfloat16* y);
  // One MoE layer: xhat [T][2560] -> y [T][2560]. Host top-10 (real_router + D2H)
  // drives the per-token expert scatter into the shared moe_window; the op
  // re-runs routing deterministically and reads expert_window + e*expert_bytes.
  void moe_block(std::uint32_t layer, int T, __nv_bfloat16* xhat, __nv_bfloat16* y);

  // Zero every recurrent state buffer (36 GDN states + 36 conv states + 12 QSA K
  // pools + 12 QSA V pools). run_sequence calls this at its top so each call is a
  // self-contained pos0 = 0 forward: the documented contract is "a length-L
  // sequence in a single round at pos0 = 0" (the QSA indexer is round-local), so
  // the GDN/conv recurrent state and the QSA KV must start fresh each call. A
  // single call is unaffected (the ctor already zeroed them); a re-prefill
  // greedy (S5) re-feeds the growing prefix and relies on this reset.
  void reset_recurrent_state(cudaStream_t s);

  // Dump the expert-usage histogram to a file (NINFER_EXPERT_USAGE_FILE, default
  // in the CWD). Called from the dtor. No-op if the histogram is empty (no MoE
  // layer ran) or the file cannot be opened.
  void dump_expert_usage() const;

  // P13 M4: seed the expert residency hierarchy from a prior run's usage
  // histogram (NINFER_EXPERT_USAGE_FILE, default ninfer-expert-usage.txt). The
  // top-N most-used (layer, expert) blobs are upserted into the GPU LRU (VRAM);
  // the broader top-R set has its mmap pages faulted into RAM so a later H2D
  // scatter runs at RAM speed. No-op if the file is absent or both seeds are
  // disabled. Called at the end of the ctor.
  void seed_expert_residency();

  const RealModelView& view_;
  std::uint32_t qsa_layer_count_;  // 12 (for block_table / page sizing)
  // Dynamic QSA paged-KV pool width (ceil(max_context / 64) pages; the default
  // 8320 -> 130 pages, byte-identical to the P13 pool). The K/V pools + the
  // identity block tables are allocated to this width in the ctor.
  std::size_t qsa_page_count_ = 0;

  // One recurrent GDN state per GDN layer, in real GDN-layer order (l=0,1,2,4,...).
  // [128, 128, 48] FP32 = 3,145,728 B each (36 total). Zeroed at construction.
  std::vector<DeviceBuffer> gdn_states_;
  // One conv state per GDN layer. [10240, 3] BF16 = 61,440 B each. Zeroed.
  std::vector<DeviceBuffer> conv_states_;
  // One SEPARATE K pool and V pool per QSA layer, each a single 65,536 B page
  // (page-major [256, 64, KV=2, P=1] BF16). Zeroed.
  std::vector<DeviceBuffer> k_pools_;
  std::vector<DeviceBuffer> v_pools_;
  // One block_table per QSA layer; page 0 = physical page 0.
  std::vector<DeviceBuffer> block_tables_;
  // QSA indexer-K pool: one paged 512-dim key pool per QSA layer (page [512, 64]
  // FP8 E4M3, 32,768 B), reusing qsa_page_count_ pages and the SAME identity
  // block_tables_[i] (pages stay aligned with the main K/V pool). Zeroed.
  std::vector<DeviceBuffer> idx_k_pools_;
  // Per-token FP8 scale, one half per 64-token page (128 B/page; Fp8 only).
  std::vector<DeviceBuffer> idx_k_scale_;
  // Persistent indexer logit scratch [chunk, n_full] FP32 (the op chunks the M
  // dim so it stays <= kIndexerLogitsBytes; reused across every QSA layer).
  DeviceBuffer idx_logits_scratch_;
  // Indexer-K pool storage dtype (the logit GEMM reads it in this dtype).
  ops::QsaIndexerKvDtype indexer_kv_dtype_ = ops::QsaIndexerKvDtype::Fp8;
  // The shared routed-expert window: 512 * 2,764,800 B ~= 1.32 GiB, re-scattered
  // per layer (M2: from the pinned expert LRU, one contiguous DMA per expert;
  // the op reads expert_window + e * kExpertBytes with the global expert id).
  DeviceBuffer moe_window_;
  // P13 M2: the scatter staging — pinned expert LRU (NINFER_EXPERT_PINNED_SLOTS
  // slots; DEFAULT 0 = the M1 pageable path, retained as the measured best on
  // the re-prefill worst case: M1 122.7 s < 2400 133.4 s < hybrid 6000 162.8 s
  // < serial-miss 6000 168.1 s; root causes in expert_pinned_slot_count()).
  // Opt-in (e.g. 6000 slots = 15.44 GiB pinned) for production decode, where
  // stable T=1 routing retains ~95%+ of the ~480-key union and a pinned hit
  // is one contiguous DMA (~26 GB/s warm) vs ~1.93 ms/expert pageable.
  ExpertPinnedLru expert_lru_;
  // P13 M4 (tier-1): the GPU-resident expert hot-set — VRAM tier of the
  // vram -> ram -> ssd hierarchy. NINFER_EXPERT_GPU_SLOTS blobs (default 0 =
  // off). When enabled, moe_block's scatter routes every selected expert
  // through upsert(); the op reads each expert from the per-expert device
  // pointer table (moe_expert_ptrs_dev_) instead of the window.
  ExpertGpuLru expert_gpu_lru_;
  // Device + pinned-host copies of the per-expert pointer table the op reads
  // (G::kNumExperts device pointers). The host copy is rebuilt each scatter
  // (window fallback for non-resident experts, the LRU slot for resident ones)
  // and DMA'd to the device table in one shot.
  DeviceBuffer moe_expert_ptrs_dev_;
  PinnedHostBuffer moe_expert_ptrs_host_;
  // The round scratch arena (sized for the largest single round this class runs).
  DeviceArena workspace_;

  PinnedHostBuffer ids_host_;
  PinnedHostBuffer router_host_;
  PinnedHostBuffer sample_host_;
  PinnedHostBuffer logits_host_;  // S4: final-position lm_head logits (kVocab BF16)

  // Expert-usage histogram: one counter per (layer, expert) pair, sized
  // G::kNumLayers * G::kNumExperts (48 x 512). Incremented in moe_block for every
  // routed expert selected by the host top-k; flushed to a file on destruction
  // (NINFER_EXPERT_USAGE_FILE, default in the CWD) so the most-active experts are
  // inspectable across runs.
  std::vector<std::uint64_t> expert_usage_;

  cudaStream_t stream_ = nullptr;
  ops::NgramRequestState ple_state_;

  // P13 M1 timing plumbing (used only when set_timings_enabled(true)): a flat
  // event pool + the (bucket, pair) segments recorded this round + the barrier
  // window pair. Segments are strictly sequential (no nesting), so one pending
  // pair at a time is enough. finalize_round reads every pair (after the round's
  // final stream sync) into timings_ and resets the per-round state.
  struct Timer {
    static constexpr int kMaxPairs = 1024;
    std::vector<cudaEvent_t> ev;  // 2 * kMaxPairs events, created on enable
    int next_pair = 0;
    struct Seg {
      int bucket;
      int pair;
    };
    std::vector<Seg> segs;
    int barrier_pair = -1;
    float barrier_wall_ms = 0.0F;
    long long union_experts = 0;
    std::chrono::steady_clock::time_point barrier_t0;
  };
  void seg_begin(int bucket);
  void seg_end();
  void barrier_begin();
  void barrier_end();
  void finalize_round();
  Timer timer_;
  RoundTimings timings_;
  bool timings_enabled_ = false;
  int seg_pair_ = -1;
  int seg_bucket_ = -1;
  int seg_trace_layer_ = -1;
};

}  // namespace ninfer::targets::qwen3_8_flash_next
