#include "targets/qwen3_6/impl/runtime/instance.h"
// CPU vision encoder (I3). Runs the same op sequence as the GPU ViT on host FP32, snapping every
// op output to the BF16 grid (snap_bf16) so the result matches the GPU encode to within a few
// BF16 ULPs. The GPU differs only in GEMM/attention accumulation order, which the BF16 storage
// boundary absorbs. Weights are read from the retained host bytes (materialized_host_*), which the
// RowSplit/W8 dequant below decodes exactly as the GPU FP32Dequant used by the linear op.

#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_cpu_simd.h"
#include "targets/qwen3_6/impl/runtime/vision_cpu_pfor.h"  // persistent worker pool (pfor_light)

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include <ninfer/targets/qwen3_6/vision_control.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

// NINFER_VISION_PROFILE=1: per-op wallclock profile of encode_cpu. Main-thread checkpoints:
// mark(name) accumulates (now - last mark) into `name`; pfor spans include the thread join, so
// the table is encode wall time. Reset at encode start, printed after the H2D handoff. Off =
// one getenv at reset + a no-op per mark (negligible).
struct VisionProfile {
    std::vector<std::pair<const char*, double>> ms;
    std::chrono::steady_clock::time_point last{};
    bool on = false;
    void reset() {
        const char* e = std::getenv("NINFER_VISION_PROFILE");
        on            = e != nullptr && e[0] != '\0';
        ms.clear();
        last = std::chrono::steady_clock::now();
    }
    void mark(const char* name) {
        if (!on) { return; }
        const auto now = std::chrono::steady_clock::now();
        const double d = std::chrono::duration<double, std::milli>(now - last).count();
        last           = now;
        for (auto& kv : ms)
            if (kv.first == name) { kv.second += d; return; }
        ms.emplace_back(name, d);
    }
    void print() const {
        if (!on) { return; }
        double total = 0.0;
        std::fprintf(stderr, "[cpu] vision profile (ms):\n");
        for (const auto& kv : ms) {
            std::fprintf(stderr, "[cpu]   %-12s %9.2f\n", kv.first, kv.second);
            total += kv.second;
        }
        std::fprintf(stderr, "[cpu]   %-12s %9.2f\n", "total", total);
    }
};
inline VisionProfile g_vision_profile;

// ---- host float-point helpers (RNE; f32_to_bf16 mirrors bench/ops/ninfer_bench_common.h) ----
inline std::uint16_t f32_to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    const std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(u >> 16);
}

inline float bf16_to_f32(std::uint16_t h) {
    const std::uint32_t u = std::uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Snap an FP32 value onto the BF16 grid (the CPU's per-op-output rounding boundary).
inline float snap_bf16(float f) { return bf16_to_f32(f32_to_bf16(f)); }

// FP16 (half) to FP32, exact for subnormals (mantissa <= 1023 => mant * 2^-24 is exact in FP32).
inline float fp16_to_f32(std::uint16_t h) {
    const bool neg          = (h & 0x8000u) != 0;
    const std::uint32_t exp  = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x03FFu;
    float value;
    if (exp == 0u) {
        value = static_cast<float>(mant) * std::ldexp(1.0F, -24);
    } else if (exp == 0x1Fu) {
        value = (mant == 0u) ? std::numeric_limits<float>::infinity()
                             : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = (1.0F + static_cast<float>(mant) / 1024.0F) *
                std::ldexp(1.0F, static_cast<int>(exp) - 15);
    }
    return neg ? -value : value;
}

// Parallel-for over count units; each unit runs body(unit). The bodies must write only
// disjoint memory per unit (shared inputs are read-only). The scalar kernels use the lighter
// pfor_light variant (threshold 64 units) of the shared persistent worker pool in
// vision_cpu_pfor.h — the old per-call std::thread spawn+join is gone (2026-09-02).

// Dequant a RowSplitK128 (Q4/Q5/Q6) or W8 RowSplitK128 weight into a flat FP32 [k*n] buffer
// (K-major, k fastest), matching the GPU FP32Dequant values. The GEMMs vectorize over N, so the
// weights must be k-major for contiguous 16-lane loads; dequants write the layout directly (the
// store is strided by n*4, which is cheap next to the per-element decode work). Flat padded
// element e = n*padded_columns + k; group g = e/group, in-group index j = e%group. The dequant is
// parallel over rows.
void dequant_weight(const Weight& w, std::vector<float>& out) {
    const std::int32_t n     = w.n;
    const std::int32_t k     = w.k;
    const std::int32_t group = w.group;               // 64 (Q4/Q5/Q6) or 32 (W8)
    const std::int32_t padded = w.padded_shape[1];
    const QType qt            = w.qtype;
    out.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
    if (cpu_simd::has_avx512f()) {
        cpu_simd::DequantKind kind;
        switch (qt) {
        case QType::Q4G64_F16S: kind = cpu_simd::DequantKind::Q4G64; break;
        case QType::Q5G64_F16S: kind = cpu_simd::DequantKind::Q5G64; break;
        case QType::Q6G64_F16S: kind = cpu_simd::DequantKind::Q6G64; break;
        case QType::W8G32_F16S: kind = cpu_simd::DequantKind::W8G32; break;
        default: throw std::logic_error("Vision CPU dequant: unsupported qtype");
        }
        cpu_simd::dequant_weight_into(kind, static_cast<const std::uint8_t*>(w.qdata),
                                      static_cast<const std::uint8_t*>(w.qhigh),
                                      static_cast<const std::uint16_t*>(w.scales), n, k, group,
                                      padded, out.data());
        return;
    }
    const std::uint8_t* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const std::uint8_t* high   = static_cast<const std::uint8_t*>(w.qhigh);
    const std::uint16_t* scales = static_cast<const std::uint16_t*>(w.scales);
    float* outdata             = out.data();
    pfor_light(static_cast<std::size_t>(n), [&](std::size_t r) {
        const std::int32_t row = static_cast<std::int32_t>(r);
        const std::size_t g0 = static_cast<std::size_t>(row) * static_cast<std::size_t>(padded) /
                               static_cast<std::size_t>(group);
        for (std::size_t gg = 0; gg * static_cast<std::size_t>(group) < static_cast<std::size_t>(k);
             ++gg) {
            const std::size_t g       = g0 + gg;
            const float scale         = fp16_to_f32(scales[g]);
            const std::size_t k0      = gg * static_cast<std::size_t>(group);
            const std::size_t kn      = std::min(k0 + static_cast<std::size_t>(group),
                                                 static_cast<std::size_t>(k));
            for (std::size_t j = k0; j < kn; ++j) {
                const std::size_t jj = j - k0;
                float wval;
                switch (qt) {
                case QType::Q4G64_F16S: {
                    const std::uint8_t c = codes[g * 32 + jj / 2];
                    const int q          = ((jj & 1) ? (c >> 4) : (c & 0x0f)) ^ 0x08;
                    wval                 = (static_cast<float>(q) - 8.0F) * scale;
                    break;
                }
                case QType::Q5G64_F16S: {
                    const std::uint8_t c = codes[g * 32 + jj / 2];
                    int q                = (jj & 1) ? (c >> 4) : (c & 0x0f);
                    q |= (static_cast<int>(high[g * 8 + (jj >> 3)] >> (jj & 7)) & 1) << 4;
                    wval                 = (static_cast<float>(q ^ 0x10) - 16.0F) * scale;
                    break;
                }
                case QType::Q6G64_F16S: {
                    const std::uint8_t c = codes[g * 32 + jj / 2];
                    int q                = (jj & 1) ? (c >> 4) : (c & 0x0f);
                    q |= (static_cast<int>(high[g * 16 + (jj >> 2)] >> ((jj & 3) << 1)) & 3) << 4;
                    wval                 = (static_cast<float>(q ^ 0x20) - 32.0F) * scale;
                    break;
                }
                case QType::W8G32_F16S: {
                    wval = static_cast<float>(static_cast<std::int8_t>(codes[g * 32 + jj])) * scale;
                    break;
                }
                default:
                    throw std::logic_error("Vision CPU dequant: unsupported qtype");
                }
                outdata[j * static_cast<std::size_t>(n) + row] = wval;
            }
        }
    });
}

// Process-lifetime cache of dequantized vision weights (2026-09-02, #19 STEP 8). encode_cpu
// used to re-dequant ALL ~111 vision weights per call (linear_into -> dequant_weight into a
// local wbuf): 71-75 ms warm / 162.82 ms cold-mmap per image (2026-09-02 A/B profile). The
// weights are immutable for the model's lifetime, so each Weight's fp32 dequant output is
// computed ONCE and reused by every subsequent encode. Keyed by the Weight pointer: this
// product loads one model per process (serve / CLI / A-B tool), so the pointer is stable for
// the cache's lifetime. Resident cost: ~1.75 GB fp32 (the full ViT+merger weight set).
// Bit-safe: an entry is the exact dequant_weight output — only the buffer lifetime moves, the
// GEMM input is unchanged. Thread-safe: the mutex guards the map; entries are never mutated
// after insertion (two concurrent first-users dequant twice and publish one; the value is
// identical). NINFER_VISION_DEQUANT_CACHE=0 restores the per-call dequant (measurement
// control; a thread-local buffer keeps the old per-call semantics).
class DequantCache {
    std::mutex mtx;
    std::unordered_map<const Weight*, std::vector<float>> entries;
    std::unordered_map<const Weight*, std::vector<std::uint32_t>> entries32p;  // k-pair bf16 (WBF16)

public:
    static DequantCache& instance() {
        static DequantCache cache;  // Meyers: first linear_into, released at exit
        return cache;
    }

    // NINFER_VISION_DEQUANT_CACHE=0 restores the per-call dequant (measurement control; a
    // thread-local buffer keeps the old per-call semantics).
    static bool cache_on() {
        static const bool v = [] {
            const char* e = std::getenv("NINFER_VISION_DEQUANT_CACHE");
            return e == nullptr || e[0] == '\0' || std::string(e) != "0";
        }();
        return v;
    }

    // NINFER_VISION_CACHE_MAX_K = the k threshold for per-call priming (prime_per_call).
    // DEFAULT (2026-09-02, #19 STEP 11 — fc2_v2_bench evidence): 65536, i.e. ABOVE every
    // vision k (max = m_fc1/m_fc2 4608), so `k > cache_max_k` is false for all weights and
    // NOTHING is primed per call — every weight is CACHED (fp32 entry + k-pair bf16 u32) and
    // run through gemm_tn_bf16w (WBF16). The STEP-9 per-call fp32 priming of fc2 is the only
    // weight the old default (2048) primed; the fc2_v2_bench (2026-09-02, same-window A/B,
    // 10/10 rounds) refuted its premise for the bf16 W: fp32 fresh-write GEMM 7.3 ms vs bf16
    // fresh-write 5.2 ms (B/A 0.734 < 0.75 -> W-traffic bound; the 18.9 MB fp32 W read 7x
    // = 132 MB vs the 9.5 MB u32 read 7x = 66.5 MB), and the fresh-write L3 effect is ~0 in
    // isolation (fp32 no-refill 7.24 ms == fp32 fresh-write 7.3 ms). Caching fc2's u32 (9.5
    // MB, immutable) is consistent with every other weight. Set NINFER_VISION_CACHE_MAX_K=2048
    // to restore the STEP-9 per-call fc2 fp32 priming (the A/B measurement control).
    static int cache_max_k() {
        static const int v = [] {
            const char* e = std::getenv("NINFER_VISION_CACHE_MAX_K");
            if (e == nullptr || e[0] == '\0') { return 65536; }
            return std::atoi(e);
        }();
        return v;
    }

    // NINFER_VISION_CACHE_MAX_W = byte threshold (0 = never prime, cache everything).
    static std::size_t cache_max_w() {
        static const std::size_t v = [] {
            const char* e = std::getenv("NINFER_VISION_CACHE_MAX_W");
            if (e == nullptr || e[0] == '\0') {
                return static_cast<std::size_t>(24u) * 1024u * 1024u;
            }
            return static_cast<std::size_t>(std::atol(e));
        }();
        return v;
    }

    // Per-weight cache policy (2026-09-02, #19 STEP 9; DEFAULT CHANGED STEP 11). With the
    // default cache_max_k() = 65536 (STEP 11, fc2_v2_bench evidence) this gate is false for
    // EVERY weight -> nothing is primed per call; every weight is cached and run through
    // gemm_tn_bf16w. The gate remains the mechanism by which NINFER_VISION_CACHE_MAX_K=2048
    // restores the STEP-9 per-call fp32 priming (fc2 only: k=4304 > 2048 and W 18.9 MB <=
    // 24 MB). Historical STEP-9 rationale (why per-call priming ever existed): the per-call
    // fresh dequant WRITE write-allocates the W into L3, and the tg-outer GEMM (k > 2048,
    // which streams the full W per tg pass) then reads it from L3 instead of DRAM — measured
    // same-window, clean window, timed encode: fc2 GEMM 256.80 per-call vs 364.15 cached
    // (27 x 9.5 vs 13.5 ms/call; the 19.8 MB W is read 7 x per call — 138 MB of L3 vs DRAM
    // traffic). STEP 11 refuted the premise for the bf16 W (see cache_max_k's comment): the
    // k-pair u32 halves the W traffic and the fresh-write effect is ~0, so caching wins.
    // The MERGER weights (m_fc1 85 MB, m_fc2 94 MB) were never prime-eligible: their W cannot
    // fit the L3-fit guard (k*n*4 <= cache_max_w, 24 MB), so they are always cached.
    // Shared with linear_into's NINFER_VISION_WBF16 gate (2026-09-02, STEP 10c): the bf16
    // path must NOT prime per call (a per-call bf16 dequant needs a bf16-output dequant
    // kernel; v1 converts the CACHED fp32 entry instead).
    static bool prime_per_call(const Weight& w) {
        return w.k > cache_max_k() && w.k != 0 &&
               static_cast<std::size_t>(w.k) * static_cast<std::size_t>(w.n) * 4u <= cache_max_w();
    }

    // The fp32 dequant of `w`: dequant + store on first use, reuse afterwards.
    const std::vector<float>& get(const Weight& w) {
        if (!cache_on() || prime_per_call(w)) {
            static thread_local std::vector<float> local;  // per-call semantics (the old wbuf)
            dequant_weight(w, local);
            return local;
        }
        std::lock_guard<std::mutex> lk(mtx);
        auto it = entries.find(&w);
        if (it != entries.end()) { return it->second; }
        std::vector<float> out;
        dequant_weight(w, out);
        return entries.emplace(&w, std::move(out)).first->second;
    }

    // The k-PAIR bf16 repack of `w` (get32p): the RNE snap (identical per-element op to
    // f32_to_bf16) of the cached fp32 dequant, packed two k rows per u32 —
    // out[kp*N + n] = (bf16 of row 2kp+1) << 16 | (bf16 of row 2kp), kp < (K+1)/2 (an odd K
    // zero-pads its last k row's high half) — computed ONCE from get() (which dequants on
    // first use) and reused afterwards. The k-pair layout is what lets gemm_tn_bf16w split the
    // two k rows into fp32 with and/shift only (no cross-lane merge — this 9950X3D's
    // allocation-dependent erratum on those ops, see vision_cpu_simd.cpp). Resident cost:
    // ~0.88 GB (u32 Kp*N ≈ u16 K*N for even K; both maps stay resident — get32p sources its
    // conversion from the fp32 entry). Feeds gemm_tn_bf16w via linear_into when WBF16 is on
    // (default; NINFER_VISION_WBF16=0 opts out; 2026-09-02, #19 STEP 10c): halves the W
    // traffic per k-step.
    // Caller contract: only for CACHED weights (!prime_per_call) — for a primed weight get()
    // dequants into a thread-local buffer per call, so get32p would re-convert that fresh
    // buffer every call (linear_into gates on !prime_per_call).
    const std::vector<std::uint32_t>& get32p(const Weight& w) {
        const std::vector<float>& f32 = get(w);
        std::lock_guard<std::mutex> lk(mtx);
        auto it = entries32p.find(&w);
        if (it != entries32p.end()) { return it->second; }
        const int K = w.k, N = w.n;
        const int Kp = (K + 1) / 2;
        std::vector<std::uint32_t> out(static_cast<std::size_t>(Kp) * N);
        auto snap = [](float f) {
            std::uint32_t u;
            std::memcpy(&u, &f, 4);
            const std::uint32_t lsb = (u >> 16) & 1u;
            return static_cast<std::uint32_t>((u + 0x7fffu + lsb) >> 16);
        };
        for (int kp = 0; kp < Kp; ++kp) {
            const float* k0 = f32.data() + static_cast<std::size_t>(2 * kp) * N;
            std::uint32_t* o0 = out.data() + static_cast<std::size_t>(kp) * N;
            if (2 * kp + 1 < K) {
                const float* k1 = f32.data() + static_cast<std::size_t>(2 * kp + 1) * N;
                for (int n = 0; n < N; ++n) { o0[n] = (snap(k1[n]) << 16) | snap(k0[n]); }
            } else {
                for (int n = 0; n < N; ++n) { o0[n] = snap(k0[n]); }
            }
        }
        return entries32p.emplace(&w, std::move(out)).first->second;
    }
};

// out[t][n] = sum_k W[k][n] * x[t][k]; W is [K*N] k-major (k fastest, see dequant_weight),
// x is [T][K] (k fastest), out is [T][N] (n fastest). Reads the weight's cached fp32 dequant
// (first use dequants; the process-lifetime cache reuses it, see DequantCache) — or, by
// default (NINFER_VISION_WBF16=0 opts out) when the weight is not a per-call primed one, the
// cached k-pair bf16 repack (DequantCache::get32p) through gemm_tn_bf16w — then reduces.
// Every output is BF16-snapped to match the GPU storage boundary. `out` must have at
// least T*N floats.
void linear_into(const Weight& w, const float* x, int T, int K, float* out) {
    const int N = w.n;
    if (K != w.k) { throw std::logic_error("Vision CPU GEMM: input K != weight k"); }
    // DEFAULT (2026-09-02, #19 STEP 10c — A/B verified: encode 0.73-0.74 s vs 0.90-0.95 s
    // fp32, A/B gate PASS with bf16-noise ULPs; NINFER_VISION_WBF16=0 opts out): run the GEMM
    // on the CACHED k-pair bf16 W (get32p) — halves the W traffic per k-step (64 B per 2-k x
    // 16-col span = 2 B per (k, column) vs gemm_tn's 4 B) and the W cache footprint (0.88 vs
    // 1.75 GB). The per-lane
    // FMA order and the fp32 accumulation are unchanged (the k-pair loop steps k by 2, FMAing
    // k0 then k0+1 per pair — the exact per-k sequence of gemm_tn); the only numeric delta vs
    // the fp32 path is the W's own bf16 snap (<= 1 bf16 ULP per element, mostly absorbed by the
    // output bf16 snap — the A/B 8-ULP gate checks).
    // STEP 11 (2026-09-02, fc2_v2_bench): the W-gate weights (K>2048, W<=24 MB — fc2) are now
    // CACHED by default too (cache_max_k() default raised above the max vision k), so they
    // take this bf16 path as well — the "v2 candidate" (a bf16-output dequant kernel for
    // per-call bf16 priming) is no longer needed: the k-pair u32 halves the W traffic and the
    // fresh-write L3 effect is ~0 (bench: fp32 fresh 7.3 vs bf16 fresh 5.2 vs fp32 no-refill
    // 7.24 ms/call). NINFER_VISION_CACHE_MAX_K=2048 restores the STEP-9 per-call primed fp32
    // path for fc2 (the A/B measurement control; it falls through to the fp32 GEMM below).
    // Gated on AVX-512 (the k-pair split is zmm ops; the scalar fallback stays fp32) and on
    // the dequant cache (get32p sources its one-time conversion from the fp32 entry).
    // Default ON (measured win, A/B 2026-09-02: encode 0.73-0.74 s vs 0.90-0.95 s fp32);
    // NINFER_VISION_WBF16=0 restores the fp32 W path.
    static const bool wbf16_on = [] {
        const char* e = std::getenv("NINFER_VISION_WBF16");
        return e == nullptr || e[0] != '0';
    }();
    if (wbf16_on && cpu_simd::has_avx512f() && DequantCache::cache_on() &&
        !DequantCache::prime_per_call(w)) {
        const std::vector<std::uint32_t>& w32p = DequantCache::instance().get32p(w);
        g_vision_profile.mark("dequant");
        cpu_simd::gemm_tn_bf16w(x, w32p.data(), out, T, N, K);
        return;
    }
    const std::vector<float>& wbuf = DequantCache::instance().get(w);
    g_vision_profile.mark("dequant");
    // NINFER_VISION_PRIME=1 (2026-09-02, #19 STEP 10): explicit L3 priming of the CACHED W.
    // (The STEP-9 premise behind this knob — the per-call fresh dequant of the K > 2048
    // (tg-outer) weights write-allocating W into L3, measured fc2 9.5 vs 13.5 ms/call — is now
    // OPT-IN: STEP 11 caches fc2 by default, so that priming only happens with
    // NINFER_VISION_CACHE_MAX_K=2048.) The K <= 2048 (ng-outer) weights are CACHED, so their
    // W sits in DRAM between calls and the ng-outer L2-slice fetch streams it from DRAM. A full
    // 16-thread scratch copy of the W immediately before the GEMM write-allocates it into L3
    // (cost: W bytes of read+write traffic). If the ng-outer GEMM is DRAM-W-bound, this should
    // transfer the fc2 priming win to qkv/fc1 (the STEP-10 experiment knob; default OFF).
    // Only the ng-outer regime qualifies (k <= 2048 = same boundary as the W-gate cache
    // default); the L3-fit guard is a no-op for the current vision shapes (largest K <= 2048 W
    // is 19.8 MB < 24 MB) but keeps the knob safe against future shapes.
    static const bool prime_on = [] {
        const char* e = std::getenv("NINFER_VISION_PRIME");
        return e != nullptr && e[0] == '1';
    }();
    if (prime_on && w.k <= 2048 &&
        static_cast<std::size_t>(w.k) * static_cast<std::size_t>(w.n) * 4u <=
            static_cast<std::size_t>(24u) * 1024u * 1024u) {
        static std::vector<float> prime_buf;  // dispatcher-thread only: pfor's barrier makes
                                              // linear_into calls sequential, no reentry.
        const std::size_t total =
            static_cast<std::size_t>(K) * static_cast<std::size_t>(N);
        prime_buf.resize(total);
        const float* src = wbuf.data();
        float* dst = prime_buf.data();
        pfor_light(total, [&](std::size_t i) { dst[i] = src[i]; });
        g_vision_profile.mark("prime");
    }
    const float* W = wbuf.data();
    if (cpu_simd::has_avx512f()) {
        cpu_simd::gemm_tn(x, W, out, T, N, K);
        return;
    }
    pfor_light(static_cast<std::size_t>(T), [&](std::size_t r) {
        const int t        = static_cast<std::int32_t>(r);
        const float* xcol  = x + static_cast<std::size_t>(t) * static_cast<std::size_t>(K);
        float* orow        = out + static_cast<std::size_t>(t) * static_cast<std::size_t>(N);
        for (int n = 0; n < N; ++n) { orow[n] = 0.0F; }
        for (int kk = 0; kk < K; ++kk) {
            const float xk   = xcol[kk];
            const float* wk  = W + static_cast<std::size_t>(kk) * static_cast<std::size_t>(N);
            for (int n = 0; n < N; ++n) { orow[n] += xk * wk[n]; }
        }
        for (int n = 0; n < N; ++n) { orow[n] = snap_bf16(orow[n]); }
    });
}

// x[t][d] += bias[d] (bias is BF16 [D]); one BF16 boundary.
void add_bias_into(const Tensor& bias, int T, int D, float* x) {
    const std::uint16_t* b = static_cast<const std::uint16_t*>(bias.data);
    std::vector<float> bf(D);
    for (int d = 0; d < D; ++d) { bf[d] = bf16_to_f32(b[d]); }
    if (cpu_simd::has_avx512f()) {
        cpu_simd::snap_add_bias_into(bf.data(), D, static_cast<std::size_t>(T) * static_cast<std::size_t>(D), x);
        return;
    }
    pfor_light(static_cast<std::size_t>(T), [&](std::size_t r) {
        const int t   = static_cast<std::int32_t>(r);
        float* row    = x + static_cast<std::size_t>(t) * static_cast<std::size_t>(D);
        for (int d = 0; d < D; ++d) { row[d] = snap_bf16(row[d] + bf[d]); }
    });
}

// out[t][d] = (x[t][d]-mean)/sqrt(var+eps)*weight[d]+bias[d]; mean/var over d (weight/bias BF16 [D]).
void layer_norm_into(const float* x, const Tensor& weight, const Tensor& bias, float eps, int T,
                     int D, float* out) {
    const std::uint16_t* w = static_cast<const std::uint16_t*>(weight.data);
    const std::uint16_t* b = static_cast<const std::uint16_t*>(bias.data);
    std::vector<float> wf(D), bf(D);
    for (int d = 0; d < D; ++d) {
        wf[d] = bf16_to_f32(w[d]);
        bf[d] = bf16_to_f32(b[d]);
    }
    const float inv_d = 1.0F / static_cast<float>(D);
    if (cpu_simd::has_avx512f()) {
        cpu_simd::layer_norm_into(x, wf.data(), bf.data(), inv_d, eps, T, D, out);
        return;
    }
    pfor_light(static_cast<std::size_t>(T), [&](std::size_t r) {
        const int t         = static_cast<std::int32_t>(r);
        const float* xr     = x + static_cast<std::size_t>(t) * static_cast<std::size_t>(D);
        float* op          = out + static_cast<std::size_t>(t) * static_cast<std::size_t>(D);
        float mean          = 0.0F;
        for (int d = 0; d < D; ++d) { mean += xr[d]; }
        mean *= inv_d;
        float var           = 0.0F;
        for (int d = 0; d < D; ++d) {
            const float dx = xr[d] - mean;
            var += dx * dx;
        }
        var *= inv_d;
        const float rstd    = 1.0F / std::sqrt(var + eps);
        for (int d = 0; d < D; ++d) {
            op[d] = snap_bf16((xr[d] - mean) * rstd * wf[d] + bf[d]);
        }
    });
}

// x[i] += y[i] (both same-shaped FP32, already BF16-snapped); one BF16 boundary.
void residual_add_into(const float* y, std::size_t count, float* x) {
    if (cpu_simd::has_avx512f()) {
        cpu_simd::snap_add_into(y, count, x);
        return;
    }
    pfor_light(count, [&](std::size_t i) { x[i] = snap_bf16(x[i] + y[i]); });
}

enum class CpuGelu { Exact, Tanh };

// x[i] = GELU(x[i]) (one BF16 boundary).
void gelu_into(float* x, std::size_t count, CpuGelu mode) {
    if (mode == CpuGelu::Tanh && cpu_simd::has_avx512f()) {
        cpu_simd::gelu_tanh_into(x, count);
        return;
    }
    const float s2 = std::sqrt(2.0F);
    const float sp = std::sqrt(2.0F / 3.14159265358979323846F);
    pfor_light(count, [&](std::size_t i) {
        const float z = x[i];
        const float r = (mode == CpuGelu::Exact)
                            ? 0.5F * z * (1.0F + std::erf(z / s2))
                            : 0.5F * z * (1.0F + std::tanh(sp * (z + 0.044715F * z * z * z)));
        x[i] = snap_bf16(r);
    });
}

// Vision 2-D RoPE in place over the q and k parts of qkv [T][3D] (d fastest). For pair i in
// [0,rotary_dim/2): axis=(i<18)?0:1, local=i%18, phi=position[t][axis]*theta^(-2*local/36);
// x[i]=x[i]*cos-x[i+R/2]*sin, x[i+R/2]=x[i+R/2]*cos+x[i]*sin. head_dim==rotary_dim==72.
// pos is the SPLIT control layout: [0..T)=y, [T..2T)=x (control.cpp appends all y's then all
// x's; the CUDA kernel reads positions[axis * tokens + token]) -- not an interleaved [T][2].
void rope_vision_into(const std::int32_t* pos, int T, int D, float* qkv) {
    constexpr int R2    = VisionScheduleConfig::rotary_dim / 2;  // 36
    constexpr int H     = VisionScheduleConfig::heads;           // 16
    constexpr int HD    = VisionScheduleConfig::head_dim;        // 72
    const float theta   = VisionScheduleConfig::rope_theta;
    if (cpu_simd::has_avx512f()) {
        cpu_simd::rope_vision_into(pos, T, D, H, HD, R2, theta, qkv);
        return;
    }
    const float inv36   = 1.0F / static_cast<float>(R2);
    std::vector<std::vector<float>> cosv(T), sinv(T);
    for (int t = 0; t < T; ++t) {
        cosv[t].resize(R2);
        sinv[t].resize(R2);
        const std::int32_t py = pos[t];
        const std::int32_t px = pos[T + t];
        for (int i = 0; i < R2; ++i) {
            const int axis  = (i < 18) ? 0 : 1;
            const int local = i % 18;
            const double phi = static_cast<double>(axis == 0 ? py : px) *
                               std::pow(static_cast<double>(theta),
                                        -2.0 * static_cast<double>(local) * inv36);
            cosv[t][i] = static_cast<float>(std::cos(phi));
            sinv[t][i] = static_cast<float>(std::sin(phi));
        }
    }
    pfor_light(static_cast<std::size_t>(T), [&](std::size_t r) {
        const int t       = static_cast<std::int32_t>(r);
        float* qrow       = qkv + static_cast<std::size_t>(t) * static_cast<std::size_t>(3 * D);
        float* krow       = qrow + D;
        const float* cv   = cosv[t].data();
        const float* sv   = sinv[t].data();
        for (int h = 0; h < H; ++h) {
            float* qv     = qrow + static_cast<std::size_t>(h) * HD;
            float* kv     = krow + static_cast<std::size_t>(h) * HD;
            for (int i = 0; i < R2; ++i) {
                const float c   = cv[i];
                const float s   = sv[i];
                const float qa  = qv[i];
                const float qb  = qv[i + R2];
                qv[i]           = snap_bf16(qa * c - qb * s);
                qv[i + R2]      = snap_bf16(qb * c + qa * s);
                const float ka  = kv[i];
                const float kb  = kv[i + R2];
                kv[i]           = snap_bf16(ka * c - kb * s);
                kv[i + R2]      = snap_bf16(kb * c + ka * s);
            }
        }
    });
}

// Packed block-diagonal dense attention (equal-length segments), D=head_dim, Hq=Hkv=heads, kvh=h.
// q/k/v read from qkv [T][3D] (d fastest): q at [0,D), k at [D,2D), v at [2D,3D), head h at h*head_dim.
// Each segment [s*L,(s+1)*L) is independent non-causal dense; out [T][D]. Stable softmax in FP32.
void attention_into(float scale, int segment_length, int T, int D, const float* qkv, float* out) {
    constexpr int H     = VisionScheduleConfig::heads;
    constexpr int HD    = VisionScheduleConfig::head_dim;
    const int SL        = segment_length;
    if (SL <= 0 || T % SL != 0) {
        throw std::invalid_argument("Vision attention: segment_length must divide token count");
    }
    if (cpu_simd::has_avx512f()) {
        cpu_simd::attention_into(scale, SL, T, D, H, HD, qkv, out);
        return;
    }
    pfor_light(static_cast<std::size_t>(T), [&](std::size_t r) {
        const int i     = static_cast<std::int32_t>(r);
        const int ib    = (i / SL) * SL;  // segment begin
        const float* qi = qkv + static_cast<std::size_t>(i) * static_cast<std::size_t>(3 * D);
        float* oi       = out + static_cast<std::size_t>(i) * static_cast<std::size_t>(D);
        std::vector<float> score(SL);
        for (int h = 0; h < H; ++h) {
            const float* qh = qi + static_cast<std::size_t>(h) * HD;
            float maxs      = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < SL; ++j) {
                const int jj     = ib + j;
                const float* kj  = qkv + static_cast<std::size_t>(jj) * static_cast<std::size_t>(3 * D) +
                                   D + static_cast<std::size_t>(h) * HD;
                float dot        = 0.0F;
                for (int d = 0; d < HD; ++d) { dot += qh[d] * kj[d]; }
                const float s    = scale * dot;
                score[j]         = s;
                if (s > maxs) { maxs = s; }
            }
            float sum           = 0.0F;
            for (int j = 0; j < SL; ++j) {
                const float e    = std::exp(score[j] - maxs);
                score[j]         = e;
                sum             += e;
            }
            const float inv      = 1.0F / sum;
            for (int d = 0; d < HD; ++d) {
                float acc        = 0.0F;
                for (int j = 0; j < SL; ++j) {
                    const int jj     = ib + j;
                    const float* vj  = qkv + static_cast<std::size_t>(jj) * static_cast<std::size_t>(3 * D) +
                                       2 * D + static_cast<std::size_t>(h) * HD;
                    acc += score[j] * inv * vj[d];
                }
                oi[static_cast<std::size_t>(h) * HD + d] = snap_bf16(acc);
            }
        }
    });
}

// Four-corner interpolated position-embedding add: x[d,p] += sum_c table[d, idx[c,p]] * w[c,p].
// table is the [hidden, position_embeddings] BF16 view (element (d,r) at d + r*hidden); one BF16
// boundary applied after the four-term sum.
// idx/wts are PATCH-MAJOR (p*4 + c) -- control.cpp appends the four corners per patch and the
// CUDA kernels read indices[patch*4 + corner]; a corner-major read here selects the wrong rows.
void pos_embed_add_into(const Tensor& table, const std::int32_t* idx, const float* wts, int P,
                        int D, float* x) {
    const std::uint16_t* tbl = static_cast<const std::uint16_t*>(table.data);
    const int R             = table.ne[1];
    pfor_light(static_cast<std::size_t>(P), [&](std::size_t r) {
        const int p   = static_cast<std::int32_t>(r);
        float* xrow   = x + static_cast<std::size_t>(p) * static_cast<std::size_t>(D);
        for (int c = 0; c < 4; ++c) {
            const int row    = idx[static_cast<std::size_t>(p) * 4 + c];
            if (row < 0 || row >= R) { throw std::out_of_range("Vision position index out of range"); }
            const float wt    = wts[static_cast<std::size_t>(p) * 4 + c];
            const std::uint16_t* t = tbl + static_cast<std::size_t>(row) * D;
            for (int d = 0; d < D; ++d) { xrow[d] += bf16_to_f32(t[d]) * wt; }
        }
        for (int d = 0; d < D; ++d) { xrow[d] = snap_bf16(xrow[d]); }
    });
}

} // namespace

void VisionContext::encode_cpu(const VisionItemView& item, Tensor& output,
                               cudaStream_t stream) const {
    const qwen3_6::VisionItemControl& control = *item.control;
    const std::size_t patches64               = control.patch_count;
    const std::size_t tokens64                = control.merged_count;
    const int patches = static_cast<std::int32_t>(patches64);
    const int tokens  = static_cast<std::int32_t>(tokens64);
    const int D    = VisionScheduleConfig::hidden;
    const int I    = VisionScheduleConfig::intermediate;
    const int PD   = VisionScheduleConfig::patch_dim;
    const int MH   = VisionScheduleConfig::merger_hidden;
    const int OH   = VisionScheduleConfig::out_hidden;
    if (patches != tokens * VisionScheduleConfig::merge_unit) {
        throw std::invalid_argument("Vision CPU requires P=merge_unit*V>0");
    }

    std::vector<float> xbuf(static_cast<std::size_t>(patches) * D);
    std::vector<float> hbuf(static_cast<std::size_t>(patches) * D);
    std::vector<float> qkvbuf(static_cast<std::size_t>(patches) * (3 * D));
    std::vector<float> attnbuf(static_cast<std::size_t>(patches) * D);
    std::vector<float> projbuf(static_cast<std::size_t>(patches) * D);
    std::vector<float> upbuf(static_cast<std::size_t>(patches) * I);
    std::vector<float> downbuf(static_cast<std::size_t>(patches) * D);
    std::vector<float> normbuf(static_cast<std::size_t>(patches) * D);
    std::vector<float> mergedbuf(static_cast<std::size_t>(tokens) * MH);
    std::vector<float> mhiddenbuf(static_cast<std::size_t>(tokens) * MH);
    std::vector<float> outbuf(static_cast<std::size_t>(tokens) * OH);
    // The weight dequants are cached process-lifetime (DequantCache) — no per-call wbuf.
    g_vision_profile.reset();

    // Oracle v2 (env-gated): dump per-stage host activations as BF16 (mirroring the GPU's
    // storage) so CPU and GPU stages can be compared stage-by-stage offline.
    const char* dump_dir = std::getenv("NINFER_VISION_DUMP");
    const bool dumping   = dump_dir != nullptr && dump_dir[0] != '\0';
    auto dump_stage = [&](const char* tag, const float* data, std::size_t elems) {
        if (!dumping) { return; }
        std::vector<std::uint16_t> host(elems);
        for (std::size_t i = 0; i < elems; ++i) { host[i] = f32_to_bf16(data[i]); }
        vision_debug_dump("cpu", tag, host.data(), host.size() * sizeof(std::uint16_t));
    };

    // Patch embedding: item.patches is BF16 [patches][patch_dim] (the frontend writes each
    // patch's 1536 features contiguously and the GPU gemm consumes the buffer as-is, x[col*k+kk]);
    // the [patches][patch_dim] FP32 GEMM input is therefore a straight conversion, NOT a transpose.
    std::vector<float> x_in(static_cast<std::size_t>(patches) * PD);
    {
        const std::uint16_t* src = item.patches.data();
        pfor_light(static_cast<std::size_t>(patches), [&](std::size_t r) {
            const int p                  = static_cast<std::int32_t>(r);
            const std::uint16_t* srcrow  = src + static_cast<std::size_t>(p) * PD;
            float* row                   = x_in.data() + static_cast<std::size_t>(p) * PD;
            for (int d = 0; d < PD; ++d) { row[d] = bf16_to_f32(srcrow[d]); }
        });
    }
    g_vision_profile.mark("x_in_conv");
    linear_into(*patch_embed_, x_in.data(), patches, PD, xbuf.data());
    g_vision_profile.mark("pe_gemm");
    add_bias_into(*patch_embed_bias_, patches, D, xbuf.data());
    g_vision_profile.mark("pe_bias");
    dump_stage("x1", xbuf.data(), static_cast<std::size_t>(patches) * D);

    // Add the four-corner interpolated position embedding (table view is the materialized
    // [hidden, position_embeddings] BF16 tensor; element (d,r) at d + r*hidden).
    pos_embed_add_into(*position_embed_, control.position_table_indices.data(),
                       control.position_table_weights.data(), patches, D, xbuf.data());
    g_vision_profile.mark("pos_embed");
    dump_stage("x2", xbuf.data(), static_cast<std::size_t>(patches) * D);

    for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
        const BlockW& block = blocks_[layer];
        // --- attention block ---
        {
            layer_norm_into(xbuf.data(), *block.norm1_weight, *block.norm1_bias,
                            VisionScheduleConfig::norm_eps, patches, D, hbuf.data());
            g_vision_profile.mark("ln1");
            linear_into(*block.qkv, hbuf.data(), patches, D, qkvbuf.data());
            g_vision_profile.mark("qkv_gemm");
            add_bias_into(*block.qkv_bias, patches, 3 * D, qkvbuf.data());
            g_vision_profile.mark("qkv_bias");
            rope_vision_into(control.position_ids.data(), patches, D, qkvbuf.data());
            g_vision_profile.mark("rope");
            attention_into(VisionScheduleConfig::attention_scale, control.segment_length, patches, D,
                           qkvbuf.data(), attnbuf.data());
            g_vision_profile.mark("attn");
            linear_into(*block.projection, attnbuf.data(), patches, D, projbuf.data());
            g_vision_profile.mark("proj_gemm");
            add_bias_into(*block.projection_bias, patches, D, projbuf.data());
            residual_add_into(projbuf.data(), static_cast<std::size_t>(patches) * D, xbuf.data());
            g_vision_profile.mark("proj_tail");
        }
        // --- mlp block ---
        {
            layer_norm_into(xbuf.data(), *block.norm2_weight, *block.norm2_bias,
                            VisionScheduleConfig::norm_eps, patches, D, hbuf.data());
            g_vision_profile.mark("ln2");
            linear_into(*block.fc1, hbuf.data(), patches, D, upbuf.data());
            g_vision_profile.mark("fc1_gemm");
            add_bias_into(*block.fc1_bias, patches, I, upbuf.data());
            g_vision_profile.mark("fc1_bias");
            gelu_into(upbuf.data(), static_cast<std::size_t>(patches) * I, CpuGelu::Tanh);
            g_vision_profile.mark("gelu");
            linear_into(*block.fc2, upbuf.data(), patches, I, downbuf.data());
            g_vision_profile.mark("fc2_gemm");
            add_bias_into(*block.fc2_bias, patches, D, downbuf.data());
            residual_add_into(downbuf.data(), static_cast<std::size_t>(patches) * D, xbuf.data());
            g_vision_profile.mark("fc2_tail");
            {
                char tag[16];
                std::snprintf(tag, sizeof(tag), "blk%02zu", layer);
                dump_stage(tag, xbuf.data(), static_cast<std::size_t>(patches) * D);
            }
        }
    }

    // --- merger ---
    layer_norm_into(xbuf.data(), *merger_.norm_weight, *merger_.norm_bias,
                    VisionScheduleConfig::norm_eps, patches, D, normbuf.data());
    g_vision_profile.mark("m_ln");
    dump_stage("m1", normbuf.data(), static_cast<std::size_t>(patches) * D);
    // merged[t][n] = normbuf[4t + n/D][n%D] for n in [0, 4D): the 2x2 merge concatenates the four
    // sub-patch features (the GPU's normalized.view({merger_hidden, tokens}) is this gather).
    {
        for (int t = 0; t < tokens; ++t) {
            float* dst = mergedbuf.data() + static_cast<std::size_t>(t) * MH;
            for (int e = 0; e < VisionScheduleConfig::merge_unit; ++e) {
                const float* src = normbuf.data() + static_cast<std::size_t>(4 * t + e) * D;
                std::memcpy(dst + static_cast<std::size_t>(e) * D, src,
                            static_cast<std::size_t>(D) * sizeof(float));
            }
        }
    }
    g_vision_profile.mark("merge");
    linear_into(*merger_.fc1, mergedbuf.data(), tokens, MH, mhiddenbuf.data());
    g_vision_profile.mark("m_fc1_gemm");
    add_bias_into(*merger_.fc1_bias, tokens, MH, mhiddenbuf.data());
    gelu_into(mhiddenbuf.data(), static_cast<std::size_t>(tokens) * MH, CpuGelu::Exact);
    g_vision_profile.mark("m_gelu");
    dump_stage("m2", mhiddenbuf.data(), static_cast<std::size_t>(tokens) * MH);
    linear_into(*merger_.fc2, mhiddenbuf.data(), tokens, MH, outbuf.data());
    g_vision_profile.mark("m_fc2_gemm");
    add_bias_into(*merger_.fc2_bias, tokens, OH, outbuf.data());
    g_vision_profile.mark("m_fc2_bias");

    // H2D handoff: outbuf [tokens][out_hidden] is exactly the contiguous [out_hidden, tokens]
    // device layout (element (n,t) at t*out_hidden + n), so convert to BF16 and copy.
    std::vector<std::uint16_t> out_bf16(static_cast<std::size_t>(tokens) * OH);
    for (std::size_t i = 0; i < out_bf16.size(); ++i) {
        out_bf16[i] = f32_to_bf16(outbuf[i]);
    }
    g_vision_profile.mark("h2d_conv");
    // Oracle: dump the exact bytes about to be handed off to the device.
    vision_debug_dump("cpu", "out", out_bf16.data(), out_bf16.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpyAsync(output.data, out_bf16.data(), out_bf16.size() * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice, stream));
    g_vision_profile.print();
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
