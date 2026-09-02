// AVX-512 blocked GEMM for the CPU vision encoder. Compiled with /arch:AVX512.
//
// C = X * W^T with X (T*K) and C (T*N) row-major fp32 and W (K*N) K-MAJOR fp32 (k fastest) —
// dequant_weight emits the weights in this layout so the FMA vectorizes over N (vectorizing K
// instead computes X[t][16m+j]*W[n][16m] — the wrong products). The register tile is 8 t rows x
// 32 n cols = 16 zmm accumulators (the per-k broadcast is the mechanism that lets one acc hold
// 16 outputs — a 16 t x 16 n tile would need 256 accs). The (tg, ng) pair order is chosen by K:
// ng-outer pins the W[ng] slice (K*32*4 B) in L2 across the 16 tg passes — measured 2026-09-02
// (isolated microbench): −30% at K=1152 (147 KB slice), flat at K=1536, +25% SLOWER at K=4304
// (554 KB, >50% of the Zen4 1 MB L2), so K <= 2048 -> ng-outer, else tg-outer (v2's L3 W stream).
// Both orders are bit-identical (per-(t,n) FMA order over k is untouched). Requires N % 16 == 0 (true for every vision N: 1152/1536/3456/4304/4608/5120). Every C element is
// BF16-snapped (identical per-lane op to f32_to_bf16 in vision_cpu_impl.h) to match the GPU
// storage boundary. Parallelizes over 128x256 output blocks with one thread per physical core
// (16 on the 9950X3D).
#include "targets/qwen3_6/impl/runtime/vision_cpu_simd.h"
#include "targets/qwen3_6/impl/runtime/vision_cpu_pfor.h"  // persistent worker pool (pfor)

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::cpu_simd {
namespace {

// RNE f32 -> bf16 grid, vectorized: per lane u += 0x7fff + lsb(u); return u >> 16 — the
// identical per-lane op as f32_to_bf16 in vision_cpu_impl.h. Both ends are REINTERPRET casts
// (castps_si512 / castsi512_ps), never conversions: cvtps_epi32 truncates toward zero (would
// zero every |value| < 1.0 lane) and cvtepi32_ps is a real vcvtdq2ps (would turn the bf16 int
// pattern 0xBE7D0000 into the float -1.09e9 instead of the float -0.4375).
inline __m512 snap_bf16_vec(__m512 v) {
    const __m512i u   = _mm512_castps_si512(v);
    const __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
    const __m512i r   = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), lsb));
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_srli_epi32(r, 16), 16));
}

// One thread per physical core (16 on the 9950X3D; SMT pairs add no fp32-FMA throughput). The
// units here are heavy (>= ~1 ms each), so parallelize as soon as there is more than one.
// pfor itself now resolves to the persistent worker pool in vision_cpu_pfor.h (detail::pfor) —
// the old per-call std::thread spawn+join cost ~1.7 ms/layer under attention alone (2026-09-02
// profile: 120 ms wall vs ~73 ms job phase sum) and sat under every GEMM call too.

constexpr int kBM = 128;  // T rows per output block
constexpr int kBN = 256;  // N rows per output block

// gemm_tn_bf16w W-split masks (see that function): W is k-pair packed u32 (get32p), so a 64-byte
// span holds 16 columns x 2 k rows — `and 0xFFFF / << 16` (k row 2kp) and `and 0xFFFF0000`
// (k row 2kp+1) give both rows as fp32 (bit-exact: bf16 IS the high half of fp32) with NO
// cross-lane merge. No VPERM2PS/VUNPCKL/VBLEND in the hot loop: on this 9950X3D (Zen 5,
// microcode 0x00010000) those corrupt the lane mapping in an allocation-dependent way
// (.logs/prim_test.cpp, 2026-09-02) — loads + and/shift + FMA are the verified-safe set.
const __m512i bf16w_lo_mask = _mm512_set1_epi32(0x0000FFFF);
const __m512i bf16w_hi_mask = _mm512_set1_epi32(0xFFFF0000);

}  // namespace

bool has_avx512f() {
    int cpu[4] = {0, 0, 0, 0};
    __cpuidex(cpu, 7, 0);
    return (cpu[1] & (1u << 16)) != 0;  // EBX bit 16 = AVX512F
}

// f32 -> bf16 RNE snap, scalar (gemm_tn epilogue tail and small kernels) — the identical per-
// lane op as f32_to_bf16 in vision_cpu_impl.h.
inline float snap_bf16_scalar(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    const std::uint32_t lsb = (u >> 16) & 1u;
    u = (u + 0x7fffu + lsb) >> 16;
    u <<= 16;
    std::memcpy(&f, &u, 4);
    return f;
}

void gemm_tn(const float* X, const float* W, float* C, int T, int N, int K) {
    const int nbT = (T + kBM - 1) / kBM;
    const int nbN = (N + kBN - 1) / kBN;
    // NINFER_GEMM_TRACE=1: per-call wall clock + per-block min/max (2026-09-02 diagnostic:
    // the in-encode GEMM profile lines show 30-65x the isolated gemm_tn time — measure the
    // in-encode gemm_tn wall + block spread to localize the gap).
    static const bool trace = [] {
        const char* e = std::getenv("NINFER_GEMM_TRACE");
        return e != nullptr && e[0] == '1';
    }();
    const auto wall0 = trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::vector<std::chrono::microseconds> blk;
    if (trace) { blk.resize(static_cast<std::size_t>(nbT) * nbN); }
    pfor(static_cast<std::size_t>(nbT * nbN), [&](std::size_t pi) {
        const auto b0 = trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const int ti = static_cast<int>(pi / static_cast<std::size_t>(nbN));
        const int ni = static_cast<int>(pi % static_cast<std::size_t>(nbN));
        const int t0 = ti * kBM;
        const int n0 = ni * kBN;
        const int tb = std::min(kBM, T - t0);
        const int nb = std::min(kBN, N - n0);

        // (tg, ng) pair schedule chosen by K. ng-outer pins the W[ng] slice (K*32*4 bytes) in L2
        // across the 16 tg passes (fetched from L3 once per ng, re-read from L2 by the other 15
        // passes); tg-outer (v2) streams the whole 1.15-4.4 MB W block from L3 per tg pass.
        // Measured 2026-09-02 (min-of-5, isolated): ng-outer −30% at K=1152 (147 KB slice), flat
        // at K=1536 (192 KB), +25% SLOWER at K=4304 (554 KB, >50% of the Zen4 1 MB L2 — the
        // strided L2 re-reads lose to the L3 stream). K <= 2048 -> ng-outer. Both orders are
        // bit-identical to each other and to the reference: the per-(t,n) FMA order over k is
        // untouched by the pair schedule. A first "v3" attempt staged the X tile transposed and
        // dropped the 8 per-k broadcasts in favor of 16 t-row zmm loads — invalid GEMM: a zmm FMA
        // pairs x-lane j with w-lane j, so a 16 t x 16 n micro-tile needs 256 accumulators, not
        // 16 (the broadcast is the mechanism that lets one acc hold 16 outputs). Dead end, kept
        // here as a record.
        // Register tile: 8 t rows x 32 n cols as NAMED accumulator locals (a00..a71 = 16 zmm,
        // t-major pairs) + 8 named xv broadcasts = 24 zmm, all register-resident. A local
        // __m512[8][2] array gets stack addresses by MSVC (the k loop then pays a load and a
        // store around every FMA — verified in the generated code); named locals stay in zmm.
        // 32 n cols keeps each k-step W fetch a contiguous 128 B span. In the last T block rows
        // beyond T-1 are clamped (their accs are never stored).
        {
            thread_local int sched[128][2];  // [pair] = {tg, ng}; <= 16 tg x 8 ng pairs per block
            int npair = 0;
            // Debug/A-B override: NINFER_VISION_GEMM_ORDER=ng|tg|auto (default auto = K <= 2048).
            // Lets the in-encode profile compare the two schedules under identical contention.
            static int ord = 0;  // 0 = uninitialized, 1 = ng, 2 = tg, 3 = auto
            if (ord == 0) {
                if (const char* e = std::getenv("NINFER_VISION_GEMM_ORDER")) {
                    if (std::strcmp(e, "ng") == 0) { ord = 1; }
                    else if (std::strcmp(e, "tg") == 0) { ord = 2; }
                }
                if (ord == 0) { ord = 3; }
            }
            if (ord == 1 || (ord == 3 && K <= 2048)) {
                for (int ng = 0; ng < nb; ng += 32) {
                    for (int tg = 0; tg < tb; tg += 8) { sched[npair][0] = tg; sched[npair][1] = ng; ++npair; }
                }
            } else {
                for (int tg = 0; tg < tb; tg += 8) {
                    for (int ng = 0; ng < nb; ng += 32) { sched[npair][0] = tg; sched[npair][1] = ng; ++npair; }
                }
            }
            for (int q = 0; q < npair; ++q) {
                const int tg = sched[q][0], ng = sched[q][1];
                const int tr = std::min(8, tb - tg);
                const int nr = std::min(32, nb - ng);
                int row[8];
                for (int t = 0; t < 8; ++t) { row[t] = std::min(t0 + tg + t, T - 1); }
                __m512 a00 = _mm512_setzero_ps(), a01 = _mm512_setzero_ps(),
                       a10 = _mm512_setzero_ps(), a11 = _mm512_setzero_ps(),
                       a20 = _mm512_setzero_ps(), a21 = _mm512_setzero_ps(),
                       a30 = _mm512_setzero_ps(), a31 = _mm512_setzero_ps(),
                       a40 = _mm512_setzero_ps(), a41 = _mm512_setzero_ps(),
                       a50 = _mm512_setzero_ps(), a51 = _mm512_setzero_ps(),
                       a60 = _mm512_setzero_ps(), a61 = _mm512_setzero_ps(),
                       a70 = _mm512_setzero_ps(), a71 = _mm512_setzero_ps();
                for (int k = 0; k < K; ++k) {
                    const __m512 x0 = _mm512_set1_ps(X[static_cast<std::size_t>(row[0]) * K + k]);
                    const __m512 x1 = _mm512_set1_ps(X[static_cast<std::size_t>(row[1]) * K + k]);
                    const __m512 x2 = _mm512_set1_ps(X[static_cast<std::size_t>(row[2]) * K + k]);
                    const __m512 x3 = _mm512_set1_ps(X[static_cast<std::size_t>(row[3]) * K + k]);
                    const __m512 x4 = _mm512_set1_ps(X[static_cast<std::size_t>(row[4]) * K + k]);
                    const __m512 x5 = _mm512_set1_ps(X[static_cast<std::size_t>(row[5]) * K + k]);
                    const __m512 x6 = _mm512_set1_ps(X[static_cast<std::size_t>(row[6]) * K + k]);
                    const __m512 x7 = _mm512_set1_ps(X[static_cast<std::size_t>(row[7]) * K + k]);
                    const float* wk = W + static_cast<std::size_t>(k) * N + n0 + ng;
                    const __m512 w0 = _mm512_loadu_ps(wk);
                    a00 = _mm512_fmadd_ps(x0, w0, a00);
                    a10 = _mm512_fmadd_ps(x1, w0, a10);
                    a20 = _mm512_fmadd_ps(x2, w0, a20);
                    a30 = _mm512_fmadd_ps(x3, w0, a30);
                    a40 = _mm512_fmadd_ps(x4, w0, a40);
                    a50 = _mm512_fmadd_ps(x5, w0, a50);
                    a60 = _mm512_fmadd_ps(x6, w0, a60);
                    a70 = _mm512_fmadd_ps(x7, w0, a70);
                    if (nr == 32) {
                        const __m512 w1 = _mm512_loadu_ps(wk + 16);
                        a01 = _mm512_fmadd_ps(x0, w1, a01);
                        a11 = _mm512_fmadd_ps(x1, w1, a11);
                        a21 = _mm512_fmadd_ps(x2, w1, a21);
                        a31 = _mm512_fmadd_ps(x3, w1, a31);
                        a41 = _mm512_fmadd_ps(x4, w1, a41);
                        a51 = _mm512_fmadd_ps(x5, w1, a51);
                        a61 = _mm512_fmadd_ps(x6, w1, a61);
                        a71 = _mm512_fmadd_ps(x7, w1, a71);
                    }
                }
                if (tr > 0) {
                    float* o = C + static_cast<std::size_t>(t0 + tg) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a00));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a01)); }
                }
                if (tr > 1) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 1) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a10));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a11)); }
                }
                if (tr > 2) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 2) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a20));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a21)); }
                }
                if (tr > 3) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 3) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a30));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a31)); }
                }
                if (tr > 4) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 4) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a40));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a41)); }
                }
                if (tr > 5) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 5) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a50));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a51)); }
                }
                if (tr > 6) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 6) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a60));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a61)); }
                }
                if (tr > 7) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 7) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a70));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a71)); }
                }
            }
        }
        if (trace) {
            blk[pi] = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - b0);
        }
    });
    if (trace) {
        auto mn = blk[0], mx = blk[0];
        for (auto v : blk) {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        const auto dt = std::chrono::steady_clock::now() - wall0;
        std::fprintf(stderr, "[gemm] T=%d N=%d K=%d: wall %7.2f ms  blocks %d  blk min %7.2f max %7.2f ms\n",
                     T, N, K, std::chrono::duration<double, std::milli>(dt).count(), nbT * nbN,
                     std::chrono::duration<double, std::milli>(mn).count(),
                     std::chrono::duration<double, std::milli>(mx).count());
    }
}

// gemm_tn with the weight k-PAIR packed u32 (DequantCache::get32p): halves the W traffic per
// k-step (64 B per 2-k x 16-col span = 2 B per (k, column) vs gemm_tn's 4 B) for the
// L2/L3-slice-driven GEMMs (see the declaration). Structure, pair schedule and epilogue are a
// copy of gemm_tn; the only difference is the k-loop, which steps k by 2 (kp < (K+1)/2; an odd
// K zero-pads its last k row, whose packed high half reads 0.0F and whose X broadcast is zero):
// one 512-bit integer load per 16 columns reads 2 k rows x 16 columns, `and 0xFFFF / << 16`
// (row 2kp) + `and 0xFFFF0000` (row 2kp+1) split the pair into fp32 with NO cross-lane merge —
// bf16 is the high half of fp32, so that split is a bit-exact bf16 -> fp32 conversion. The
// per-lane FMA sequence is k0, k0+1, k0+2, ... (k0 then k0+1 per kp) — the exact per-k order of
// gemm_tn — so the numerics are identical to gemm_tn on the same W values; the only difference
// is the W's own bf16 snap (get32p). No VPERM2PS/VUNPCKL/VBLEND anywhere in the hot loop: on
// this 9950X3D (Zen 5, microcode 0x00010000) the cross-lane merge ops corrupt the lane mapping
// in an allocation-dependent way (.logs/prim_test.cpp) — loads + and/shift + FMA is the
// verified-safe set. The x broadcasts sit next to each row's FMAs to hold the live zmm count at
// 16 acc + 4 w + 2 x (no spills on the 32-register Zen 5). The nr == 16 tail (N % 32 != 0, the
// fc1 N=4304 last block) does the same with ONE 64-byte load; a01..a71 stay gated on nr == 32,
// as in gemm_tn.
void gemm_tn_bf16w(const float* X, const std::uint32_t* W, float* C, int T, int N, int K) {
    const int nbT = (T + kBM - 1) / kBM;
    const int nbN = (N + kBN - 1) / kBN;
    const int Kp = (K + 1) / 2;
    pfor(static_cast<std::size_t>(nbT * nbN), [&](std::size_t pi) {
        const int ti = static_cast<int>(pi / static_cast<std::size_t>(nbN));
        const int ni = static_cast<int>(pi % static_cast<std::size_t>(nbN));
        const int t0 = ti * kBM;
        const int n0 = ni * kBN;
        const int tb = std::min(kBM, T - t0);
        const int nb = std::min(kBN, N - n0);
        {
            thread_local int sched[128][2];  // [pair] = {tg, ng}
            int npair = 0;
            static int ord = 0;  // 0 = uninitialized, 1 = ng, 2 = tg, 3 = auto (K <= 2048)
            if (ord == 0) {
                if (const char* e = std::getenv("NINFER_VISION_GEMM_ORDER")) {
                    if (std::strcmp(e, "ng") == 0) { ord = 1; }
                    else if (std::strcmp(e, "tg") == 0) { ord = 2; }
                }
                if (ord == 0) { ord = 3; }
            }
            if (ord == 1 || (ord == 3 && K <= 2048)) {
                for (int ng = 0; ng < nb; ng += 32) {
                    for (int tg = 0; tg < tb; tg += 8) { sched[npair][0] = tg; sched[npair][1] = ng; ++npair; }
                }
            } else {
                for (int tg = 0; tg < tb; tg += 8) {
                    for (int ng = 0; ng < nb; ng += 32) { sched[npair][0] = tg; sched[npair][1] = ng; ++npair; }
                }
            }
            for (int q = 0; q < npair; ++q) {
                const int tg = sched[q][0], ng = sched[q][1];
                const int tr = std::min(8, tb - tg);
                const int nr = std::min(32, nb - ng);
                int row[8];
                for (int t = 0; t < 8; ++t) { row[t] = std::min(t0 + tg + t, T - 1); }
                __m512 a00 = _mm512_setzero_ps(), a01 = _mm512_setzero_ps(),
                       a10 = _mm512_setzero_ps(), a11 = _mm512_setzero_ps(),
                       a20 = _mm512_setzero_ps(), a21 = _mm512_setzero_ps(),
                       a30 = _mm512_setzero_ps(), a31 = _mm512_setzero_ps(),
                       a40 = _mm512_setzero_ps(), a41 = _mm512_setzero_ps(),
                       a50 = _mm512_setzero_ps(), a51 = _mm512_setzero_ps(),
                       a60 = _mm512_setzero_ps(), a61 = _mm512_setzero_ps(),
                       a70 = _mm512_setzero_ps(), a71 = _mm512_setzero_ps();
                for (int kp = 0; kp < Kp; ++kp) {
                    const int k0 = 2 * kp;
                    const bool has_k1 = k0 + 1 < K;
                    const std::uint32_t* wk = W + static_cast<std::size_t>(kp) * N + n0 + ng;
                    if (nr == 32) {
                        // 32 cols x 2 k rows = 128 B: two 512-bit integer loads, four w's.
                        const __m512i v0 = _mm512_loadu_epi32(reinterpret_cast<const __m512i*>(wk));
                        const __m512i v1 = _mm512_loadu_epi32(reinterpret_cast<const __m512i*>(wk + 16));
                        const __m512 wa0 = _mm512_castsi512_ps(
                            _mm512_slli_epi32(_mm512_and_si512(v0, bf16w_lo_mask), 16));
                        const __m512 wb0 = _mm512_castsi512_ps(_mm512_and_si512(v0, bf16w_hi_mask));
                        const __m512 wa1 = _mm512_castsi512_ps(
                            _mm512_slli_epi32(_mm512_and_si512(v1, bf16w_lo_mask), 16));
                        const __m512 wb1 = _mm512_castsi512_ps(_mm512_and_si512(v1, bf16w_hi_mask));
                        const __m512 x0a = _mm512_set1_ps(X[static_cast<std::size_t>(row[0]) * K + k0]);
                        const __m512 x0b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[0]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a00 = _mm512_fmadd_ps(x0a, wa0, a00);
                        a00 = _mm512_fmadd_ps(x0b, wb0, a00);
                        a01 = _mm512_fmadd_ps(x0a, wa1, a01);
                        a01 = _mm512_fmadd_ps(x0b, wb1, a01);
                        const __m512 x1a = _mm512_set1_ps(X[static_cast<std::size_t>(row[1]) * K + k0]);
                        const __m512 x1b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[1]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a10 = _mm512_fmadd_ps(x1a, wa0, a10);
                        a10 = _mm512_fmadd_ps(x1b, wb0, a10);
                        a11 = _mm512_fmadd_ps(x1a, wa1, a11);
                        a11 = _mm512_fmadd_ps(x1b, wb1, a11);
                        const __m512 x2a = _mm512_set1_ps(X[static_cast<std::size_t>(row[2]) * K + k0]);
                        const __m512 x2b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[2]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a20 = _mm512_fmadd_ps(x2a, wa0, a20);
                        a20 = _mm512_fmadd_ps(x2b, wb0, a20);
                        a21 = _mm512_fmadd_ps(x2a, wa1, a21);
                        a21 = _mm512_fmadd_ps(x2b, wb1, a21);
                        const __m512 x3a = _mm512_set1_ps(X[static_cast<std::size_t>(row[3]) * K + k0]);
                        const __m512 x3b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[3]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a30 = _mm512_fmadd_ps(x3a, wa0, a30);
                        a30 = _mm512_fmadd_ps(x3b, wb0, a30);
                        a31 = _mm512_fmadd_ps(x3a, wa1, a31);
                        a31 = _mm512_fmadd_ps(x3b, wb1, a31);
                        const __m512 x4a = _mm512_set1_ps(X[static_cast<std::size_t>(row[4]) * K + k0]);
                        const __m512 x4b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[4]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a40 = _mm512_fmadd_ps(x4a, wa0, a40);
                        a40 = _mm512_fmadd_ps(x4b, wb0, a40);
                        a41 = _mm512_fmadd_ps(x4a, wa1, a41);
                        a41 = _mm512_fmadd_ps(x4b, wb1, a41);
                        const __m512 x5a = _mm512_set1_ps(X[static_cast<std::size_t>(row[5]) * K + k0]);
                        const __m512 x5b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[5]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a50 = _mm512_fmadd_ps(x5a, wa0, a50);
                        a50 = _mm512_fmadd_ps(x5b, wb0, a50);
                        a51 = _mm512_fmadd_ps(x5a, wa1, a51);
                        a51 = _mm512_fmadd_ps(x5b, wb1, a51);
                        const __m512 x6a = _mm512_set1_ps(X[static_cast<std::size_t>(row[6]) * K + k0]);
                        const __m512 x6b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[6]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a60 = _mm512_fmadd_ps(x6a, wa0, a60);
                        a60 = _mm512_fmadd_ps(x6b, wb0, a60);
                        a61 = _mm512_fmadd_ps(x6a, wa1, a61);
                        a61 = _mm512_fmadd_ps(x6b, wb1, a61);
                        const __m512 x7a = _mm512_set1_ps(X[static_cast<std::size_t>(row[7]) * K + k0]);
                        const __m512 x7b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[7]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a70 = _mm512_fmadd_ps(x7a, wa0, a70);
                        a70 = _mm512_fmadd_ps(x7b, wb0, a70);
                        a71 = _mm512_fmadd_ps(x7a, wa1, a71);
                        a71 = _mm512_fmadd_ps(x7b, wb1, a71);
                    } else {
                        // nr == 16 (fc1 N=4304 tail): 16 cols x 2 k rows = 64 B, ONE load.
                        const __m512i v0 = _mm512_loadu_epi32(reinterpret_cast<const __m512i*>(wk));
                        const __m512 wa0 = _mm512_castsi512_ps(
                            _mm512_slli_epi32(_mm512_and_si512(v0, bf16w_lo_mask), 16));
                        const __m512 wb0 = _mm512_castsi512_ps(_mm512_and_si512(v0, bf16w_hi_mask));
                        const __m512 x0a = _mm512_set1_ps(X[static_cast<std::size_t>(row[0]) * K + k0]);
                        const __m512 x0b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[0]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a00 = _mm512_fmadd_ps(x0a, wa0, a00);
                        a00 = _mm512_fmadd_ps(x0b, wb0, a00);
                        const __m512 x1a = _mm512_set1_ps(X[static_cast<std::size_t>(row[1]) * K + k0]);
                        const __m512 x1b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[1]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a10 = _mm512_fmadd_ps(x1a, wa0, a10);
                        a10 = _mm512_fmadd_ps(x1b, wb0, a10);
                        const __m512 x2a = _mm512_set1_ps(X[static_cast<std::size_t>(row[2]) * K + k0]);
                        const __m512 x2b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[2]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a20 = _mm512_fmadd_ps(x2a, wa0, a20);
                        a20 = _mm512_fmadd_ps(x2b, wb0, a20);
                        const __m512 x3a = _mm512_set1_ps(X[static_cast<std::size_t>(row[3]) * K + k0]);
                        const __m512 x3b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[3]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a30 = _mm512_fmadd_ps(x3a, wa0, a30);
                        a30 = _mm512_fmadd_ps(x3b, wb0, a30);
                        const __m512 x4a = _mm512_set1_ps(X[static_cast<std::size_t>(row[4]) * K + k0]);
                        const __m512 x4b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[4]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a40 = _mm512_fmadd_ps(x4a, wa0, a40);
                        a40 = _mm512_fmadd_ps(x4b, wb0, a40);
                        const __m512 x5a = _mm512_set1_ps(X[static_cast<std::size_t>(row[5]) * K + k0]);
                        const __m512 x5b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[5]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a50 = _mm512_fmadd_ps(x5a, wa0, a50);
                        a50 = _mm512_fmadd_ps(x5b, wb0, a50);
                        const __m512 x6a = _mm512_set1_ps(X[static_cast<std::size_t>(row[6]) * K + k0]);
                        const __m512 x6b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[6]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a60 = _mm512_fmadd_ps(x6a, wa0, a60);
                        a60 = _mm512_fmadd_ps(x6b, wb0, a60);
                        const __m512 x7a = _mm512_set1_ps(X[static_cast<std::size_t>(row[7]) * K + k0]);
                        const __m512 x7b = has_k1
                            ? _mm512_set1_ps(X[static_cast<std::size_t>(row[7]) * K + k0 + 1])
                            : _mm512_setzero_ps();
                        a70 = _mm512_fmadd_ps(x7a, wa0, a70);
                        a70 = _mm512_fmadd_ps(x7b, wb0, a70);
                    }
                }
                if (tr > 0) {
                    float* o = C + static_cast<std::size_t>(t0 + tg) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a00));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a01)); }
                }
                if (tr > 1) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 1) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a10));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a11)); }
                }
                if (tr > 2) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 2) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a20));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a21)); }
                }
                if (tr > 3) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 3) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a30));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a31)); }
                }
                if (tr > 4) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 4) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a40));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a41)); }
                }
                if (tr > 5) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 5) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a50));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a51)); }
                }
                if (tr > 6) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 6) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a60));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a61)); }
                }
                if (tr > 7) {
                    float* o = C + static_cast<std::size_t>(t0 + tg + 7) * N + n0 + ng;
                    _mm512_storeu_ps(o, snap_bf16_vec(a70));
                    if (nr == 32) { _mm512_storeu_ps(o + 16, snap_bf16_vec(a71)); }
                }
            }
        }
    });
}

// f32 -> bf16 RNE snap, 256-bit variant of snap_bf16_vec (head_dim == 8 tail) — the identical
// per-lane op as f32_to_bf16 in vision_cpu_impl.h.
inline __m256 snap_bf16_vec16(__m256 v) {
    const __m256i u   = _mm256_castps_si256(v);
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
    const __m256i r   = _mm256_add_epi32(u, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), lsb));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_srli_epi32(r, 16), 16));
}

void attention_into(float scale, int SL, int T, int D, int H, int HD,
                    const float* qkv, float* out) {
    const int segs  = T / SL;
    const int j16   = SL / 16;  // full 512-bit j chunks (scores/keys along the segment axis)
    const int d16   = HD / 16;  // full 512-bit d chunks (values along the head_dim axis)
    const int drem  = HD % 16;  // head_dim tail: one 256-bit chunk when it is 8, else scalar
    const int dy    = (drem >= 8) ? 8 : 0;
    const std::size_t s3D = static_cast<std::size_t>(3) * D;
    // kT/sc rows are padded to a 32-column multiple: the QK tile's last ng group (width 8..31
    // when SL % 32 != 0) reads and stores full 16-lane halves unconditionally, and the pad
    // lanes (j >= SL) are never consumed — softmax/PV read [0, SL) only. Keeps every 512-bit
    // op in-buffer for any SL (a "last group is exactly 16 wide" assumption breaks the w1
    // load for SL % 32 in 8..24: OOB read on the last kT row + cross-row sc stores).
    const int SLp = (SL + 31) & ~31;
    // Phase timing (env NINFER_VISION_ATTN_PHASE=1): per-call sum over all (seg, head) jobs of
    // staging / QK / softmax / PV wall time, printed at the end of the call (stderr).
    static std::atomic<long long> ph_ns[4];
    static int ph_on = 0;  // 0 = uninit, 1 = off, 2 = on
    if (ph_on == 0) { ph_on = std::getenv("NINFER_VISION_ATTN_PHASE") ? 2 : 1; }
    const bool ph = (ph_on == 2);
    for (int k = 0; k < 4; ++k) { ph_ns[k].store(0, std::memory_order_relaxed); }
    // NINFER_VISION_VSTAGE (2026-09-02, #19 STEP 13): 0 = the A/B control (read v straight
    // from the strided qkv, the pre-STEP-13 behavior); 1/unset (default) = stage v to a
    // contiguous [SL][HD] buffer (faster PV). Same binary toggles it at runtime, so
    // same-window alternating A/B pairs isolate the v-staging effect from the machine's
    // fast/slow window drift.
    static int vstage_on = -1;
    if (vstage_on < 0) {
        const char* e = std::getenv("NINFER_VISION_VSTAGE");
        vstage_on = (e != nullptr && e[0] == '0') ? 0 : 1;
    }
    const bool vstage = (vstage_on == 1);
    pfor(static_cast<std::size_t>(segs * H), [&](std::size_t pi) {
        const int s  = static_cast<int>(pi / static_cast<std::size_t>(H));
        const int h  = static_cast<int>(pi % static_cast<std::size_t>(H));
        const int ib = s * SL;  // segment begin
        long long a0 = 0, a1 = 0, a2 = 0, a3 = 0, te = 0, tp = 0;
        const auto ns_ = [] {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count();
        };
        if (ph) { te = ns_(); }

        // Per (segment, head) staging: k transposed to kT [HD][SLp] (SLp-padded) so the QK
        // tile vectorizes over 16 consecutive j per k-step. v is staged to a contiguous
        // [SL][HD] block (the vbuf block below; NINFER_VISION_VSTAGE=0 restores the pre-STEP-13
        // strided-qkv PV read as the A/B control).
        thread_local std::vector<float> kT;
        if (kT.size() < static_cast<std::size_t>(HD) * SLp) {
            kT.resize(static_cast<std::size_t>(HD) * SLp);
            std::memset(kT.data(), 0, kT.size() * sizeof(float));  // zero the pad lanes
        }
        for (int j = 0; j < SL; ++j) {
            const float* krow = qkv + static_cast<std::size_t>(ib + j) * s3D + D + h * HD;
            for (int d = 0; d < HD; ++d) {
                kT[static_cast<std::size_t>(d) * SLp + j] = krow[d];
            }
        }
        // v staged to a CONTIGUOUS [SL][HD] block (2026-09-02 phase split: PV was 3.6x the QK
        // for equal FLOPs — the PV read v straight from the strided qkv, stride s3D ~ 13.8 KB
        // between v-rows, which defeats prefetch and scatters each head's 253 KB v-matrix over
        // a 12 MB span; across the 16 parallel head jobs that is ~192 MB, thrashing L3). The
        // v block is fixed per (segment, head) — stage it ONCE per job here, exactly like kT
        // — and the PV reads the 253 KB buffer (L2-resident on the job's core, sequential ->
        // prefetch-friendly). BIT-IDENTICAL: same v values, same ascending-j FMA order; only
        // the read source changes.
        thread_local std::vector<float> vbuf;
        if (vstage) {
            if (vbuf.size() < static_cast<std::size_t>(SL) * HD) {
                vbuf.resize(static_cast<std::size_t>(SL) * HD);
            }
            for (int j = 0; j < SL; ++j) {
                const float* vrow = qkv + static_cast<std::size_t>(ib + j) * s3D + 2 * D + h * HD;
                for (int d = 0; d < HD; ++d) {
                    vbuf[static_cast<std::size_t>(j) * HD + d] = vrow[d];
                }
            }
        }
        if (ph) { tp = ns_(); a0 += tp - te; te = tp; }

        // t-blocked structure (2026-09-02 phase split: the old per-row QK streamed the whole
        // 253 KB kT per i row = 222 MB per job from L2; the tile reads kT once per t-block).
        // Per 128-row block: QK tile -> scores [tb][SLp] FP32 -> vectorized row softmax (with
        // inv folded into the row) -> per-row PV. Bit-identical to the per-row version: the
        // per-(t,j) FMA set and the ascending-d / ascending-j order are unchanged.
        thread_local std::vector<float> sc;
        const __m512 scalev = _mm512_set1_ps(scale);
        const __m512 log2ev = _mm512_set1_ps(1.4426950408889634F);  // exp(x) = exp2(x*log2(e))
        const int qh = h * HD;
        for (int tb0 = 0; tb0 < SL; tb0 += 128) {
            const int tb = std::min(128, SL - tb0);
            if (sc.size() < static_cast<std::size_t>(tb) * SLp) {
                sc.resize(static_cast<std::size_t>(tb) * SLp);
            }
            float* sc0 = sc.data();
            if (ph) { te = ns_(); }
            // QK tile: scores[t][j] = scale * sum_d X[t][d] * kT[d][j] — gemm_tn's 8t x 32n
            // register tile (16 named accs; ng-outer pins the W[ng] slice, HD*32*4 ~ 9 KB, in
            // L1 across the 16 tg passes). Named locals stay in zmm (an __m512 array gets
            // stack addresses from MSVC — the gemm_tn asm check). X rows are q rows at stride
            // s3D, W rows are kT rows at stride SL.
            for (int ng = 0; ng < SL; ng += 32) {
                for (int tg = 0; tg < tb; tg += 8) {
                    const int tr = std::min(8, tb - tg);
                    int row[8];
                    for (int t = 0; t < 8; ++t) { row[t] = std::min(tg + t, tb - 1); }
                    __m512 a00 = _mm512_setzero_ps(), a01 = _mm512_setzero_ps(),
                           a10 = _mm512_setzero_ps(), a11 = _mm512_setzero_ps(),
                           a20 = _mm512_setzero_ps(), a21 = _mm512_setzero_ps(),
                           a30 = _mm512_setzero_ps(), a31 = _mm512_setzero_ps(),
                           a40 = _mm512_setzero_ps(), a41 = _mm512_setzero_ps(),
                           a50 = _mm512_setzero_ps(), a51 = _mm512_setzero_ps(),
                           a60 = _mm512_setzero_ps(), a61 = _mm512_setzero_ps(),
                           a70 = _mm512_setzero_ps(), a71 = _mm512_setzero_ps();
                    const float* wk = kT.data() + ng;  // kT[0][ng]
                    for (int d = 0; d < HD; ++d) {
                        const __m512 x0 = _mm512_set1_ps(qkv[static_cast<std::size_t>(ib + tb0 + row[0]) * s3D + qh + d]);
                        const __m512 x1 = _mm512_set1_ps(qkv[static_cast<std::size_t>(ib + tb0 + row[1]) * s3D + qh + d]);
                        const __m512 x2 = _mm512_set1_ps(qkv[static_cast<std::size_t>(ib + tb0 + row[2]) * s3D + qh + d]);
                        const __m512 x3 = _mm512_set1_ps(qkv[static_cast<std::size_t>(ib + tb0 + row[3]) * s3D + qh + d]);
                        const __m512 x4 = _mm512_set1_ps(qkv[static_cast<std::size_t>(ib + tb0 + row[4]) * s3D + qh + d]);
                        const __m512 x5 = _mm512_set1_ps(qkv[static_cast<std::size_t>(ib + tb0 + row[5]) * s3D + qh + d]);
                        const __m512 x6 = _mm512_set1_ps(qkv[static_cast<std::size_t>(ib + tb0 + row[6]) * s3D + qh + d]);
                        const __m512 x7 = _mm512_set1_ps(qkv[static_cast<std::size_t>(ib + tb0 + row[7]) * s3D + qh + d]);
                        const __m512 w0 = _mm512_loadu_ps(wk);
                        a00 = _mm512_fmadd_ps(x0, w0, a00);
                        a10 = _mm512_fmadd_ps(x1, w0, a10);
                        a20 = _mm512_fmadd_ps(x2, w0, a20);
                        a30 = _mm512_fmadd_ps(x3, w0, a30);
                        a40 = _mm512_fmadd_ps(x4, w0, a40);
                        a50 = _mm512_fmadd_ps(x5, w0, a50);
                        a60 = _mm512_fmadd_ps(x6, w0, a60);
                        a70 = _mm512_fmadd_ps(x7, w0, a70);
                        const __m512 w1 = _mm512_loadu_ps(wk + 16);
                        a01 = _mm512_fmadd_ps(x0, w1, a01);
                        a11 = _mm512_fmadd_ps(x1, w1, a11);
                        a21 = _mm512_fmadd_ps(x2, w1, a21);
                        a31 = _mm512_fmadd_ps(x3, w1, a31);
                        a41 = _mm512_fmadd_ps(x4, w1, a41);
                        a51 = _mm512_fmadd_ps(x5, w1, a51);
                        a61 = _mm512_fmadd_ps(x6, w1, a61);
                        a71 = _mm512_fmadd_ps(x7, w1, a71);
                        wk += SLp;
                    }
                    // Epilogue: scale + FP32 store into the scores row (16 j per acc). Both
                    // 16-lane halves store unconditionally: the last ng group's pad lanes
                    // (j >= SL) land in the SLp pad, which softmax/PV never read.
                    if (tr > 0) {
                        float* o = sc0 + static_cast<std::size_t>(tg + 0) * SLp + ng;
                        _mm512_storeu_ps(o, _mm512_mul_ps(a00, scalev));
                        _mm512_storeu_ps(o + 16, _mm512_mul_ps(a01, scalev));
                    }
                    if (tr > 1) {
                        float* o = sc0 + static_cast<std::size_t>(tg + 1) * SLp + ng;
                        _mm512_storeu_ps(o, _mm512_mul_ps(a10, scalev));
                        _mm512_storeu_ps(o + 16, _mm512_mul_ps(a11, scalev));
                    }
                    if (tr > 2) {
                        float* o = sc0 + static_cast<std::size_t>(tg + 2) * SLp + ng;
                        _mm512_storeu_ps(o, _mm512_mul_ps(a20, scalev));
                        _mm512_storeu_ps(o + 16, _mm512_mul_ps(a21, scalev));
                    }
                    if (tr > 3) {
                        float* o = sc0 + static_cast<std::size_t>(tg + 3) * SLp + ng;
                        _mm512_storeu_ps(o, _mm512_mul_ps(a30, scalev));
                        _mm512_storeu_ps(o + 16, _mm512_mul_ps(a31, scalev));
                    }
                    if (tr > 4) {
                        float* o = sc0 + static_cast<std::size_t>(tg + 4) * SLp + ng;
                        _mm512_storeu_ps(o, _mm512_mul_ps(a40, scalev));
                        _mm512_storeu_ps(o + 16, _mm512_mul_ps(a41, scalev));
                    }
                    if (tr > 5) {
                        float* o = sc0 + static_cast<std::size_t>(tg + 5) * SLp + ng;
                        _mm512_storeu_ps(o, _mm512_mul_ps(a50, scalev));
                        _mm512_storeu_ps(o + 16, _mm512_mul_ps(a51, scalev));
                    }
                    if (tr > 6) {
                        float* o = sc0 + static_cast<std::size_t>(tg + 6) * SLp + ng;
                        _mm512_storeu_ps(o, _mm512_mul_ps(a60, scalev));
                        _mm512_storeu_ps(o + 16, _mm512_mul_ps(a61, scalev));
                    }
                    if (tr > 7) {
                        float* o = sc0 + static_cast<std::size_t>(tg + 7) * SLp + ng;
                        _mm512_storeu_ps(o, _mm512_mul_ps(a70, scalev));
                        _mm512_storeu_ps(o + 16, _mm512_mul_ps(a71, scalev));
                    }
                }
            }
            if (ph) { tp = ns_(); a1 += tp - te; te = tp; }
            // Softmax per scores row: vectorized max/exp/sum (the scalar left-to-right scans
            // were dependent-latency chains — ~11k of the row's ~12k cycles, measured in the
            // 2026-09-02 phase split), then inv folded into the row so the PV reads pre-scaled
            // scores and drops the per-j mul. The max is exact (order-independent); the sum
            // reorders association (chunk order + tree reduce) -> inv at ~1e-7 relative, far
            // below the 8-ULP A/B gate.
            for (int i = 0; i < tb; ++i) {
                float* r = sc0 + static_cast<std::size_t>(i) * SLp;
                __m512 mx = _mm512_loadu_ps(r);
                for (int jc = 1; jc < j16; ++jc) {
                    mx = _mm512_max_ps(mx, _mm512_loadu_ps(r + jc * 16));
                }
                float maxs = _mm512_reduce_max_ps(mx);
                for (int j = j16 * 16; j < SL; ++j) {
                    if (r[j] > maxs) { maxs = r[j]; }
                }
                const __m512 maxv = _mm512_set1_ps(maxs);
                __m512 sv = _mm512_setzero_ps();
                for (int jc = 0; jc < j16; ++jc) {
                    float* p = r + jc * 16;
                    const __m512 e = _mm512_exp2_ps(_mm512_mul_ps(
                        _mm512_sub_ps(_mm512_loadu_ps(p), maxv), log2ev));
                    _mm512_storeu_ps(p, e);
                    sv = _mm512_add_ps(sv, e);
                }
                for (int j = j16 * 16; j < SL; ++j) { r[j] = std::exp(r[j] - maxs); }
                float sum = _mm512_reduce_add_ps(sv);
                for (int j = j16 * 16; j < SL; ++j) { sum += r[j]; }
                const float inv = 1.0F / sum;
                const __m512 invv = _mm512_set1_ps(inv);
                for (int jc = 0; jc < j16; ++jc) {
                    float* p = r + jc * 16;
                    _mm512_storeu_ps(p, _mm512_mul_ps(_mm512_loadu_ps(p), invv));
                }
                for (int j = j16 * 16; j < SL; ++j) { r[j] *= inv; }
            }
            if (ph) { tp = ns_(); a2 += tp - te; te = tp; }
            // PV per scores row: out[d] = sum_j srow[j] * v[j][d], srow = scores * inv. Named
            // d-chunk accumulators (4 zmm + 1 ymm tail) cover all HD per j; v is read directly
            // from qkv (v rows are contiguous HD-float spans at stride s3D — the vbuf staging
            // is gone). Bit-identical to the old per-row PV: same per-(i,d) FMA set, ascending
            // j, same pv[j]*inv products (the inv fold is the same IEEE mul).
            for (int i = 0; i < tb; ++i) {
                const float* srow = sc0 + static_cast<std::size_t>(i) * SLp;
                float* orow = out + static_cast<std::size_t>(ib + tb0 + i) * static_cast<std::size_t>(D) +
                              static_cast<std::size_t>(h) * HD;
                __m512 va0 = _mm512_setzero_ps(), va1 = _mm512_setzero_ps(),
                       va2 = _mm512_setzero_ps(), va3 = _mm512_setzero_ps();
                __m256 vah = _mm256_setzero_ps();
                for (int j = 0; j < SL; ++j) {
                    const float pf = srow[j];
                    const float* vr = vstage ? (vbuf.data() + static_cast<std::size_t>(j) * HD)
                                             : (qkv + static_cast<std::size_t>(ib + j) * s3D + 2 * D + h * HD);
                    const __m512 pb = _mm512_set1_ps(pf);
                    if (d16 >= 1) { va0 = _mm512_fmadd_ps(pb, _mm512_loadu_ps(vr + 0), va0); }
                    if (d16 >= 2) { va1 = _mm512_fmadd_ps(pb, _mm512_loadu_ps(vr + 16), va1); }
                    if (d16 >= 3) { va2 = _mm512_fmadd_ps(pb, _mm512_loadu_ps(vr + 32), va2); }
                    if (d16 >= 4) { va3 = _mm512_fmadd_ps(pb, _mm512_loadu_ps(vr + 48), va3); }
                    if (dy) {
                        vah = _mm256_fmadd_ps(_mm256_set1_ps(pf), _mm256_loadu_ps(vr + d16 * 16), vah);
                    }
                }
                if (d16 >= 1) { _mm512_storeu_ps(orow + 0, snap_bf16_vec(va0)); }
                if (d16 >= 2) { _mm512_storeu_ps(orow + 16, snap_bf16_vec(va1)); }
                if (d16 >= 3) { _mm512_storeu_ps(orow + 32, snap_bf16_vec(va2)); }
                if (d16 >= 4) { _mm512_storeu_ps(orow + 48, snap_bf16_vec(va3)); }
                if (dy) { _mm256_storeu_ps(orow + d16 * 16, snap_bf16_vec16(vah)); }
                for (int d = dy; d < drem; ++d) {  // degenerate tail (head_dim % 16 in 1..7)
                    float a = 0.0F;
                    for (int j = 0; j < SL; ++j) {
                        const float v = vstage ? vbuf[static_cast<std::size_t>(j) * HD + d16 * 16 + d]
                                               : qkv[static_cast<std::size_t>(ib + j) * s3D + 2 * D +
                                                     h * HD + d16 * 16 + d];
                        a += srow[j] * v;
                    }
                    orow[d16 * 16 + d] = snap_bf16_scalar(a);
                }
            }
            if (ph) { tp = ns_(); a3 += tp - te; te = tp; }
        }
        ph_ns[0].fetch_add(a0, std::memory_order_relaxed);
        ph_ns[1].fetch_add(a1, std::memory_order_relaxed);
        ph_ns[2].fetch_add(a2, std::memory_order_relaxed);
        ph_ns[3].fetch_add(a3, std::memory_order_relaxed);
    });
    if (ph) {
        std::fprintf(stderr,
                     "attn phases us: staging %lld qk %lld softmax %lld pv %lld (jobs %d)\n",
                     ph_ns[0].load() / 1000, ph_ns[1].load() / 1000,
                     ph_ns[2].load() / 1000, ph_ns[3].load() / 1000, segs * H);
    }
}

// ---- elementwise AVX-512 passes (16-lane, one BF16 snap boundary per element, same per-element
// formula as the scalar kernels in vision_cpu_impl.h) ----

// x[i] = snap(x[i] + y[i]); both FP32, same count.
void snap_add_into(const float* y, std::size_t count, float* x) {
    pfor((count + 15) / 16, [&](std::size_t b) {
        const std::size_t i0 = b * 16;
        if (i0 + 16 <= count) {
            _mm512_storeu_ps(x + i0,
                             snap_bf16_vec(_mm512_add_ps(_mm512_loadu_ps(x + i0),
                                                         _mm512_loadu_ps(y + i0))));
            return;
        }
        for (std::size_t i = i0; i < count; ++i) { x[i] = snap_bf16_scalar(x[i] + y[i]); }
    });
}

// x[t][d] = snap(x[t][d] + bias[d]); bias is FP32 [D], x is [T*D] (count = T*D).
void snap_add_bias_into(const float* bias, int D, std::size_t count, float* x) {
    const std::size_t Ds = static_cast<std::size_t>(D);
    pfor((count + 15) / 16, [&](std::size_t b) {
        const std::size_t i0 = b * 16;
        if (i0 + 16 <= count && i0 % Ds + 16 <= Ds) {
            // Fast path: the 16-lane block stays inside one row (no D-boundary crossing), so the
            // 16 bias values are contiguous at i0 % D.
            _mm512_storeu_ps(x + i0, snap_bf16_vec(
                                         _mm512_add_ps(_mm512_loadu_ps(x + i0),
                                                       _mm512_loadu_ps(bias + i0 % Ds))));
            return;
        }
        for (std::size_t i = i0; i < count; ++i) { x[i] = snap_bf16_scalar(x[i] + bias[i % Ds]); }
    });
}

// x[i] = GELU_tanh(x[i]) = z*(1 - 1/(exp2(u*2*log2e)+1)), u = sp*(z + 0.044715*z^3),
// sp = sqrt(2/pi): algebraically the same curve as 0.5*z*(1 + tanh(u)) (tanh(u) = 1 - 2/(e^2u+1)),
// computed with the hardware exp2 instead of the libm tanh (both land on the same BF16 grid to
// within a few FP32 ulps; the A/B ULP gate absorbs it).
void gelu_tanh_into(float* x, std::size_t count) {
    constexpr float kSp  = 0.7978845608028654F;  // sqrt(2/pi)
    constexpr float kC   = 0.044715F;
    constexpr float kL2E = 2.0F * 1.4426950408889634F;  // 2*log2(e)
    const __m512 spv  = _mm512_set1_ps(kSp);
    const __m512 cv   = _mm512_set1_ps(kC);
    const __m512 l2ev = _mm512_set1_ps(kL2E);
    const __m512 onev = _mm512_set1_ps(1.0F);
    pfor((count + 15) / 16, [&](std::size_t b) {
        const std::size_t i0 = b * 16;
        if (i0 + 16 <= count) {
            const __m512 z   = _mm512_loadu_ps(x + i0);
            const __m512 z3  = _mm512_mul_ps(_mm512_mul_ps(z, z), z);
            const __m512 u   = _mm512_mul_ps(spv, _mm512_add_ps(z, _mm512_mul_ps(cv, z3)));
            const __m512 e   = _mm512_exp2_ps(_mm512_mul_ps(u, l2ev));
            const __m512 inv = _mm512_div_ps(onev, _mm512_add_ps(e, onev));
            _mm512_storeu_ps(x + i0, snap_bf16_vec(_mm512_mul_ps(z, _mm512_sub_ps(onev, inv))));
            return;
        }
        for (std::size_t i = i0; i < count; ++i) {
            const float z = x[i];
            x[i]          = snap_bf16_scalar(0.5F * z * (1.0F + std::tanh(kSp * (z + kC * z * z * z))));
        }
    });
}

// out[t][d] = (x[t][d]-mean)*rstd*wf[d] + bf[d]; mean/var over d, rstd = 1/sqrt(var+eps).
// wf/bf are FP32 [D] (already converted; the scalar path converts the same way). Four
// independent 16-lane accumulators reduce at the end — the FP32 summation order differs from
// the scalar loop by a few ulps, invisible after the BF16 snap. Requires D % 16 == 0 for the
// vector body (1152/4608 both are); the scalar tail covers any remainder.
void layer_norm_into(const float* x, const float* wf, const float* bf, float inv_d, float eps,
                     int T, int D, float* out) {
    const int d16 = D / 16;
    pfor(static_cast<std::size_t>(T), [&](std::size_t t) {
        const float* xr = x + t * D;
        float* op       = out + t * D;
        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
        int d = 0;
        for (; d + 4 <= d16; d += 4) {
            s0 = _mm512_add_ps(s0, _mm512_loadu_ps(xr + d * 16));
            s1 = _mm512_add_ps(s1, _mm512_loadu_ps(xr + (d + 1) * 16));
            s2 = _mm512_add_ps(s2, _mm512_loadu_ps(xr + (d + 2) * 16));
            s3 = _mm512_add_ps(s3, _mm512_loadu_ps(xr + (d + 3) * 16));
        }
        for (; d < d16; ++d) { s0 = _mm512_add_ps(s0, _mm512_loadu_ps(xr + d * 16)); }
        const float mean =
            _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3))) *
            inv_d;
        const __m512 mb = _mm512_set1_ps(mean);
        s0 = s1 = s2 = s3 = _mm512_setzero_ps();
        d = 0;
        for (; d + 4 <= d16; d += 4) {
            const __m512 v0 = _mm512_sub_ps(_mm512_loadu_ps(xr + d * 16), mb);
            const __m512 v1 = _mm512_sub_ps(_mm512_loadu_ps(xr + (d + 1) * 16), mb);
            const __m512 v2 = _mm512_sub_ps(_mm512_loadu_ps(xr + (d + 2) * 16), mb);
            const __m512 v3 = _mm512_sub_ps(_mm512_loadu_ps(xr + (d + 3) * 16), mb);
            s0 = _mm512_fmadd_ps(v0, v0, s0);
            s1 = _mm512_fmadd_ps(v1, v1, s1);
            s2 = _mm512_fmadd_ps(v2, v2, s2);
            s3 = _mm512_fmadd_ps(v3, v3, s3);
        }
        for (; d < d16; ++d) {
            const __m512 v = _mm512_sub_ps(_mm512_loadu_ps(xr + d * 16), mb);
            s0 = _mm512_fmadd_ps(v, v, s0);
        }
        const float var  =
            _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3))) *
            inv_d;
        const float rstd = 1.0F / std::sqrt(var + eps);
        const __m512 rb   = _mm512_set1_ps(rstd);
        for (int dd = 0; dd < d16; ++dd) {
            const __m512 v  = _mm512_loadu_ps(xr + dd * 16);
            const __m512 wv = _mm512_loadu_ps(wf + dd * 16);
            const __m512 bv = _mm512_loadu_ps(bf + dd * 16);
            const __m512 o  = _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(v, mb), rb), wv), bv);
            _mm512_storeu_ps(op + dd * 16, snap_bf16_vec(o));
        }
        for (int dd = d16 * 16; dd < D; ++dd) {  // degenerate tail (D % 16 in 1..15)
            op[dd] = snap_bf16_scalar((xr[dd] - mean) * rstd * wf[dd] + bf[dd]);
        }
    });
}

// Vision 2-D RoPE in place over the q and k parts of qkv [T][3D] — same formula, cos/sin
// staging (double-precision pow/cos/sin) and BF16 snap as rope_vision_into in vision_cpu_impl.h.
// pos is the SPLIT layout [0..T)=y, [T..2T)=x; the cos/sin table for row t has rotary_dim/2
// entries shared by every head (entry i: axis = i<R2/2 ? 0 : 1, local = i % (R2/2)).
void rope_vision_into(const std::int32_t* pos, int T, int D, int H, int HD, int R2, float theta,
                      float* qkv) {
    const float inv_r2 = 1.0F / static_cast<float>(R2);
    pfor(static_cast<std::size_t>(T), [&](std::size_t r) {
        const int t  = static_cast<int>(r);
        const int py = pos[t], px = pos[T + t];
        std::vector<float> cv(R2), sv(R2);
        for (int i = 0; i < R2; ++i) {
            const int axis  = (i < R2 / 2) ? 0 : 1;
            const int local = i % (R2 / 2);
            const double phi =
                static_cast<double>(axis == 0 ? py : px) *
                std::pow(static_cast<double>(theta), -2.0 * static_cast<double>(local) * inv_r2);
            cv[i] = static_cast<float>(std::cos(phi));
            sv[i] = static_cast<float>(std::sin(phi));
        }
        float* qrow = qkv + static_cast<std::size_t>(t) * static_cast<std::size_t>(3 * D);
        float* krow = qrow + D;
        for (int h = 0; h < H; ++h) {
            float* qv = qrow + static_cast<std::size_t>(h) * HD;
            float* kv = krow + static_cast<std::size_t>(h) * HD;
            for (int i0 = 0; i0 < R2; i0 += 16) {
                if (i0 + 16 <= R2) {
                    const __m512 c = _mm512_loadu_ps(cv.data() + i0);
                    const __m512 s = _mm512_loadu_ps(sv.data() + i0);
                    const __m512 qa = _mm512_loadu_ps(qv + i0);
                    const __m512 qb = _mm512_loadu_ps(qv + i0 + R2);
                    _mm512_storeu_ps(qv + i0, snap_bf16_vec(_mm512_sub_ps(_mm512_mul_ps(qa, c),
                                                                          _mm512_mul_ps(qb, s))));
                    _mm512_storeu_ps(qv + i0 + R2, snap_bf16_vec(_mm512_add_ps(_mm512_mul_ps(qb, c),
                                                                               _mm512_mul_ps(qa, s))));
                    const __m512 ka = _mm512_loadu_ps(kv + i0);
                    const __m512 kb = _mm512_loadu_ps(kv + i0 + R2);
                    _mm512_storeu_ps(kv + i0, snap_bf16_vec(_mm512_sub_ps(_mm512_mul_ps(ka, c),
                                                                          _mm512_mul_ps(kb, s))));
                    _mm512_storeu_ps(kv + i0 + R2, snap_bf16_vec(_mm512_add_ps(_mm512_mul_ps(kb, c),
                                                                               _mm512_mul_ps(ka, s))));
                } else {
                    for (int i = i0; i < R2; ++i) {
                        const float c = cv[i], s = sv[i];
                        const float qa = qv[i], qb = qv[i + R2];
                        qv[i]          = snap_bf16_scalar(qa * c - qb * s);
                        qv[i + R2]     = snap_bf16_scalar(qb * c + qa * s);
                        const float ka = kv[i], kb = kv[i + R2];
                        kv[i]          = snap_bf16_scalar(ka * c - kb * s);
                        kv[i + R2]     = snap_bf16_scalar(kb * c + ka * s);
                    }
                }
            }
        }
    });
}

// ---- AVX-512 dequant: 16-row blocks, k-outer, so the [k*N + n] stores are 16 contiguous
// floats each (the scalar kernel writes one 4 B element per 64 B line — 16x line traffic; that
// RFO flood is most of the scalar dequant time). Same per-element decode as dequant_weight in
// vision_cpu_impl.h (Q4/Q5/Q6 nibble + high-bit fields, W8 int8, fp16 group scale).
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
        value = (1.0F + static_cast<float>(mant) / 1024.0F) * std::ldexp(1.0F, static_cast<int>(exp) - 15);
    }
    return neg ? -value : value;
}

void dequant_weight_into(DequantKind kind, const std::uint8_t* codes, const std::uint8_t* high,
                         const std::uint16_t* scales, int n, int k, int group, int padded,
                         float* out) {
    const std::size_t N = static_cast<std::size_t>(n);
    const std::size_t K = static_cast<std::size_t>(k);
    const std::size_t G = static_cast<std::size_t>(group);
    const std::size_t P = static_cast<std::size_t>(padded);
    const std::size_t blocks = N / 16 + (N % 16 ? 1 : 0);
    pfor(blocks, [&](std::size_t bi) {
        const std::size_t n0 = bi * 16;
        const std::size_t nr = std::min<std::size_t>(16, N - n0);
        std::size_t g0[16];
        for (std::size_t r = 0; r < 16; ++r) {
            // Clamp the phantom rows of a trailing partial block to the last real row — their
            // codes/scales would otherwise read out of bounds; the lanes are never stored.
            g0[r] = std::min(n0 + r, N - 1) * P / G;
        }
        const std::size_t gcount = K / G + (K % G ? 1 : 0);
        float scals[256][16];  // [group][row] — 256 groups covers k <= 8192 at group 32
        for (std::size_t gg = 0; gg < gcount; ++gg) {
            for (std::size_t r = 0; r < 16; ++r) { scals[gg][r] = fp16_to_f32(scales[g0[r] + gg]); }
        }
        for (std::size_t gg = 0; gg < gcount; ++gg) {
            const std::size_t k0 = gg * G;
            const std::size_t gk = std::min(G, K - k0);
            const std::uint8_t* cp[16];
            const std::uint8_t* hp[16];
            for (std::size_t r = 0; r < 16; ++r) {
                cp[r] = codes + (g0[r] + gg) * 32;
                hp[r] = (kind == DequantKind::Q5G64) ? high + (g0[r] + gg) * 8
                       : (kind == DequantKind::Q6G64) ? high + (g0[r] + gg) * 16
                                                      : cp[r];
            }
            if (kind == DequantKind::W8G32) {
                for (std::size_t jj = 0; jj < gk; ++jj) {
                    std::int32_t q[16];
                    for (std::size_t r = 0; r < 16; ++r) { q[r] = static_cast<std::int8_t>(cp[r][jj]); }
                    const __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_epi32(q)),
                                                  _mm512_loadu_ps(scals[gg]));
                    if (nr == 16) { _mm512_storeu_ps(out + (k0 + jj) * N + n0, v); }
                    else {
                        float tmp[16];
                        _mm512_storeu_ps(tmp, v);
                        for (std::size_t r = 0; r < nr; ++r) { out[(k0 + jj) * N + n0 + r] = tmp[r]; }
                    }
                }
            } else if (kind == DequantKind::Q4G64) {
                for (std::size_t jj = 0; jj < gk; ++jj) {
                    std::int32_t q[16];
                    for (std::size_t r = 0; r < 16; ++r) {
                        const std::uint8_t c = cp[r][jj / 2];
                        q[r] = (static_cast<int>((jj & 1) ? (c >> 4) : (c & 0x0f)) ^ 0x08) - 8;
                    }
                    const __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_epi32(q)),
                                                  _mm512_loadu_ps(scals[gg]));
                    if (nr == 16) { _mm512_storeu_ps(out + (k0 + jj) * N + n0, v); }
                    else {
                        float tmp[16];
                        _mm512_storeu_ps(tmp, v);
                        for (std::size_t r = 0; r < nr; ++r) { out[(k0 + jj) * N + n0 + r] = tmp[r]; }
                    }
                }
            } else if (kind == DequantKind::Q5G64) {
                for (std::size_t jj = 0; jj < gk; ++jj) {
                    std::int32_t q[16];
                    for (std::size_t r = 0; r < 16; ++r) {
                        const std::uint8_t c   = cp[r][jj / 2];
                        const int n4           = (jj & 1) ? (c >> 4) : (c & 0x0f);
                        const int bit          = (hp[r][jj >> 3] >> (jj & 7)) & 1;
                        q[r]                    = ((n4 | bit << 4) ^ 0x10) - 16;
                    }
                    const __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_epi32(q)),
                                                  _mm512_loadu_ps(scals[gg]));
                    if (nr == 16) { _mm512_storeu_ps(out + (k0 + jj) * N + n0, v); }
                    else {
                        float tmp[16];
                        _mm512_storeu_ps(tmp, v);
                        for (std::size_t r = 0; r < nr; ++r) { out[(k0 + jj) * N + n0 + r] = tmp[r]; }
                    }
                }
            } else {  // Q6G64
                for (std::size_t jj = 0; jj < gk; ++jj) {
                    std::int32_t q[16];
                    for (std::size_t r = 0; r < 16; ++r) {
                        const std::uint8_t c   = cp[r][jj / 2];
                        const int n4           = (jj & 1) ? (c >> 4) : (c & 0x0f);
                        const int h2           = (hp[r][jj >> 2] >> ((jj & 3) << 1)) & 3;
                        q[r]                    = ((n4 | h2 << 4) ^ 0x20) - 32;
                    }
                    const __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_epi32(q)),
                                                  _mm512_loadu_ps(scals[gg]));
                    if (nr == 16) { _mm512_storeu_ps(out + (k0 + jj) * N + n0, v); }
                    else {
                        float tmp[16];
                        _mm512_storeu_ps(tmp, v);
                        for (std::size_t r = 0; r < nr; ++r) { out[(k0 + jj) * N + n0 + r] = tmp[r]; }
                    }
                }
            }
        }
    });
}

}  // namespace ninfer::targets::qwen3_6::detail::cpu_simd
