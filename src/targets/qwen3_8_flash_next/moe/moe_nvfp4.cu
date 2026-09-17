// P8: paged NVFP4 sparse MoE (512 experts x top-10 x 640) — target-private kernels.
//
// v1 contract (see include/ninfer/ops/sparse_moe_nvfp4.h). Both routes are
// oracle-qualified against an FP64 reference on the DECODED weights; the
// production mma path (block-scale FP4 x FP4) is deferred to P13 — this Op
// computes by dequantizing each weight K16 group (code * e4m3) and each
// activation (W4A16: BF16 direct; W4A4: quantize-then-dequant) on the fly and
// accumulating in FP32, applying the per-object divisor in the FP32 epilogue.
//
// Layout facts (canonical blockscale-k16-m128x4-v1, docs/maintainer/storage-
// layouts.md §4 — the on-disk weight CODES are natural row-major; only the
// weight SCALES are swizzled):
//   codes[n][k]          = codes[n * (K/2) + k/2], low nibble = even k, high = odd
//   scales slab (row n, k64 group g = k/64) =
//     scales + ((n/128) * (K/64) + g) * 512
//          + ((n & 127) & 31) * 16 + ((n & 127) >> 5) * 4
//     (4 consecutive bytes = the E4M3 scales of the k64's 4 K16 subgroups;
//      subgroup = (k & 63) >> 4)
//
// Compact columns: the op treats the routed top-K selections and the shared
// expert uniformly as C = T*(K+1) columns. Column c < T*K is routed
// (token t = c/K, rank j = c%K, expert = ids[t*K+j]); column c >= T*K is the
// shared expert (token t = c - T*K). The per-column expert is resolved by pure
// arithmetic, so no scatter/scan is needed.

#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <ninfer/ops/sparse_moe_nvfp4.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

using ninfer::DType;
using ninfer::Tensor;
using ninfer::WorkspaceArena;
using ninfer::ops::SparseMoeNvfp4Epilogue;
using ninfer::ops::SparseMoeNvfp4Geometry;
using ninfer::ops::SparseMoeNvfp4Plane;
using ninfer::ops::SparseMoeNvfp4Route;
using ninfer::ops::SparseMoeNvfp4Weights;

constexpr int kThreads = 256;

__device__ __forceinline__ float moe_silu(float x) {
    return x / (1.0F + __expf(-x));
}

__device__ __forceinline__ float moe_sigmoid(float x) {
    return 1.0F / (1.0F + __expf(-x));
}

// One weight element (code * e4m3), natural codes + swizzled scales. K = the
// plane's reduction extent (must be a multiple of 64).
__device__ __forceinline__ float moe_dequant_weight(const std::uint8_t* __restrict__ codes,
                                                    const std::uint8_t* __restrict__ scales,
                                                    int n, int k, int K) {
    const std::uint8_t cbyte =
        __ldg(codes + static_cast<std::int64_t>(n) * (K / 2) + (k >> 1));
    const float2 cv          = ninfer::ops::detail::decode_nvfp4_e2m1x2(cbyte);
    const float code         = (k & 1) ? cv.y : cv.x;
    const int slab = ((n / 128) * (K / 64) + (k / 64)) * 512;
    const int off  = ((n & 127) & 31) * 16 + ((n & 127) >> 5) * 4 + ((k & 63) >> 4);
    const float scale = ninfer::ops::detail::decode_nvfp4_e4m3(__ldg(scales + slab + off));
    return code * scale;
}

// Dequant one NVFP4 activation element for column/row k. The activation planes
// are written natural, never swizzled: codes_row[k/2] (low nibble = even k,
// high = odd), scales_row[k/16]. codes_row / scales_row point at the start of
// the activation's row (per-token for the input, per-column for h2/aq2).
__device__ __forceinline__ float moe_dequant_act_row(const std::uint8_t* __restrict__ codes_row,
                                                     const std::uint8_t* __restrict__ scales_row,
                                                     int k) {
    const std::uint8_t cb = __ldg(codes_row + (k >> 1));
    const float2 cv       = ninfer::ops::detail::decode_nvfp4_e2m1x2(cb);
    const float code      = (k & 1) ? cv.y : cv.x;
    const float scale     = ninfer::ops::detail::decode_nvfp4_e4m3(__ldg(scales_row + (k >> 4)));
    return code * scale;
}

// Resolve a compact column to (token, is_shared, routed expert id).
__device__ __forceinline__ void moe_resolve_col(int c, int T, int K,
                                                const std::int32_t* __restrict__ ids,
                                                int& token, int& is_shared, int& expert) {
    if (c < T * K) {
        token     = c / K;
        is_shared = 0;
        expert    = ids[token * K + (c - token * K)];
    } else {
        token     = c - T * K;
        is_shared = 1;
        expert    = 0;
    }
}

// Per-(t, r) router work: r in [0, E) = expert logit, r == E = shared gate.
__global__ __launch_bounds__(kThreads) void moe_router_logit_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint16_t* __restrict__ router,
    const std::uint16_t* __restrict__ shared_gate, std::int32_t T, std::int32_t E,
    std::int32_t K_in, float* __restrict__ logits, float* __restrict__ shared_gate_score) {
    const int task = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    const int total = T * (E + 1);
    if (task >= total) { return; }
    const int t  = task / (E + 1);
    const int r  = task - t * (E + 1);
    const __nv_bfloat16* xr = x + static_cast<std::int64_t>(t) * K_in;
    float acc             = 0.0F;
    if (r < E) {
        const std::uint16_t* wr = router + static_cast<std::int64_t>(r) * K_in;
// No-FMA: __fmul_rn then __fadd_rn is bit-identical to the host oracle's
// separate mul + add (MSVC /fp:precise, no contraction), so the top-K selection
// (ids) matches exactly.
// __ushort_as_bfloat16 is a BIT cast; an implicit ushort->__nv_bfloat16
// conversion instead promotes numerically (ushort bits become their integer
// value), inflating every router logit ~1e4x.
#pragma unroll 4
        for (int k = 0; k < K_in; ++k) {
            acc = __fadd_rn(acc,
                            __fmul_rn(__bfloat162float(xr[k]),
                                     __bfloat162float(__ushort_as_bfloat16(__ldg(wr + k)))));
        }
        logits[static_cast<std::int64_t>(t) * E + r] = acc;
    } else {
        const std::uint16_t* g = shared_gate;
#pragma unroll 4
        for (int k = 0; k < K_in; ++k) {
            acc = __fadd_rn(acc,
                            __fmul_rn(__bfloat162float(xr[k]),
                                     __bfloat162float(__ushort_as_bfloat16(__ldg(g + k)))));
        }
        shared_gate_score[t] = moe_sigmoid(acc);
    }
}

// Per-token softmax-over-E + top-K selection (value DESC, id ASC) +
// renormalized alpha. ids/alpha written with ids ASCENDING.
__global__ __launch_bounds__(kThreads) void moe_router_topk_kernel(
    const float* __restrict__ logits, std::int32_t T, std::int32_t E, std::int32_t K,
    std::int32_t* __restrict__ ids, float* __restrict__ alpha) {
    const int t = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (t >= T) { return; }
    const float* row = logits + static_cast<std::int64_t>(t) * E;

    float m = -INFINITY;
#pragma unroll 4
    for (int e = 0; e < E; ++e) {
        m = fmaxf(m, row[e]);
    }

    // K rounds of argmax with the (value DESC, id ASC) tie-break. K <= 32.
    constexpr int kMaxK = 32;
    std::int32_t sel_e[kMaxK];
    float sel_v[kMaxK];
    for (int s = 0; s < K; ++s) {
        float best_v = -INFINITY;
        int best_e   = -1;
#pragma unroll 4
        for (int e = 0; e < E; ++e) {
            bool taken = false;
            for (int q = 0; q < s; ++q) {
                if (sel_e[q] == e) { taken = true; break; }
            }
            if (taken) { continue; }
            const float v = row[e];
            if (v > best_v || (v == best_v && (best_e < 0 || e < best_e))) {
                best_v = v;
                best_e = e;
            }
        }
        sel_e[s] = best_e;
        sel_v[s] = best_v;
    }

    // Sort the K selected by expert id ascending (K <= 32, insertion sort).
    for (int a = 1; a < K; ++a) {
        const int ee = sel_e[a];
        const float vv = sel_v[a];
        int b         = a;
        while (b > 0 && sel_e[b - 1] > ee) {
            sel_e[b] = sel_e[b - 1];
            sel_v[b] = sel_v[b - 1];
            --b;
        }
        sel_e[b] = ee;
        sel_v[b] = vv;
    }

    // alpha_j = exp(v_j - m) / Sum_selected exp(v_j - m).
    float ssum = 0.0F;
    float exps[kMaxK];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        exps[j] = __expf(sel_v[j] - m);
        ssum += exps[j];
    }
    std::int32_t* ids_row = ids + static_cast<std::int64_t>(t) * K;
    float* alpha_row      = alpha + static_cast<std::int64_t>(t) * K;
#pragma unroll
    for (int j = 0; j < K; ++j) {
        ids_row[j]   = sel_e[j];
        alpha_row[j] = ssum != 0.0F ? exps[j] / ssum : 0.0F;
    }
}

// W4A4 input quantization: x [K_in, T] (x[k][t] = x[t*K_in + k]) ->
// aq_codes [T][K_in/2] + aq_scales [T][K_in/16]. Divisor 1.0 (weights carry
// the divisor).
__global__ __launch_bounds__(kThreads) void moe_quant_x_kernel(
    const __nv_bfloat16* __restrict__ x, std::uint8_t* __restrict__ aq_codes,
    std::uint8_t* __restrict__ aq_scales, std::int32_t T, std::int32_t K_in) {
    const int kGroupsPerRow = K_in / 16;
    const int task = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    const int total = T * kGroupsPerRow;
    if (task >= total) { return; }
    const int t     = task / kGroupsPerRow;
    const int group = task - t * kGroupsPerRow;
    const ninfer::ops::detail::Nvfp4QuantizedK16 q =
        ninfer::ops::detail::quantize_nvfp4_k16(x + static_cast<std::int64_t>(t) * K_in + group * 16,
                                                1.0F);
    std::uint8_t* cd = aq_codes + static_cast<std::int64_t>(t) * (K_in / 2) + group * 8;
    cd[0] = static_cast<std::uint8_t>(q.codes_lo & 0xffu);
    cd[1] = static_cast<std::uint8_t>((q.codes_lo >> 8) & 0xffu);
    cd[2] = static_cast<std::uint8_t>((q.codes_lo >> 16) & 0xffu);
    cd[3] = static_cast<std::uint8_t>((q.codes_lo >> 24) & 0xffu);
    cd[4] = static_cast<std::uint8_t>(q.codes_hi & 0xffu);
    cd[5] = static_cast<std::uint8_t>((q.codes_hi >> 8) & 0xffu);
    cd[6] = static_cast<std::uint8_t>((q.codes_hi >> 16) & 0xffu);
    cd[7] = static_cast<std::uint8_t>((q.codes_hi >> 24) & 0xffu);
    aq_scales[static_cast<std::int64_t>(t) * kGroupsPerRow + group] = q.scale;
}

struct MoegemmArgs {
    const std::uint8_t* codes_base;   // shared plane codes (for is_shared columns)
    const std::uint8_t* scales_base;  // shared plane scales
    float shared_divisor;
    const std::uint8_t* expert_window;
    std::int64_t expert_bytes;
    const std::uint8_t* const* expert_ptrs;  // per-expert device table; null = window
    std::int64_t plane_codes_off;     // plane offset within an expert blob (codes)
    std::int64_t plane_scales_off;    // plane offset within an expert blob (scales)
    float routed_divisor;
    const __nv_bfloat16* x;           // activation rows (W4A16); null for W4A4
    const std::uint8_t* aq_codes;     // W4A4 activation codes
    const std::uint8_t* aq_scales;    // W4A4 activation scales
    bool w4a4;
    bool act_row_is_col;   // gu: activation row = token; dn: activation row = column
    bool store_f32;        // gu: h BF16; dn: y FP32
    std::int32_t T;
    std::int32_t Ktop;
    std::int32_t C;
    std::int32_t N;
    std::int32_t K;
};

// One GEMM over compact columns: out[n][c] (c-fastest) = alpha * sum_k
// act_k(c) * W_e(c)[n][k], alpha = 1/divisor. The stored NVFP4 word is the
// weight_scale_divisor (a large normalization constant); the reconstructed
// weight is code*scale/divisor, so the epilogue divides (mirrors the 27B
// nvfp4_w4a4 alpha = 1/(input_div*weight_div); W4A4 input_div = 1.0). act =
// x column token (gu) or h2/aq2 row column c (dn); W4A16 reads BF16 directly,
// W4A4 dequantizes the natural quantized codes.
__global__ __launch_bounds__(kThreads) void moe_gemm_kernel(
    MoegemmArgs args, const std::int32_t* __restrict__ ids,
    void* __restrict__ out) {
    const int idx  = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    const int total = args.N * args.C;
    if (idx >= total) { return; }
    const int n = idx % args.N;
    const int c = idx / args.N;

    int token = 0, is_shared = 0, expert = 0;
    moe_resolve_col(c, args.T, args.Ktop, ids, token, is_shared, expert);
    const int act_row = args.act_row_is_col ? c : token;

    const std::uint8_t* codes;
    const std::uint8_t* scales;
    float divisor;
    if (is_shared) {
        codes   = args.codes_base;
        scales  = args.scales_base;
        divisor = args.shared_divisor;
    } else {
        const std::uint8_t* blob =
            (args.expert_ptrs != nullptr) ? args.expert_ptrs[expert]
                                          : args.expert_window + expert * args.expert_bytes;
        codes   = blob + args.plane_codes_off;
        scales  = blob + args.plane_scales_off;
        divisor = args.routed_divisor;
    }

    float acc = 0.0F;
    if (args.w4a4) {
        const std::uint8_t* aq_c =
            args.aq_codes + static_cast<std::int64_t>(act_row) * (args.K / 2);
        const std::uint8_t* aq_s =
            args.aq_scales + static_cast<std::int64_t>(act_row) * (args.K / 16);
#pragma unroll 4
        for (int k = 0; k < args.K; ++k) {
            acc += moe_dequant_act_row(aq_c, aq_s, k) *
                   moe_dequant_weight(codes, scales, n, k, args.K);
        }
    } else {
        const __nv_bfloat16* xr = args.x + static_cast<std::int64_t>(act_row) * args.K;
#pragma unroll 4
        for (int k = 0; k < args.K; ++k) {
            acc += __bfloat162float(__ldg(xr + k)) * moe_dequant_weight(codes, scales, n, k, args.K);
        }
    }
    const float alpha = 1.0F / divisor;
    if (args.store_f32) {
        static_cast<float*>(out)[static_cast<std::int64_t>(c) * args.N + n] = acc * alpha;
    } else {
        static_cast<__nv_bfloat16*>(out)[static_cast<std::int64_t>(c) * args.N + n] =
            __float2bfloat16_rn(acc * alpha);
    }
}

// silu_mul: h2[i][c] = silu(h[i][c]) * h[i+I][c], i in [0, I), h c-fastest
// [2I][C]. W4A16: store BF16 h2 [I][C]. W4A4: re-quantize h2 (per K16 over the
// I dim, per column) to aq2 codes/scales.
__global__ __launch_bounds__(kThreads) void moe_silu_mul_kernel(
    const __nv_bfloat16* __restrict__ h, std::int32_t C, std::int32_t I, bool w4a4,
    __nv_bfloat16* __restrict__ h2, std::uint8_t* __restrict__ aq2_codes,
    std::uint8_t* __restrict__ aq2_scales) {
    if (w4a4) {
        const int kGroups = I / 16;
        const int task        = static_cast<int>(blockIdx.x) * kThreads +
                                 static_cast<int>(threadIdx.x);
        const int total       = C * kGroups;
        if (task >= total) { return; }
        const int c     = task / kGroups;
        const int group = task - c * kGroups;
        const int i0    = group * 16;
        // Gather the 16 h2 values for this column/K16 group.
        __nv_bfloat16 buf[16];
        const __nv_bfloat16* hg = h + static_cast<std::int64_t>(c) * (2 * I) + i0;
        const __nv_bfloat16* hu = hg + I;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const float g   = __bfloat162float(hg[s]);
            const float up  = __bfloat162float(hu[s]);
            buf[s]          = __float2bfloat16_rn(moe_silu(g) * up);
        }
        const ninfer::ops::detail::Nvfp4QuantizedK16 q =
            ninfer::ops::detail::quantize_nvfp4_k16(buf, 1.0F);
        std::uint8_t* cd = aq2_codes + static_cast<std::int64_t>(c) * (I / 2) + group * 8;
        cd[0] = static_cast<std::uint8_t>(q.codes_lo & 0xffu);
        cd[1] = static_cast<std::uint8_t>((q.codes_lo >> 8) & 0xffu);
        cd[2] = static_cast<std::uint8_t>((q.codes_lo >> 16) & 0xffu);
        cd[3] = static_cast<std::uint8_t>((q.codes_lo >> 24) & 0xffu);
        cd[4] = static_cast<std::uint8_t>(q.codes_hi & 0xffu);
        cd[5] = static_cast<std::uint8_t>((q.codes_hi >> 8) & 0xffu);
        cd[6] = static_cast<std::uint8_t>((q.codes_hi >> 16) & 0xffu);
        cd[7] = static_cast<std::uint8_t>((q.codes_hi >> 24) & 0xffu);
        aq2_scales[static_cast<std::int64_t>(c) * kGroups + group] = q.scale;
    } else {
        const int task = static_cast<int>(blockIdx.x) * kThreads +
                         static_cast<int>(threadIdx.x);
        const int total = C * I;
        if (task >= total) { return; }
        const int c = task / I;
        const int i = task - c * I;
        const __nv_bfloat16* hp = h + static_cast<std::int64_t>(c) * (2 * I);
        const float g          = __bfloat162float(hp[i]);
        const float up         = __bfloat162float(hp[i + I]);
        h2[static_cast<std::int64_t>(c) * I + i] =
            __float2bfloat16_rn(moe_silu(g) * up);
    }
}

// Merge: out[t][k] = sum_j alpha[t][j] * y[c(t,j)][k] + sg[t] * y[c(t,shared)][k]
// (+ residual[t][k]). y c-fastest [K_in][C].
__global__ __launch_bounds__(kThreads) void moe_merge_kernel(
    const float* __restrict__ y, const float* __restrict__ alpha,
    const float* __restrict__ shared_gate_score, const std::int32_t* __restrict__ ids,
    const __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out,
    std::int32_t T, std::int32_t K, std::int32_t K_in, bool add_residual) {
    const int task  = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    const int total = T * K_in;
    if (task >= total) { return; }
    const int t  = task / K_in;
    const int k  = task - t * K_in;

    float acc = 0.0F;
    const float* alpha_row = alpha + static_cast<std::int64_t>(t) * K;
    const int shared_col   = T * K + t;
#pragma unroll 4
    for (int j = 0; j < K; ++j) {
        const int c = t * K + j;
        acc += alpha_row[j] * __ldg(y + static_cast<std::int64_t>(c) * K_in + k);
    }
    acc += shared_gate_score[t] * __ldg(y + static_cast<std::int64_t>(shared_col) * K_in + k);
    if (add_residual) {
        acc += __bfloat162float(__ldg(residual + static_cast<std::int64_t>(t) * K_in + k));
    }
    out[static_cast<std::int64_t>(t) * K_in + k] = __float2bfloat16_rn(acc);
}

// Workspace layout (route-independent maximum; all terms scale with T, so the
// max over T in [min_tokens, max_tokens] is at T = max_tokens).
struct WorkspaceLayout {
    std::size_t logits;
    std::size_t sg;
    std::size_t ids;
    std::size_t alpha;
    std::size_t aq;     // aq_codes + aq_scales (W4A4)
    std::size_t h;      // gu output, BF16 [2I][C]
    std::size_t mid;    // max(h2 BF16 [I][C], aq2 codes+scales)
    std::size_t y;      // down output, FP32 [K_in][C]
    std::size_t total;
};

[[nodiscard]] WorkspaceLayout plan_layout(const SparseMoeNvfp4Geometry& g, int T) {
    const int E = g.experts, K = g.topk, K_in = g.input_dim, I = g.intermediate;
    const std::size_t C = static_cast<std::size_t>(T) * (K + 1);
    WorkspaceLayout L;
    L.logits = static_cast<std::size_t>(T) * E * sizeof(float);
    L.sg     = static_cast<std::size_t>(T) * sizeof(float);
    L.ids    = static_cast<std::size_t>(T) * K * sizeof(std::int32_t);
    L.alpha  = static_cast<std::size_t>(T) * K * sizeof(float);
    L.aq     = static_cast<std::size_t>(T) * (K_in / 2) + static_cast<std::size_t>(T) * (K_in / 16);
    L.h      = C * (2 * I) * sizeof(__nv_bfloat16);
    const std::size_t h2   = C * I * sizeof(__nv_bfloat16);
    const std::size_t aq2  = C * (I / 2) + C * (I / 16);
    L.mid                  = std::max(h2, aq2);
    L.y                    = C * K_in * sizeof(float);
    auto align256          = [](std::size_t b) { return (b + 255) & ~static_cast<std::size_t>(255); };
    L.total = align256(L.logits) + align256(L.sg) + align256(L.ids) + align256(L.alpha) +
              align256(L.aq) + align256(L.h) + align256(L.mid) + align256(L.y);
    return L;
}

}  // namespace

namespace ninfer::ops {

std::size_t sparse_moe_nvfp4_workspace_capacity_bytes(const SparseMoeNvfp4Geometry& g,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    const std::uint32_t E = g.experts, K = g.topk, K_in = g.input_dim, I = g.intermediate;
    if (E < 1 || K == 0 || K > 32 || K_in % 64 != 0 || K_in < 128 || K_in > 4096 ||
        I % 64 != 0 || I < 64 || I > 4096 || min_tokens < 1 || min_tokens > 2048 ||
        max_tokens < min_tokens || max_tokens > 2048) {
        throw std::invalid_argument("sparse_moe_nvfp4: geometry/interval out of domain");
    }
    return plan_layout(g, max_tokens).total;
}

void sparse_moe_nvfp4(const Tensor& x, const SparseMoeNvfp4Geometry& geo,
                      const SparseMoeNvfp4Weights& w, SparseMoeNvfp4Route route,
                      SparseMoeNvfp4Epilogue epilogue, const Tensor* residual,
                      Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::uint32_t E = geo.experts, K = geo.topk, K_in = geo.input_dim, I = geo.intermediate;
    const int T = x.ne[1];
    if (E < 1 || K == 0 || K > 32 || K_in % 64 != 0 || K_in < 128 || K_in > 4096 ||
        I % 64 != 0 || I < 64 || I > 4096 || T < 1 || T > 2048 || x.ne[0] != static_cast<int32_t>(K_in)) {
        throw std::invalid_argument("sparse_moe_nvfp4: x shape / geometry out of domain");
    }
    const bool w4a4          = (route == SparseMoeNvfp4Route::W4A4);
    const bool add_residual  = (epilogue == SparseMoeNvfp4Epilogue::AddResidual);
    const WorkspaceLayout L  = plan_layout(geo, T);

    auto scope = workspace.scope();
    auto logits = workspace.alloc_bytes(L.logits);
    auto sg     = workspace.alloc_bytes(L.sg);
    auto ids    = workspace.alloc_bytes(L.ids);
    auto alpha  = workspace.alloc_bytes(L.alpha);
    auto aq     = workspace.alloc_bytes(L.aq);
    auto h      = workspace.alloc_bytes(L.h);
    auto mid    = workspace.alloc_bytes(L.mid);
    auto y      = workspace.alloc_bytes(L.y);

    const __nv_bfloat16* xp   = static_cast<const __nv_bfloat16*>(x.data);
    __nv_bfloat16* destp      = static_cast<__nv_bfloat16*>(destination.data);
    const __nv_bfloat16* res  = add_residual ? static_cast<const __nv_bfloat16*>(residual->data)
                                             : nullptr;
    float* dlogits          = static_cast<float*>(logits.data);
    float* dsg                = static_cast<float*>(sg.data);
    std::int32_t* dids        = static_cast<std::int32_t*>(ids.data);
    float* dalpha             = static_cast<float*>(alpha.data);
    std::uint8_t* daq_codes   = static_cast<std::uint8_t*>(aq.data);
    std::uint8_t* daq_scales  = daq_codes + static_cast<std::size_t>(T) * (K_in / 2);
    __nv_bfloat16* dh         = static_cast<__nv_bfloat16*>(h.data);
    __nv_bfloat16* dh2        = static_cast<__nv_bfloat16*>(mid.data);
    std::uint8_t* daq2_codes  = static_cast<std::uint8_t*>(mid.data);
    std::uint8_t* daq2_scales = daq2_codes + static_cast<std::size_t>(T) * (K + 1) * (I / 2);
    float* dy           = static_cast<float*>(y.data);

    const int C = T * (static_cast<int>(K) + 1);

    // 1. Router logits + shared gate.
    {
        const int total = T * (static_cast<int>(E) + 1);
        const int blocks = (total + kThreads - 1) / kThreads;
        moe_router_logit_kernel<<<blocks, kThreads, 0, stream>>>(
            xp, w.router_bf16, w.shared_gate_bf16, T, static_cast<int>(E), K_in,
            dlogits, dsg);
    }
    // 2. Top-K selection + alpha.
    {
        const int blocks = (T + kThreads - 1) / kThreads;
        moe_router_topk_kernel<<<blocks, kThreads, 0, stream>>>(
            dlogits, T, static_cast<int>(E), static_cast<int>(K), dids, dalpha);
    }

    // 3. W4A4 input quantization.
    if (w4a4) {
        const int total = T * (K_in / 16);
        const int blocks = (total + kThreads - 1) / kThreads;
        moe_quant_x_kernel<<<blocks, kThreads, 0, stream>>>(xp, daq_codes, daq_scales, T, K_in);
    }

    // 4. gate_up GEMM -> h [2I][C].
    {
        const auto planes = sparse_moe_nvfp4_planes(geo);
        MoegemmArgs args;
        args.codes_base    = w.shared_gu_codes;
        args.scales_base   = w.shared_gu_scales;
        args.shared_divisor = w.shared_gu_divisor;
        args.expert_window = w.expert_window;
        args.expert_bytes  = static_cast<std::int64_t>(sparse_moe_nvfp4_expert_bytes(geo));
        args.expert_ptrs   = w.expert_ptrs;
        args.plane_codes_off = static_cast<std::int64_t>(planes.gu_codes);
        args.plane_scales_off = static_cast<std::int64_t>(planes.gu_scales);
        args.routed_divisor   = w.routed_gu_divisor;
        args.x               = w4a4 ? nullptr : xp;
        args.aq_codes        = w4a4 ? daq_codes : nullptr;
        args.aq_scales       = w4a4 ? daq_scales : nullptr;
        args.w4a4            = w4a4;
        args.act_row_is_col  = false;
        args.store_f32       = false;
        args.T               = T;
        args.Ktop            = static_cast<int>(K);
        args.C               = C;
        args.N               = 2 * static_cast<int>(I);
        args.K               = K_in;
        const int total = (2 * static_cast<int>(I)) * C;
        const int blocks = (total + kThreads - 1) / kThreads;
        moe_gemm_kernel<<<blocks, kThreads, 0, stream>>>(args, dids, dh);
    }

    // 5. silu_mul -> h2 (BF16) or aq2 (W4A4).
    {
        const int total = w4a4 ? C * (I / 16) : C * static_cast<int>(I);
        const int blocks = (total + kThreads - 1) / kThreads;
        moe_silu_mul_kernel<<<blocks, kThreads, 0, stream>>>(dh, C, static_cast<int>(I), w4a4,
                                                             dh2, daq2_codes, daq2_scales);
    }

    // 6. down GEMM -> y [K_in][C].
    {
        const auto planes = sparse_moe_nvfp4_planes(geo);
        MoegemmArgs args;
        args.codes_base    = w.shared_dn_codes;
        args.scales_base   = w.shared_dn_scales;
        args.shared_divisor = w.shared_dn_divisor;
        args.expert_window = w.expert_window;
        args.expert_bytes  = static_cast<std::int64_t>(sparse_moe_nvfp4_expert_bytes(geo));
        args.expert_ptrs   = w.expert_ptrs;
        args.plane_codes_off = static_cast<std::int64_t>(planes.dn_codes);
        args.plane_scales_off = static_cast<std::int64_t>(planes.dn_scales);
        args.routed_divisor   = w.routed_dn_divisor;
        args.x               = w4a4 ? nullptr : dh2;
        args.aq_codes        = w4a4 ? daq2_codes : nullptr;
        args.aq_scales       = w4a4 ? daq2_scales : nullptr;
        args.w4a4            = w4a4;
        args.act_row_is_col  = true;
        args.store_f32       = true;
        args.T               = T;
        args.Ktop            = static_cast<int>(K);
        args.C               = C;
        args.N               = K_in;
        args.K               = static_cast<int>(I);
        const int total = K_in * C;
        const int blocks = (total + kThreads - 1) / kThreads;
        moe_gemm_kernel<<<blocks, kThreads, 0, stream>>>(args, dids, dy);
    }

    // 7. Merge -> destination.
    {
        const int total = T * K_in;
        const int blocks = (total + kThreads - 1) / kThreads;
        moe_merge_kernel<<<blocks, kThreads, 0, stream>>>(
            dy, dalpha, dsg, dids, res, destp, T, static_cast<int>(K), K_in, add_residual);
    }

    // Diagnostic (VERIFY-gated, off by default): expose the device top-K ids and
    // router logits so a caller can compare them against its own host top-K / router.
    // A non-null out-param is the sole enable signal; production callers leave null.
    // dids/dlogits are written by kernels 1-2 and untouched by 3-7, so the drain
    // below (on the same stream) makes them stable for the D2H.
    if (w.diag_ids_out != nullptr || w.diag_logits_out != nullptr) {
        const cudaError_t se = cudaStreamSynchronize(stream);
        if (se != cudaSuccess) {
            throw std::runtime_error(std::string("sparse_moe_nvfp4 diag sync: ") +
                                     cudaGetErrorString(se));
        }
        if (w.diag_ids_out != nullptr) {
            const cudaError_t ce = cudaMemcpy(
                w.diag_ids_out, dids,
                sizeof(std::int32_t) * static_cast<std::size_t>(T) * K,
                cudaMemcpyDeviceToHost);
            if (ce != cudaSuccess) {
                throw std::runtime_error(std::string("sparse_moe_nvfp4 diag ids D2H: ") +
                                         cudaGetErrorString(ce));
            }
        }
        if (w.diag_logits_out != nullptr) {
            const cudaError_t ce = cudaMemcpy(
                w.diag_logits_out, dlogits,
                sizeof(float) * static_cast<std::size_t>(T) * E,
                cudaMemcpyDeviceToHost);
            if (ce != cudaSuccess) {
                throw std::runtime_error(std::string("sparse_moe_nvfp4 diag logits D2H: ") +
                                         cudaGetErrorString(ce));
            }
        }
    }
}

}  // namespace ninfer::ops
