#pragma once
// AVX-512 SIMD kernels for the CPU vision encoder (vision_cpu_impl.h). This translation unit is
// compiled with /arch:AVX512 (per-file COMPILE_FLAGS in CMakeLists.txt); callers dispatch at
// runtime via has_avx512f() and fall back to the scalar kernels when the CPU lacks AVX-512F,
// so the library stays correct on non-AVX-512 hosts.

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_6::detail::cpu_simd {

// cpuid leaf 7, EBX bit 16 (AVX512F).
bool has_avx512f();

// Elementwise 16-lane passes. Each applies ONE BF16 snap boundary per element (same RNE op as
// snap_bf16 in vision_cpu_impl.h) with the identical per-element formula as the scalar kernel
// it dispatches — the FP32 values may differ from the scalar path by a few ulps (libm tanh vs
// hardware exp2 in GELU, different summation order in LayerNorm), which the A/B ULP gate
// absorbs.
void snap_add_into(const float* y, std::size_t count, float* x);      // x = snap(x + y)
void snap_add_bias_into(const float* bias, int D, std::size_t count, float* x);  // x[t][d] = snap(x[t][d]+bias[d])
void gelu_tanh_into(float* x, std::size_t count);                     // x = GELU_tanh(x)
void layer_norm_into(const float* x, const float* wf, const float* bf, float inv_d, float eps,
                     int T, int D, float* out);
void rope_vision_into(const std::int32_t* pos, int T, int D, int H, int HD, int R2, float theta,
                      float* qkv);

// RowSplitK128 dequant into the [k*n] K-major buffer (same layout/numerics as dequant_weight in
// vision_cpu_impl.h): flat padded element e = n*padded + k, group g = e/group. The 16-row-block
// k-outer loop makes the [k*N + n] stores 16 contiguous floats each (the scalar kernel's one
// 4 B store per 64 B line is its main cost).
enum class DequantKind { Q4G64, Q5G64, Q6G64, W8G32 };
void dequant_weight_into(DequantKind kind, const std::uint8_t* codes, const std::uint8_t* high,
                         const std::uint16_t* scales, int n, int k, int group, int padded,
                         float* out);

// C[t][n] = sum_k W[k][n] * X[t][k], with X (T*K) and C (T*N) row-major fp32 and W (K*N)
// K-major fp32 (k fastest — dequant_weight emits this layout). Every C element is BF16-snapped
// (RNE, identical per-lane op to f32_to_bf16 in vision_cpu_impl.h) to match the GPU storage
// boundary. Requires N % 16 == 0 (a 512-bit fp32 register is 16 lanes wide; true for every
// vision N: 1152/1536/3456/4304/4608/5120). Parallelizes over 128x256 output blocks with one
// thread per physical core (16 on the 9950X3D).
void gemm_tn(const float* X, const float* W, float* C, int T, int N, int K);

// gemm_tn with the weight k-PAIR packed u32 (DequantCache::get32p in vision_cpu_impl.h):
// W[kp*N + n] = (bf16 of row 2kp+1 << 16) | bf16 of row 2kp, kp < (K+1)/2 — an odd K zero-pads
// the last k row. The 2-k step over a 32-column span is 128 B instead of 256 B (2 B per
// (k, column), half of gemm_tn): two 512-bit integer loads each read 16 u32 = 16 columns x 2 k
// rows, and `and 0xFFFF / << 16` + `and 0xFFFF0000` split the two k rows into fp32 with NO
// cross-lane merge. bf16 is the high half of fp32, so that split is a BIT-EXACT fp32 conversion
// — the only numeric difference vs gemm_tn is the bf16 snap of W itself (<= 1 bf16 ULP per
// weight element). The k-pair layout is deliberate, not just the packing: on this 9950X3D
// (Zen 5, microcode 0x00010000) the cross-lane merge ops (VPERM2PS with derived operands,
// VUNPCKL/H even with plain loads — and one VPERMPS allocation) corrupt the lane mapping
// (allocation-dependent erratum, .logs/prim_test.cpp), so the hot loop only runs
// loads + and/shift + FMA. Same layout, pair schedule, accumulation order (k step 2 keeps the
// exact per-k FMA sequence), epilogue and N % 16 == 0 requirement; X/C unchanged; every C
// element BF16-snapped as in gemm_tn.
void gemm_tn_bf16w(const float* X, const std::uint32_t* W, float* C, int T, int N, int K);

// Packed block-diagonal dense attention, AVX-512. Same layout and numerics as attention_into in
// vision_cpu_impl.h: q\k\v read from qkv [T][3D] (q at [0,D), k at [D,2D), v at [2D,3D); head h
// at [h*head_dim,(h+1)*head_dim) of each third; segments of segment_length rows are independent
// non-causal dense (T must be a multiple of segment_length); stable FP32 softmax; out [T][D]
// BF16-snapped per element. Each (segment, head) job re-stages its k block (transposed to
// [head_dim][segment_length]) and v block (packed [segment_length][head_dim]) into contiguous
// buffers: the packed qkv rows are 3*D floats apart, so reading them head-wise directly touches
// ~50 cache lines per 288 B of head data (the scalar kernel is L3-bandwidth-bound on exactly
// that traffic).
void attention_into(float scale, int segment_length, int T, int D, int heads, int head_dim,
                    const float* qkv, float* out);

}  // namespace ninfer::targets::qwen3_6::detail::cpu_simd
