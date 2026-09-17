// Flash-Next P9 — target-private mini leaf declarations.
//
// All buffers are device pointers, laid out exactly as the P9 design contract
// pins them (see out/flash_next_dev/P9_design.md). Every leaf is bit-
// deterministic: fixed accumulation order, no atomics, no data-dependent
// scheduling. The FP8 row-scale convention (code plane [N*K] u8, scale plane
// [N] BF16 at align_up(N*K, 256)) is the artifact's RowScaleV1 layout.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// ---------------------------------------------------------------------------
// linear + embedding
// ---------------------------------------------------------------------------

// Embedding gather: h[t, c] = decode(code[id[t] * 256 + c]) * bf16(scale[id[t]]).
// `code` [256][256] u8, `scale` [256] BF16, `h` [T][256] BF16 token-major.
void mini_embedding_gather(const std::int32_t* ids, const std::uint8_t* code,
                           const std::uint16_t* scale, int T, __nv_bfloat16* h,
                           cudaStream_t stream);

// FP8 row-scale linear: out[t, n] = bf16( Sum_k bf16(x[t,k]) *
// e4m3(code[n*K + k]) * bf16(scale[n]) ), FP32 accumulation, k ascending.
// x [T][K] token-major BF16; out [T][N] token-major BF16.
void mini_fp8_linear(const __nv_bfloat16* x, int T, int K, const std::uint8_t* code,
                     const std::uint16_t* scale, int N, __nv_bfloat16* out,
                     cudaStream_t stream);

// BF16 router pre-pass (MoE decode staging): logits[t, e] =
// Sum_k bf16(router[e, k]) * bf16(x[t, k]), FP32 accumulation, k ascending.
// x [T][256] token-major; router [8][256] BF16; out [T][8] FP32.
void mini_router(const __nv_bfloat16* x, int T, const __nv_bfloat16* router,
                 float* logits, cudaStream_t stream);

// ---------------------------------------------------------------------------
// layout / hyper-connection glue
// ---------------------------------------------------------------------------

// out[k, t] = x[t, k] (BF16, token-major -> channel-major).
void mini_transpose_tm_cm(const __nv_bfloat16* x, int T, int K, __nv_bfloat16* out,
                          cudaStream_t stream);

// out[t, k] = x[k, t] (BF16, channel-major -> token-major).
void mini_transpose_cm_tm(const __nv_bfloat16* x, int T, int K, __nv_bfloat16* out,
                          cudaStream_t stream);

// stream[t, b*256 + c] = f32(h[t, c]) for every branch b in [0,4) (v1
// reduction: the embedding is replicated into all 4 branches).
void mini_stream_init(const __nv_bfloat16* h, int T, float* stream, cudaStream_t stream_);

// h_final[t, c] = bf16( mean_{b in [0,4)} stream[t, b*256 + c] ).
void mini_final_collapse(const float* stream, int T, __nv_bfloat16* h_final,
                         cudaStream_t stream_);

// block_out[t, c] = f32(y[t, c])  ->  [T, 256] FP32, row stride 256 (the
// gated_residual write call consumes block_out[t, s%H] with H=256).
void mini_block_out(const __nv_bfloat16* y, int T, float* block_out, cudaStream_t stream_);

// out[t, c] = bf16(x[t, c])  ->  [T, H] BF16 from [T, H] FP32 (converts a
// gated_residual read_out [T,256] FP32 into the BF16 x the block leaves eat).
void mini_f32_to_bf16(const float* x, int T, int H, __nv_bfloat16* out,
                      cudaStream_t stream);

// x[t, c] = bf16( f32(x[t, c]) + mem[t, c % 128] ) in place (PLE add, layer 1).
void mini_ple_add(__nv_bfloat16* x, const float* mem, int T, cudaStream_t stream_);

// ---------------------------------------------------------------------------
// GDN block
// ---------------------------------------------------------------------------

// Gating: for h in [0,4): a_h[t] = Sum_k bf16(a_weight[h,k]) * bf16(x[t,k]);
// b_h[t] likewise; g[h,t] = -expf(A_log[h]) * softplus(a_h[t] + dt_bias[h]);
// beta[h,t] = sigmoid(b_h[t]). x [T][256] token-major; g/beta [4][T] FP32.
void mini_gdn_gating(const __nv_bfloat16* x, int T, const __nv_bfloat16* a_weight,
                     const __nv_bfloat16* b_weight, const float* a_log,
                     const float* dt_bias, float* g, float* beta, cudaStream_t stream);

// Zero-pad pack (GDN): q_op[d, h, t] = conv_out[32*h + d, t] for d in [0,32),
// h in [0,2); k_op[d, h, t] = conv_out[64 + 32*h + d, t]; v_op[d, h, t] =
// conv_out[128 + 32*h + d, t] for h in [0,4); every other element of the
// [128, H, T] d-fastest buffers is zero. `conv_out` is channel-major [256, T].
void mini_gdn_pack(const __nv_bfloat16* conv_out, int T, __nv_bfloat16* q_op,
                   __nv_bfloat16* k_op, __nv_bfloat16* v_op, cudaStream_t stream);

// GDN z slice: z[t, d] = qkvz[t, 256 + d] — gathers the last 128 channels of
// each [T][384] qkvz row into a contiguous [T][128] token-major buffer.
void mini_gdn_zslice(const __nv_bfloat16* qkvz, int T, __nv_bfloat16* z,
                     cudaStream_t stream);

// Fused target-private gated RMSNorm (the design's step 6): one thread per
// (t, h). x = o[d, h, t] (d in [0,32), o [128,4,T] d-fastest);
// onx[t, h*32 + d] = bf16( x * inv_r * bf16w[d] * silu(f32(z[t, h*32 + d])) ),
// inv_r = 1 / sqrt( (1/32) * Sum_d x^2 + 1e-6 ). All FP32 internally.
void mini_gated_rmsnorm(const __nv_bfloat16* o, const __nv_bfloat16* z,
                        const __nv_bfloat16* w, int T, __nv_bfloat16* onx,
                        cudaStream_t stream);

// ---------------------------------------------------------------------------
// QSA block
// ---------------------------------------------------------------------------

// Per-head RMSNorm (P7 convention): x[t, h, d] = bf16( f32(x) * g[d] /
// sqrt( (1/32) * Sum_d f32(x)^2 + 1e-6 ) ), in place over [T][H][32]
// token-major. `g` [32] BF16.
void mini_head_rmsnorm(__nv_bfloat16* x, int T, int H, const __nv_bfloat16* g,
                       cudaStream_t stream);

// v1 RoPE (D=32, R=8, theta = 10000.0): for m in [0,4), angle =
// (pos0 + t) * 10000^(-2m/8); q'[m] = q[m]*cos - q[m+4]*sin;
// q'[m+4] = q[m]*sin + q[m+4]*cos; dims [8,32) pass through. Applied in
// place to q [T][Hq][32] and k [T][Hk][32] (token-major, positions absolute
// from the sequence frontier).
void mini_rope(__nv_bfloat16* q, int T, int Hq, int pos0, __nv_bfloat16* k, int Hk,
               cudaStream_t stream);

// Paged KV append (mini, D=32, KV=2): the lane's 16,384 B page holds the K
// plane (first 8,192 B) and the V plane (second 8,192 B); each plane is
// addressed by paged_kv_element_offset<32,2> on its own base (P7 semantics).
// k/v [T][2][32] token-major; token t is written at absolute position
// pos + t. `block_table` maps page index -> physical 16,384 B page.
void mini_kv_append(const __nv_bfloat16* k, const __nv_bfloat16* v, int T, int pos,
                    const std::int32_t* block_table, __nv_bfloat16* kv,
                    cudaStream_t stream);

// Mini indexer (round-local, 32-dim chunks): z[t, c] = Sum_k bf16(qk_proj[c,k])
// * bf16(x[t,k]) in FP32; q = rms32(z[t, 0:32], q_norm); the 4 k chunks are
// rms32(z[t, 32 + 32c : 64 + 32c], k_norm) (k_norm shared);
// score[t, i] = i <= t ? (1/sqrt(32)) * Sum_c Sum_d qnorm[t,d] *
// knorm_c[i,d] (c = 0..3 — the sum spans ALL four key chunks, the P7
// contract) : -inf; ids[t, :] = the min(counts[t], B=64) positions with
// the HIGHEST scores (score DESC, position DESC on ties), stored ascending;
// counts[t] = min(B, t + 1). Under the P9 cap (B >= T) the dense shortcut
// ids = 0..t is active; the topk path exists with the same contract.
void mini_indexer(const __nv_bfloat16* x, int T, const std::uint16_t* qk_proj,
                  const std::uint16_t* q_norm, const std::uint16_t* k_norm,
                  float* z, float* scores, std::int32_t* ids, std::int32_t* counts,
                  cudaStream_t stream);

// Mini sparse GQA (D=32, Q=4, KV=2, g = h/2, scale 1/sqrt(32)):
// s_j = scale * Sum_d q[t,h,d] * K[p_j, g, d] over the selected positions;
// online softmax; out[t, h, d] = bf16( (1/z) * Sum_j u_j * V[p_j, g, d] ).
// ids [T][64] ascending, counts [T]; page reads use the mini KV layout (K
// plane first 8,192 B, V plane second 8,192 B of each 16,384 B page).
void mini_sparse_gqa(const __nv_bfloat16* q, int T, const std::int32_t* ids,
                     const std::int32_t* counts, const std::int32_t* block_table,
                     const __nv_bfloat16* k_pages, const __nv_bfloat16* v_pages,
                     __nv_bfloat16* out, cudaStream_t stream);

}  // namespace ninfer::targets::qwen3_8_flash_next::detail
