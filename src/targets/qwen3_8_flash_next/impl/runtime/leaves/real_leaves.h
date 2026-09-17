// Flash-Next P12 S2 — target-private REAL glue leaf declarations.
//
// Real-geometry counterparts of the P9 mini leaves (mini_leaves.h) that do
// not map to a reusable generic op. Every leaf takes its dims as runtime
// parameters (no mini constants baked in); the only fixed geometry is the
// real GDN/QSA invariants that the P6/P7 op contracts pin (GDN state dim
// 128, QSA head dim 256 / rotary 64, paged-KV plane [256,64,KV,P]). Buffers
// are device pointers, laid out exactly as the real op contracts pin them.
//
// FP8 row-scale convention (RowScaleV1): code plane [N*K] u8, scale plane
// [N] BF16 at align_up(N*K, 256). The generic leaves (fp8_linear, transposes,
// f32_to_bf16) are REUSED unchanged; only the glue below is re-expressed at
// real geometry. Every leaf is bit-deterministic: fixed accumulation order,
// no atomics, no data-dependent scheduling.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// ---------------------------------------------------------------------------
// linear + embedding
// ---------------------------------------------------------------------------

// Embedding gather: h[t, c] = bf16( e4m3(code[id[t]*K + c]) * bf16(scale[id[t]]) ).
// `code` [V][K] u8, `scale` [V] BF16, `h` [T][K] BF16 token-major. K is the
// hidden width (2560 real).
void real_embedding_gather(const std::int32_t* ids, const std::uint8_t* code,
                           const std::uint16_t* scale, int T, int K, __nv_bfloat16* h,
                           cudaStream_t stream);

// BF16 router pre-pass (MoE decode staging): logits[t, e] = Sum_k
// bf16(router[e*K + k]) * bf16(x[t, k]), FP32 accumulation, k ascending.
// x [T][K] token-major; router [E][K] BF16; out [T][E] FP32. Real: K=2560,
// E=512.
void real_router(const __nv_bfloat16* x, int T, int K, int E, const __nv_bfloat16* router,
                 float* logits, cudaStream_t stream);

// ---------------------------------------------------------------------------
// layout / hyper-connection glue
// ---------------------------------------------------------------------------

// stream[t, b*H + c] = f32(h[t, c]) for every branch b in [0, B) (the
// embedding is replicated into all B branches). H hidden width, B HC count.
void real_stream_init(const __nv_bfloat16* h, int T, int H, int B, float* stream,
                      cudaStream_t stream_);

// h_final[t, c] = bf16( mean_{b in [0,B)} stream[t, b*H + c] ).
void real_final_collapse(const float* stream, int T, int H, int B, __nv_bfloat16* h_final,
                         cudaStream_t stream_);

// block_out[t, c] = f32(y[t, c])  ->  [T, H] FP32 (the gated_residual write
// call consumes block_out[t, s%H]).
void real_block_out(const __nv_bfloat16* y, int T, int H, float* block_out,
                    cudaStream_t stream_);

// x[t, c] = bf16( f32(x[t, c]) + mem[t, c%E] ) in place (PLE add, layer 1).
// x [T][H] BF16; mem [T][E] FP32 (the NgramEmbedding gather row, E the PLE
// embed dim; E<=H, c wraps mod E).
void real_ple_add(__nv_bfloat16* x, const float* mem, int T, int H, int E,
                  cudaStream_t stream_);

// ---------------------------------------------------------------------------
// GDN block
// ---------------------------------------------------------------------------

// Gating: for h in [0,H): a_h[t] = Sum_k bf16(a_weight[h,k]) * bf16(x[t,k]);
// b_h[t] likewise; g[h,t] = -expf(A_log[h]) * softplus(a_h[t] + dt_bias[h]);
// beta[h,t] = sigmoid(b_h[t]). x [T][K] token-major; a/b_weight [H][K] BF16
// (a_weight rows [0,H), b_weight rows [H,2H) of the fused a_b_projection);
// g/beta [H][T] FP32. Real: K=2560, H=48 (value heads).
void real_gdn_gating(const __nv_bfloat16* x, int T, int K, int H,
                     const __nv_bfloat16* a_weight, const __nv_bfloat16* b_weight,
                     const float* a_log, const float* dt_bias, float* g, float* beta,
                     cudaStream_t stream);

// Zero-pad pack (GDN). The op buffers are [S,H,T] d-fastest (memory
// (t*Hh+h)*S + d); the conv projection is channel-major [C,T] (chan*T + t),
// laid out [ q : Hqk*D | k : Hqk*D | v : Hv*D ]. For h in [0,Hqk):
//   q_op[d,h,t] = conv_out[(h*D + d, t)]         (d in [0,D))
//   k_op[d,h,t] = conv_out[((Hqk+h)*D + d, t)]   (d in [0,D))
// for h in [0,Hv):
//   v_op[d,h,t] = conv_out[((2*Hqk+h)*D + d, t)] (d in [0,D))
// every other element (d in [D,S)) is zero. Real: D=S=128 (no pad), Hqk=16,
// Hv=48, C=10240.
void real_gdn_pack(const __nv_bfloat16* conv_out, int T, int D, int S, int Hqk, int Hv,
                   __nv_bfloat16* q_op, __nv_bfloat16* k_op, __nv_bfloat16* v_op,
                   cudaStream_t stream);

// GDN z slice: z[t, d] = qkvz[t, ROW - ZC + d] for d in [0, ZC) — gathers the
// last ZC channels of each [T][ROW] qkvz row into a contiguous [T][ZC]
// token-major buffer. Real: ROW=16384, ZC=6144 (=Hv*D).
void real_gdn_zslice(const __nv_bfloat16* qkvz, int T, int ROW, int ZC, __nv_bfloat16* z,
                     cudaStream_t stream);

// Fused target-private gated RMSNorm: one thread per (t, h). x = o[(t*H+h)*S
// + d] (d in [0,D), o the GDN out [S,H,T] d-fastest); onx[t, h*D + d] =
// bf16( x * inv_r * bf16(w[d]) * silu(f32(z[t, h*D + d])) ), inv_r =
// 1 / sqrt( (1/D) * Sum_d x^2 + 1e-6 ). All FP32 internally. Real: H=48,
// D=S=128.
void real_gated_rmsnorm(const __nv_bfloat16* o, const __nv_bfloat16* z,
                        const __nv_bfloat16* w, int T, int H, int D, int S,
                        __nv_bfloat16* onx, cudaStream_t stream);

// ---------------------------------------------------------------------------
// QSA block
// ---------------------------------------------------------------------------

// Per-head RMSNorm (P7 convention): x[t, h, d] = bf16( f32(x) * g[d] /
// sqrt( (1/D) * Sum_d f32(x)^2 + 1e-6 ) ), in place over [T][H][D]
// token-major. `g` [D] BF16. Real: D=256 (the QSA head dim; read-twice,
// write-once in place — safe because each d is read before its own store and
// no other d reads it).
void real_head_rmsnorm(__nv_bfloat16* x, int T, int H, int D, const __nv_bfloat16* g,
                       cudaStream_t stream);

// RoPE (split-half NeoX, D=256, R=64, theta=10000): for i in [0, R/2),
// angle = (pos0 + t) * 10000^(-2i/R); x[i] = x[i]*cos - x[i+R/2]*sin;
// x[i+R/2] = x[i+R/2]*cos + x[i]*sin; dims [R, D) pass through. Applied in
// place to q [T][Hq][D] and k [T][Hk][D] (token-major, positions absolute
// from the sequence frontier). Matches the shared rope op's D256/R64 Text
// domain.
void real_rope(__nv_bfloat16* q, int T, int Hq, int pos0, __nv_bfloat16* k, int Hk,
               cudaStream_t stream);

// Paged KV append (real, D=256): K and V live in SEPARATE page pools
// (k_pages / v_pages), each a page-major [256, 64, KV, P] BF16 plane
// addressed by the paged_kv_element_offset<256,KV> formula with the head
// extent at runtime (P7 separate-plane semantics). k/v [T][KV][D] token-
// major; token t is written at absolute position pos + t. `block_table` maps
// page index -> physical page.
void real_kv_append(const __nv_bfloat16* k, const __nv_bfloat16* v, int T, int pos, int KV,
                    const std::int32_t* block_table, __nv_bfloat16* k_pages,
                    __nv_bfloat16* v_pages, cudaStream_t stream);

// QSA output-side head reduction 48 -> 24 (S4-confirmed v1: SwiGLU split-half
// gate). The QSA block produces Q=48 query-head rows [T][Q][D] (the
// qsa_sparse_gqa out) but the layer's output projection takes 24*D=6144
// channels. reduced[t, h*D + d] = bf16( f32(qsa[t, h*D + d]) *
// silu( f32(qsa[t, (h + Q/2)*D + d]) ) ) for h in [0, Q/2), d in [0, D).
// qsa [T][Q][D] token-major; reduced [T][Q/2][D] token-major. Q must be even
// (real: Q=48, D=256 -> reduced [T][24][256] = [T][6144]).
void real_qsa_gate(const __nv_bfloat16* qsa, int T, int Q, int D, __nv_bfloat16* reduced,
                   cudaStream_t stream);

}  // namespace ninfer::targets::qwen3_8_flash_next::detail
