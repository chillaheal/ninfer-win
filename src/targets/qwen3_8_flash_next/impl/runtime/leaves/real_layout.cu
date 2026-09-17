// Flash-Next P12 S2 — real layout / hyper-connection / GDN-pack glue leaves
// (target-private).
//
// Real-geometry generalizations of the P9 mini stream_init / final_collapse /
// block_out / ple_add / gdn_pack / gdn_zslice leaves: the hidden width H, the
// HC branch count B, the PLE embed dim E, and the GDN pack dims (D, S, Hqk,
// Hv) and z-slice extents (ROW, ZC) are all runtime parameters. Every store is
// single-thread-owned (no atomics); the GDN pack zero-fills the [D, S) pad
// lanes (a no-op at the real D = S = 128).

#include "targets/qwen3_8_flash_next/impl/runtime/leaves/real_leaves.h"

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace {

// stream[t, b*H + c] = f32(h[t, c]) for every branch b in [0, B).
__global__ void real_stream_init_kernel(const __nv_bfloat16* __restrict__ h, int T, int H,
                                        int B, float* __restrict__ stream) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * B * H) {
    return;
  }
  const int b = idx / H;
  const int c = idx - b * H;
  const int t = b / B;
  stream[idx] = __bfloat162float(h[static_cast<std::size_t>(t) * H + c]);
}

// h_final[t, c] = bf16( (1/B) * Sum_b stream[t, b*H + c] ).
__global__ void real_final_collapse_kernel(const float* __restrict__ stream, int T, int H,
                                           int B, __nv_bfloat16* __restrict__ h_final) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * H) {
    return;
  }
  const int c = idx % H;
  const int t = idx / H;
  const float* row = stream + static_cast<std::size_t>(t) * B * H + c;
  float acc = 0.0F;
  for (int b = 0; b < B; ++b) {
    acc += row[b * H];
  }
  h_final[idx] = __float2bfloat16(acc * (1.0F / static_cast<float>(B)));
}

// block_out[t, c] = f32(y[t, c]).
__global__ void real_block_out_kernel(const __nv_bfloat16* __restrict__ y, int T, int H,
                                      float* __restrict__ block_out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * H) {
    return;
  }
  block_out[idx] = __bfloat162float(y[idx]);
}

// x[t, c] = bf16( f32(x[t, c]) + mem[t, c % E] ) in place.
__global__ void real_ple_add_kernel(__nv_bfloat16* __restrict__ x,
                                    const float* __restrict__ mem, int T, int H, int E) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * H) {
    return;
  }
  const int c = idx % H;
  const int t = idx / H;
  const float m = mem[static_cast<std::size_t>(t) * E + (c % E)];
  x[idx] = __float2bfloat16(__bfloat162float(x[idx]) + m);
}

// One pack region: op[d-fastest (t*Hh+h)*S + d] = (d < D) ?
// conv_out[(chan_off + h*D + d) * T + t] : 0. Launched once per op (q/k/v)
// with the matching Hh and chan_off.
__global__ void real_gdn_pack_kernel(const __nv_bfloat16* __restrict__ conv_out, int T, int D,
                                     int S, int Hh, int chan_off,
                                     __nv_bfloat16* __restrict__ op) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Hh * S * T) {
    return;
  }
  const int d = idx % S;
  const int rem = idx / S;
  const int h = rem % Hh;
  const int t = rem / Hh;
  if (d < D) {
    const std::size_t chan = static_cast<std::size_t>(chan_off + h * D + d);
    op[idx] = conv_out[chan * T + t];
  } else {
    op[idx] = __float2bfloat16(0.0F);
  }
}

// z[t, d] = qkvz[t, ROW - ZC + d].
__global__ void real_gdn_zslice_kernel(const __nv_bfloat16* __restrict__ qkvz, int T, int ROW,
                                       int ZC, __nv_bfloat16* __restrict__ z) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * ZC) {
    return;
  }
  const int d = idx % ZC;
  const int t = idx / ZC;
  z[idx] = qkvz[static_cast<std::size_t>(t) * ROW + (ROW - ZC) + d];
}

}  // namespace

void real_stream_init(const __nv_bfloat16* h, int T, int H, int B, float* stream,
                      cudaStream_t stream_) {
  const int total = T * B * H;
  const int blocks = (total + 255) / 256;
  real_stream_init_kernel<<<blocks, 256, 0, stream_>>>(h, T, H, B, stream);
}

void real_final_collapse(const float* stream, int T, int H, int B, __nv_bfloat16* h_final,
                         cudaStream_t stream_) {
  const int total = T * H;
  const int blocks = (total + 255) / 256;
  real_final_collapse_kernel<<<blocks, 256, 0, stream_>>>(stream, T, H, B, h_final);
}

void real_block_out(const __nv_bfloat16* y, int T, int H, float* block_out,
                    cudaStream_t stream_) {
  const int total = T * H;
  const int blocks = (total + 255) / 256;
  real_block_out_kernel<<<blocks, 256, 0, stream_>>>(y, T, H, block_out);
}

void real_ple_add(__nv_bfloat16* x, const float* mem, int T, int H, int E,
                  cudaStream_t stream_) {
  const int total = T * H;
  const int blocks = (total + 255) / 256;
  real_ple_add_kernel<<<blocks, 256, 0, stream_>>>(x, mem, T, H, E);
}

void real_gdn_pack(const __nv_bfloat16* conv_out, int T, int D, int S, int Hqk, int Hv,
                   __nv_bfloat16* q_op, __nv_bfloat16* k_op, __nv_bfloat16* v_op,
                   cudaStream_t stream) {
  // q: Hh=Hqk, chan_off=0. k: Hh=Hqk, chan_off=Hqk*D. v: Hh=Hv,
  // chan_off=2*Hqk*D.
  {
    const int total = Hqk * S * T;
    const int blocks = (total + 255) / 256;
    real_gdn_pack_kernel<<<blocks, 256, 0, stream>>>(conv_out, T, D, S, Hqk, 0, q_op);
  }
  {
    const int total = Hqk * S * T;
    const int blocks = (total + 255) / 256;
    real_gdn_pack_kernel<<<blocks, 256, 0, stream>>>(conv_out, T, D, S, Hqk, Hqk * D, k_op);
  }
  {
    const int total = Hv * S * T;
    const int blocks = (total + 255) / 256;
    real_gdn_pack_kernel<<<blocks, 256, 0, stream>>>(conv_out, T, D, S, Hv, 2 * Hqk * D, v_op);
  }
}

void real_gdn_zslice(const __nv_bfloat16* qkvz, int T, int ROW, int ZC, __nv_bfloat16* z,
                     cudaStream_t stream) {
  const int total = T * ZC;
  const int blocks = (total + 255) / 256;
  real_gdn_zslice_kernel<<<blocks, 256, 0, stream>>>(qkvz, T, ROW, ZC, z);
}

}  // namespace ninfer::targets::qwen3_8_flash_next::detail
