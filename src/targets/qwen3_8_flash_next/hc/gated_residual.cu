// P5 v1 gated_residual (HyperConnection) kernel — contract in
// include/ninfer/ops/gated_residual.h. One fused kernel: one block (1024
// threads) per token; the stream is split into S/1024 elements per thread.
// Phases: low-rank down + GELU (threads 0..R-1) -> branch RMS partials +
// block reduce -> per-element gate (low-rank up + sigmoid) + write
// (sigmoid-gated inject) + deterministic branch mix (fixed four-term sum
// of shared gnv entries — no float atomics, so reruns are bit-equal).

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <ninfer/ops/gated_residual.h>

namespace ninfer::ops {

namespace {

constexpr int kGrB = 4;
constexpr int kGrThreads = 1024;
constexpr int kGrMaxLowrank = 320;
constexpr int kGrMaxHidden = 2560;

// The rms-partial and gate*normed phase regions are disjoint in time; they
// share one __shared__ union (max region = S = 10240 floats = 40 KB).
union GrSharedU {
  float rms_part[kGrB][kGrThreads];
  float gnv[10240];
};

__global__ void gr_kernel(const float* __restrict__ stream,
                          const float* __restrict__ bout,
                          const float* __restrict__ wdown,
                          const float* __restrict__ wup,
                          const float* __restrict__ winj,
                          const float* __restrict__ normw,
                          float* __restrict__ xout, float* __restrict__ sout,
                          int H, int S, int R, int ep, bool norm_branches) {
  // Phase regions are disjoint in time; the rms partials and the per-element
  // gate*normed values share one __shared__ union (max region = S floats).
  // The mix is a deterministic fixed-order sum (no float atomics ->
  // bit-equal reruns).
  __shared__ GrSharedU sh_u;
  __shared__ float sh_a[kGrMaxLowrank];
  __shared__ float sh_rms[kGrB];
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const float* st = stream + static_cast<std::size_t>(t) * S;

  if (tid < R) {
    float acc = 0.f;
    const float* w = wdown + static_cast<std::size_t>(tid) * S;
    for (int c = 0; c < S; ++c) acc = fmaf(w[c], st[c], acc);
    const float x = acc;
    sh_a[tid] = 0.5f * x * (1.f + erff(x * 0.70710678118654752440f));
  }
  __syncthreads();

  float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
  for (int k = 0; k < ep; ++k) {
    const int s = tid + k * kGrThreads;
    const float v = st[s];
    const int b = s / H;
    if (b == 0) p0 = fmaf(v, v, p0);
    else if (b == 1) p1 = fmaf(v, v, p1);
    else if (b == 2) p2 = fmaf(v, v, p2);
    else p3 = fmaf(v, v, p3);
  }
  sh_u.rms_part[0][tid] = p0;
  sh_u.rms_part[1][tid] = p1;
  sh_u.rms_part[2][tid] = p2;
  sh_u.rms_part[3][tid] = p3;
  __syncthreads();
  for (int off = kGrThreads / 2; off > 0; off >>= 1) {
    if (tid < off) {
      sh_u.rms_part[0][tid] += sh_u.rms_part[0][tid + off];
      sh_u.rms_part[1][tid] += sh_u.rms_part[1][tid + off];
      sh_u.rms_part[2][tid] += sh_u.rms_part[2][tid + off];
      sh_u.rms_part[3][tid] += sh_u.rms_part[3][tid + off];
    }
    __syncthreads();
  }
  if (tid < kGrB)
    sh_rms[tid] =
        norm_branches ? sqrtf(sh_u.rms_part[tid][0] / static_cast<float>(H)) : 1.0f;
  __syncthreads();

  for (int k = 0; k < ep; ++k) {
    const int s = tid + k * kGrThreads;
    float u = 0.f;
    const float* w = wup + static_cast<std::size_t>(s) * R;
    for (int r = 0; r < R; ++r) u = fmaf(w[r], sh_a[r], u);
    const float g = 1.0f / (1.0f + expf(-u));
    const int b = s / H;
    const int c = s - b * H;
    const float nv = norm_branches
                         ? st[s] * (1.0f / fmaxf(sh_rms[b], 1e-8f)) * normw[s]
                         : st[s];
    sh_u.gnv[s] = g * nv;
    const float gi = 1.0f / (1.0f + expf(-winj[static_cast<std::size_t>(b) * S + s]));
    sout[static_cast<std::size_t>(t) * S + s] =
        st[s] + gi * bout[static_cast<std::size_t>(t) * H + c];
  }
  __syncthreads();
  // H can exceed the 1024 thread count (prod H = 2560): stride over all
  // columns, fixed four-term order (bit-equal reruns).
  for (int c = tid; c < H; c += kGrThreads)
    xout[static_cast<std::size_t>(t) * H + c] =
        sh_u.gnv[c] + sh_u.gnv[c + H] + sh_u.gnv[c + 2 * H] +
        sh_u.gnv[c + 3 * H];
}

}  // namespace

[[noreturn]] void gr_fail(const char* what) {
  std::fprintf(stderr, "gated_residual: %s\n", what);
  std::abort();
}

void gr_launch(const GatedResidualParams& p, cudaStream_t stream) {
  const int S = p.hc_count * p.hidden;
  const int ep = S / kGrThreads;
  dim3 grid(p.tokens);
  dim3 block(kGrThreads);
  gr_kernel<<<grid, block, 0, stream>>>(
      p.stream_dev, p.block_out_dev, p.w_down_dev, p.w_up_dev, p.w_inj_dev,
      p.norm_w_dev, p.read_out_dev, p.stream_out_dev, p.hidden, S, p.lowrank,
      ep, p.norm_branches);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) gr_fail(cudaGetErrorString(err));
}

void gated_residual(const GatedResidualParams& p, cudaStream_t stream) {
  const int S = p.hc_count * p.hidden;
  if (p.tokens <= 0) gr_fail("tokens must be >= 1");
  if (p.hc_count != kGrB) gr_fail("v1 supports hc_count == 4 only");
  if (p.hidden <= 0 || p.hidden > kGrMaxHidden) gr_fail("hidden out of v1 range");
  if (S % kGrThreads != 0 || S > 10240) gr_fail("stream width out of v1 range");
  if (p.lowrank <= 0 || p.lowrank > kGrMaxLowrank)
    gr_fail("lowrank out of v1 range");
  if (!p.stream_dev || !p.block_out_dev || !p.w_down_dev || !p.w_up_dev ||
      !p.w_inj_dev || !p.read_out_dev || !p.stream_out_dev)
    gr_fail("missing buffer");
  if (p.norm_branches && !p.norm_w_dev) gr_fail("missing norm_w");
  if (p.stream_out_dev == p.stream_dev)
    gr_fail("stream_out must be distinct from stream (per-element in-place "
            "mixing is not defined)");
  gr_launch(p, stream);
}

}  // namespace ninfer::ops
