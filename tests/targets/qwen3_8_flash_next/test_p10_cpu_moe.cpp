// P10 gate — CPU MoE forward (W4A16 NVFP4, P3.2) vs the GPU sparse_moe_nvfp4
// op on the SAME device router logits.
//
// Contract: CpuMoe::run consumes the D2H'd device logits (diag_logits_out), so
// its top-k runs on bit-identical inputs to the device's. Then:
//   - ids  : EXACT vs the device ids (diag_ids_out) — same logits, exact
//            top-k replica;
//   - alpha: pointwise {5e-5, 1e-3} (host std::expf vs device __expf);
//   - y    : reduction {1.5e-2, 1e-4, 1.5e-2} vs the GPU op output (the same
//            gates p8 uses for the op vs its FP64 oracle).
// The weight planes are the p8 fixture's raw NVFP4 planes (same splitmix64
// seed); the routed planes point into the p8 expert window blob with the
// op's per-expert stride, so CpuMoe dequantizes exactly what the device GEMM
// dequantizes.

#include <ninfer/ops/sparse_moe_nvfp4.h>

#include "ops/op_tester.h"
#include "targets/qwen3_8_flash_next/impl/runtime/cpu_moe.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;
using ninfer::targets::qwen3_8_flash_next::CpuMoe;
using ninfer::targets::qwen3_8_flash_next::CpuMoePlane;
using ninfer::targets::qwen3_8_flash_next::CpuMoeWeights;

namespace {

int failures = 0;

// --- deterministic RNG (p8 fixture, same seed) ------------------------------
std::uint64_t s_rng = 0x85a308d313198a2eULL;
std::uint64_t splitmix64(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
float uniform_f32(float lo, float hi) {
  const double u = static_cast<double>(splitmix64(s_rng) >> 11) / static_cast<double>(1ULL << 53);
  return static_cast<float>(lo + u * (hi - lo));
}

// --- NVFP4 host codecs (match nvfp4_codec.cuh / moe_nvfp4.cu exactly) --------
constexpr float kE2m1[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};

float e2m1_decode(std::uint8_t nib) {
  const float m = kE2m1[nib & 7u];
  return (nib & 8u) ? -m : m;
}

float e4m3_decode(std::uint8_t b) {
  const float s = (b & 0x80u) ? -1.0F : 1.0F;
  const int e   = (b >> 3) & 0xF;
  const int m   = b & 0x7;
  const float v = (e == 0) ? (static_cast<float>(m) * 0.015625F)
                           : ((1.0F + 0.125F * static_cast<float>(m)) * std::ldexp(1.0F, e - 7));
  return s * v;
}

// cvt.rn.satfinite.e4m3 of a positive finite f: grid-nearest, ties to the even code.
std::uint8_t e4m3_encode(float f) {
  if (!std::isfinite(f) || f <= 0.0F) return 0;
  if (f >= 448.0F) return 0x7Eu;
  std::uint8_t best = 0;
  float bd          = std::numeric_limits<float>::infinity();
  for (int c = 0; c <= 126; ++c) {
    const float d = std::fabs(e4m3_decode(static_cast<std::uint8_t>(c)) - f);
    if (d < bd || (d == bd && (best & 1) && !(c & 1))) {
      bd   = d;
      best = static_cast<std::uint8_t>(c);
    }
  }
  return best;
}

std::uint8_t e2m1_encode(float x) {
  const bool neg = (x < 0.0F);
  const float a  = std::fabs(x);
  int best = 0;
  float bd = std::numeric_limits<float>::infinity();
  for (int c = 0; c < 8; ++c) {
    const float d = std::fabs(kE2m1[c] - a);
    if (d < bd || (d == bd && (best & 1) && !(c & 1))) {
      bd   = d;
      best = c;
    }
  }
  return static_cast<std::uint8_t>((best & 7) | (neg ? 8u : 0u));
}

// One K16 group -> the 16 DEQUANTIZED floats (code * e4m3 scale).
void quantize_k16_dequant(const float* src, float divisor, float* out16) {
  float max_abs = 0.0F;
  for (int i = 0; i < 16; ++i) max_abs = fmaxf(max_abs, fabsf(src[i]));
  const std::uint8_t sb = e4m3_encode(divisor * max_abs / 6.0F);
  if (sb == 0) {
    for (int i = 0; i < 16; ++i) out16[i] = 0.0F;
    return;
  }
  const float dec = e4m3_decode(sb);
  for (int i = 0; i < 16; ++i) {
    const float q   = src[i] * divisor / dec;
    out16[i]        = e2m1_decode(e2m1_encode(q)) * dec;  // code * scale
  }
}

// --- canonical plane writer (natural codes, row-tile-relative swizzled scales)
struct Plane {
  std::vector<std::uint8_t> codes;    // N*(K/2)
  std::vector<std::uint8_t> scales;   // N*K/16
};

Plane write_plane(const std::vector<float>& f, int N, int K) {
  Plane p;
  p.codes.assign(static_cast<std::size_t>(N) * (K / 2), 0);
  p.scales.assign(static_cast<std::size_t>(N) * (K / 16), 0);
  const int gpr = K / 16;
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < gpr; ++g) {
      float out16[16];
      quantize_k16_dequant(f.data() + static_cast<std::size_t>(n) * K + g * 16, 1.0F, out16);
      float max_abs = 0.0F;
      for (int s = 0; s < 16; ++s)
        max_abs = fmaxf(max_abs, fabsf(f[static_cast<std::size_t>(n) * K + g * 16 + s]));
      const std::uint8_t sb  = e4m3_encode(max_abs / 6.0F);
      const float dec        = e4m3_decode(sb);
      std::uint8_t* cd       = p.codes.data() + (static_cast<std::size_t>(n) * K + g * 16) / 2;
      if (sb == 0) {
        for (int s = 0; s < 8; ++s) cd[s] = 0;
      } else {
        for (int s = 0; s < 16; ++s) {
          const std::uint8_t nib =
              e2m1_encode(f[static_cast<std::size_t>(n) * K + g * 16 + s] / dec);
          cd[s >> 1] = (s & 1) ? static_cast<std::uint8_t>((cd[s >> 1] & 0x0Fu) | (nib << 4))
                               : static_cast<std::uint8_t>((cd[s >> 1] & 0xF0u) | nib);
        }
      }
      const int slab = ((n / 128) * (K / 64) + (g / 4)) * 512;
      const int off  = ((n & 127) & 31) * 16 + ((n & 127) >> 5) * 4 + (g % 4);
      p.scales[static_cast<std::size_t>(slab) + off] = sb;
    }
  }
  return p;
}

// One expert blob from FP32 source matrices gu [2I][K_in], dn [K_in][I], laid out
// [gu_codes][gu_scales][dn_codes][dn_scales] (the canonical per-expert slice).
std::vector<std::uint8_t> build_expert_blob(const std::vector<float>& gu, int I, int K_in,
                                            const std::vector<float>& dn) {
  const Plane g = write_plane(gu, 2 * I, K_in);
  const Plane d = write_plane(dn, K_in, I);
  std::vector<std::uint8_t> blob(g.codes.size() + g.scales.size() + d.codes.size() +
                                 d.scales.size());
  std::size_t o = 0;
  auto cp = [&](const std::vector<std::uint8_t>& v) {
    std::copy(v.begin(), v.end(), blob.begin() + o);
    o += v.size();
  };
  cp(g.codes);
  cp(g.scales);
  cp(d.codes);
  cp(d.scales);
  return blob;
}

// --- weight generation (p8 fixture) -----------------------------------------
struct Weights {
  int E = 0, K = 0, K_in = 0, I = 0;
  std::vector<float> x, router, sgate;  // BF16-exact (rounded)
  Plane sguc, sdnc;                     // shared gu / dn (codes+scales)
  std::vector<std::uint8_t> ewin;       // routed window [E][expert_bytes]
};

Weights make_weights(int E, int K, int K_in, int I, int T) {
  Weights w;
  w.E = E;
  w.K = K;
  w.K_in = K_in;
  w.I = I;

  w.x.assign(static_cast<std::size_t>(T) * K_in, 0.0F);
  for (auto& v : w.x) v = uniform_f32(-1.0F, 1.0F);
  round_to_bf16(w.x);
  w.router.assign(static_cast<std::size_t>(E) * K_in, 0.0F);
  for (auto& v : w.router) v = uniform_f32(-0.6F, 0.6F);
  round_to_bf16(w.router);
  w.sgate.assign(K_in, 0.0F);
  for (auto& v : w.sgate) v = uniform_f32(-0.6F, 0.6F);
  round_to_bf16(w.sgate);

  std::vector<float> sgu(static_cast<std::size_t>(2 * I) * K_in);
  for (auto& v : sgu) v = uniform_f32(-2.0F, 2.0F);
  w.sguc = write_plane(sgu, 2 * I, K_in);
  std::vector<float> sd(static_cast<std::size_t>(K_in) * I);
  for (auto& v : sd) v = uniform_f32(-2.0F, 2.0F);
  w.sdnc = write_plane(sd, K_in, I);

  const auto geo = SparseMoeNvfp4Geometry{static_cast<std::uint32_t>(E), static_cast<std::uint32_t>(K),
                                          static_cast<std::uint32_t>(K_in), static_cast<std::uint32_t>(I)};
  const std::size_t eb = sparse_moe_nvfp4_expert_bytes(geo);
  w.ewin.assign(static_cast<std::size_t>(E) * eb, 0);
  for (int e = 0; e < E; ++e) {
    std::vector<float> gu_src(static_cast<std::size_t>(2 * I) * K_in);
    for (auto& v : gu_src) v = uniform_f32(-2.0F, 2.0F);
    std::vector<float> dn_src(static_cast<std::size_t>(K_in) * I);
    for (auto& v : dn_src) v = uniform_f32(-2.0F, 2.0F);
    const std::vector<std::uint8_t> blob = build_expert_blob(gu_src, I, K_in, dn_src);
    std::copy(blob.begin(), blob.end(), w.ewin.begin() + static_cast<std::size_t>(e) * eb);
  }
  return w;
}

struct Divs {
  float sgu = 1.0F, sdn = 1.0F, rgu = 1.0F, rdn = 1.0F;
};

// --- GPU op reference (W4A16 / Identity / no residual, diag armed) -----------
struct OpRef {
  std::vector<double> out;    // [T][K_in]
  std::vector<int> ids;       // [T][K] (device top-k)
  std::vector<float> logits;  // [T][E] (device router logits)
  std::vector<double> alpha;  // [T][K]
};

OpRef run_op(const Weights& w, int T, const Divs& d, cudaStream_t stream) {
  const auto geo = SparseMoeNvfp4Geometry{static_cast<std::uint32_t>(w.E), static_cast<std::uint32_t>(w.K),
                                          static_cast<std::uint32_t>(w.K_in), static_cast<std::uint32_t>(w.I)};
  const int K_in = w.K_in;

  DeviceBuffer d_x      = to_device_bf16(w.x);
  DeviceBuffer d_router = to_device_bf16(w.router);
  DeviceBuffer d_sgate  = to_device_bf16(w.sgate);
  DeviceBuffer d_sguc   = to_device(w.sguc.codes);
  DeviceBuffer d_sgus   = to_device(w.sguc.scales);
  DeviceBuffer d_sdnc   = to_device(w.sdnc.codes);
  DeviceBuffer d_sdns   = to_device(w.sdnc.scales);
  DeviceBuffer d_ewin   = to_device(w.ewin);
  DeviceBuffer d_ids    = to_device(std::vector<std::int32_t>(static_cast<std::size_t>(T) * w.K, 0));
  DeviceBuffer d_logits = to_device(std::vector<float>(static_cast<std::size_t>(T) * w.E, 0.0F));

  SparseMoeNvfp4Weights W;
  W.router_bf16       = static_cast<const std::uint16_t*>(d_router.p);
  W.shared_gate_bf16  = static_cast<const std::uint16_t*>(d_sgate.p);
  W.shared_gu_codes   = static_cast<const std::uint8_t*>(d_sguc.p);
  W.shared_gu_scales  = static_cast<const std::uint8_t*>(d_sgus.p);
  W.shared_dn_codes   = static_cast<const std::uint8_t*>(d_sdnc.p);
  W.shared_dn_scales  = static_cast<const std::uint8_t*>(d_sdns.p);
  W.shared_gu_divisor = d.sgu;
  W.shared_dn_divisor = d.sdn;
  W.expert_window     = static_cast<const std::uint8_t*>(d_ewin.p);
  W.routed_gu_divisor = d.rgu;
  W.routed_dn_divisor = d.rdn;
  W.diag_ids_out      = static_cast<std::int32_t*>(d_ids.p);
  W.diag_logits_out   = static_cast<float*>(d_logits.p);

  DeviceArena ws(sparse_moe_nvfp4_workspace_capacity_bytes(geo, 1, T));
  GuardedDeviceBuffer gdest(static_cast<std::size_t>(T) * K_in * sizeof(std::uint16_t));
  Tensor xt(d_x.p, DType::BF16, {K_in, T});
  Tensor dt(gdest.data(), DType::BF16, {K_in, T});

  sparse_moe_nvfp4(xt, geo, W, SparseMoeNvfp4Route::W4A16, SparseMoeNvfp4Epilogue::Identity,
                   nullptr, dt, ws, stream);
  cuda_check_last_launch("sparse_moe_nvfp4");
  cuda_synchronize(stream);
  failures += gdest.verify_guards("p10 dest guards");

  // plan_layout cumulative 256-alignment (p8 order: logits, sg, ids, alpha).
  const auto align256 = [](std::size_t b) { return (b + 255) & ~static_cast<std::size_t>(255); };
  const std::size_t b_logits = static_cast<std::size_t>(T) * w.E * sizeof(float);
  const std::size_t b_sg     = static_cast<std::size_t>(T) * sizeof(float);
  const std::size_t b_ids    = static_cast<std::size_t>(T) * w.K * sizeof(std::int32_t);
  const std::size_t o_ids    = align256(b_logits) + align256(b_sg);
  const std::size_t o_alpha  = o_ids + align256(b_ids);
  const char* base           = static_cast<const char*>(ws.base());
  OpRef o;
  o.out    = from_device_bf16(gdest.data(), static_cast<std::size_t>(T) * K_in);
  o.ids    = from_device<int>(d_ids.p, static_cast<std::size_t>(T) * w.K);
  o.logits = from_device<float>(d_logits.p, static_cast<std::size_t>(T) * w.E);
  const std::vector<float> af = from_device<float>(base + o_alpha, static_cast<std::size_t>(T) * w.K);
  o.alpha.assign(af.begin(), af.end());
  return o;
}

// --- case driver: GPU op vs CpuMoe on the device logits ----------------------
void check(const Weights& w, int T, const Divs& d, const char* label, cudaStream_t stream) {
  const std::string name = std::string(label);
  const OpRef got        = run_op(w, T, d, stream);

  // Host inputs for CpuMoe: BF16 x / shared_gate (the same bytes the op read).
  std::vector<std::uint16_t> x_bf(w.x.size());
  for (std::size_t i = 0; i < x_bf.size(); ++i) x_bf[i] = f32_to_bf16(w.x[static_cast<std::size_t>(i)]);
  std::vector<std::uint16_t> sg_bf(w.sgate.size());
  for (std::size_t i = 0; i < sg_bf.size(); ++i) sg_bf[i] = f32_to_bf16(w.sgate[static_cast<std::size_t>(i)]);

  const auto geo = SparseMoeNvfp4Geometry{static_cast<std::uint32_t>(w.E), static_cast<std::uint32_t>(w.K),
                                          static_cast<std::uint32_t>(w.K_in), static_cast<std::uint32_t>(w.I)};
  const auto pl       = sparse_moe_nvfp4_planes(geo);
  const std::size_t eb = sparse_moe_nvfp4_expert_bytes(geo);

  CpuMoeWeights cw;
  cw.shared_gu_codes  = w.sguc.codes.data();
  cw.shared_gu_scales = w.sguc.scales.data();
  cw.shared_dn_codes  = w.sdnc.codes.data();
  cw.shared_dn_scales = w.sdnc.scales.data();
  cw.shared_gu_divisor = d.sgu;
  cw.shared_dn_divisor = d.sdn;
  const std::uint8_t* ew = w.ewin.data();
  cw.routed_gu_codes  = CpuMoePlane{ew + pl.gu_codes, static_cast<std::int64_t>(eb)};
  cw.routed_gu_scales = CpuMoePlane{ew + pl.gu_scales, static_cast<std::int64_t>(eb)};
  cw.routed_dn_codes  = CpuMoePlane{ew + pl.dn_codes, static_cast<std::int64_t>(eb)};
  cw.routed_dn_scales = CpuMoePlane{ew + pl.dn_scales, static_cast<std::int64_t>(eb)};
  cw.routed_gu_divisor = d.rgu;
  cw.routed_dn_divisor = d.rdn;

  CpuMoe cm(w.E, w.K, w.K_in, w.I, 64);
  std::vector<std::uint16_t> y_c(static_cast<std::size_t>(T) * w.K_in);
  std::vector<std::int32_t> ids_c(static_cast<std::size_t>(T) * w.K);
  std::vector<float> alpha_c(static_cast<std::size_t>(T) * w.K);
  cm.run(got.logits.data(), T, x_bf.data(), sg_bf.data(), cw, y_c.data(), ids_c.data(),
         alpha_c.data());

  failures += verify_exact((name + " ids").c_str(),
                           std::vector<int>(ids_c.begin(), ids_c.end()), got.ids);

  std::vector<double> alpha_cd(alpha_c.begin(), alpha_c.end());
  failures += verify_pointwise(name + " alpha", alpha_cd, got.alpha,
                               PointwiseCriterion{5e-5, 1e-3});

  std::vector<double> y_cd(y_c.size());
  for (std::size_t i = 0; i < y_c.size(); ++i) y_cd[i] = bf16_to_f32(y_c[i]);
  failures += verify_reduction(name + " out", y_cd, got.out,
                               ReductionCriterion{1.5e-2, 1e-4, 1.5e-2});
}

}  // namespace

int main() {
  if (cuda_unavailable()) {
    std::cout << "SKIP: no usable CUDA device\n";
    return 77;
  }
  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

  {  // (a) mini E=8 / topk=2 / K_in=640 / I=128.
    const Weights mini = make_weights(8, 2, 640, 128, 4);
    check(mini, 1, {}, "a-mini-t1", stream);
    check(mini, 4, {}, "a-mini-t4", stream);
    // Divisor case: the GEMM epilogues divide by the stored weight_scale_divisor
    // (exercises both the shared and routed divisor paths).
    Divs dv;
    dv.sdn = 1e-3F;
    dv.rdn = 1e-3F;
    check(mini, 4, dv, "a-mini-dn1e-3-t4", stream);
  }

  {  // (a2) forward geometry E=8 / topk=2 / K_in=256 / I=64.
    const Weights fgeo = make_weights(8, 2, 256, 64, 4);
    check(fgeo, 1, {}, "a2-fgeo-t1", stream);
    check(fgeo, 4, {}, "a2-fgeo-t4", stream);
  }

  // Prod (512 experts, ~1.4 GB window + CpuMoe dequant) — set NINFER_P10_SKIP_PROD=1
  // to run only the mini cases when the GPU is held by a concurrent process.
  if (std::getenv("NINFER_P10_SKIP_PROD") == nullptr) {
    const Weights prod = make_weights(512, 10, 2560, 640, 4);
    check(prod, 1, {}, "b-prod-t1", stream);
    check(prod, 4, {}, "b-prod-t4", stream);
  } else {
    std::cout << "SKIP prod (NINFER_P10_SKIP_PROD set)\n";
  }

  cudaStreamDestroy(stream);
  std::cout << (failures == 0 ? "OK" : "FAIL") << " flash next P10 cpu moe\n";
  return failures == 0 ? 0 : 1;
}
