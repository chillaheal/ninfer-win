// P5 — gated_residual (HyperConnection) v1 contract test.
//
// Sections:
//   (a) mini decode T=1          FP64 oracle, tight tol
//   (b) mini prefill T=8         FP64 oracle, tight tol
//   (c) mini prefill T=64        FP64 oracle, tight tol
//   (d) production shape H=2560  FP64 oracle, GEMV-widened tol (one call)
//   (e) non-finite policy        zero stream -> finite + x==0; +inf input ->
//                                per-element pattern match vs FP64 oracle
//   (f) determinism              two identical runs -> bit-equal
//
// Mini geometry (make_mini_artifact): hidden=256, hc_count=4, stream=1024,
// lowrank=32. Production: hidden=2560, stream=10240, lowrank=320.
// No artifact I/O: random fp32 weights (the P9 binder does the bf16->fp32
// conversion of the real objects), so no fixture dependency and no skip.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include <ninfer/ops/gated_residual.h>

namespace {

int g_failures = 0;

void fail_or_count(const std::string& msg) {
  std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
  ++g_failures;
}

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "FATAL: %s\n", msg.c_str());
  std::exit(2);
}

void cuda_or_throw(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}

// xorshift32 -> uniform float in [0, 1)
struct Xorshift {
  std::uint32_t s;
  explicit Xorshift(std::uint32_t seed) : s(seed ? seed : 0x9e3779b9u) {}
  float next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return static_cast<float>(s * 0x1p-32);
  }
  float sym() { return 2.0f * next() - 1.0f; }  // [-1, 1)
};

double gelu_exact_d(double x) {
  return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440));
}
double sigmoid_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Naive FP64 oracle of the documented v1 contract. Returns x [T*H] and
// stream' [T*S].
void fp64_oracle(int T, int H, const std::vector<float>& stream,
                 const std::vector<float>& bout, const std::vector<float>& wdown,
                 const std::vector<float>& wup, const std::vector<float>& winj,
                 const std::vector<float>& normw, bool norm_branches,
                 std::vector<double>* x, std::vector<double>* sout) {
  const int S = 4 * H;
  const int R = static_cast<int>(wdown.size() / S);
  x->assign(static_cast<std::size_t>(T) * H, 0.0);
  sout->assign(static_cast<std::size_t>(T) * S, 0.0);
  std::vector<double> a(static_cast<std::size_t>(T) * R);
  for (int t = 0; t < T; ++t) {
    const float* st = stream.data() + static_cast<std::size_t>(t) * S;
    for (int r = 0; r < R; ++r) {
      double h = 0.0;
      for (int c = 0; c < S; ++c)
        h += static_cast<double>(wdown[static_cast<std::size_t>(r) * S + c]) *
             static_cast<double>(st[c]);
      a[static_cast<std::size_t>(t) * R + r] = gelu_exact_d(h);
    }
    double rms[4] = {0.0, 0.0, 0.0, 0.0};
    for (int b = 0; b < 4; ++b) {
      double ss = 0.0;
      for (int c = 0; c < H; ++c) {
        const double v = st[b * H + c];
        ss += v * v;
      }
      rms[b] = std::sqrt(ss / H);
    }
    for (int c = 0; c < H; ++c) {
      double xv = 0.0;
      for (int b = 0; b < 4; ++b) {
        const int s = b * H + c;
        double u = 0.0;
        for (int r = 0; r < R; ++r)
          u += static_cast<double>(wup[static_cast<std::size_t>(s) * R + r]) *
               a[static_cast<std::size_t>(t) * R + r];
        const double g = sigmoid_d(u);
        const double nv = norm_branches
                              ? (static_cast<double>(st[s]) /
                                     (rms[b] > 1e-8 ? rms[b] : 1e-8) *
                                 static_cast<double>(normw[s]))
                              : static_cast<double>(st[s]);
        xv += g * nv;
      }
      (*x)[static_cast<std::size_t>(t) * H + c] = xv;
    }
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < H; ++c) {
        const int s = b * H + c;
        const double gi =
            sigmoid_d(static_cast<double>(winj[static_cast<std::size_t>(b) * S + s]));
        (*sout)[static_cast<std::size_t>(t) * S + s] =
            static_cast<double>(st[s]) +
            gi * static_cast<double>(bout[static_cast<std::size_t>(t) * H + c]);
      }
    }
  }
}

void check_close_enough(const std::vector<float>& got, const std::vector<double>& ref,
                        const char* what, double abs_tol, double rel_tol) {
  double max_ref = 0.0;
  for (const double v : ref) max_ref = std::max(max_ref, std::fabs(v));
  const double tol = abs_tol + rel_tol * max_ref;
  double worst = 0.0;
  std::size_t where = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = std::fabs(static_cast<double>(got[i]) - ref[i]);
    if (d > worst) {
      worst = d;
      where = i;
    }
  }
  if (worst > tol) {
    fail_or_count(std::string(what) + ": max abs diff " + std::to_string(worst) +
                  " at " + std::to_string(where) + " (tolerance " +
                  std::to_string(tol) + ", max ref " + std::to_string(max_ref) +
                  ")");
  }
}

struct Shape {
  int T, H, R;
};

size_t bytes(int n, int el) { return static_cast<size_t>(n) * el * sizeof(float); }

// Runs one section: random (or caller-provided) data, op, D2H, oracle check.
// Returns true if the section passed. abs/rel tolerance per section.
bool run_section(const Shape& sh, const char* what, double abs_tol, double rel_tol,
                 std::uint32_t seed, bool norm_branches) {
  const int S = 4 * sh.H;
  Xorshift rng(seed);
  std::vector<float> stream(static_cast<std::size_t>(sh.T) * S),
      bout(static_cast<std::size_t>(sh.T) * sh.H),
      wdown(static_cast<std::size_t>(sh.R) * S),
      wup(static_cast<std::size_t>(S) * sh.R),
      winj(4 * S), normw(S);
  for (auto& v : stream) v = rng.sym();
  for (auto& v : bout) v = rng.sym();
  for (auto& v : wdown) v = rng.sym();
  for (auto& v : wup) v = rng.sym();
  for (auto& v : winj) v = rng.sym();
  for (auto& v : normw) v = 0.5f + rng.sym();  // gamma in [0, 2)

  float *d_stream, *d_bout, *d_wdown, *d_wup, *d_winj, *d_normw, *d_x, *d_sout;
  if (cudaMalloc(&d_stream, bytes(sh.T, S)) != cudaSuccess ||
      cudaMalloc(&d_bout, bytes(sh.T, sh.H)) != cudaSuccess ||
      cudaMalloc(&d_wdown, bytes(sh.R, S)) != cudaSuccess ||
      cudaMalloc(&d_wup, bytes(S, sh.R)) != cudaSuccess ||
      cudaMalloc(&d_winj, bytes(4, S)) != cudaSuccess ||
      cudaMalloc(&d_normw, bytes(1, S)) != cudaSuccess ||
      cudaMalloc(&d_x, bytes(sh.T, sh.H)) != cudaSuccess ||
      cudaMalloc(&d_sout, bytes(sh.T, S)) != cudaSuccess)
    die(std::string(what) + ": cudaMalloc (buffers)");

  cudaMemcpy(d_stream, stream.data(), bytes(sh.T, S), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bout, bout.data(), bytes(sh.T, sh.H), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wdown, wdown.data(), bytes(sh.R, S), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wup, wup.data(), bytes(S, sh.R), cudaMemcpyHostToDevice);
  cudaMemcpy(d_winj, winj.data(), bytes(4, S), cudaMemcpyHostToDevice);
  cudaMemcpy(d_normw, normw.data(), bytes(1, S), cudaMemcpyHostToDevice);

  // The delta window brackets the Op call ONLY: cudaMemGetInfo also sees a
  // concurrent serve process' VRAM on this card, so the section's own
  // setup/teardown allocations are excluded. The Op allocates nothing.
  size_t free_before = 0, free_after = 0;
  cudaMemGetInfo(&free_before, nullptr);
  ninfer::ops::GatedResidualParams p;
  p.tokens = sh.T;
  p.hidden = sh.H;
  p.hc_count = 4;
  p.lowrank = sh.R;
  p.stream_dev = d_stream;
  p.block_out_dev = d_bout;
  p.w_down_dev = d_wdown;
  p.w_up_dev = d_wup;
  p.w_inj_dev = d_winj;
  p.norm_w_dev = d_normw;
  p.read_out_dev = d_x;
  p.stream_out_dev = d_sout;
  p.norm_branches = norm_branches;
  ninfer::ops::gated_residual(p, 0);
  cudaDeviceSynchronize();
  cuda_or_throw(what);
  cudaMemGetInfo(&free_after, nullptr);
  const long long delta_kb =
      static_cast<long long>(free_before - free_after) / 1024;
  if (delta_kb > 16 * 1024)
    fail_or_count(std::string(what) +
                  ": device delta " + std::to_string(delta_kb) + " KiB > 16 MiB");

  std::vector<float> x(static_cast<std::size_t>(sh.T) * sh.H),
      sout(static_cast<std::size_t>(sh.T) * S);
  cudaMemcpy(x.data(), d_x, bytes(sh.T, sh.H), cudaMemcpyDeviceToHost);
  cudaMemcpy(sout.data(), d_sout, bytes(sh.T, S), cudaMemcpyDeviceToHost);

  std::vector<double> rx, rsout;
  fp64_oracle(sh.T, sh.H, stream, bout, wdown, wup, winj, normw, norm_branches, &rx,
              &rsout);
  check_close_enough(x, rx, (std::string(what) + " read").c_str(), abs_tol, rel_tol);
  check_close_enough(sout, rsout, (std::string(what) + " write").c_str(), abs_tol,
                     rel_tol);

  cudaFree(d_stream);
  cudaFree(d_bout);
  cudaFree(d_wdown);
  cudaFree(d_wup);
  cudaFree(d_winj);
  cudaFree(d_normw);
  cudaFree(d_x);
  cudaFree(d_sout);
  return true;
}

}  // namespace

int main() {
  if (cudaSetDevice(0) != cudaSuccess) die("cudaSetDevice");
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  std::printf("P5 gated_residual test on %s\n", prop.name);

  // (a) decode T=1, (b) prefill T=8, (c) prefill T=64 — mini geometry.
  for (const Shape s : {Shape{1, 256, 32}, Shape{8, 256, 32}, Shape{64, 256, 32}}) {
    const char* what = s.T == 1 ? "decode T=1 (a)" : (s.T == 8 ? "prefill T=8 (b)"
                                                               : "prefill T=64 (c)");
    run_section(s, what, 1e-6, 1e-4, 0x5a5a5a5a + s.T * 7, true);
  }

  // (d) production shape, one call: H=2560, S=10240, R=320. GEMV accumulation
  // widens the fp32 error budget vs the FP64 oracle (down GEMV: 10240 terms).
  run_section(Shape{16, 2560, 320}, "prod shape (d)", 5e-4, 1e-3, 0x5eed5eed, true);

  // (e) non-finite policy.
  {
    const int T = 2, H = 256, S = 1024, R = 32;
    Xorshift rng(0xefee);
    std::vector<float> stream(static_cast<std::size_t>(T) * S),
        bout(static_cast<std::size_t>(T) * H, 0.5f),
        wdown(static_cast<std::size_t>(R) * S),
        wup(static_cast<std::size_t>(S) * R), winj(4 * S), normw(S, 1.0f);
    for (auto& v : wdown) v = rng.sym();
    for (auto& v : wup) v = rng.sym();
    for (auto& v : winj) v = rng.sym();

    // (e1) all-zero stream -> finite outputs, x == 0 exactly.
    {
      float *ds, *db, *dw1, *dw2, *dw3, *dn, *dx, *dso;
      const size_t bT = sizeof(float) * S;
      const size_t bB = sizeof(float) * H;
      if (cudaMalloc(&ds, bT * T) != cudaSuccess ||
          cudaMalloc(&db, bB * T) != cudaSuccess ||
          cudaMalloc(&dw1, sizeof(float) * R * S) != cudaSuccess ||
          cudaMalloc(&dw2, sizeof(float) * S * R) != cudaSuccess ||
          cudaMalloc(&dw3, sizeof(float) * 4 * S) != cudaSuccess ||
          cudaMalloc(&dn, sizeof(float) * S) != cudaSuccess ||
          cudaMalloc(&dx, bB * T) != cudaSuccess ||
          cudaMalloc(&dso, bT * T) != cudaSuccess)
        die("e1: cudaMalloc buffers");
      cudaMemset(ds, 0, bT * T);
      cudaMemcpy(db, bout.data(), bB * T, cudaMemcpyHostToDevice);
      cudaMemcpy(dw1, wdown.data(), sizeof(float) * R * S, cudaMemcpyHostToDevice);
      cudaMemcpy(dw2, wup.data(), sizeof(float) * S * R, cudaMemcpyHostToDevice);
      cudaMemcpy(dw3, winj.data(), sizeof(float) * 4 * S, cudaMemcpyHostToDevice);
      cudaMemcpy(dn, normw.data(), sizeof(float) * S, cudaMemcpyHostToDevice);
      ninfer::ops::GatedResidualParams p;
      p.tokens = T;
      p.hidden = H;
      p.lowrank = R;
      p.stream_dev = ds;
      p.block_out_dev = db;
      p.w_down_dev = dw1;
      p.w_up_dev = dw2;
      p.w_inj_dev = dw3;
      p.norm_w_dev = dn;
      p.read_out_dev = dx;
      p.stream_out_dev = dso;
      ninfer::ops::gated_residual(p, 0);
      cudaDeviceSynchronize();
      cuda_or_throw("e1");
      std::vector<float> x(static_cast<std::size_t>(T) * H),
          so(static_cast<std::size_t>(T) * S);
      cudaMemcpy(x.data(), dx, bB * T, cudaMemcpyDeviceToHost);
      cudaMemcpy(so.data(), dso, bT * T, cudaMemcpyDeviceToHost);
      for (const float v : x)
        if (!std::isfinite(v) || v != 0.0f) {
          fail_or_count("e1 zero-stream: x not finite or not exactly 0");
          break;
        }
      for (const float v : so)
        if (!std::isfinite(v)) {
          fail_or_count("e1 zero-stream: stream' not finite");
          break;
        }
      for (auto* p : {ds, db, dw1, dw2, dw3, dn, dx, dso}) cudaFree(p);
    }

    // (e2) stream[t0, s0] = +inf. The inf also enters the low-rank down GEMV
    // (0*inf = NaN; gelu(-inf) = -inf*(1-1) = NaN), so the exact NaN positions
    // are weight-dependent: assert per-element PATTERN agreement (finite/
    // NaN/+-inf) against the FP64 oracle (same IEEE rules), closeness where
    // both are finite, and +inf at the injected position in stream'.
    {
      const int t0 = 1, s0 = H / 2;  // branch 0, c = s0
      std::fill(stream.begin(), stream.end(), 0.25f);
      stream[static_cast<std::size_t>(t0) * S + s0] = INFINITY;
      float *ds, *db, *dw1, *dw2, *dw3, *dn, *dx, *dso;
      const size_t bT = sizeof(float) * S;
      const size_t bB = sizeof(float) * H;
      if (cudaMalloc(&ds, bT * T) != cudaSuccess ||
          cudaMalloc(&db, bB * T) != cudaSuccess ||
          cudaMalloc(&dw1, sizeof(float) * R * S) != cudaSuccess ||
          cudaMalloc(&dw2, sizeof(float) * S * R) != cudaSuccess ||
          cudaMalloc(&dw3, sizeof(float) * 4 * S) != cudaSuccess ||
          cudaMalloc(&dn, sizeof(float) * S) != cudaSuccess ||
          cudaMalloc(&dx, bB * T) != cudaSuccess ||
          cudaMalloc(&dso, bT * T) != cudaSuccess)
        die("e2: cudaMalloc buffers");
      cudaMemcpy(ds, stream.data(), bT * T, cudaMemcpyHostToDevice);
      cudaMemcpy(db, bout.data(), bB * T, cudaMemcpyHostToDevice);
      cudaMemcpy(dw1, wdown.data(), sizeof(float) * R * S, cudaMemcpyHostToDevice);
      cudaMemcpy(dw2, wup.data(), sizeof(float) * S * R, cudaMemcpyHostToDevice);
      cudaMemcpy(dw3, winj.data(), sizeof(float) * 4 * S, cudaMemcpyHostToDevice);
      cudaMemcpy(dn, normw.data(), sizeof(float) * S, cudaMemcpyHostToDevice);
      ninfer::ops::GatedResidualParams p;
      p.tokens = T;
      p.hidden = H;
      p.lowrank = R;
      p.stream_dev = ds;
      p.block_out_dev = db;
      p.w_down_dev = dw1;
      p.w_up_dev = dw2;
      p.w_inj_dev = dw3;
      p.norm_w_dev = dn;
      p.read_out_dev = dx;
      p.stream_out_dev = dso;
      ninfer::ops::gated_residual(p, 0);
      cudaDeviceSynchronize();
      cuda_or_throw("e2");
      std::vector<float> x(static_cast<std::size_t>(T) * H),
          so(static_cast<std::size_t>(T) * S);
      cudaMemcpy(x.data(), dx, bB * T, cudaMemcpyDeviceToHost);
      cudaMemcpy(so.data(), dso, bT * T, cudaMemcpyDeviceToHost);
      std::vector<double> rx, rsout;
      fp64_oracle(T, H, stream, bout, wdown, wup, winj, normw, true, &rx, &rsout);
      auto pattern_bad = [](const float* got, const std::vector<double>& ref,
                            double tol_base) {
        int bad = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
          const float g = got[i];
          const double r = ref[i];
          if (std::isfinite(g) && std::isfinite(r)) {
            const double tol = tol_base + 1e-3 * std::fabs(r);
            if (std::fabs(static_cast<double>(g) - r) > tol) ++bad;
          } else if (std::isnan(g) && std::isnan(r)) {
          } else if (std::isinf(g) && std::isinf(r) &&
                     std::signbit(g) == std::signbit(r)) {
          } else {
            ++bad;
          }
        }
        return bad;
      };
      const int bx = pattern_bad(x.data(), rx, 1e-6);
      const int bs = pattern_bad(so.data(), rsout, 1e-6);
      if (bx || bs)
        fail_or_count("e2: pattern disagreement with oracle: x=" +
                      std::to_string(bx) + " stream'=" + std::to_string(bs));
      if (so[static_cast<std::size_t>(t0) * S + s0] != INFINITY)
        fail_or_count("e2: stream'[s0] expected +inf, got " +
                      std::to_string(so[static_cast<std::size_t>(t0) * S + s0]));
      for (auto* q : {ds, db, dw1, dw2, dw3, dn, dx, dso}) cudaFree(q);
    }
  }

  // (f) determinism: two identical mini T=8 runs -> bit-equal.
  {
    const Shape sh{8, 256, 32};
    const int S = 4 * sh.H;
    Xorshift rng(0xdeadbeef);
    std::vector<float> stream(static_cast<std::size_t>(sh.T) * S),
        bout(static_cast<std::size_t>(sh.T) * sh.H),
        wdown(static_cast<std::size_t>(sh.R) * S),
        wup(static_cast<std::size_t>(S) * sh.R), winj(4 * S), normw(S);
    for (auto& v : stream) v = rng.sym();
    for (auto& v : bout) v = rng.sym();
    for (auto& v : wdown) v = rng.sym();
    for (auto& v : wup) v = rng.sym();
    for (auto& v : winj) v = rng.sym();
    for (auto& v : normw) v = 0.5f + rng.sym();
    float *ds, *db, *dw1, *dw2, *dw3, *dn, *dx, *dso;
    const size_t bT = sizeof(float) * S;
    const size_t bB = sizeof(float) * sh.H;
    if (cudaMalloc(&ds, bT * sh.T) != cudaSuccess ||
        cudaMalloc(&db, bB * sh.T) != cudaSuccess ||
        cudaMalloc(&dw1, sizeof(float) * sh.R * S) != cudaSuccess ||
        cudaMalloc(&dw2, sizeof(float) * S * sh.R) != cudaSuccess ||
        cudaMalloc(&dw3, sizeof(float) * 4 * S) != cudaSuccess ||
        cudaMalloc(&dn, sizeof(float) * S) != cudaSuccess ||
        cudaMalloc(&dx, bB * sh.T) != cudaSuccess ||
        cudaMalloc(&dso, bT * sh.T) != cudaSuccess)
      die("f: cudaMalloc");
    cudaMemcpy(ds, stream.data(), bT * sh.T, cudaMemcpyHostToDevice);
    cudaMemcpy(db, bout.data(), bB * sh.T, cudaMemcpyHostToDevice);
    cudaMemcpy(dw1, wdown.data(), sizeof(float) * sh.R * S, cudaMemcpyHostToDevice);
    cudaMemcpy(dw2, wup.data(), sizeof(float) * S * sh.R, cudaMemcpyHostToDevice);
    cudaMemcpy(dw3, winj.data(), sizeof(float) * 4 * S, cudaMemcpyHostToDevice);
    cudaMemcpy(dn, normw.data(), sizeof(float) * S, cudaMemcpyHostToDevice);
    ninfer::ops::GatedResidualParams p;
    p.tokens = sh.T;
    p.hidden = sh.H;
    p.lowrank = sh.R;
    p.stream_dev = ds;
    p.block_out_dev = db;
    p.w_down_dev = dw1;
    p.w_up_dev = dw2;
    p.w_inj_dev = dw3;
    p.norm_w_dev = dn;
    p.read_out_dev = dx;
    p.stream_out_dev = dso;
    ninfer::ops::gated_residual(p, 0);
    std::vector<float> x1(static_cast<std::size_t>(sh.T) * sh.H),
        so1(static_cast<std::size_t>(sh.T) * S);
    cudaMemcpy(x1.data(), dx, bB * sh.T, cudaMemcpyDeviceToHost);
    cudaMemcpy(so1.data(), dso, bT * sh.T, cudaMemcpyDeviceToHost);
    ninfer::ops::gated_residual(p, 0);
    cudaDeviceSynchronize();
    cuda_or_throw("f");
    std::vector<float> x2(x1.size()), so2(so1.size());
    cudaMemcpy(x2.data(), dx, bB * sh.T, cudaMemcpyDeviceToHost);
    cudaMemcpy(so2.data(), dso, bT * sh.T, cudaMemcpyDeviceToHost);
    if (memcmp(x1.data(), x2.data(), x1.size() * sizeof(float)) != 0 ||
        memcmp(so1.data(), so2.data(), so1.size() * sizeof(float)) != 0)
      fail_or_count("f: two identical runs are not bit-equal");
    for (auto* q : {ds, db, dw1, dw2, dw3, dn, dx, dso}) cudaFree(q);
  }

  if (g_failures == 0) {
    std::printf("P5 gated_residual test: ALL PASS\n");
    return 0;
  }
  std::printf("P5 gated_residual test: %d failure(s)\n", g_failures);
  return 1;
}
