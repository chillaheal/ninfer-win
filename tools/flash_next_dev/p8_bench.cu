// P8 microbench — W4A4 vs W4A16 decode timing for the NVFP4 sparse MoE.
//
// Geometry (production): E=512, K=10, K_in=2560, I=640. Decode T=1.
// Weight/value buffers are filled with a fixed non-degenerate pattern (timing
// only — the values do not change the per-route kernel work). Each route is
// warmed up (kWarm iters) then timed over kIters iters with CUDA events. The
// per-route us/iter and the winner are written to out/flash_next_dev/P8.log
// and stdout.
//
// The winner is recorded in the target dispatch (moe/moe_nvfp4_dispatch.cpp,
// which exposes the winning route only). The public Op keeps both routes.

#include <ninfer/ops/sparse_moe_nvfp4.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;

namespace {

constexpr int            kE     = 512;
constexpr int            kK     = 10;
constexpr int            kKIn   = 2560;
constexpr int            kI     = 640;
constexpr int            kT     = 1;
constexpr int            kWarm  = 10;
constexpr int            kIters = 200;
constexpr std::uint8_t   kFill  = 0x5A;
// The expert window only needs to cover the K routed experts the op actually
// reads. With the uniform kFill the router logits are a full E-way tie, so the
// documented (value DESC, id ASC) topk tie-break selects ids 0..K-1; 64 experts
// is a wide margin. The window size does NOT affect per-route timing — the op
// reads the same K experts + the E-wide router either way — so this keeps the
// device footprint small enough to run with the live serve resident.
constexpr int            kWinExperts = 64;

// bf16 bit pattern -> the float value it represents (shift into the top 16 bits).
inline float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
  float               f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

inline std::string fmt(double v) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.4f", v);
  return std::string(buf);
}

double time_route(SparseMoeNvfp4Route route,
                  const SparseMoeNvfp4Geometry& geo,
                  const SparseMoeNvfp4Weights& w,
                  const Tensor& x,
                  Tensor& dest,
                  WorkspaceArena& ws,
                  cudaStream_t stream) {
  for (int i = 0; i < kWarm; ++i) {
    sparse_moe_nvfp4(x, geo, w, route, SparseMoeNvfp4Epilogue::Identity, nullptr, dest, ws, stream);
  }
  cudaStreamSynchronize(stream);

  cudaEvent_t ev0 = nullptr;
  cudaEvent_t ev1 = nullptr;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  cudaEventRecord(ev0, stream);
  for (int i = 0; i < kIters; ++i) {
    sparse_moe_nvfp4(x, geo, w, route, SparseMoeNvfp4Epilogue::Identity, nullptr, dest, ws, stream);
  }
  cudaEventRecord(ev1, stream);
  cudaEventSynchronize(ev1);
  float ms = 0.0F;
  cudaEventElapsedTime(&ms, ev0, ev1);
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  return (static_cast<double>(ms) * 1000.0) / static_cast<double>(kIters);
}

} // namespace

int main() {
  const SparseMoeNvfp4Geometry geo{static_cast<std::uint32_t>(kE),
                                   static_cast<std::uint32_t>(kK),
                                   static_cast<std::uint32_t>(kKIn),
                                   static_cast<std::uint32_t>(kI)};
  const std::uint64_t expert_bytes = sparse_moe_nvfp4_expert_bytes(geo);

  // Device buffers (values non-degenerate; timing only).
  const std::uint64_t gu_codes = static_cast<std::uint64_t>(kI) * kKIn;
  const std::uint64_t dn_codes = gu_codes / 2;
  DeviceBuffer xbuf(static_cast<std::size_t>(kKIn) * kT * sizeof(std::uint16_t));
  DeviceBuffer router(static_cast<std::size_t>(kE) * kKIn * sizeof(std::uint16_t));
  DeviceBuffer sgate(static_cast<std::size_t>(kKIn) * sizeof(std::uint16_t));
  DeviceBuffer sgu_c(gu_codes);
  DeviceBuffer sgu_s(gu_codes / 8);
  DeviceBuffer sdn_c(dn_codes);
  DeviceBuffer sdn_s(dn_codes / 8);
  DeviceBuffer ewin(static_cast<std::size_t>(kWinExperts) * expert_bytes);
  DeviceBuffer dest(static_cast<std::size_t>(kKIn) * kT * sizeof(std::uint16_t));
  // x is filled with O(1) BF16 (0x3F80 = 1.0) so the W4A4 activation quantizes
  // to non-zero codes and the |out|max sanity check is meaningful; the weight
  // planes keep the kFill byte pattern (non-degenerate E2M1 codes + E4M3 scales).
  std::vector<std::uint16_t> hx(static_cast<std::size_t>(kKIn) * kT, 0x3F80);
  xbuf.copy_from_host(hx.data(), hx.size() * sizeof(std::uint16_t));
  router.fill(kFill);
  sgate.fill(kFill);
  sgu_c.fill(kFill);
  sgu_s.fill(kFill);
  sdn_c.fill(kFill);
  sdn_s.fill(kFill);
  ewin.fill(kFill);

  SparseMoeNvfp4Weights w;
  w.router_bf16      = static_cast<const std::uint16_t*>(router.p);
  w.shared_gate_bf16 = static_cast<const std::uint16_t*>(sgate.p);
  w.shared_gu_codes  = static_cast<const std::uint8_t*>(sgu_c.p);
  w.shared_gu_scales = static_cast<const std::uint8_t*>(sgu_s.p);
  w.shared_dn_codes  = static_cast<const std::uint8_t*>(sdn_c.p);
  w.shared_dn_scales = static_cast<const std::uint8_t*>(sdn_s.p);
  w.shared_gu_divisor = 1.0F;
  w.shared_dn_divisor = 1.0F;
  w.expert_window     = static_cast<const std::uint8_t*>(ewin.p);
  w.routed_gu_divisor = 1.0F;
  w.routed_dn_divisor = 1.0F;

  WorkspaceArena ws(sparse_moe_nvfp4_workspace_capacity_bytes(geo, kT, kT));
  Tensor         x_t(xbuf.p, DType::BF16, {kKIn, kT});
  Tensor         dt(dest.p, DType::BF16, {kKIn, kT});
  cudaStream_t   stream = nullptr;
  cudaStreamCreate(&stream);

  const double w4a4_us  = time_route(SparseMoeNvfp4Route::W4A4, geo, w, x_t, dt, ws, stream);
  const double w4a16_us = time_route(SparseMoeNvfp4Route::W4A16, geo, w, x_t, dt, ws, stream);

  // Sanity: confirm the last run produced a non-trivial destination.
  std::vector<std::uint16_t> host(static_cast<std::size_t>(kKIn) * kT);
  dest.copy_to_host(host.data(), host.size() * sizeof(std::uint16_t));
  double amax = 0.0;
  for (std::uint16_t b : host) {
    amax = std::fmax(amax, std::fabs(bf16_to_float(b)));
  }

  const bool        w4a4_wins = (w4a4_us <= w4a16_us);
  const std::string winner = w4a4_wins ? "W4A4" : "W4A16";

  std::string log;
  log += "P8 W4A4 vs W4A16 microbench (NVFP4 sparse MoE)\n";
  log += "  geometry: E=512 K=10 K_in=2560 I=640 T=1 (decode)\n";
  log += "  expert window: " + std::to_string(kWinExperts) +
         " experts (topk tie-break selects ids 0..K-1; window size does not "
         "affect per-route timing)\n";
  log += "  iters: warm=" + std::to_string(kWarm) + " timed=" + std::to_string(kIters) +
         " (CUDA events, ms/iters*1000)\n";
  log += "  conditions: live serve resident (GPU ~31 GB/32 GB) — absolute us "
         "reflects contention; the W4A4-vs-W4A16 RANKING is the dispatch basis\n";
  log += "  W4A4:  " + fmt(w4a4_us) + " us/iter\n";
  log += "  W4A16: " + fmt(w4a16_us) + " us/iter\n";
  log += "  |out|max after last (W4A16) run: " + fmt(amax) + "\n";
  log += "  WINNER: " + winner + "\n";
  std::printf("%s", log.c_str());

  const std::filesystem::path log_path(
      std::filesystem::path(NINFER_SOURCE_DIR) / "out" / "flash_next_dev" / "P8.log");
  std::ofstream f(log_path, std::ios::binary | std::ios::trunc);
  if (f) {
    f << log;
    std::printf("[bench] wrote %s\n", log_path.string().c_str());
  } else {
    std::fprintf(stderr, "[bench] could not open %s for writing\n", log_path.string().c_str());
  }

  cudaStreamDestroy(stream);
  return 0;
}
