#pragma once
// P3.2 — CPU MoE forward (W4A16 NVFP4) for the cold MoE layers offloaded from
// the GPU (Unsloth `-ncmoe` style). Host-only C++ — no CUDA.
//
// Math contract (matches moe/moe_nvfp4.cu's W4A16 path):
//   - dequantized weight W[n][k] = e2m1(code) * e4m3(scale): bit-identical to
//     the device (both a single F32 multiply; both LUT decodes are exact).
//   - GEMM rows accumulate F32 in the same k order as the device's contracted
//     FMA chain (fmaf); a 4-lane SIMD reduction is tree-reduced (rel error
//     ~1e-6, far inside the p8 {1.5e-2, 1e-4, 1.5e-2} tolerance).
//   - top-k is the EXACT replica of kernel 2 (value-DESC/id-ASC argmax,
//     insertion sort, ids written ascending) over the caller's F32 logits —
//     in production those are the D2H'd device router logits, so the CPU and
//     device top-k consume bit-identical inputs.
//   - the exp-based terms (alpha, silu, sigmoid) use std::expf where the
//     device uses __expf: ~1-2 ulp drift, which flips a bf16 output only on a
//     rounding boundary (accepted; the GPU op and this path are compared
//     within the p8 tolerance in test_p10).
//
// Domain (same as sparse_moe_nvfp4): K_in % 64 == 0 in [128, 4096];
// I % 64 == 0 in [64, 4096]; 1 <= K <= 32; T in [1, 2048].

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer::targets::qwen3_8_flash_next {

// One routed plane: expert e's bytes live at base + e * expert_stride
// (the host mmap bases from RoutedHostBases; per-expert plane stride).
struct CpuMoePlane {
  const std::uint8_t* base = nullptr;
  std::int64_t expert_stride = 0;
};

struct CpuMoeWeights {
  // Shared expert: host-resident copies (D2H'd once at init; the shared
  // planes exist only in the device arena).
  const std::uint8_t* shared_gu_codes = nullptr;
  const std::uint8_t* shared_gu_scales = nullptr;
  const std::uint8_t* shared_dn_codes = nullptr;
  const std::uint8_t* shared_dn_scales = nullptr;
  float shared_gu_divisor = 1.0F;
  float shared_dn_divisor = 1.0F;

  // Routed experts: the four host mmap plane bases + per-expert stride.
  CpuMoePlane routed_gu_codes;
  CpuMoePlane routed_gu_scales;
  CpuMoePlane routed_dn_codes;
  CpuMoePlane routed_dn_scales;
  float routed_gu_divisor = 1.0F;
  float routed_dn_divisor = 1.0F;
};

class CpuMoe {
 public:
  // experts/topk/input_dim/intermediate = the MoE geometry (E, K, K_in, I).
  // expert_cache = LRU slots of DEQUANTIZED F32 routed experts (gu+dn each);
  // the shared expert holds a dedicated non-evictable slot.
  CpuMoe(int experts, int topk, int input_dim, int intermediate,
         int expert_cache = 64);
  ~CpuMoe();
  CpuMoe(const CpuMoe&) = delete;
  CpuMoe& operator=(const CpuMoe&) = delete;

  // logits [T][E] F32 (the device router output, already on the host);
  // x [T][K_in] BF16 token-major; shared_gate [K_in] BF16;
  // y out [T][K_in] BF16 token-major; ids [T][K] (ascending) and alpha [T][K]
  // are optional diagnostics. threads = 0 picks
  // hardware_concurrency()/2 clamped to [1, 16].
  void run(const float* logits, int tokens, const std::uint16_t* x,
           const std::uint16_t* shared_gate, const CpuMoeWeights& w,
           std::uint16_t* y, std::int32_t* ids_out = nullptr,
           float* alpha_out = nullptr, int threads = 0);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ninfer::targets::qwen3_8_flash_next
