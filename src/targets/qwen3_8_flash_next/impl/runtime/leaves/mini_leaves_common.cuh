// Flash-Next P9 — shared __device__ decode/math helpers for the mini leaves.
//
// The decoders mirror the host codecs in tests/targets/qwen3_8_flash_next
// (P8 test_p8_sparse_moe.cpp) so device and host reference agree bit-for-bit
// on every representable code.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// OCP e4m3fn: bit 7 sign, bits 6-3 exponent (bias 7), bits 2-0 mantissa.
// Exponent 0 -> mantissa * 2^-6; otherwise (1 + mantissa/8) * 2^(exp - 7).
__device__ __forceinline__ float e4m3_decode(std::uint8_t code) {
  const std::uint32_t sign = (code >> 7) & 1u;
  const std::uint32_t exp = (code >> 3) & 0xFu;
  const std::uint32_t mant = code & 7u;
  float value;
  if (exp == 0u) {
    value = static_cast<float>(mant) * 0.015625F;  // 2^-6
  } else {
    value = (1.0F + static_cast<float>(mant) * 0.125F) *
            exp2f(static_cast<float>(exp) - 7.0F);
  }
  return sign ? -value : value;
}

// Decode a BF16 code to FP32. BF16 is exactly the top 16 bits of an IEEE-754
// single (sign/exp/shared mantissa), so decoding is lossless: shift left 16.
// (We don't touch __nv_bfloat16::__x — that member is private in CUDA 13.3.)
__device__ __forceinline__ float bf16_decode(std::uint16_t bits) {
  const std::uint32_t f = static_cast<std::uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

__device__ __forceinline__ float mini_sigmoid(float x) {
  return 1.0F / (1.0F + expf(-x));
}

// softplus(x) = log1p(exp(x)); for large x exp overflows, so the identity
// softplus(x) = x applies in FP32 well before x > 20.
__device__ __forceinline__ float mini_softplus(float x) {
  return x > 20.0F ? x : log1pf(expf(x));
}

__device__ __forceinline__ float mini_silu(float x) {
  return x / (1.0F + expf(-x));
}

}  // namespace ninfer::targets::qwen3_8_flash_next::detail
