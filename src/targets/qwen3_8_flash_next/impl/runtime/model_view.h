// Flash-Next P9 — ModelView: the bound weight bag for the mini Program.
//
// Pure pointer bag: the device weight bytes are owned by the loader's
// MaterializedArtifact device arena (lifetime = LoadedModel, which outlives
// the Program); the host-converted buffers (HC FP32 weights, the transposed
// GDN conv) are DeviceBuffers owned by LoadedModel. The pager and the PLE
// table are also owned by LoadedModel; ModelView only references them.
//
// Tensor names/shapes are the mini artifact's (see
// out/flash_next_dev/qwen3_8_flash_next_mini.inventory.json). FP8 row-scale
// layout: code plane [N*K] u8, scale plane [N] BF16 at align_up(N*K, 256) —
// the artifact's RowScaleV1. NVFP4 block-scale layout: the canonical
// BlockScaleK16M128x4V1 planes (code N*K/2 packed E2M1, scales N*K/16
// swizzled, one FP32 divisor word after the scale plane).

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "core/arena.h"
#include "ninfer/ops/ngram_embedding.h"
#include "targets/qwen3_8_flash_next/paging/pager.h"

namespace ninfer::targets::qwen3_8_flash_next {

// Locked mini geometry (inventory.json).
namespace mini {
inline constexpr std::uint32_t kHidden          = 256;
inline constexpr std::uint32_t kVocab           = 256;
inline constexpr std::uint32_t kLayers          = 4;
inline constexpr std::uint32_t kGdnLayers       = 3;  // layers 0..2
inline constexpr std::uint32_t kQsaLayer        = 3;
inline constexpr std::uint32_t kPleLayer        = 1;
inline constexpr std::uint32_t kQsaQHeads       = 4;
inline constexpr std::uint32_t kQsaKvHeads      = 2;
inline constexpr std::uint32_t kQsaHeadDim      = 32;
inline constexpr std::uint32_t kQsaIndexerChunks = 4;
inline constexpr std::uint32_t kGdnQkHeads      = 2;
inline constexpr std::uint32_t kGdnValueHeads   = 4;
inline constexpr std::uint32_t kGdnStateDim     = 128;
inline constexpr std::uint32_t kGdnConvChannels = 256;
inline constexpr std::uint32_t kGdnConvWidth    = 4;
inline constexpr std::uint32_t kGdnQkvzRows     = 384;  // 2*128 (q+k) + 4*32 (v)
inline constexpr std::uint32_t kGdnOutputCols   = 128;  // 4 value heads x 32
inline constexpr std::uint32_t kGdnAbRows       = 8;    // [0,4) = a, [4,8) = b
inline constexpr std::uint32_t kHcCount         = 4;
inline constexpr std::uint32_t kStreamWidth     = 1024; // 4 branches x 256
inline constexpr std::uint32_t kHcLowRank       = 32;
inline constexpr std::uint32_t kMoEExperts      = 8;
inline constexpr std::uint32_t kMoETopK         = 2;
inline constexpr std::uint32_t kMoEIntermediate = 64;
inline constexpr std::uint32_t kPleConvKernel   = 4;
inline constexpr std::uint32_t kPleEmbedDim     = 128;
}  // namespace mini

// FP8_E4M3FN row-scale weight (RowScaleV1).
struct Fp8RowScaleWeight {
    const std::uint8_t* codes = nullptr;  // [rows*cols]
    const std::uint16_t* scale = nullptr; // [rows] BF16
    std::uint32_t rows        = 0;
    std::uint32_t cols        = 0;
};

// Plain BF16 weight (ContiguousLeV1), row-major.
struct Bf16Weight {
    const std::uint16_t* data = nullptr;  // [rows*cols]
    std::uint32_t rows        = 0;
    std::uint32_t cols        = 0;
};

// One hyper-connection set, converted to FP32 on the device (the
// gated_residual op's row-major layout):
//   w_down [R,S] = [32,1024], w_up [S,R] = [1024,32],
//   w_inj [B,S] = [4,1024], norm_w [S] = [1024].
struct HcWeights {
    const float* w_down  = nullptr;
    const float* w_up    = nullptr;
    const float* w_inj   = nullptr;
    const float* norm_w  = nullptr;
};

struct GdnLayerWeights {
    Fp8RowScaleWeight qkvz;   // [384,256]
    Fp8RowScaleWeight output; // [256,128]
    Bf16Weight a_b_projection;  // [8,256]
    const float* a_log    = nullptr;  // [4] FP32
    const float* dt_bias  = nullptr;  // [4] FP32
    const std::uint16_t* conv = nullptr;  // [256,4] channel-first (transposed)
    const std::uint16_t* norm = nullptr;  // [32] BF16
};

struct QsaLayerWeights {
    Fp8RowScaleWeight query;   // [256,256]
    Fp8RowScaleWeight key;     // [64,256]
    Fp8RowScaleWeight value;   // [64,256]
    Fp8RowScaleWeight output;  // [256,128]
    const std::uint16_t* query_norm = nullptr;  // [32] BF16
    const std::uint16_t* key_norm   = nullptr;  // [32] BF16
    Bf16Weight indexer_qk_proj;  // [160,256]
    const std::uint16_t* indexer_q_norm = nullptr;  // [32] BF16
    const std::uint16_t* indexer_k_norm = nullptr;  // [32] BF16 (shared over 4 chunks)
};

struct MoeLayerWeights {
    const std::uint16_t* router_bf16 = nullptr;  // [8][256] BF16
    const std::uint16_t* shared_gate_bf16 = nullptr;  // [256] BF16
    // Shared expert (device-resident, canonical block-scale layout).
    const std::uint8_t* shared_gu_codes  = nullptr;  // [128][128] packed E2M1
    const std::uint8_t* shared_gu_scales = nullptr;  // swizzled plane
    const std::uint8_t* shared_dn_codes  = nullptr;  // [256][32] packed E2M1
    const std::uint8_t* shared_dn_scales = nullptr;
    float shared_gu_divisor = 0.0F;
    float shared_dn_divisor = 0.0F;
    // Routed experts (P3 pager device window). Filled by create_program
    // after the 32-expert construction staging: [8][27,648] contiguous,
    // expert-ascending (sparse_moe_nvfp4 contract).
    const std::uint8_t* expert_window = nullptr;
    float routed_gu_divisor = 0.0F;
    float routed_dn_divisor = 0.0F;
};

struct LayerWeights {
    std::optional<GdnLayerWeights> gdn;   // layers 0..2
    std::optional<QsaLayerWeights> qsa;   // layer 3
    HcWeights attn_hc;
    HcWeights mlp_hc;
    MoeLayerWeights mlp;
};

struct ModelView {
    Fp8RowScaleWeight embedding;    // [256,256]
    Fp8RowScaleWeight output_head;  // [256,256]

    std::array<LayerWeights, mini::kLayers> layers;

    // PLE (layer 1): host table + host conv weights [4][128] FP32 (the first
    // 128 channels of each tap of ple/convolution [4,1024] BF16).
    const ops::NgramEmbeddingTable* ple_table = nullptr;
    std::array<std::array<float, mini::kPleEmbedDim>, mini::kPleConvKernel> ple_conv_weights;

    // Paged routed experts (owned by LoadedModel; the Program stages on the
    // pager and reads the expert windows off its device window).
    const paging::ExpertArchive* archive = nullptr;
    paging::Pager* pager = nullptr;

    // The loader's device arena backing every materialized weight —
    // memory_summary() reports its peak. Non-const:
    // Program::reset_memory_peaks() resets the arena's peak counter in place.
    // (The host-converted buffers are separate LoadedModel-owned DeviceBuffers.)
    DeviceArena* weights_arena = nullptr;
};

}  // namespace ninfer::targets::qwen3_8_flash_next
