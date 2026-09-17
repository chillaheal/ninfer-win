// Flash-Next S1 gate (b) — the real-geometry ModelView.
//
// Reuses the dimension-generic weight structs from model_view.h (Fp8RowScaleWeight,
// Bf16Weight, HcWeights, GdnLayerWeights, QsaLayerWeights, MoeLayerWeights,
// LayerWeights) but carries the REAL geometry:
//   * layers is a runtime std::vector (48), not the mini std::array<…, 4>;
//   * the PLE conv weights are a runtime std::vector<float> (embed_dim is read
//     from the .ngram table, not the mini compile-time constant);
//   * the routed-expert archive/pager are null for gate (b) — S3 wires the real
//     pager over the artifact's mmap'd expert slices.
//
// Lifetime: identical to the mini ModelView — the device weight bytes are owned
// by the loader's MaterializedArtifact device arena (lifetime = RealLoadedModel),
// the host-converted buffers (HC FP32, transposed GDN conv) are DeviceBuffers
// owned by RealLoadedModel, and the PLE table is owned by RealLoadedModel.
// RealModelView only references them.

#pragma once

#include <cstdint>
#include <vector>

#include "targets/qwen3_8_flash_next/impl/runtime/model_view.h"  // weight structs

namespace ninfer::targets::qwen3_8_flash_next {

// Host-side bases of one layer's fused 512-expert routed objects (the .ninfer
// mmap, never H2D'd). The S3 RealProgram scatters the per-token selected
// experts from these host bases into the device moe_window. codes_base is the
// fused object's code plane; scales_base = codes_base + the NVFP4 scale-plane
// offset. The per-expert stride within each plane (512 experts fused) is:
//   gu: codes 1,638,400 B/expert, scales 204,800 B/expert   (rows 512*1280, cols 2560)
//   dn: codes 819,200 B/expert,   scales 102,400 B/expert   (rows 512*2560, cols 640)
// 128-row swizzle slabs align to expert boundaries (1280 = 10*128, 2560 = 20*128),
// so each expert's scale block is contiguous.
struct RoutedHostBases {
  const std::uint8_t* gu_codes = nullptr;
  const std::uint8_t* gu_scales = nullptr;
  const std::uint8_t* dn_codes = nullptr;
  const std::uint8_t* dn_scales = nullptr;
};

// The real weight bag for the 48-layer flash_next Program. Weight struct
// layouts are identical to the mini ones (dimension-generic); only the layer
// count and the PLE embed dim are real-sized.
struct RealModelView {
  Fp8RowScaleWeight embedding;    // [248320, 2560]
  Fp8RowScaleWeight output_head;  // [248320, 2560]

  std::vector<LayerWeights> layers;  // 48 (gdn at l%4!=3, qsa at l%4==3)

  // PLE (layer 1): host table + host conv weights [4][embed_dim] FP32 (the
  // first embed_dim channels of each tap of ple/convolution [4, 10240] BF16).
  // embed_dim = NgramStoreInfo::embed_dim() = heads*cols (16*160 = 2560 = hidden).
  const ops::NgramEmbeddingTable* ple_table = nullptr;
  std::vector<float> ple_conv_weights;  // [4 * ple_conv_channels] FP32, row-major
  std::uint32_t ple_conv_channels = 0;  // = embed_dim (2560)

  // Paged routed experts (owned by RealLoadedModel; S3's concern). Null for
  // gate (b) — the 69 GB routed experts stay mmap'd in the .ninfer and the real
  // pager (wrapping the artifact's expert slices) is wired in S3.
  const paging::ExpertArchive* archive = nullptr;
  paging::Pager* pager = nullptr;

  // Per-layer host bases of the fused routed objects (see RoutedHostBases).
  // Populated from the artifact Reader's mmap in S3; null for gate (b). The
  // backing mmap is owned by RealLoadedModel (the durable Reader).
  std::vector<RoutedHostBases> routed_host;

  // The loader's device arena backing every materialized weight —
  // memory_summary() reports its peak. The host-converted buffers are
  // separate RealLoadedModel-owned DeviceBuffers (outside this arena).
  DeviceArena* weights_arena = nullptr;
};

}  // namespace ninfer::targets::qwen3_8_flash_next
