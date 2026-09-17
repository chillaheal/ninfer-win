// Flash-Next S1 gate (b) — the real-geometry device loader.
//
// RealLoadedModel::open mirrors the mini LoadedModel::open (bindings.cpp) but at
// REAL geometry (detail::Geometry) and with the mini loader's PROVEN materialization
// placement (not the gate-(a) canonical residency placement):
//   * hyper-connection / GDN-conv / PLE-conv BF16 tensors -> retain_on_host, then
//     converted into RealLoadedModel-owned DeviceBuffers (HC -> FP32, GDN conv ->
//     channel-first BF16, PLE conv -> host FP32) because the ops want those
//     layouts (HcWeights holds const float*; the GDN conv is channel-first);
//   * the global mixer, the routed-expert fused NVFP4 objects, and the
//     input_scale_divisor scalars -> validate_only (unused by the v1 W4A16 path;
//     the pager supplies the routed experts in S3);
//   * everything else (embedding/output head, MoE router+shared, GDN QKVZ/out/norm,
//     QSA projections, the non-text vision/MTP GpuResident tensors) -> device.
//
// The routed experts (69 GB) are NOT pinned or H2D'd: they stay mmap'd inside the
// .ninfer (the materialize step never uploads the validate_only objects), which is
// exactly the "mmap the 69 GB experts (not pinned)" the gate specifies. The 51.8 GB
// .ngram is opened by NgramEmbeddingTable as a read-only mmap (not pinned).

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/ngram_embedding.h"
#include "targets/qwen3_8_flash_next/impl/runtime/real_model_view.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"

namespace ninfer::targets::qwen3_8_flash_next {

// Device-footprint summary logged by the gate (b) driver.
struct RealLoadSummary {
  std::uint64_t device_capacity_bytes = 0;  // planned backbone device capacity
  std::uint64_t h2d_bytes = 0;              // bytes actually uploaded to the device
  std::uint64_t arena_peak_bytes = 0;       // the device arena's peak_used()
  std::uint64_t derived_bytes = 0;          // host-converted DeviceBuffers (HC FP32 + GDN conv)
  std::uint64_t total_device_bytes = 0;     // arena peak + derived
  std::size_t object_count = 0;
  std::size_t device_objects = 0;
  std::size_t host_objects = 0;
  std::size_t layer_count = 0;
  std::uint32_t ple_embed_dim = 0;          // from the .ngram table (2560)
  std::uint64_t ngram_bytes = 0;            // the .ngram file size (mmap'd, not pinned)
};

class RealLoadedModel {
 public:
  ~RealLoadedModel();

  // Loads the real artifact to the device (H2D backbone) + mmaps the .ngram.
  // Throws on identity/shape mismatch, a non-finite divisor, or a CUDA error.
  static std::unique_ptr<RealLoadedModel> open(const std::filesystem::path& artifact_path,
                                               const std::filesystem::path& ngram_path,
                                               DeviceContext& device);

  const RealModelView& view() const noexcept { return view_; }
  DeviceArena& weights_arena() noexcept { return backing_.device_arena(); }
  const RealLoadSummary& summary() const noexcept { return summary_; }
  // The artifact Reader (its mmap backs the routed-expert scatter source the
  // S3 RealProgram reads per layer). Kept alive for the model's lifetime so the
  // host bases in view().routed_host stay valid.
  const artifact::Reader& reader() const noexcept { return *reader_; }

 private:
  RealLoadedModel() = default;

  RealModelView view_{};
  RealLoadSummary summary_{};
  artifact::MaterializedArtifact backing_;
  std::vector<DeviceBuffer> derived_;  // HC FP32 + transposed GDN conv (own cudaMallocs)
  std::unique_ptr<ops::NgramEmbeddingTable> ple_table_;
  std::unique_ptr<artifact::Reader> reader_;
};

}  // namespace ninfer::targets::qwen3_8_flash_next
