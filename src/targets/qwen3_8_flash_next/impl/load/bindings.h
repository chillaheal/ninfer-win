// Flash-Next P9 — loader: materializes the mini .ninfer, derives the
// host->device converted weights (hyper-connection FP32, GDN conv transpose),
// opens the PLE ngram table + the expert archive/pager, and fills the ModelView
// the mini Program binds.
//
// Lifetime contract: every pointer the Program dereferences (device weight
// planes, the converted buffers, the PLE table, the archive) is owned by a
// LoadedModel and valid for its lifetime. create_program() copies the
// ModelView (a pointer bag); it does not take ownership. The pager is
// constructed with device_slots = 32 (all routed experts device-resident for
// the mini) and is staged by create_program's prefill.

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/ngram_embedding.h"
#include "artifact/materializer.h"
#include "targets/qwen3_8_flash_next/impl/runtime/model_view.h"
#include "targets/qwen3_8_flash_next/paging/pager.h"

namespace ninfer::targets::qwen3_8_flash_next {

// Owns everything the ModelView references. Move-only (a materialized device
// arena + a paged archive cannot be copied).
class LoadedModel {
 public:
  ~LoadedModel();

  // Materializes `artifact_path` (the mini .ninfer) into the device weight
  // arena, converts the host-resident hyper-connection / GDN-conv tensors to
  // device FP32 / channel-first BF16, opens `ngram_path` (the .ngram PLE
  // sidecar) and `archive_path` (the .exarch expert archive), and builds the
  // pager (device_slots = 32). `device` supplies the H2D stream. Throws
  // artifact::ArtifactError on an identity/format/shape mismatch and
  // std::runtime_error on an archive/pager fault.
  static std::unique_ptr<LoadedModel> open(const std::filesystem::path& artifact_path,
                                           const std::filesystem::path& ngram_path,
                                           const std::filesystem::path& archive_path,
                                           DeviceContext& device);

  // The bound weight bag; its pointers are valid while *this is alive.
  const ModelView& view() const noexcept { return view_; }

  // The device arena backing every materialized device tensor (memory_summary
  // peak). Non-const accessor: the arena's peak counter is reset in place by
  // Program::reset_memory_peaks(). The host-converted buffers live in separate
  // LoadedModel-owned DeviceBuffers, not this arena.
  DeviceArena& weights_arena() noexcept { return backing_.device_arena(); }

 private:
  LoadedModel() = default;

  ModelView view_{};
  artifact::MaterializedArtifact backing_;  // owns the device weight arena
  std::vector<DeviceBuffer> derived_;       // HC FP32 + GDN conv (host->device)
  std::unique_ptr<ops::NgramEmbeddingTable> ple_table_;
  std::unique_ptr<paging::ExpertArchive> archive_;
  std::unique_ptr<paging::Pager> pager_;
};

}  // namespace ninfer::targets::qwen3_8_flash_next
