#pragma once
// Flash-Next S1 — real-geometry loader, gate (a).
//
// CPU header-only bind of the REAL 180B artifact: every object is consumed and
// placed through the Binder (the finish() invariant: no unconsumed, no
// unplaced) WITHOUT materialization, so no GPU / VRAM is touched. The 1235 text
// tensors bind at their Geometry-derived shapes (the meaningful shape check);
// the 367 non-text (vision / MTP) tensors are consumed shape-agnostically from
// their own descriptors. Placement is residency-driven (detail::residency_class)
// so the plan's device set is the full GpuResident backbone (text + mtp + vision
// = 1504 objects, ~6.02 GiB per the P0 audit / P11). Gate (b) (device load +
// ModelView) builds on this and is a separate increment.

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace ninfer::targets::qwen3_8_flash_next {

// Summary of a gate-(a) header-only bind of the real artifact.
struct RealBindSummary {
  std::size_t object_count = 0;           // total Binder objects (1608)
  std::size_t resource_count = 0;        // frontend resources (6)
  std::size_t text_tensor_count = 0;     // explicitly bound text tensors (1235)
  std::size_t nontext_tensor_count = 0;  // vision + MTP, enumerated (367)
  std::size_t device_count = 0;          // objects placed on device (all GpuResident: 1504)
  std::size_t validate_count = 0;        // objects validate-only (6 resources + 98 HostExperts = 104)
  std::uint64_t device_capacity_bytes = 0;  // backbone device footprint (plan)
};

// Opens the artifact header, binds + places every object (no materialize), and
// returns the summary. Throws on identity mismatch, a text shape/format/layout
// mismatch, or if the Binder.finish() invariant fails. GPU-independent.
RealBindSummary real_bind_only(const std::filesystem::path& artifact_path);

}  // namespace ninfer::targets::qwen3_8_flash_next
