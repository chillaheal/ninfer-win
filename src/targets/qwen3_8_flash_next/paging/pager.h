// P3: expert pager -- the only reason a 32 GB card can run Flash-Next.
//
// The model's 98 fused expert objects (~69.4 GB of NVFP4 packed bytes) are
// paged, not resident: a pinned-RAM host window faults them in lazily from
// the mmap'd archive, and a small fixed-slot GPU window (LRU) holds whatever
// the next layer needs. Staging is an async H2D copy on a dedicated copy
// stream with its own events; the compute stream never touches these copies,
// and they are never part of a CUDA graph capture. Packed bytes on the device
// are byte-identical to the archive (no runtime repack).

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace ninfer::targets::qwen3_8_flash_next::paging {

// Archive file layout (little-endian):
//   bytes [0,8)   magic "EXARCH\0\1"
//   bytes [8,12)  u32 num_layers
//   bytes [12,16) u32 experts_per_layer
//   bytes [16,24) u64 bytes_per_expert
//   bytes [24,40) reserved (zero)
//   bytes [40..)  num_layers * experts_per_layer * bytes_per_expert,
//                 layer-major expert-minor:
//                 offset = (layer * experts_per_layer + expert) * bytes_per_expert
//
// An ExpertArchive is a read-only mmap of such a file. In production the
// packed bytes are exactly the .ninfer expert-object slices (P8 wires the
// slice mapping); the bytes staged to the device are never repacked.
constexpr std::uint8_t kArchiveMagic[8] = {'E', 'X', 'A', 'R', 'C', 'H', 0, 1};
constexpr std::size_t kArchiveHeaderBytes = 40;

enum class StageStatus {
  Hit,                // already device-resident; no transfer issued
  Miss,               // H2D copy issued on the copy stream (visible after wait())
  DeviceSlotOom,      // the expert blob does not fit the device slot window
  OverlappingStage,   // a copy for the same (layer, expert) is still in flight
};

struct Counters {
  std::uint64_t hits = 0;      // resolutions already device-resident
  std::uint64_t misses = 0;    // H2D transfers issued
  std::uint64_t h2d_bytes = 0; // total bytes moved host -> device
  double stall_ms = 0.0;       // accumulated event-measured wait() time
};

class ExpertArchive {
 public:
  ~ExpertArchive();
  ExpertArchive(ExpertArchive&&) noexcept;
  ExpertArchive& operator=(ExpertArchive&&) noexcept;

  // Opens the file-backed archive. Throws std::runtime_error on a missing
  // file, a bad magic, or a truncated header/payload extent.
  static ExpertArchive open(const std::filesystem::path& path);

  std::uint32_t num_layers() const;
  std::uint32_t experts_per_layer() const;
  std::uint64_t bytes_per_expert() const;

  // Contiguous read-only span of one expert's packed bytes. Throws
  // std::invalid_argument out of range.
  std::span<const std::uint8_t> expert(std::uint32_t layer, std::uint32_t expert_id) const;

 private:
  struct Impl;
  explicit ExpertArchive(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

struct PagerConfig {
  std::size_t host_slots = 1;       // experts resident in the pinned window
  std::size_t device_slots = 1;     // experts resident in the GPU window
  // Per-slot device bytes. 0 = auto-size to the archive's bytes_per_expert;
  // a value below it makes every stage fail with DeviceSlotOom.
  std::uint64_t device_slot_bytes = 0;
};

class Pager {
 public:
  ~Pager();
  Pager(Pager&&) noexcept;
  Pager& operator=(Pager&&) noexcept;

  Pager(const ExpertArchive& archive, const PagerConfig& config);

  // Stage one expert. Hit = no transfer. Miss = async H2D issued; the bytes
  // are device-visible after wait(). A stage for a key whose copy is still in
  // flight fails with OverlappingStage (no silent double-copy).
  StageStatus stage(std::uint32_t layer, std::uint32_t expert_id);

  // Blocks until every in-flight copy completes; accumulates stall_ms.
  void wait();

  // Verification accessor: copies the device-resident expert back over the
  // copy stream and spans the readback buffer. Throws std::runtime_error if
  // the expert is not device-resident. The span is valid until the next
  // device_bytes() call.
  std::span<const std::uint8_t> device_bytes(std::uint32_t layer,
                                             std::uint32_t expert_id) const;

  // Direct device pointer to a device-resident expert's slot (no copy, no
  // sync). Throws std::runtime_error if the expert is not resident. Valid
  // while the expert stays resident (a later Miss may evict the slot).
  const std::uint8_t* device_window(std::uint32_t layer, std::uint32_t expert_id) const;

  Counters counters() const;
  std::size_t pending() const;  // in-flight copies
  const ExpertArchive& archive() const;

 private:
  struct Impl;
  explicit Pager(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Prefill stages ALL experts of the layer (prefill needs the full set).
class PrefillWindow {
 public:
  explicit PrefillWindow(Pager& pager) : pager_(pager) {}

  // Stages every expert of the layer; returns the number of misses issued.
  std::size_t stage(std::uint32_t layer);
  void wait() { pager_.wait(); }
  Pager& pager() const { return pager_; }

 private:
  Pager& pager_;
};

// Decode stages only the routed expert ids.
class DecodeFetch {
 public:
  explicit DecodeFetch(Pager& pager) : pager_(pager) {}

  std::size_t stage(std::uint32_t layer, std::span<const std::uint32_t> expert_ids);
  void wait() { pager_.wait(); }
  Pager& pager() const { return pager_; }

 private:
  Pager& pager_;
};

}  // namespace ninfer::targets::qwen3_8_flash_next::paging
