#include "pager.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <windows.h>

namespace ninfer::targets::qwen3_8_flash_next::paging {

namespace {

// Throws std::runtime_error only when error != cudaSuccess.
void cuda_or_throw(cudaError_t error, const char* what) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
  }
}

std::uint64_t make_key(std::uint32_t layer, std::uint32_t expert_id) {
  return (static_cast<std::uint64_t>(layer) << 32) | expert_id;
}

struct Slot {
  std::uint64_t key = 0;
  bool in_use = false;
  std::uint64_t tick = 0;
};

// First free slot, else the least-recently-used one.
std::size_t acquire_slot(std::vector<Slot>& slots) {
  for (std::size_t i = 0; i < slots.size(); ++i) {
    if (!slots[i].in_use) return i;
  }
  std::size_t victim = 0;
  for (std::size_t i = 1; i < slots.size(); ++i) {
    if (slots[i].tick < slots[victim].tick) victim = i;
  }
  return victim;
}

}  // namespace

// ---------------------------------------------------------------------------
// ExpertArchive
// ---------------------------------------------------------------------------

struct ExpertArchive::Impl {
  HANDLE file = INVALID_HANDLE_VALUE;
  HANDLE mapping = nullptr;
  const std::uint8_t* base = nullptr;
  std::uint32_t num_layers = 0;
  std::uint32_t experts_per_layer = 0;
  std::uint64_t bytes_per_expert = 0;

  ~Impl() {
    if (base != nullptr) UnmapViewOfFile(base);
    if (mapping != nullptr) CloseHandle(mapping);
    if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
  }
};

ExpertArchive::ExpertArchive(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

ExpertArchive::~ExpertArchive() = default;
ExpertArchive::ExpertArchive(ExpertArchive&&) noexcept = default;
ExpertArchive& ExpertArchive::operator=(ExpertArchive&&) noexcept = default;

ExpertArchive ExpertArchive::open(const std::filesystem::path& path) {
  auto impl = std::make_unique<Impl>();

  impl->file = CreateFileW(path.wstring().c_str(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (impl->file == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("expert archive missing: " + path.string());
  }
  LARGE_INTEGER size{};
  if (!GetFileSizeEx(impl->file, &size) || size.QuadPart < 0) {
    throw std::runtime_error("expert archive unreadable size: " + path.string());
  }
  const std::uint64_t file_size = static_cast<std::uint64_t>(size.QuadPart);
  if (file_size < kArchiveHeaderBytes) {
    throw std::runtime_error("expert archive truncated: header (" + path.string() +
                             ", " + std::to_string(file_size) + " bytes)");
  }
  impl->mapping = CreateFileMappingW(impl->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (impl->mapping == nullptr) {
    throw std::runtime_error("expert archive mapping failed: " + path.string());
  }
  impl->base = static_cast<const std::uint8_t*>(
      MapViewOfFile(impl->mapping, FILE_MAP_READ, 0, 0, 0));
  if (impl->base == nullptr) {
    throw std::runtime_error("expert archive view failed: " + path.string());
  }
  if (std::memcmp(impl->base, kArchiveMagic, sizeof(kArchiveMagic)) != 0) {
    throw std::runtime_error("expert archive bad magic: " + path.string());
  }
  std::uint32_t num_layers = 0, experts_per_layer = 0;
  std::uint64_t bytes_per_expert = 0;
  std::memcpy(&num_layers, impl->base + 8, sizeof(num_layers));
  std::memcpy(&experts_per_layer, impl->base + 12, sizeof(experts_per_layer));
  std::memcpy(&bytes_per_expert, impl->base + 16, sizeof(bytes_per_expert));
  const std::uint64_t payload =
      static_cast<std::uint64_t>(num_layers) * experts_per_layer * bytes_per_expert;
  if (kArchiveHeaderBytes + payload > file_size) {
    throw std::runtime_error("expert archive truncated: payload wants " +
                             std::to_string(kArchiveHeaderBytes + payload) +
                             " bytes, file has " + std::to_string(file_size));
  }
  impl->num_layers = num_layers;
  impl->experts_per_layer = experts_per_layer;
  impl->bytes_per_expert = bytes_per_expert;
  return ExpertArchive(std::move(impl));
}

std::uint32_t ExpertArchive::num_layers() const { return impl_->num_layers; }
std::uint32_t ExpertArchive::experts_per_layer() const { return impl_->experts_per_layer; }
std::uint64_t ExpertArchive::bytes_per_expert() const { return impl_->bytes_per_expert; }

std::span<const std::uint8_t> ExpertArchive::expert(std::uint32_t layer,
                                                    std::uint32_t expert_id) const {
  if (layer >= impl_->num_layers || expert_id >= impl_->experts_per_layer) {
    throw std::invalid_argument("expert archive: (layer, expert) out of range");
  }
  const std::uint64_t offset = kArchiveHeaderBytes +
      (static_cast<std::uint64_t>(layer) * impl_->experts_per_layer + expert_id) *
          impl_->bytes_per_expert;
  return std::span<const std::uint8_t>(impl_->base + offset, impl_->bytes_per_expert);
}

// ---------------------------------------------------------------------------
// Pager
// ---------------------------------------------------------------------------

struct Pager::Impl {
  const ExpertArchive& archive;
  PagerConfig config;

  std::vector<Slot> host_slots;
  std::vector<Slot> device_slots;
  void* host_base = nullptr;       // pinned: host_slots * bytes_per_expert
  void* device_base = nullptr;     // device_slots * config.device_slot_bytes
  cudaStream_t stream = nullptr;   // dedicated copy stream (never the compute stream)
  cudaEvent_t ev_start = nullptr;  // wave timing (event-measured stall_ms)
  cudaEvent_t ev_stop = nullptr;
  std::vector<std::uint64_t> pending;
  std::vector<char> host_live;      // host slot has an in-flight H2D (drain before reuse)
  std::uint64_t tick = 0;
  Counters counters;
  mutable std::vector<std::uint8_t> readback;

  explicit Impl(const ExpertArchive& archive_ref, const PagerConfig& config_ref)
      : archive(archive_ref), config(config_ref) {
    if (config.host_slots == 0 || config.device_slots == 0) {
      throw std::invalid_argument("pager: host_slots and device_slots must be >= 1");
    }
    if (config.device_slot_bytes == 0) {
      config.device_slot_bytes = archive.bytes_per_expert();
    }
    const std::uint64_t bytes_per_expert = archive.bytes_per_expert();

    host_slots.resize(config.host_slots);
    device_slots.resize(config.device_slots);
    host_live.assign(config.host_slots, 0);
    cuda_or_throw(cudaHostAlloc(&host_base, config.host_slots * bytes_per_expert,
                                cudaHostAllocDefault),
                  "cudaHostAlloc host window");
    cuda_or_throw(
        cudaMalloc(&device_base, config.device_slots * config.device_slot_bytes),
        "cudaMalloc device window");
    cuda_or_throw(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags copy stream");
    cuda_or_throw(cudaEventCreate(&ev_start), "cudaEventCreate start");
    cuda_or_throw(cudaEventCreate(&ev_stop), "cudaEventCreate stop");
  }

  ~Impl() {
    if (ev_start != nullptr) cudaEventDestroy(ev_start);
    if (ev_stop != nullptr) cudaEventDestroy(ev_stop);
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (device_base != nullptr) cudaFree(device_base);
    if (host_base != nullptr) cudaFreeHost(host_base);
  }

  StageStatus stage(std::uint32_t layer, std::uint32_t expert_id) {
    const std::uint64_t key = make_key(layer, expert_id);
    const std::uint64_t bytes_per_expert = archive.bytes_per_expert();

    // In flight before resident: a copy still pending for this key is not
    // device-visible, so it must fail as OverlappingStage, not short-circuit
    // to Hit (Hit = already visible).
    if (std::find(pending.begin(), pending.end(), key) != pending.end()) {
      return StageStatus::OverlappingStage;
    }
    for (Slot& slot : device_slots) {
      if (slot.in_use && slot.key == key) {
        slot.tick = ++tick;
        ++counters.hits;
        return StageStatus::Hit;
      }
    }
    if (bytes_per_expert > config.device_slot_bytes) {
      return StageStatus::DeviceSlotOom;
    }

    // Host fault (mmap -> pinned, LRU eviction), then async H2D on the copy
    // stream (the compute stream and any graph capture never see this copy).
    const auto source = archive.expert(layer, expert_id);
    const std::size_t host_slot = acquire_slot(host_slots);
    // An earlier H2D sourced from this host slot may still be pending on the copy
    // stream. The async DMA reads the pinned buffer at execution time, not at
    // enqueue, so overwriting it here would corrupt that in-flight copy. With
    // host_slots == 1 every consecutive stage reuses the same slot, so drain the
    // copy stream before writing. (Correct for any host_slots count.)
    if (host_live[host_slot]) wait();
    std::memcpy(static_cast<std::uint8_t*>(host_base) + host_slot * bytes_per_expert,
                source.data(), bytes_per_expert);
    host_slots[host_slot].key = key;
    host_slots[host_slot].in_use = true;
    host_slots[host_slot].tick = ++tick;

    const std::size_t device_slot = acquire_slot(device_slots);
    device_slots[device_slot].key = key;
    device_slots[device_slot].in_use = true;
    device_slots[device_slot].tick = tick;

    pending.push_back(key);
    if (pending.size() == 1) {
      cuda_or_throw(cudaEventRecord(ev_start, stream), "cudaEventRecord start");
    }
    cudaError_t error = cudaMemcpyAsync(
        static_cast<std::uint8_t*>(device_base) + device_slot * config.device_slot_bytes,
        static_cast<std::uint8_t*>(host_base) + host_slot * bytes_per_expert,
        bytes_per_expert, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) {
      pending.pop_back();
      device_slots[device_slot].in_use = false;
      cuda_or_throw(error, "cudaMemcpyAsync H2D");
    }
    ++counters.misses;
    counters.h2d_bytes += bytes_per_expert;
    host_live[host_slot] = 1;
    return StageStatus::Miss;
  }

  void wait() {
    if (pending.empty()) return;
    cuda_or_throw(cudaEventRecord(ev_stop, stream), "cudaEventRecord stop");
    cuda_or_throw(cudaEventSynchronize(ev_stop), "cudaEventSynchronize stop");
    float ms = 0.0f;
    cuda_or_throw(cudaEventElapsedTime(&ms, ev_start, ev_stop),
                  "cudaEventElapsedTime stall");
    counters.stall_ms += static_cast<double>(ms);
    pending.clear();
    std::fill(host_live.begin(), host_live.end(), 0);
  }

  std::span<const std::uint8_t> device_bytes(std::uint32_t layer,
                                             std::uint32_t expert_id) const {
    const std::uint64_t key = make_key(layer, expert_id);
    std::size_t slot = 0;
    bool found = false;
    for (std::size_t i = 0; i < device_slots.size(); ++i) {
      if (device_slots[i].in_use && device_slots[i].key == key) {
        slot = i;
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::runtime_error("pager: expert (layer " + std::to_string(layer) +
                               ", expert " + std::to_string(expert_id) +
                               ") is not device-resident");
    }
    const std::uint64_t bytes_per_expert = archive.bytes_per_expert();
    readback.resize(bytes_per_expert);
    cuda_or_throw(cudaMemcpyAsync(readback.data(),
                                  static_cast<const std::uint8_t*>(device_base) +
                                      slot * config.device_slot_bytes,
                                  bytes_per_expert, cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync D2H readback");
    cuda_or_throw(cudaStreamSynchronize(stream), "cudaStreamSynchronize readback");
    return std::span<const std::uint8_t>(readback);
  }

  const std::uint8_t* device_window(std::uint32_t layer, std::uint32_t expert_id) const {
    const std::uint64_t key = make_key(layer, expert_id);
    for (std::size_t i = 0; i < device_slots.size(); ++i) {
      if (device_slots[i].in_use && device_slots[i].key == key) {
        return static_cast<const std::uint8_t*>(device_base) +
               i * config.device_slot_bytes;
      }
    }
    throw std::runtime_error("pager: expert (layer " + std::to_string(layer) +
                             ", expert " + std::to_string(expert_id) +
                             ") is not device-resident");
  }
};

Pager::Pager(const ExpertArchive& archive, const PagerConfig& config)
    : impl_(std::make_unique<Impl>(archive, config)) {}

Pager::Pager(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Pager::~Pager() = default;
Pager::Pager(Pager&&) noexcept = default;
Pager& Pager::operator=(Pager&&) noexcept = default;

StageStatus Pager::stage(std::uint32_t layer, std::uint32_t expert_id) {
  return impl_->stage(layer, expert_id);
}

void Pager::wait() { impl_->wait(); }

std::span<const std::uint8_t> Pager::device_bytes(std::uint32_t layer,
                                                  std::uint32_t expert_id) const {
  return impl_->device_bytes(layer, expert_id);
}

const std::uint8_t* Pager::device_window(std::uint32_t layer, std::uint32_t expert_id) const {
  return impl_->device_window(layer, expert_id);
}

Counters Pager::counters() const { return impl_->counters; }
std::size_t Pager::pending() const { return impl_->pending.size(); }
const ExpertArchive& Pager::archive() const { return impl_->archive; }

std::size_t PrefillWindow::stage(std::uint32_t layer) {
  std::size_t misses = 0;
  for (std::uint32_t expert = 0; expert < pager_.archive().experts_per_layer(); ++expert) {
    if (pager_.stage(layer, expert) == StageStatus::Miss) ++misses;
  }
  return misses;
}

std::size_t DecodeFetch::stage(std::uint32_t layer,
                               std::span<const std::uint32_t> expert_ids) {
  std::size_t misses = 0;
  for (const std::uint32_t expert : expert_ids) {
    if (expert >= pager_.archive().experts_per_layer()) {
      throw std::invalid_argument("decode fetch: expert id out of range");
    }
    if (pager_.stage(layer, expert) == StageStatus::Miss) ++misses;
  }
  return misses;
}

}  // namespace ninfer::targets::qwen3_8_flash_next::paging
