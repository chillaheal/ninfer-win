#pragma once
// P2 binder: maps artifact objects to the canonical residency classes and opens
// the out-of-artifact handles LAZILY:
//   - PLE .ngram  -> Win32 file + PAGE_READONLY mapping, NO MapViewOfFile
//   - experts     -> a second read handle on the .ninfer (archive) for P3's
//                    direct host->GPU page staging
//
// No device allocation and no page commit happens here: a mapping without a
// view commits nothing, and the archive handle only opens the file.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <windows.h>

namespace ninfer::targets::qwen3_8_flash_next {

struct PlannedAllocation {
    std::string name;
    std::uint64_t bytes = 0;
};

// RAII over the .ngram sidecar: file handle + whole-file PAGE_READONLY mapping.
// opened() is false until open(path) succeeds; bytes() is the full file size.
// The view stays unmapped until view() (P4's NgramEmbeddingTable consumes it).
class PleMmapHandle {
public:
    PleMmapHandle() = default;
    PleMmapHandle(PleMmapHandle&& other) noexcept;
    PleMmapHandle& operator=(PleMmapHandle&& other) noexcept;
    ~PleMmapHandle();

    PleMmapHandle(const PleMmapHandle&) = delete;
    PleMmapHandle& operator=(const PleMmapHandle&) = delete;

    // Throws std::runtime_error on missing file, short header, bad magic, or
    // directory out of range. Leaves no view mapped (lazy-commit contract).
    void open(const std::filesystem::path& path);

    // Maps the whole file (first call, 64-bit-safe via MapViewOfFileEx) and
    // returns the read-only base. Throws std::runtime_error if not opened.
    [[nodiscard]] const std::uint8_t* view();

    [[nodiscard]] bool opened() const noexcept { return mapping_ != nullptr; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

private:
    void close() noexcept;

    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
    const std::uint8_t* view_ = nullptr;
    std::uint64_t bytes_ = 0;
};

// RAII over a plain read handle on the .ninfer artifact (P3 direct page reads).
class ArchiveHandle {
public:
    ArchiveHandle() = default;
    ArchiveHandle(ArchiveHandle&& other) noexcept;
    ArchiveHandle& operator=(ArchiveHandle&& other) noexcept;
    ~ArchiveHandle();

    ArchiveHandle(const ArchiveHandle&) = delete;
    ArchiveHandle& operator=(const ArchiveHandle&) = delete;

    void open(const std::filesystem::path& path);  // throws std::runtime_error

    [[nodiscard]] bool opened() const noexcept { return file_ != INVALID_HANDLE_VALUE; }

private:
    HANDLE file_ = INVALID_HANDLE_VALUE;
};

struct BindResult {
    std::vector<PlannedAllocation> gpu_planned;  // GpuResident objects (sizes only)
    std::vector<PlannedAllocation> host_experts;  // HostExperts objects (sizes only)
    std::size_t resource_count = 0;               // frontend resources (validated only)
    std::uint64_t gpu_weight_bytes = 0;
    std::uint64_t host_expert_bytes = 0;
    PleMmapHandle ple;
    ArchiveHandle archive;
};

// Opens <artifact> (.ninfer) + its .ngram sidecar. Verifies the artifact
// identity is qwen3.8-flash-next/nvfp4, walks every object through the
// residency rule, and opens the lazy handles. Throws std::runtime_error on any
// contract violation.
[[nodiscard]] BindResult bind_artifact(const std::filesystem::path& artifact_path,
                                       const std::filesystem::path& ngram_path);

}  // namespace ninfer::targets::qwen3_8_flash_next
