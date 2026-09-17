#include <ninfer/targets/qwen3_8_flash_next/binder.h>

#include <cstring>
#include <stdexcept>

#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/config.h"

namespace ninfer::targets::qwen3_8_flash_next {

namespace {

[[noreturn]] void fail(std::string message) {
    throw std::runtime_error("flash_next binder: " + std::move(message));
}

std::uint8_t ngram_magic[8] = {'N', 'N', 'G', 'R', 'A', 'M', 0, 1};

}  // namespace

PleMmapHandle::PleMmapHandle(PleMmapHandle&& other) noexcept
    : file_(other.file_),
      mapping_(other.mapping_),
      view_(other.view_),
      bytes_(other.bytes_) {
    other.file_ = INVALID_HANDLE_VALUE;
    other.mapping_ = nullptr;
    other.view_ = nullptr;
    other.bytes_ = 0;
}

PleMmapHandle& PleMmapHandle::operator=(PleMmapHandle&& other) noexcept {
    if (this != &other) {
        close();
        file_ = other.file_;
        mapping_ = other.mapping_;
        view_ = other.view_;
        bytes_ = other.bytes_;
        other.file_ = INVALID_HANDLE_VALUE;
        other.mapping_ = nullptr;
        other.view_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

PleMmapHandle::~PleMmapHandle() { close(); }

void PleMmapHandle::close() noexcept {
    if (view_ != nullptr) {
        UnmapViewOfFile(view_);
        view_ = nullptr;
    }
    if (mapping_ != nullptr) {
        CloseHandle(mapping_);
        mapping_ = nullptr;
    }
    if (file_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE;
    }
    bytes_ = 0;
}

const std::uint8_t* PleMmapHandle::view() {
    if (mapping_ == nullptr) {
        fail("PLE handle not opened");
    }
    if (view_ == nullptr) {
        const void* base = MapViewOfFileEx(mapping_, FILE_MAP_READ, 0, 0, 0, 0);
        if (base == nullptr) {
            fail("MapViewOfFileEx failed (error " + std::to_string(GetLastError()) + ")");
        }
        view_ = static_cast<const std::uint8_t*>(base);
    }
    return view_;
}

void PleMmapHandle::open(const std::filesystem::path& path) {
    const HANDLE file =
        CreateFileW(path.wstring().c_str(), GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        fail("cannot open PLE sidecar '" + path.string() + "' (error " +
             std::to_string(GetLastError()) + ")");
    }
    LARGE_INTEGER file_size{};
    if (!GetFileSizeEx(file, &file_size)) {
        CloseHandle(file);
        fail("GetFileSizeEx failed for '" + path.string() + "'");
    }
    std::uint8_t header[16] = {};
    DWORD read = 0;
    if (!ReadFile(file, header, sizeof(header), &read, nullptr) || read != sizeof(header)) {
        CloseHandle(file);
        fail("ngram header too small in '" + path.string() + "'");
    }
    if (std::memcmp(header, ngram_magic, sizeof(ngram_magic)) != 0) {
        CloseHandle(file);
        fail("bad ngram magic in '" + path.string() + "'");
    }
    std::uint64_t json_bytes = 0;
    std::memcpy(&json_bytes, header + 8, sizeof(json_bytes));
    if (16 + json_bytes > static_cast<std::uint64_t>(file_size.QuadPart)) {
        CloseHandle(file);
        fail("ngram directory exceeds file in '" + path.string() + "'");
    }
    const HANDLE mapping =
        CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        CloseHandle(file);
        fail("CreateFileMappingW failed for '" + path.string() + "' (error " +
             std::to_string(GetLastError()) + ")");
    }
    file_ = file;
    mapping_ = mapping;
    bytes_ = static_cast<std::uint64_t>(file_size.QuadPart);
}

ArchiveHandle::ArchiveHandle(ArchiveHandle&& other) noexcept : file_(other.file_) {
    other.file_ = INVALID_HANDLE_VALUE;
}

ArchiveHandle& ArchiveHandle::operator=(ArchiveHandle&& other) noexcept {
    if (this != &other) {
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        file_ = other.file_;
        other.file_ = INVALID_HANDLE_VALUE;
    }
    return *this;
}

ArchiveHandle::~ArchiveHandle() {
    if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
}

void ArchiveHandle::open(const std::filesystem::path& path) {
    const HANDLE file =
        CreateFileW(path.wstring().c_str(), GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        fail("cannot open archive handle on '" + path.string() + "' (error " +
             std::to_string(GetLastError()) + ")");
    }
    file_ = file;
}

BindResult bind_artifact(const std::filesystem::path& artifact_path,
                         const std::filesystem::path& ngram_path) {
    if (!std::filesystem::exists(artifact_path)) {
        fail("missing artifact '" + artifact_path.string() + "'");
    }
    if (!std::filesystem::exists(ngram_path)) {
        fail("missing ngram sidecar '" + ngram_path.string() + "'");
    }
    artifact::Reader reader(artifact_path);
    const auto& identity = reader.identity();
    if (identity.model_id != "qwen3.8-flash-next" || identity.weights_id != "nvfp4") {
        fail("artifact identity '" + identity.model_id + "/" + identity.weights_id +
             "' is not qwen3.8-flash-next/nvfp4");
    }

    BindResult result;
    for (const auto& object : reader.objects()) {
        const auto name = std::string(artifact::object_name(object));
        const auto bytes = artifact::object_bytes(object);
        switch (detail::residency_class(name)) {
            case detail::ResidencyClass::GpuResident:
                result.gpu_planned.push_back({name, bytes});
                result.gpu_weight_bytes += bytes;
                break;
            case detail::ResidencyClass::HostExperts:
                result.host_experts.push_back({name, bytes});
                result.host_expert_bytes += bytes;
                break;
            case detail::ResidencyClass::Resource:
                ++result.resource_count;
                break;
        }
    }
    result.archive.open(artifact_path);
    result.ple.open(ngram_path);
    return result;
}

}  // namespace ninfer::targets::qwen3_8_flash_next
