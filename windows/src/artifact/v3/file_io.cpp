#include "artifact/v3/file_io.h"

#include "artifact/v3/framing.h"
#include "artifact/v3/schema.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact::v3 {
namespace {

#if defined(_WIN32)
std::string win32_error_message(DWORD code) {
    char* buffer = nullptr;
    const DWORD length = ::FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<char*>(&buffer), 0, nullptr);
    std::string message;
    if (length != 0 && buffer != nullptr) {
        message.assign(buffer, static_cast<std::size_t>(length));
        while (!message.empty() &&
               (message.back() == '\r' || message.back() == '\n' || message.back() == ' ')) {
            message.pop_back();
        }
        ::LocalFree(buffer);
    }
    if (message.empty()) {
        message = "unknown error";
    }
    return message;
}

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " +
                        win32_error_message(::GetLastError()));
}

// Positions `handle` before `offset` and reads up to `length` bytes into `buffer`.
// Windows pages reads through the OS cache, so every read is served from the cache
// and the O_DIRECT fast path of the POSIX build is a no-op on this platform.
bool read_at(HANDLE handle, std::uint64_t offset, void* buffer, std::size_t length,
              std::size_t& read_bytes) {
    LARGE_INTEGER position {};
    position.QuadPart = static_cast<LONGLONG>(offset);
    if (!::SetFilePointerEx(handle, position, nullptr, FILE_BEGIN)) {
        return false;
    }
    DWORD bytes_read = 0;
    const bool ok = ::ReadFile(handle, buffer, static_cast<DWORD>(length), &bytes_read, nullptr);
    read_bytes = static_cast<std::size_t>(bytes_read);
    return ok;
}
#else
[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}
#endif

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
#ifdef _WIN32
    HANDLE handle = ::CreateFileW(path_.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        fail(path_, "open");
    }
    fd_ = reinterpret_cast<std::intptr_t>(handle);

    LARGE_INTEGER size {};
    if (::GetFileSizeEx(reinterpret_cast<HANDLE>(fd_), &size) == 0 || size.QuadPart < 0) {
        ::CloseHandle(reinterpret_cast<HANDLE>(fd_));
        fd_ = -1;
        fail(path_, "fstat");
    }
    const DWORD attributes = ::GetFileAttributesW(path_.wstring().c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        ::CloseHandle(reinterpret_cast<HANDLE>(fd_));
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(size.QuadPart);
#else
    fd_ = static_cast<std::intptr_t>(::open(path_.c_str(), O_RDONLY | O_CLOEXEC));
    if (fd_ < 0) {
        fail(path_, "open");
    }

    struct stat status {};
    if (::fstat(static_cast<int>(fd_), &status) != 0) {
        const int error = errno;
        ::close(static_cast<int>(fd_));
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(static_cast<int>(fd_));
        fd_   = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
#endif
}

InputFile::~InputFile() {
#ifdef _WIN32
    if (direct_fd_ != -1) {
        ::CloseHandle(reinterpret_cast<HANDLE>(direct_fd_));
    }
    if (fd_ != -1) {
        ::CloseHandle(reinterpret_cast<HANDLE>(fd_));
    }
#else
    if (direct_fd_ >= 0) {
        ::close(static_cast<int>(direct_fd_));
    }
    if (fd_ >= 0) {
        ::close(static_cast<int>(fd_));
    }
#endif
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        std::size_t read = 0;
#ifdef _WIN32
        if (!read_at(reinterpret_cast<HANDLE>(fd_), offset, destination.data(), count, read)) {
            fail(path_, "read");
        }
#else
        const long result = ::pread(static_cast<int>(fd_), destination.data(), count,
                                file_offset(offset));
        if (result < 0) {
            if (errno == EINTR) {
                continue;
        }
        fail(path_, "pread");
        }
        read = static_cast<std::size_t>(result);
#endif
        if (read == 0) {
            throw ArtifactError(path_.string() + ": unexpected EOF");
        }
        offset += read;
        destination = destination.subspan(read);
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset,
                          std::span<std::byte> destination) const {
    if (destination.empty()) {
        return 0;
    }
#ifdef _WIN32
    if (direct_fd_ == -1) {
        HANDLE handle = ::CreateFileW(path_.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ,
                                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            fail(path_, "open direct");
        }
        direct_fd_ = reinterpret_cast<std::intptr_t>(handle);
    }
    std::size_t total = 0;
    while (total < destination.size()) {
        std::size_t read = 0;
        if (!read_at(reinterpret_cast<HANDLE>(direct_fd_), offset + total,
                         destination.data() + total, destination.size() - total, read)) {
            break;
        }
        if (read == 0) {
            break;
        }
        total += read;
    }
    return total;
#else
    if (direct_fd_ < 0) {
        direct_fd_ = static_cast<std::intptr_t>(::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT));
        if (direct_fd_ < 0) {
            fail(path_, "open direct");
        }
    }
    std::size_t total = 0;
    while (total < destination.size()) {
        const auto count =
            std::min<std::size_t>(destination.size() - total, 64ULL * 1024 * 1024);
        const long result =
            ::pread(static_cast<int>(direct_fd_), destination.data() + total, count,
                       file_offset(offset + total));
        if (result < 0) {
            if (errno == EINTR) {
                continue;
        }
        fail(path_, "direct pread");
        }
        if (result == 0) {
            break;
        }
        total += static_cast<std::size_t>(result);
    }
    return total;
#endif
}

} // namespace ninfer::artifact::v3
