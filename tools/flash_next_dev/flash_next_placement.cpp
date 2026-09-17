// P11 CLI placement dry-run for the flash_next family.
//
// Pure planning tool: it takes the artifact's inventoried byte counts (the P0
// audit sums) plus the launch options, runs the P2 planner (plan_placement),
// and prints the human-readable placement table (render_placement). It never
// touches a device, clock, or the real artifact -- the same inputs always
// print the same table. It exists so that, BEFORE the GPU ever sees the real
// file, an operator can see which dimension (GPU VRAM vs host pinned RAM) a
// given launch configuration will overflow on a given machine.
//
// Exit codes: 0 = placement FITs, 3 = placement OVERFLOW (the planner named
// the budget that exceeded), 2 = bad usage / unparseable argument.
//
// This is the P11 deliverable; P12 wires the friendlier serve flags
// (--expert-device-window / --expert-host-all / --ple-mmap) into the serve
// through the same render_placement so the live launch prints this table.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

#include <ninfer/targets/qwen3_8_flash_next/planner.h>
#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/types.h>

namespace {

using ninfer::targets::qwen3_8_flash_next::plan_placement;
using ninfer::targets::qwen3_8_flash_next::render_placement;
using ninfer::targets::qwen3_8_flash_next::Package;
using ninfer::targets::qwen3_8_flash_next::PlacementError;
using ninfer::targets::qwen3_8_flash_next::PlacementRequest;

void usage(std::FILE* out) {
    std::fputs(
        "usage: flash_next_placement "
        "--max-context N --kv-dtype fp8|bf16 "
        "--backbone-bytes N --expert-host-bytes N "
        "--device-budget-bytes N --system-ram-bytes N "
        "[--max-concurrency N] [--ple-bytes N] [--expert-window-bytes N] "
        "[--workspace-bytes N] [--gdn-state-bytes N] [--host-kv] [--expert-mmap]\n"
        "  N is a base-10 byte / token count. "
        "Exit 0 = FIT, 3 = OVERFLOW, 2 = bad usage.\n"
        "  Experts policy: default = --expert-host-all (pin the full expert set to "
        "host RAM; overflows host_pinned when it exceeds physical RAM); "
        "--expert-mmap = the real loader path (spills the set budget-driven, RAM "
        "first, SSD/page-cache when RAM cannot hold it).\n",
        out);
}

bool parse_uint64(std::string_view text, std::uint64_t* out) {
    if (text.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(std::string(text).c_str(), &end, 10);
    if (end == nullptr || *end != '\0') return false;
    *out = static_cast<std::uint64_t>(value);
    return true;
}

    // Returns the argv value for "--key" (the following token). `provided` is
    // set true when the key was present; false when absent. Returns false only
    // on a malformed "--key" with no following value.
bool find_value(int argc, char** argv, const char* key, std::string* out, bool* provided) {
    *provided = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) {
            if (i + 1 >= argc) return false;  // missing value
            *out = argv[i + 1];
            *provided = true;
            return true;
        }
    }
    return true;  // key absent (ok)
}

bool find_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
    // Emit the exact render_placement bytes (LF line endings, no CRLF text-mode
    // translation) so the P11 snapshot test compares stdout byte-for-byte. A
    // text-mode stdout on a pipe turns each "\n" into "\r\n".
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    PlacementRequest request;
    std::string value;
    bool provided = false;

    // Required scalar fields.
    std::uint32_t max_context = 0;
    std::uint64_t raw = 0;
    std::string kv_dtype;
    bool kv_dtype_provided = false;

    if (!find_value(argc, argv, "--max-context", &value, &provided) || !provided ||
        !parse_uint64(value, &raw) || raw > 4294967295u) {
        usage(stderr);
        return 2;
    }
    max_context = static_cast<std::uint32_t>(raw);

    if (!find_value(argc, argv, "--kv-dtype", &value, &kv_dtype_provided) ||
        !kv_dtype_provided || (value != "fp8" && value != "bf16")) {
        usage(stderr);
        return 2;
    }
    kv_dtype = value;

    for (const char* key : {"--backbone-bytes", "--expert-host-bytes",
                            "--device-budget-bytes", "--system-ram-bytes"}) {
        if (!find_value(argc, argv, key, &value, &provided) || !provided ||
            !parse_uint64(value, &raw)) {
            usage(stderr);
            return 2;
        }
        if (std::strcmp(key, "--backbone-bytes") == 0) {
            request.backbone_weight_bytes = raw;
        } else if (std::strcmp(key, "--expert-host-bytes") == 0) {
            request.expert_host_bytes = raw;
        } else if (std::strcmp(key, "--device-budget-bytes") == 0) {
            request.device_budget_bytes = raw;
        } else {
            request.system_ram_bytes = raw;
        }
    }

    // Optional fields (defaults match PlacementRequest).
    if (find_value(argc, argv, "--max-concurrency", &value, &provided) && provided) {
        if (!parse_uint64(value, &raw) || raw == 0) {
            usage(stderr);
            return 2;
        }
        request.max_concurrency = static_cast<std::uint32_t>(raw);
    }
    const auto read_optional = [&](const char* key, std::uint64_t* out) {
        bool p = false;
        if (!find_value(argc, argv, key, &value, &p) || !p) return true;
        if (!parse_uint64(value, &raw)) return false;
        *out = raw;
        return true;
    };
    if (!read_optional("--ple-bytes", &request.ple_bytes) ||
        !read_optional("--expert-window-bytes", &request.expert_window_bytes) ||
        !read_optional("--workspace-bytes", &request.workspace_bytes) ||
        !read_optional("--gdn-state-bytes", &request.gdn_state_bytes)) {
        usage(stderr);
        return 2;
    }

    request.max_context = max_context;
    request.host_kv = find_flag(argc, argv, "--host-kv");
    // Default is --expert-host-all (pin the set to RAM); --expert-mmap switches to
    // the real loader's budget-driven spill (RAM first, SSD/mmap as the last resort).
    request.expert_host_all = !find_flag(argc, argv, "--expert-mmap");
    // Share the exact production KV bytes-per-token with the runtime rather
    // than duplicating the constants here.
    request.kv_bytes_per_token = kv_dtype == "fp8"
        ? Package::kv_bytes_per_token(ninfer::KvCacheStorage::Fp8E4M3Row256)
        : Package::kv_bytes_per_token(ninfer::KvCacheStorage::BFloat16);

    const auto result = plan_placement(request);
    const std::string text = render_placement(request, result);
    std::fputs(text.c_str(), stdout);
    std::fflush(stdout);
    if (const auto* error = std::get_if<PlacementError>(&result)) {
        (void)error;
        return 3;
    }
    (void)result;
    return 0;
}
