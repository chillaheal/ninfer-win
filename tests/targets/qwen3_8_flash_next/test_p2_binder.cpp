// P2: binder -- artifact objects mapped to residency classes, lazy handles.
//
// * Every mini object lands in exactly one residency class; the fused expert
//   matrices (and only they) are host_experts, frontend/* are resources.
// * Byte sums are self-consistent (totals == sum of the per-object plans).
// * The PLE handle is a real mapping whose size equals the .ngram file size,
//   the archive handle is open, and NOTHING was committed: the working-set
//   delta across the whole bind stays below 2 MiB.
// * A missing .ngram sidecar throws.
//
// Skips (exit 77) when the mini fixture is missing; generate it with
// `python -m tools.convert.qwen3_8_flash_next.make_mini_artifact`.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <windows.h>
#include <psapi.h>

#include <ninfer/targets/qwen3_8_flash_next/binder.h>

namespace {

using BindResult = ninfer::targets::qwen3_8_flash_next::BindResult;
using ninfer::targets::qwen3_8_flash_next::bind_artifact;

int failures = 0;

void fail_or_count(const std::string& message) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
}

[[noreturn]] void die(const std::string& message) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
}

bool is_expert_name(const std::string& name) {
    const std::string_view view = name;
    return view.ends_with("/mlp/experts/gate_up") || view.ends_with("/mlp/experts/down");
}

std::uint64_t working_set_bytes() {
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        die("GetProcessMemoryInfo");
    }
    return counters.WorkingSetSize;
}

}  // namespace

int main() {
    const std::filesystem::path out_dir =
        std::filesystem::path(NINFER_SOURCE_DIR) / "out" / "flash_next_dev";
    const std::filesystem::path ninfer_path = out_dir / "qwen3_8_flash_next_mini.ninfer";
    const std::filesystem::path ngram_path = out_dir / "qwen3_8_flash_next_mini.ngram";
    if (!std::filesystem::exists(ninfer_path) || !std::filesystem::exists(ngram_path)) {
        return 77;  // fixture not built
    }

    const std::uint64_t before = working_set_bytes();

    BindResult bound;
    try {
        bound = bind_artifact(ninfer_path, ngram_path);
    } catch (const std::exception& e) {
        die(std::string("bind_artifact threw: ") + e.what());
    }

    // 119 mini objects, each in exactly one residency class.
    const std::uint64_t total_objects =
        bound.gpu_planned.size() + bound.host_experts.size() + bound.resource_count;
    if (total_objects != 119) {
        fail_or_count("object count across classes = " + std::to_string(total_objects) +
                      " (want 119)");
    }

    // Fused expert matrices are the only host class; everything else is GPU.
    if (bound.host_experts.empty()) {
        fail_or_count("no host_expert objects found in the mini artifact");
    }
    for (const auto& plan : bound.host_experts) {
        if (!is_expert_name(plan.name)) {
            fail_or_count("host_expert is not a fused expert matrix: " + plan.name);
        }
    }
    for (const auto& plan : bound.gpu_planned) {
        if (plan.name.rfind("frontend/", 0) == 0 || is_expert_name(plan.name)) {
            fail_or_count("misclassified GPU object: " + plan.name);
        }
    }

    // Byte sums are self-consistent.
    std::uint64_t gpu_sum = 0;
    for (const auto& plan : bound.gpu_planned) gpu_sum += plan.bytes;
    if (gpu_sum != bound.gpu_weight_bytes) {
        fail_or_count("gpu_weight_bytes != sum of gpu_planned bytes");
    }
    std::uint64_t host_sum = 0;
    for (const auto& plan : bound.host_experts) host_sum += plan.bytes;
    if (host_sum != bound.host_expert_bytes) {
        fail_or_count("host_expert_bytes != sum of host_experts bytes");
    }

    // Handles: PLE mapping sized to the file, archive handle open.
    if (!bound.ple.opened()) {
        fail_or_count("PLE mapping is not open");
    } else if (bound.ple.bytes() != std::filesystem::file_size(ngram_path)) {
        fail_or_count("PLE mapping size != .ngram file size");
    }
    if (!bound.archive.opened()) {
        fail_or_count("archive handle is not open");
    }

    // Missing sidecar throws.
    try {
        static_cast<void>(bind_artifact(ninfer_path, out_dir / "does_not_exist.ngram"));
        fail_or_count("bind_artifact accepted a missing .ngram sidecar");
    } catch (const std::exception&) {
        // expected
    }

    // Lazy: nothing was committed by the bind.
    const std::uint64_t after = working_set_bytes();
    if (after > before && after - before > 2ull * 1024 * 1024) {
        fail_or_count("bind committed more than 2 MiB of working set");
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d binder check(s) failed\n", failures);
        return 1;
    }
    std::fprintf(stderr, "P2 binder: all checks passed\n");
    return 0;
}
