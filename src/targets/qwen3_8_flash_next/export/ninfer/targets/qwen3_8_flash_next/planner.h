#pragma once
// P2 startup planner: pure byte arithmetic.
//
// No device access, no clocks, no I/O -- the same input always yields the same
// output (deterministic contract, spec P2). It takes the artifact's inventoried
// byte counts plus the launch options and either produces a placement table or
// names the budget that overflowed (the CLI surfaces that message verbatim).

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {

enum class Residency : std::uint8_t {
    Gpu,         // device-resident (weights, KV, expert staging window, workspace)
    HostPinned,  // host pinned memory (fused experts; KV when host_kv)
    Mmap,        // page-cache backed file mapping (PLE .ngram; never counted)
};

[[nodiscard]] std::string_view residency_name(Residency residency);

struct PlacementEntry {
    Residency residency = Residency::Gpu;
    std::string budget;  // "backbone" | "kv_cache" | "indexer_kv" | "expert_window" |
                         // "workspace" | "gdn_state" | "experts" | "ple"
    std::uint64_t bytes = 0;
};

struct PlacementTable {
    std::vector<PlacementEntry> entries;
    std::uint64_t kv_pool_tokens = 0;             // max_context x max_concurrency
    std::uint64_t gpu_total_bytes = 0;
    std::uint64_t host_pinned_total_bytes = 0;
    std::uint64_t mmap_total_bytes = 0;
};

struct PlacementError {
    std::string budget;  // the budget that overflowed ("" when no single entry did)
    std::uint64_t required_bytes = 0;
    std::uint64_t available_bytes = 0;

    [[nodiscard]] std::string message() const;
};

struct PlacementRequest {
    std::uint64_t device_budget_bytes = 0;   // GPU budget (free VRAM in production)
    std::uint64_t system_ram_bytes = 0;      // host budget for pinned allocations
    std::uint64_t backbone_weight_bytes = 0;  // gpu_resident object bytes
    std::uint64_t expert_host_bytes = 0;      // host_experts object bytes (69.363 GB real)
    std::uint64_t ple_bytes = 0;              // .ngram size (always mmap, page-cache backed)
    std::uint64_t kv_bytes_per_token = 0;
    // Indexer-K pool per-token footprint (512-dim key: 1 B/elem FP8 or 2 B/elem BF16,
    // + one 2 B half scale per token). 0 disables the pool in the table (the FP8
    // default is populated by Package::bind; kept 0 by the dev dry-run tool).
    std::uint64_t indexer_kv_bytes_per_token = 0;
    std::uint32_t max_context = 0;
    std::uint32_t max_concurrency = 1;
    std::uint64_t expert_window_bytes = 0;    // GPU staging window for paged expert pages
    bool host_kv = false;                     // KV on host pinned memory instead of GPU
    // Where the full expert set (69.36 GB real) is placed. true = --expert-host-all:
    // pin every expert to host RAM (the fastest config; overflows host_pinned when
    // the set is larger than physical RAM). false = the real loader's mmap path: the
    // planner then spills the set budget-driven, RAM first and SSD/mmap as the last
    // resort when RAM cannot hold it.
    bool expert_host_all = true;
    std::uint64_t workspace_bytes = 0;
    // Total GDN recurrent state across all GDN layers and the concurrent
    // sequences (one FP32 [128,128,48] state per GDN layer per sequence).
    // Callers publish it from Package::gdn_state_bytes(max_concurrency).
    std::uint64_t gdn_state_bytes = 0;
};

// Deterministic: same input -> same output.
[[nodiscard]] std::variant<PlacementTable, PlacementError> plan_placement(
    const PlacementRequest& request);

// Human-readable placement table for the CLI dry-run. Pure formatting (no
// device access): the same (request, result) always renders the same string.
// Shared by the dry-run tool (tools/flash_next_dev/flash_next_placement.cpp)
// and the P11 snapshot test, which compare the tool's stdout byte-for-byte
// against this in-process rendering.
[[nodiscard]] std::string render_placement(
    const PlacementRequest& request,
    const std::variant<PlacementTable, PlacementError>& result);

}  // namespace ninfer::targets::qwen3_8_flash_next
