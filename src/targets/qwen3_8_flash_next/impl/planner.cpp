#include <ninfer/targets/qwen3_8_flash_next/planner.h>

#include <utility>

namespace ninfer::targets::qwen3_8_flash_next {

std::string_view residency_name(Residency residency) {
    switch (residency) {
        case Residency::Gpu:
            return "gpu";
        case Residency::HostPinned:
            return "host_pinned";
        case Residency::Mmap:
            return "mmap";
    }
    return "unknown";
}

std::string PlacementError::message() const {
    std::string text = "placement overflow: budget '" + budget + "' exceeded: total required " +
                       std::to_string(required_bytes) + " bytes but only " +
                       std::to_string(available_bytes) + " bytes available";
    return text;
}

namespace {

// Fixed evaluation order: the device budget is charged in this sequence and the
// first entry whose cumulative total crosses the budget is the named overflow.
const char* kGpuBudgetOrder[] = {"backbone", "kv_cache", "indexer_kv", "expert_window",
                                 "workspace", "gdn_state"};

const PlacementEntry* find_entry(const PlacementTable& table, std::string_view budget) {
    for (const auto& entry : table.entries) {
        if (entry.budget == budget) return &entry;
    }
    return nullptr;
}

}  // namespace

std::variant<PlacementTable, PlacementError> plan_placement(const PlacementRequest& request) {
    const std::uint64_t pool_tokens =
        static_cast<std::uint64_t>(request.max_context) * request.max_concurrency;
    const std::uint64_t kv_bytes = pool_tokens * request.kv_bytes_per_token;
    const std::uint64_t indexer_kv_bytes = pool_tokens * request.indexer_kv_bytes_per_token;

    PlacementTable table;
    table.kv_pool_tokens = pool_tokens;
    table.entries.push_back(
        {Residency::Gpu, "backbone", request.backbone_weight_bytes});
    table.entries.push_back({request.host_kv ? Residency::HostPinned : Residency::Gpu,
                             "kv_cache", kv_bytes});
    // Indexer-K pool: always device-resident (the qsa_indexer logit GEMM reads it on
    // the GPU). Emitted only when the caller populates the per-token footprint
    // (Package::bind from the FP8/BF16 default); the dev dry-run tool leaves it 0,
    // so its rendered table stays byte-identical to the pool-free case.
    if (request.indexer_kv_bytes_per_token > 0) {
        table.entries.push_back({Residency::Gpu, "indexer_kv", indexer_kv_bytes});
    }
    table.entries.push_back(
        {Residency::Gpu, "expert_window", request.expert_window_bytes});
    table.entries.push_back({Residency::Gpu, "workspace", request.workspace_bytes});
    table.entries.push_back({Residency::Gpu, "gdn_state", request.gdn_state_bytes});
    // The full expert set (69.36 GB real) is the unit the launch config decides
    // where to put. --expert-host-all (default) pins it to host RAM; that is the
    // fastest config and overflows host_pinned when the set exceeds physical RAM
    // (the P11 finding, preserved). Otherwise the planner spills it budget-driven:
    // RAM first (host_pinned), and the SSD/mmap page-cache tier as the last resort
    // when RAM cannot hold it. Deterministic and real-model-aware -- on a box whose
    // RAM is smaller than the set, the mmap path lands the experts on SSD, not a
    // host_pinned overflow.
    const std::uint64_t host_if_pinned =
        request.expert_host_bytes + (request.host_kv ? kv_bytes : 0ull);
    const Residency expert_residency =
        !request.expert_host_all && host_if_pinned > request.system_ram_bytes
            ? Residency::Mmap
            : Residency::HostPinned;
    table.entries.push_back({expert_residency, "experts", request.expert_host_bytes});
    table.entries.push_back({Residency::Mmap, "ple", request.ple_bytes});

    std::uint64_t gpu_total = 0;
    std::uint64_t host_total = 0;
    std::uint64_t mmap_total = 0;
    for (const auto& entry : table.entries) {
        if (entry.residency == Residency::Gpu) {
            gpu_total += entry.bytes;
        } else if (entry.residency == Residency::HostPinned) {
            host_total += entry.bytes;
        } else {
            mmap_total += entry.bytes;
        }
    }
    table.gpu_total_bytes = gpu_total;
    table.host_pinned_total_bytes = host_total;
    table.mmap_total_bytes = mmap_total;

    if (gpu_total > request.device_budget_bytes) {
        std::uint64_t running = 0;
        std::string_view budget = kGpuBudgetOrder[0];
        for (const char* name : kGpuBudgetOrder) {
            const auto* entry = find_entry(table, name);
            if (entry == nullptr) continue;
            running += entry->bytes;
            budget = name;
            if (running > request.device_budget_bytes) break;
        }
        return PlacementError{std::string(budget), gpu_total, request.device_budget_bytes};
    }
    if (host_total > request.system_ram_bytes) {
        return PlacementError{"host_pinned", host_total, request.system_ram_bytes};
    }
    return table;
}

std::string render_placement(const PlacementRequest& request,
                             const std::variant<PlacementTable, PlacementError>& result) {
    std::string out;
    out += "flash-next placement dry-run\n";

    if (const auto* error = std::get_if<PlacementError>(&result)) {
        out += "  verdict: OVERFLOW\n";
        out += "  budget: " + error->budget + "\n";
        out += "  required_bytes: " + std::to_string(error->required_bytes) + "\n";
        out += "  available_bytes: " + std::to_string(error->available_bytes) + "\n";
        out += "  " + error->message() + "\n";
        return out;
    }

    const auto& table = std::get<PlacementTable>(result);
    const std::uint64_t device_free =
        table.gpu_total_bytes <= request.device_budget_bytes
            ? request.device_budget_bytes - table.gpu_total_bytes
            : 0;
    const std::uint64_t host_free =
        table.host_pinned_total_bytes <= request.system_ram_bytes
            ? request.system_ram_bytes - table.host_pinned_total_bytes
            : 0;

    out += "  verdict: FIT\n";
    out += "  kv_pool_tokens: " + std::to_string(table.kv_pool_tokens) + "\n";
    out += "  device_budget_bytes: " + std::to_string(request.device_budget_bytes) + "\n";
    out += "    gpu_used_bytes: " + std::to_string(table.gpu_total_bytes) + "\n";
    out += "    gpu_free_bytes: " + std::to_string(device_free) + "\n";
    out += "  host_ram_bytes: " + std::to_string(request.system_ram_bytes) + "\n";
    out += "    host_pinned_used_bytes: " + std::to_string(table.host_pinned_total_bytes) + "\n";
    out += "    host_pinned_free_bytes: " + std::to_string(host_free) + "\n";
    out += "  mmap_total_bytes: " + std::to_string(table.mmap_total_bytes) +
           " (page-cache, never counted)\n";
    out += "  entries:\n";
    for (const auto& entry : table.entries) {
        out += "    " + entry.budget + "  " + std::string(residency_name(entry.residency)) +
               "  " + std::to_string(entry.bytes) + " bytes\n";
    }
    return out;
}

}  // namespace ninfer::targets::qwen3_8_flash_next
