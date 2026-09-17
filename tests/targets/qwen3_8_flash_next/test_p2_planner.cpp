// P2: startup planner -- pure byte arithmetic, no device access.
//
// Synthetic byte counts only (spec P2 test list):
//  1. backbone 12 GiB + expert window 2.5 GiB + KV 8 GiB + workspace 4 GiB -> OK
//  2. 262144-token BF16 QSA KV pool -> OK-or-fail; on failure the message
//     must name the KV budget
//  3. a 28 GiB PLE must appear as mmap, never in the GPU total
//  4. 58 GiB of experts must appear as host_pinned, never in the GPU total
//  5. a deterministic KV overflow names the kv_cache budget and is
//     reproducible (same input -> same output)

#include <cstdint>
#include <cstdio>
#include <string>
#include <variant>

#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen3_8_flash_next/planner.h>

namespace {

using FlashNextPackage = ninfer::targets::qwen3_8_flash_next::Package;
using Entry = ninfer::targets::qwen3_8_flash_next::PlacementEntry;
using Error = ninfer::targets::qwen3_8_flash_next::PlacementError;
using Request = ninfer::targets::qwen3_8_flash_next::PlacementRequest;
using Residency = ninfer::targets::qwen3_8_flash_next::Residency;
using Table = ninfer::targets::qwen3_8_flash_next::PlacementTable;
using ninfer::KvCacheStorage;
using ninfer::targets::qwen3_8_flash_next::plan_placement;

constexpr std::uint64_t GiB = 1ull << 30;
constexpr std::uint64_t MiB = 1ull << 20;

int failures = 0;

[[noreturn]] void fail(const std::string& message) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
}

void expect_true(bool condition, const std::string& what) {
    if (!condition) fail(what);
}

const Entry* entry_of(const Table& table, const char* budget) {
    for (const auto& entry : table.entries) {
        if (entry.budget == budget) return &entry;
    }
    fail(std::string("placement table has no '") + budget + "' entry");
}

}  // namespace

int main() {
    // KV model: 12 full/QSA attention layers x K+V x 2 KV heads x 256 head dim.
    expect_true(FlashNextPackage::kv_bytes_per_token(KvCacheStorage::BFloat16) == 24576,
                "BF16 kv_bytes_per_token != 24576");
    expect_true(FlashNextPackage::kv_bytes_per_token(KvCacheStorage::Fp8E4M3Row256) == 12288,
                "FP8 kv_bytes_per_token != 12288");
    expect_true(FlashNextPackage::kv_bytes_per_token(KvCacheStorage::Int8Group64) == 12288,
                "Int8 kv_bytes_per_token != 12288");

    // 1. The spec's OK case: everything fits the 32 GiB device budget.
    {
        Request request;
        request.device_budget_bytes = 32 * GiB;
        request.system_ram_bytes    = 64 * GiB;
        request.backbone_weight_bytes = 12 * GiB;
        request.expert_window_bytes   = 5 * MiB * 512;  // 2.5 GiB
        request.kv_bytes_per_token    = 8 * GiB / 1024;  // 1024 tokens -> 8 GiB KV
        request.max_context           = 1024;
        request.max_concurrency       = 1;
        request.workspace_bytes       = 4 * GiB;
        request.expert_host_bytes     = 0;
        request.ple_bytes             = 16 * MiB;

        const auto outcome = plan_placement(request);
        const auto* table = std::get_if<Table>(&outcome);
        expect_true(table != nullptr, "case 1: placement did not fit (want OK)");
        expect_true(table->kv_pool_tokens == 1024, "case 1: kv_pool_tokens != 1024");
        expect_true(entry_of(*table, "backbone")->bytes == 12 * GiB &&
                        entry_of(*table, "backbone")->residency == Residency::Gpu,
                    "case 1: backbone entry wrong");
        expect_true(entry_of(*table, "kv_cache")->bytes == 8 * GiB &&
                        entry_of(*table, "kv_cache")->residency == Residency::Gpu,
                    "case 1: kv_cache entry wrong");
        expect_true(entry_of(*table, "expert_window")->bytes == 5 * MiB * 512 &&
                        entry_of(*table, "expert_window")->residency == Residency::Gpu,
                    "case 1: expert_window entry wrong");
        expect_true(entry_of(*table, "workspace")->bytes == 4 * GiB &&
                        entry_of(*table, "workspace")->residency == Residency::Gpu,
                    "case 1: workspace entry wrong");
        expect_true(table->gpu_total_bytes == 12 * GiB + 8 * GiB + 5 * MiB * 512 + 4 * GiB,
                    "case 1: gpu_total_bytes wrong");
        expect_true(table->host_pinned_total_bytes == 0, "case 1: host_pinned total != 0");
        expect_true(table->mmap_total_bytes == 16 * MiB, "case 1: mmap total wrong");
    }

    // 2. 262144-token BF16 QSA KV pool at max_concurrency 1: OK-or-fail; on
    //    failure the overflowing budget must be the KV budget.
    {
        Request request;
        request.device_budget_bytes = 32 * GiB;
        request.system_ram_bytes    = 64 * GiB;
        request.backbone_weight_bytes = 12 * GiB;
        request.expert_window_bytes   = 5 * MiB * 512;  // 2.5 GiB
        request.kv_bytes_per_token    = FlashNextPackage::kv_bytes_per_token(KvCacheStorage::BFloat16);
        request.max_context           = 262144;
        request.max_concurrency       = 1;
        request.workspace_bytes       = 4 * GiB;
        request.expert_host_bytes     = 58 * GiB;
        request.ple_bytes             = 16 * MiB;

        const auto outcome = plan_placement(request);
        if (const auto* error = std::get_if<Error>(&outcome)) {
            expect_true(error->budget == "kv_cache",
                        "case 2: failure did not name the kv_cache budget");
            expect_true(error->required_bytes > error->available_bytes,
                        "case 2: required <= available in failure");
        } else {
            const Table& table = std::get<Table>(outcome);
            expect_true(table.kv_pool_tokens == 262144, "case 2: kv_pool_tokens != 262144");
            expect_true(table.gpu_total_bytes <= request.device_budget_bytes,
                        "case 2: gpu total exceeds budget in OK table");
        }
    }

    // 3. A 28 GiB PLE is mmap: it never enters the GPU or host-pinned totals.
    {
        Request request;
        request.device_budget_bytes = 8 * GiB;
        request.system_ram_bytes    = 64 * GiB;
        request.backbone_weight_bytes = 1 * GiB;
        request.kv_bytes_per_token    = 24576;
        request.max_context           = 16;
        request.max_concurrency       = 1;
        request.ple_bytes             = 28 * GiB;

        const auto outcome = plan_placement(request);
        const auto* table = std::get_if<Table>(&outcome);
        expect_true(table != nullptr, "case 3: placement did not fit (mmap must be free)");
        expect_true(entry_of(*table, "ple")->residency == Residency::Mmap,
                    "case 3: PLE is not mmap");
        expect_true(table->mmap_total_bytes == 28 * GiB, "case 3: mmap total wrong");
        expect_true(table->gpu_total_bytes < 28 * GiB,
                    "case 3: PLE leaked into the GPU total");
    }

    // 4. 58 GiB of experts are host_pinned: they never enter the GPU total.
    {
        Request request;
        request.device_budget_bytes = 8 * GiB;
        request.system_ram_bytes    = 64 * GiB;
        request.backbone_weight_bytes = 1 * GiB;
        request.kv_bytes_per_token    = 24576;
        request.max_context           = 16;
        request.max_concurrency       = 1;
        request.expert_host_bytes     = 58 * GiB;

        const auto outcome = plan_placement(request);
        const auto* table = std::get_if<Table>(&outcome);
        expect_true(table != nullptr, "case 4: placement did not fit (want OK at 64 GiB RAM)");
        expect_true(entry_of(*table, "experts")->residency == Residency::HostPinned,
                    "case 4: experts are not host_pinned");
        expect_true(table->host_pinned_total_bytes == 58 * GiB,
                    "case 4: host_pinned total wrong");
        expect_true(table->gpu_total_bytes < 58 * GiB,
                    "case 4: experts leaked into the GPU total");

        // The same experts do not fit a 32 GiB RAM budget.
        request.system_ram_bytes = 32 * GiB;
        const auto tight = plan_placement(request);
        const auto* error = std::get_if<Error>(&tight);
        expect_true(error != nullptr, "case 4b: 58 GiB experts fit 32 GiB RAM (want fail)");
        expect_true(error->budget == "host_pinned",
                    "case 4b: failure did not name the host_pinned budget");
        expect_true(error->required_bytes == 58 * GiB, "case 4b: required bytes wrong");
    }

    // 5. Deterministic KV overflow: 131072 x 8 pool at 24576 B/token pushes
    //    past the 32 GiB device budget; the named budget is kv_cache, and the
    //    second call reproduces the result exactly.
    {
        Request request;
        request.device_budget_bytes = 32 * GiB;
        request.system_ram_bytes    = 64 * GiB;
        request.backbone_weight_bytes = 12 * GiB;
        request.expert_window_bytes   = 5 * MiB * 512;  // 2.5 GiB
        request.kv_bytes_per_token    = FlashNextPackage::kv_bytes_per_token(KvCacheStorage::BFloat16);
        request.max_context           = 131072;
        request.max_concurrency       = 8;
        request.workspace_bytes       = 4 * GiB;

        const auto first = plan_placement(request);
        const auto* error = std::get_if<Error>(&first);
        expect_true(error != nullptr,
                    "case 5: 131072 x 8 KV pool fits 32 GiB (want overflow)");
        expect_true(error->budget == "kv_cache",
                    "case 5: overflow did not name the kv_cache budget");
        const std::uint64_t pool = 131072ull * 8;
        const std::uint64_t kv   = pool * 24576;
        expect_true(error->required_bytes == 12 * GiB + kv + 5 * MiB * 512 + 4 * GiB,
                    "case 5: required bytes wrong");
        expect_true(error->available_bytes == 32 * GiB, "case 5: available bytes wrong");
        expect_true(error->message().find("kv_cache") != std::string::npos,
                    "case 5: message does not name the kv_cache budget");

        const auto second = plan_placement(request);
        const auto* error2 = std::get_if<Error>(&second);
        expect_true(error2 != nullptr, "case 5b: second call did not fail");
        expect_true(error2->budget == error->budget &&
                        error2->required_bytes == error->required_bytes &&
                        error2->available_bytes == error->available_bytes,
                    "case 5b: planner is not deterministic");
    }

    // 6. GDN state at the P6 real formula: the 36 GDN layers (48 text layers
    //    minus the 12 full-attention ones) each hold a per-sequence FP32
    //    [128,128,48] recurrent state = 786,432 FP32 elements = 3,145,728 bytes.
    expect_true(FlashNextPackage::gdn_state_bytes_per_layer() ==
                    48ull * 128ull * 128ull * 4ull,
                "gdn_state_bytes_per_layer != 48 x 128 x 128 x 4 (3145728)");
    expect_true(FlashNextPackage::gdn_state_bytes(1) ==
                    36ull * 48ull * 128ull * 128ull * 4ull,
                "gdn_state_bytes(1) != 36 GDN layers x per-layer bytes");
    expect_true(FlashNextPackage::gdn_state_bytes(2) ==
                    2 * FlashNextPackage::gdn_state_bytes(1),
                "gdn_state_bytes not linear in max_concurrency");
    {
        const std::uint64_t gdn8 = FlashNextPackage::gdn_state_bytes(8);
        const std::uint64_t kv = 16ull * 24576ull;  // max_context x bytes/token
        Request request;
        request.device_budget_bytes = 1 * GiB + gdn8 + kv;
        request.system_ram_bytes    = 64 * GiB;
        request.backbone_weight_bytes = 1 * GiB;
        request.kv_bytes_per_token    = 24576;
        request.max_context           = 16;
        request.max_concurrency       = 1;
        request.gdn_state_bytes       = gdn8;

        const auto outcome = plan_placement(request);
        const auto* table = std::get_if<Table>(&outcome);
        expect_true(table != nullptr, "case 6: gdn_state placement did not fit (want OK)");
        expect_true(entry_of(*table, "gdn_state")->residency == Residency::Gpu,
                    "case 6: gdn_state is not GPU-resident");
        expect_true(entry_of(*table, "gdn_state")->bytes == gdn8,
                    "case 6: gdn_state entry bytes wrong");
        expect_true(table->gpu_total_bytes ==
                        1 * GiB + 16ull * 24576ull + gdn8,
                    "case 6: gpu_total_bytes missing gdn_state");
        expect_true(table->host_pinned_total_bytes == 0,
                    "case 6: gdn_state leaked into host-pinned");

        // One byte tighter: the deterministic overflow must name gdn_state.
        request.device_budget_bytes = 1 * GiB + gdn8 + kv - 1;
        const auto tight = plan_placement(request);
        const auto* error = std::get_if<Error>(&tight);
        expect_true(error != nullptr, "case 6b: tightening by 1 byte still fits");
        expect_true(error->budget == "gdn_state",
                    "case 6b: overflow did not name the gdn_state budget");
    }

    std::fprintf(stderr, "P2 planner: all checks passed\n");
    return 0;
}
