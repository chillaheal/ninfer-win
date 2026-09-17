// P11: 32 GB placement vs REAL inventory bytes.
//
// Before the GPU sees the real file, the planner must predict survival. The P0
// audit measured the flash_next residency byte sums; this test plugs those exact
// numbers into the P2 planner, driven by THIS machine's real budgets (32 GB
// card / 68.4 GB physical RAM).
//
//  T1 recommended config          -> GPU placement FITS the 32 GB card
//  T2 recommended config + REAL RAM -> host experts FAIL loud (host_pinned)
//  T3 spec 262144/bf16/MTP conc 1  -> GPU FITS this 32 GB card (the honest note:
//     the spec's "negative" does not overflow the GPU at conc 1; host RAM does)
//  T4 262144/bf16 conc 8          -> a genuine GPU KV overflow, FAILS loud and
//     deterministically (names kv_cache)
//  T5 CLI dry-run                 -> flash_next_placement.exe stdout == the
//     in-process render_placement (byte-for-byte snapshot)
//
// The spec's ~58 GB expert estimate is NOT used here; the measured ~69.4 GB
// (69,363,302,792 B) from the P0 audit is (agent_prompt.md P11 carry-forward).

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen3_8_flash_next/planner.h>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

using FlashNextPackage = ninfer::targets::qwen3_8_flash_next::Package;
using Entry            = ninfer::targets::qwen3_8_flash_next::PlacementEntry;
using Error            = ninfer::targets::qwen3_8_flash_next::PlacementError;
using Request          = ninfer::targets::qwen3_8_flash_next::PlacementRequest;
using Residency        = ninfer::targets::qwen3_8_flash_next::Residency;
using Table            = ninfer::targets::qwen3_8_flash_next::PlacementTable;
using Outcome          = std::variant<Table, Error>;
using ninfer::KvCacheStorage;
using ninfer::targets::qwen3_8_flash_next::plan_placement;
using ninfer::targets::qwen3_8_flash_next::render_placement;

constexpr std::uint64_t GiB = 1ull << 30;

// ---- real P0-audit byte sums (qwen3_8_flash_next-artifact-audit.md §5) --------
constexpr std::uint64_t kBackboneGpuBytes = 6021195832ull;      // gpu_resident (1504 objs)
constexpr std::uint64_t kExpertHostBytes  = 69363302792ull;     // host_experts (98 objs)
constexpr std::uint64_t kPleBytes         = 51840558936ull;     // mmap .ngram
// Per 512-expert layer: gate_up 943718404 + down 471859204 (uniform across the
// 48 text + 1 MTP layers). The expert-window bytes = window x this payload.
constexpr std::uint64_t kPerLayerExpertPayload = 1415577608ull;

// ---- this machine's real budgets (verified 2026-09-12) ----------------------
constexpr std::uint64_t kDeviceBudgetBytes = 34359738368ull;  // 32 GiB RTX 5090
constexpr std::uint64_t kSystemRamBytes    = 68384739328ull;  // physical RAM (63.67 GiB)

// A large-enough RAM budget to isolate the GPU dimension in T1/T3/T4.
constexpr std::uint64_t kLargeRamBytes = 512ull * GiB;

int failures = 0;

void fail(const std::string& message) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
}

void expect_true(bool condition, const std::string& what) {
    if (!condition) fail(what);
}

const Entry* entry_of(const Table& table, const char* budget) {
    for (const auto& entry : table.entries) {
        if (entry.budget == budget) return &entry;
    }
    fail(std::string("placement table has no '") + budget + "' entry");
    return nullptr;
}

std::filesystem::path tool_exe() {
    return std::filesystem::path(NINFER_SOURCE_DIR) / "build" / "tests" / "flash_next_placement.exe";
}

#if defined(_WIN32)
// Spawn the dry-run tool with `cmd` (full command line), capture stdout via an
// anonymous pipe (P9 pattern: a file handle on hStdOutput captured 0 bytes for
// the CLI, a pipe captures the full stream). Returns stdout + sets *exit_code.
std::string run_tool_capture(const std::wstring& cmd, int* exit_code) {
    *exit_code = -1;
    SECURITY_ATTRIBUTES sa{};
    sa.nLength      = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE read_pipe = nullptr, write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) return "";
    STARTUPINFOW si{};
    si.cb       = sizeof(si);
    si.dwFlags  = STARTF_USESTDHANDLES;
    si.hStdOutput = write_pipe;
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(nullptr, const_cast<wchar_t*>(cmd.c_str()), nullptr, nullptr,
                                   /*bInheritHandles=*/TRUE, 0, nullptr, nullptr, &si, &pi);
    CloseHandle(write_pipe);  // parent keeps only the read end
    if (!ok) {
        CloseHandle(read_pipe);
        return "";
    }
    std::string out;
    std::vector<char> buf(65536);
    for (;;) {
        DWORD got = 0;
        if (!ReadFile(read_pipe, buf.data(), static_cast<DWORD>(buf.size()), &got, nullptr) ||
            got == 0)
            break;
        out.append(buf.data(), static_cast<std::size_t>(got));
    }
    CloseHandle(read_pipe);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    *exit_code = static_cast<int>(code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return out;
}
#endif

// Build the recommended-config request (32768 ctx / conc 1 / fp8 / expert
// window 2 / GDN state for conc 1) with the real audit bytes. `system_ram` is
// the only knob that differs across T1 (large) and T2 (real).
Request recommended_request(std::uint64_t system_ram) {
    Request request;
    request.device_budget_bytes   = kDeviceBudgetBytes;
    request.system_ram_bytes      = system_ram;
    request.backbone_weight_bytes = kBackboneGpuBytes;
    request.expert_host_bytes     = kExpertHostBytes;
    request.ple_bytes             = kPleBytes;
    request.kv_bytes_per_token    = FlashNextPackage::kv_bytes_per_token(KvCacheStorage::Fp8E4M3Row256);
    request.max_context           = 32768;
    request.max_concurrency       = 1;
    request.expert_window_bytes   = 2 * kPerLayerExpertPayload;
    request.host_kv               = false;
    request.workspace_bytes       = 0;
    request.gdn_state_bytes       = FlashNextPackage::gdn_state_bytes(1);
    return request;
}

void check_t5_cli_snapshot() {
#if defined(_WIN32)
    const std::filesystem::path exe = tool_exe();
    if (!std::filesystem::exists(exe)) {
        fail("(T5) CLI dry-run exe missing: " + exe.string());
        return;
    }
    // Same overflow config as T2 (recommended + real RAM -> host_pinned).
    const Request request = recommended_request(kSystemRamBytes);
    const Outcome outcome = plan_placement(request);
    const std::string expected = render_placement(request, outcome);

    const std::wstring cmd =
        L"\"" + exe.wstring() + L"\" "
        L"--max-context 32768 --max-concurrency 1 --kv-dtype fp8 "
        L"--backbone-bytes " + std::to_wstring(kBackboneGpuBytes) + L" "
        L"--expert-host-bytes " + std::to_wstring(kExpertHostBytes) + L" "
        L"--ple-bytes " + std::to_wstring(kPleBytes) + L" "
        L"--expert-window-bytes " + std::to_wstring(2 * kPerLayerExpertPayload) + L" "
        L"--workspace-bytes 0 "
        L"--gdn-state-bytes " + std::to_wstring(FlashNextPackage::gdn_state_bytes(1)) + L" "
        L"--device-budget-bytes " + std::to_wstring(kDeviceBudgetBytes) + L" "
        L"--system-ram-bytes " + std::to_wstring(kSystemRamBytes);

    int rc = -1;
    const std::string actual = run_tool_capture(cmd, &rc);
    expect_true(rc == 3,
                "(T5) tool exit " + std::to_string(rc) + " (want 3 for overflow)");
    expect_true(actual == expected,
                "(T5) tool stdout != in-process render_placement\n--- tool ---\n" + actual +
                "\n--- expected ---\n" + expected);
    if (rc == 3 && actual == expected) {
        std::fprintf(stderr, "(T5) CLI dry-run snapshot OK (%zu bytes, exit 3)\n", actual.size());
    }
#else
    std::fprintf(stderr, "(T5) skipped (non-Windows)\n");
#endif
}

}  // namespace

int main() {
    // Sanity: the real GDN-state formula and KV bytes-per-token agree with the
    // values the recommended config relies on.
    expect_true(FlashNextPackage::kv_bytes_per_token(KvCacheStorage::Fp8E4M3Row256) == 12288,
                "FP8 kv_bytes_per_token != 12288");
    expect_true(FlashNextPackage::kv_bytes_per_token(KvCacheStorage::BFloat16) == 24576,
                "BF16 kv_bytes_per_token != 24576");
    expect_true(FlashNextPackage::gdn_state_bytes(1) == 113246208ull,
                "gdn_state_bytes(1) != 113246208");

    const std::uint64_t gdn1 = FlashNextPackage::gdn_state_bytes(1);
    const std::uint64_t window2 = 2 * kPerLayerExpertPayload;
    const std::uint64_t kv_fp8_32k = 32768ull * 12288ull;      // 402,653,184

    // T1: recommended config, large RAM (isolate GPU) -> FITS the 32 GB card.
    {
        const Request request = recommended_request(kLargeRamBytes);
        const auto outcome = plan_placement(request);
        const auto* table = std::get_if<Table>(&outcome);
        expect_true(table != nullptr, "(T1) recommended config did not fit the 32 GB GPU");
        if (table != nullptr) {
            const std::uint64_t want_gpu =
                kBackboneGpuBytes + kv_fp8_32k + window2 + 0 + gdn1;  // 9,368,250,440
            expect_true(table->kv_pool_tokens == 32768, "(T1) kv_pool_tokens != 32768");
            expect_true(table->gpu_total_bytes == want_gpu,
                        "(T1) gpu_total_bytes != " + std::to_string(want_gpu));
            expect_true(table->gpu_total_bytes <= kDeviceBudgetBytes,
                        "(T1) gpu total exceeds the 32 GiB budget");
            expect_true(table->host_pinned_total_bytes == kExpertHostBytes,
                        "(T1) host_pinned total != expert host bytes");
            expect_true(table->mmap_total_bytes == kPleBytes, "(T1) mmap total != PLE bytes");
            expect_true(entry_of(*table, "backbone")->residency == Residency::Gpu &&
                            entry_of(*table, "backbone")->bytes == kBackboneGpuBytes,
                        "(T1) backbone entry wrong");
            expect_true(entry_of(*table, "kv_cache")->residency == Residency::Gpu &&
                            entry_of(*table, "kv_cache")->bytes == kv_fp8_32k,
                        "(T1) kv_cache entry wrong");
            expect_true(entry_of(*table, "expert_window")->bytes == window2,
                        "(T1) expert_window entry wrong");
            expect_true(entry_of(*table, "gdn_state")->bytes == gdn1,
                        "(T1) gdn_state entry wrong");
            expect_true(entry_of(*table, "experts")->residency == Residency::HostPinned &&
                            entry_of(*table, "experts")->bytes == kExpertHostBytes,
                        "(T1) experts entry wrong");
            expect_true(entry_of(*table, "ple")->residency == Residency::Mmap,
                        "(T1) ple is not mmap");
        }
    }

    // T2: recommended config + REAL RAM -> host experts FAIL loud (host_pinned).
    // This is the binding finding: 69.36 GB experts exceed the 68.4 GB RAM, so
    // the literal "--expert-host-all" recommended flag does not fit on THIS
    // machine; the GPU does, the host does not.
    {
        const Request request = recommended_request(kSystemRamBytes);
        const auto outcome = plan_placement(request);
        const auto* error = std::get_if<Error>(&outcome);
        expect_true(error != nullptr,
                    "(T2) recommended config + real RAM fit (want host_pinned overflow)");
        if (error != nullptr) {
            expect_true(error->budget == "host_pinned",
                        "(T2) overflow did not name the host_pinned budget");
            expect_true(error->required_bytes == kExpertHostBytes,
                        "(T2) required bytes != expert host bytes");
            expect_true(error->available_bytes == kSystemRamBytes,
                        "(T2) available bytes != real RAM");
            const std::string msg = error->message();
            expect_true(msg.find("host_pinned") != std::string::npos,
                        "(T2) message does not name host_pinned");
            expect_true(msg.find(std::to_string(kExpertHostBytes)) != std::string::npos,
                        "(T2) message missing the required byte count");
            expect_true(msg.find(std::to_string(kSystemRamBytes)) != std::string::npos,
                        "(T2) message missing the available byte count");
        }
    }

    // T3: the spec's 262144/bf16/MTP config (conc 1) FITS the 32 GB GPU -- the
    // honest note that its real blocker on this machine is host RAM (T2), not
    // the GPU. MTP is already baked into the real backbone + expert byte sums.
    {
        Request request;
        request.device_budget_bytes   = kDeviceBudgetBytes;
        request.system_ram_bytes      = kLargeRamBytes;
        request.backbone_weight_bytes = kBackboneGpuBytes;
        request.expert_host_bytes     = kExpertHostBytes;
        request.ple_bytes             = kPleBytes;
        request.kv_bytes_per_token    = FlashNextPackage::kv_bytes_per_token(KvCacheStorage::BFloat16);
        request.max_context           = 262144;
        request.max_concurrency       = 1;
        request.expert_window_bytes   = window2;
        request.host_kv               = false;
        request.workspace_bytes       = 0;
        request.gdn_state_bytes       = gdn1;

        const auto outcome = plan_placement(request);
        const auto* table = std::get_if<Table>(&outcome);
        expect_true(table != nullptr, "(T3) 262144/bf16 conc1 did not fit the 32 GB GPU");
        if (table != nullptr) {
            const std::uint64_t kv = 262144ull * 24576ull;  // 6,442,450,944
            const std::uint64_t want_gpu = kBackboneGpuBytes + kv + window2 + 0 + gdn1;
            expect_true(table->kv_pool_tokens == 262144, "(T3) kv_pool_tokens != 262144");
            expect_true(table->gpu_total_bytes == want_gpu,
                        "(T3) gpu_total_bytes != " + std::to_string(want_gpu));
            expect_true(table->gpu_total_bytes <= kDeviceBudgetBytes,
                        "(T3) gpu total exceeds the 32 GiB budget");
        }
    }

    // T4: a genuine GPU KV overflow (262144/bf16 conc 8) FAILS loud and
    // deterministically, naming the kv_cache budget.
    {
        Request request;
        request.device_budget_bytes   = kDeviceBudgetBytes;
        request.system_ram_bytes      = kLargeRamBytes;
        request.backbone_weight_bytes = kBackboneGpuBytes;
        request.expert_host_bytes     = kExpertHostBytes;
        request.ple_bytes             = kPleBytes;
        request.kv_bytes_per_token    = FlashNextPackage::kv_bytes_per_token(KvCacheStorage::BFloat16);
        request.max_context           = 262144;
        request.max_concurrency       = 8;
        request.expert_window_bytes   = window2;
        request.host_kv               = false;
        request.workspace_bytes       = 0;
        request.gdn_state_bytes       = FlashNextPackage::gdn_state_bytes(8);

        const std::uint64_t kv = 262144ull * 8 * 24576ull;               // 51,539,607,552
        const std::uint64_t gdn8 = FlashNextPackage::gdn_state_bytes(8);  // 905,969,664
        const std::uint64_t want_required = kBackboneGpuBytes + kv + window2 + 0 + gdn8;

        const auto first = plan_placement(request);
        const auto* error = std::get_if<Error>(&first);
        expect_true(error != nullptr, "(T4) 262144/bf16 conc8 fit the 32 GB GPU (want overflow)");
        if (error != nullptr) {
            expect_true(error->budget == "kv_cache",
                        "(T4) overflow did not name the kv_cache budget");
            expect_true(error->required_bytes == want_required,
                        "(T4) required bytes != " + std::to_string(want_required));
            expect_true(error->available_bytes == kDeviceBudgetBytes,
                        "(T4) available bytes != 32 GiB");
            expect_true(error->message().find("kv_cache") != std::string::npos,
                        "(T4) message does not name kv_cache");
        }

        const auto second = plan_placement(request);
        const auto* error2 = std::get_if<Error>(&second);
        expect_true(error2 != nullptr, "(T4b) second call did not fail");
        if (error != nullptr && error2 != nullptr) {
            expect_true(error2->budget == error->budget &&
                            error2->required_bytes == error->required_bytes &&
                            error2->available_bytes == error->available_bytes,
                        "(T4b) planner is not deterministic");
        }
    }

    // T6: the real loader's mmap-experts path at 100k context. expert_host_all=false
    // -> the 69.36 GB expert set does NOT fit this machine's 68.4 GB RAM, so the
    // planner spills it to the SSD/mmap tier (page-cache) instead of overflowing
    // host_pinned; the GPU carries the 100k KV. This is the config that makes 100k
    // usable, and the planner must predict FIT.
    {
        Request request;
        request.device_budget_bytes   = kDeviceBudgetBytes;
        request.system_ram_bytes      = kSystemRamBytes;
        request.backbone_weight_bytes = kBackboneGpuBytes;
        request.expert_host_bytes     = kExpertHostBytes;
        request.ple_bytes             = kPleBytes;
        request.kv_bytes_per_token    = FlashNextPackage::kv_bytes_per_token(KvCacheStorage::Fp8E4M3Row256);
        request.max_context           = 100000;
        request.max_concurrency       = 1;
        request.expert_window_bytes   = window2;
        request.host_kv               = false;
        request.expert_host_all       = false;  // the real loader mmaps the expert set
        request.workspace_bytes       = 0;
        request.gdn_state_bytes       = gdn1;

        const auto outcome = plan_placement(request);
        const auto* table = std::get_if<Table>(&outcome);
        expect_true(table != nullptr, "(T6) 100k mmap-experts did not fit (want FIT)");
        if (table != nullptr) {
            const std::uint64_t kv = 100000ull * 12288ull;  // 1,228,800,000
            const std::uint64_t want_gpu = kBackboneGpuBytes + kv + window2 + 0 + gdn1;
            expect_true(table->kv_pool_tokens == 100000, "(T6) kv_pool_tokens != 100000");
            expect_true(table->gpu_total_bytes == want_gpu,
                        "(T6) gpu_total_bytes != " + std::to_string(want_gpu));
            expect_true(table->gpu_total_bytes <= kDeviceBudgetBytes,
                        "(T6) gpu total exceeds the 32 GiB budget");
            expect_true(table->host_pinned_total_bytes == 0,
                        "(T6) host_pinned total != 0 (experts should have spilled to mmap)");
            expect_true(table->mmap_total_bytes == kExpertHostBytes + kPleBytes,
                        "(T6) mmap total != experts + ple");
            expect_true(entry_of(*table, "experts")->residency == Residency::Mmap,
                        "(T6) experts not on the mmap/SSD tier");
            expect_true(entry_of(*table, "ple")->residency == Residency::Mmap,
                        "(T6) ple not on the mmap/SSD tier");
        }
    }

    // T7: budget-driven spill in the other direction -- on a machine with enough
    // RAM the mmap-experts path puts the full set in RAM (host_pinned) rather than
    // spilling to SSD. This is the "dynamic" dimension: the residency adapts to the
    // RAM budget (RAM first, SSD as the last resort).
    {
        Request request;
        request.device_budget_bytes   = kDeviceBudgetBytes;
        request.system_ram_bytes      = kLargeRamBytes;  // 512 GiB, plenty for 69.36 GB
        request.backbone_weight_bytes = kBackboneGpuBytes;
        request.expert_host_bytes     = kExpertHostBytes;
        request.ple_bytes             = kPleBytes;
        request.kv_bytes_per_token    = FlashNextPackage::kv_bytes_per_token(KvCacheStorage::Fp8E4M3Row256);
        request.max_context           = 32768;
        request.max_concurrency       = 1;
        request.expert_window_bytes   = window2;
        request.host_kv               = false;
        request.expert_host_all       = false;
        request.workspace_bytes       = 0;
        request.gdn_state_bytes       = gdn1;

        const auto outcome = plan_placement(request);
        const auto* table = std::get_if<Table>(&outcome);
        expect_true(table != nullptr, "(T7) large-RAM mmap-experts did not fit (want FIT)");
        if (table != nullptr) {
            expect_true(entry_of(*table, "experts")->residency == Residency::HostPinned,
                        "(T7) experts not in RAM on a large-RAM machine (want host_pinned)");
            expect_true(table->host_pinned_total_bytes == kExpertHostBytes,
                        "(T7) host_pinned total != expert bytes");
            expect_true(table->mmap_total_bytes == kPleBytes,
                        "(T7) mmap total != ple only (experts should be in RAM)");
        }
    }

    check_t5_cli_snapshot();

    if (failures == 0) {
        std::fprintf(stderr, "P11 placement: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "P11 placement: %d failure(s)\n", failures);
    return 1;
}
