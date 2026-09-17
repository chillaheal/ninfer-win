#pragma once
// Flash-Next Package: identity + the runtime family contract (P9).
//
// Keeps the P2 planning surface (resolve_weights, per-token KV / GDN-state
// byte planning, the budget `plan` shell) and adds the non-templated family
// type aliases + the load/run factories the generic engine
// (targets::registry + runtime::EngineCore) needs to materialize the mini
// model and drive its first forward. The family types live in
// impl/runtime/family.h (namespace ninfer::targets::qwen3_8_flash_next); the
// Package re-exports them under the alias set EngineCore<Instance> derives
// its working types from (mirroring qwen3_6_27b::Package).

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>

#include "ninfer/types.h"
#include "ninfer/targets/qwen3_8_flash_next/binder.h"
#include "ninfer/targets/qwen3_8_flash_next/planner.h"
#include "targets/qwen3_8_flash_next/impl/runtime/family.h"

namespace ninfer {
struct DeviceContext;
}  // namespace ninfer

namespace ninfer::artifact {
struct ArtifactIdentity;
}  // namespace ninfer::artifact

namespace ninfer::targets::qwen3_8_flash_next {

class LoadedModel;  // impl/load/bindings.h (self-materializes the mini .ninfer)

struct Package {
    static constexpr std::string_view model_id   = "qwen3.8-flash-next";
    static constexpr std::string_view weights_id = "nvfp4";
    static constexpr std::string_view target_key = "qwen3_8_flash_next";

    using WeightsProfile = qwen3_8_flash_next::WeightsProfile;

    // Family contract aliases (EngineCore / ResourceManager / MaterializationPlanner
    // / RequestRecord derive their working types from Package through these).
    using Program                   = qwen3_8_flash_next::Program;
    using SequencePlanner           = qwen3_8_flash_next::SequencePlanner;
    using SequencePlan              = qwen3_8_flash_next::SequencePlan;
    using RequestBasePlan           = qwen3_8_flash_next::RequestBasePlan;
    using AdmissionCandidate        = qwen3_8_flash_next::AdmissionCandidate;
    using ResourcePlan              = qwen3_8_flash_next::ResourcePlan;
    using PersistentBackfillProof   = qwen3_8_flash_next::PersistentBackfillProof;
    using SequenceHandle            = qwen3_8_flash_next::SequenceHandle;
    using ContinuationHandle        = qwen3_8_flash_next::ContinuationHandle;
    using SharedPrefixHandle        = qwen3_8_flash_next::SharedPrefixHandle;
    using CaptureOffer              = qwen3_8_flash_next::CaptureOffer;
    using CacheSessionKey           = qwen3_8_flash_next::PreparedSessionKey;
    using ContinuationSummary       = qwen3_8_flash_next::ContinuationSummary;
    using SharedPrefixSummary       = qwen3_8_flash_next::SharedPrefixSummary;
    using PressurePlanningSession   = qwen3_8_flash_next::PressurePlanningSession;
    using PressureTargetHandle      = qwen3_8_flash_next::PressureTargetHandle;
    using MaterializationResult     = qwen3_8_flash_next::MaterializationResult;
    using ContextTransactionProgress = qwen3_8_flash_next::ContextTransactionProgress;
    using CaptureAssessment         = qwen3_8_flash_next::CaptureAssessment;
    using ActiveCaptureResult       = qwen3_8_flash_next::ActiveCaptureResult;
    using PendingBatch              = qwen3_8_flash_next::PendingBatch;
    using StartResult               = qwen3_8_flash_next::StartResult;
    using PrefillProgress           = qwen3_8_flash_next::PrefillProgress;
    using CommitResult              = qwen3_8_flash_next::CommitResult;
    using DiscardResult             = qwen3_8_flash_next::DiscardResult;
    using FinishResult              = qwen3_8_flash_next::FinishResult;
    using AbortResult               = qwen3_8_flash_next::AbortResult;
    using ReleaseResult             = qwen3_8_flash_next::ReleaseResult;

    // Runtime family objects.
    using LoadedModel   = qwen3_8_flash_next::LoadedModel;
    using Frontend      = qwen3_8_flash_next::Frontend;
    using PreparedPrompt   = qwen3_8_flash_next::PreparedPrompt;
    using OutputSession    = qwen3_8_flash_next::OutputSession;
    using PublishedOutput  = qwen3_8_flash_next::PublishedOutput;

    // P9 factories.
    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static std::unique_ptr<LoadedModel> construct_loaded_model(
        const std::filesystem::path& artifact_path, const std::filesystem::path& ngram_path,
        const std::filesystem::path& archive_path, DeviceContext& device);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model,
                                                const EngineOptions& options);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program> create_program(const LoadedModel& model,
                                                                 SequencePlan&& plan,
                                                                 DeviceContext& device);

    // Throws std::runtime_error for any other (model_id, weights_id) pair,
    // mirroring the 27B resolve_weights error contract.
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);

    // Per-token KV bytes for the 12 full/QSA attention layers (GQA: 2 KV
    // heads x 256 head dim): BF16 -> 24576, Int8/FP8 -> 12288. The scale-plane
    // overhead of the group/row-compressed storages is a P7 paged-KV
    // refinement of this planning model.
    [[nodiscard]] static std::uint64_t kv_bytes_per_token(KvCacheStorage storage);

    // Per-layer, per-sequence GDN (gated delta net) recurrent state bytes:
    // FP32 [128, 128, 48 value heads] = 3,145,728 bytes (P6 real formula; the
    // gated_delta_net Op's state layout at the production head counts).
    [[nodiscard]] static std::uint64_t gdn_state_bytes_per_layer();

    // Total GDN state across all 36 GDN layers (48 text layers minus the 12
    // full-attention ones) for `max_concurrency` concurrent sequences.
    [[nodiscard]] static std::uint64_t gdn_state_bytes(std::uint32_t max_concurrency);

    // P2 shell: bind the artifact (lazy handles, planned allocations only) and
    // plan the placement against the given budgets. No device access, no page
    // commit. Throws std::runtime_error (message names the overflowing budget)
    // when the placement does not fit.
    struct Shell {
        BindResult binder;
        PlacementTable placement;
    };

    [[nodiscard]] static Shell plan(const std::filesystem::path& artifact_path,
                                    const std::filesystem::path& ngram_path,
                                    const EngineOptions& options,
                                    std::uint64_t device_budget_bytes,
                                    std::uint64_t system_ram_bytes,
                                    std::uint64_t expert_window_bytes,
                                    bool host_kv,
                                    std::uint64_t workspace_bytes);
};

}  // namespace ninfer::targets::qwen3_8_flash_next
