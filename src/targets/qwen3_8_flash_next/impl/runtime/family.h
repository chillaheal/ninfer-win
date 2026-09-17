#pragma once

// Flash-Next (qwen3.8-flash-next) family contract types — P9 v1.
//
// Non-templated, family-owned. Mirrors the qwen3_6 export/runtime.h shapes without
// instantiating any qwen3_6 family class. v1 scope: no captures (CaptureOffer is never
// emitted), no engine-side catalog (finish always returns Released, no continuation is
// published), prefix reuse is purely Program-internal. v1-unreachable Program methods
// exist with exact signatures and throw std::logic_error with a v1-scope note.

#include "ninfer/ops/qsa_indexer_k_append.h"
#include "ninfer/types.h"
#include "runtime/contract/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer {
struct DeviceContext;
} // namespace ninfer

namespace ninfer::runtime {
struct ContextMachineCostModel;
} // namespace ninfer::runtime

namespace ninfer::targets::qwen3_8_flash_next {

enum class WeightsProfile : std::uint8_t {
    Nvfp4,
};

namespace detail {
struct ProgramImpl;
struct SequencePlanImpl;
struct SequencePlannerImpl;
struct PressurePlanningSessionImpl
    {};  // v1: empty. Pressure planning is unreachable (see the "v1-unreachable"
         // note at PressurePlanningSession); the complete (empty) type keeps the
         // session's unique_ptr<Impl> destructor well-formed in every TU.
struct RuntimeContractAccess;
} // namespace detail

// ---------------------------------------------------------------------------
// Checkpoint / summary value types
// ---------------------------------------------------------------------------

struct PrefixShortlistKey {
    std::uint64_t digest     = 0;
    std::uint32_t frontier   = 0;
    std::uint32_t identity_tag = 0;

    bool operator==(const PrefixShortlistKey&) const noexcept = default;
};

struct TargetKVRequirement {
    std::uint32_t main_frontier    = 0;
    std::uint32_t backend_frontier = 0;
    std::uint32_t main_pages       = 0;
    std::uint32_t backend_pages    = 0;

    bool operator==(const TargetKVRequirement&) const noexcept = default;
};

struct CheckpointSummary {
    runtime::CheckpointRef ref;
    runtime::CheckpointScope scope = runtime::CheckpointScope::Private;
    PrefixShortlistKey shortlist_key;
    runtime::ReplicaResidency state_residency = runtime::ReplicaResidency::DeviceOnly;
    TargetKVRequirement required_kv;
    runtime::PrefillWork rebuild_work;

    bool operator==(const CheckpointSummary&) const noexcept = default;
};

struct ContinuationSummary {
    std::optional<CheckpointSummary> endpoint;
    std::optional<CheckpointSummary> rewrite;
    std::vector<CheckpointSummary> long_anchors;
    std::uint32_t active_references = 0;

    bool operator==(const ContinuationSummary&) const noexcept = default;
};

struct SharedPrefixSummary {
    CheckpointSummary checkpoint;
    std::uint32_t active_references = 0;

    bool operator==(const SharedPrefixSummary&) const noexcept = default;
};

struct PhysicalUsageSnapshot {
    std::uint64_t resource_revision       = 0;
    std::uint32_t device_state_slots      = 0;
    std::uint32_t host_state_slots        = 0;
    std::uint32_t device_main_kv_pages    = 0;
    std::uint32_t device_backend_kv_pages = 0;
    std::size_t host_kv_bytes             = 0;

    bool operator==(const PhysicalUsageSnapshot&) const noexcept = default;
};

enum class TextPhase : std::uint8_t {
    Prefill,
    Verify,
};

enum class CaptureStatePlacement : std::uint8_t {
    DeviceFork,
    HostSnapshot,
};

inline constexpr std::size_t kPreparedSessionKeyCapacity = kMaximumContextCacheSessionKeyBytes;

struct PreparedSessionKey {
    std::uint16_t size = 0;
    std::array<char, kPreparedSessionKeyCapacity> bytes{};

    [[nodiscard]] std::string_view view() const noexcept { return {bytes.data(), size}; }

    bool operator==(const PreparedSessionKey&) const noexcept = default;
};

struct PreparedCacheOpportunity {
    PromptCacheMarkerKind kind = PromptCacheMarkerKind::SharedStablePrefix;
    std::uint32_t frontier     = 0;
    std::uint32_t input_order  = 0;

    bool operator==(const PreparedCacheOpportunity&) const noexcept = default;
};

struct PreparedContextCache {
    std::optional<PreparedSessionKey> session_key;
    runtime::RetentionClass retention = runtime::RetentionClass::RecentPrivate;
    std::vector<PreparedCacheOpportunity> opportunities;
    bool update_session_index = true;
};

// ---------------------------------------------------------------------------
// PreparedPrompt (v1: stub-tokenized token ids; real tokenizer in P10)
// ---------------------------------------------------------------------------

struct PreparedPrompt {
    PreparedPrompt() noexcept              = default;
    PreparedPrompt(PreparedPrompt&&) noexcept = default;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept = default;
    PreparedPrompt(const PreparedPrompt&)           = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept { return summary_; }
    [[nodiscard]] const PromptPreparationStats&
    preparation_stats() const noexcept {
        return preparation_;
    }
    [[nodiscard]] const std::vector<TokenId>& token_ids() const noexcept { return token_ids_; }

    PromptSummary summary_;
    PromptPreparationStats preparation_;
    std::vector<TokenId> token_ids_;
};

// ---------------------------------------------------------------------------
// PublishedOutput / OutputSession
// ---------------------------------------------------------------------------

class PublishedOutput {
public:
    PublishedOutput()                                  = default;
    PublishedOutput(const PublishedOutput&)            = default;
    PublishedOutput& operator=(const PublishedOutput&) = default;
    PublishedOutput(PublishedOutput&&) noexcept        = default;
    PublishedOutput& operator=(PublishedOutput&&) noexcept = default;

    [[nodiscard]] bool empty() const noexcept { return values_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

    [[nodiscard]] auto begin() noexcept { return values_.begin(); }
    [[nodiscard]] auto end() noexcept { return values_.end(); }
    [[nodiscard]] auto begin() const noexcept { return values_.begin(); }
    [[nodiscard]] auto end() const noexcept { return values_.end(); }

    [[nodiscard]] OutputDelta& back() noexcept { return values_.back(); }
    [[nodiscard]] const OutputDelta& back() const noexcept { return values_.back(); }

    void clear() noexcept { values_.clear(); }
    void push_back(OutputDelta value) { values_.push_back(std::move(value)); }

private:
    std::vector<OutputDelta> values_;
};

class OutputSession {
public:
    OutputSession() noexcept = default;
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;
    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    [[nodiscard]] runtime::OutputDecision
    preview_model(std::span<const TokenId> tokens, std::uint32_t total_budget_remaining,
                  FinishReason limit_reason);
    [[nodiscard]] std::uint32_t
    model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept;
    [[nodiscard]] std::span<const TokenId> pending_control_tokens() const noexcept;
    [[nodiscard]] runtime::OutputDecision
    preview_control(std::span<const TokenId> tokens, std::uint32_t total_budget_remaining);
    void validate_generation_capacity(std::uint32_t effective_output_tokens) const;
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview() noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    friend class Frontend;
};

// ---------------------------------------------------------------------------
// Frontend (v1 stub tokenizer: UTF-8 bytes -> token ids = byte values, vocab 256)
// ---------------------------------------------------------------------------

struct FrontendOptions {
    std::uint32_t max_context = 2048;
};

class Frontend {
public:
    Frontend()                                  = default;
    Frontend(const Frontend&)                   = default;
    Frontend& operator=(const Frontend&)        = default;
    Frontend(Frontend&&) noexcept               = default;
    Frontend& operator=(Frontend&&) noexcept    = default;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt
    prepare_tokens(std::vector<TokenId> token_ids, bool allow_prefix_identity = true) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] OutputSession
    make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                        const OutputOptions& output            = {},
                        const ThinkingControlOptions& thinking = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;

private:
    class Impl;
    std::shared_ptr<const Impl> impl_;

    friend Frontend make_frontend(FrontendOptions options);
    friend Frontend make_real_frontend(std::string tokenizer_json,
                                       std::string tokenizer_config_json,
                                       std::string generation_config_json,
                                       std::string chat_template_jinja, FrontendOptions options);
};

[[nodiscard]] Frontend make_frontend(FrontendOptions options);
[[nodiscard]] Frontend make_real_frontend(std::string tokenizer_json,
                                          std::string tokenizer_config_json,
                                          std::string generation_config_json,
                                          std::string chat_template_jinja, FrontendOptions options);

// ---------------------------------------------------------------------------
// SequencePlan / SequencePlanner
// ---------------------------------------------------------------------------

class SequencePlan {
public:
    SequencePlan() noexcept             = default;
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    SequencePlan(const SequencePlan&)           = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;
    ~SequencePlan();

    [[nodiscard]] std::uint32_t capacity() const;
    [[nodiscard]] std::uint32_t kv_capacity() const;
    [[nodiscard]] std::uint32_t max_concurrency() const;
    [[nodiscard]] std::size_t device_reservation_bytes() const;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const;

    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl> impl_;
};

class SequencePlanner {
public:
    SequencePlanner() noexcept              = default;
    SequencePlanner(SequencePlanner&&) noexcept = default;
    SequencePlanner& operator=(SequencePlanner&&) noexcept = default;
    SequencePlanner(const SequencePlanner&)           = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;
    ~SequencePlanner();

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const;
    [[nodiscard]] SequencePlan finalize(std::uint32_t main_page_groups) &&;

    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl> impl_;

    friend SequencePlanner
    make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                          WeightsProfile weights_profile);
};

// ---------------------------------------------------------------------------
// Admission / resource planning
// ---------------------------------------------------------------------------

struct RequestBasePlan {
    RequestBasePlan() noexcept             = default;
    RequestBasePlan(RequestBasePlan&&) noexcept = default;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept = default;
    RequestBasePlan(const RequestBasePlan&)           = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept { return summary_; }
    [[nodiscard]] const PreparedContextCache&
    context_cache() const noexcept {
        return context_cache_;
    }
    [[nodiscard]] std::optional<PrefixShortlistKey>
    prefix_shortlist_key(std::uint32_t) const noexcept;

    runtime::RequestPlanSummary summary_;
    PreparedContextCache context_cache_;

    // v1 prefix-reuse key material: FNV-1a 64 digests over the first f prompt token
    // ids (f = 1..8) + the constant identity tag. Computed in plan_request.
    std::array<std::uint64_t, 8> prefix_digests_{};
    std::uint32_t prefix_identity_tag_ = 0;
};

struct AdmissionCandidate {
    AdmissionCandidate() noexcept               = default;
    AdmissionCandidate(AdmissionCandidate&&) noexcept = default;
    AdmissionCandidate& operator=(AdmissionCandidate&&) noexcept = default;
    AdmissionCandidate(const AdmissionCandidate&)            = delete;
    AdmissionCandidate& operator=(const AdmissionCandidate&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept { return summary_; }
    [[nodiscard]] const runtime::IdentityMaterializationAssessment&
    identity_assessment() const noexcept {
        return identity_;
    }

    runtime::RequestPlanSummary summary_;
    runtime::IdentityMaterializationAssessment identity_;
};

class ResourcePlan {
public:
    ResourcePlan(ResourcePlan&&) noexcept = default;
    ResourcePlan& operator=(ResourcePlan&&) noexcept = default;
    ResourcePlan(const ResourcePlan&)            = delete;
    ResourcePlan& operator=(const ResourcePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary&
    summary() const noexcept {
        return admission_.summary();
    }
    [[nodiscard]] bool needs_transfer() const noexcept { return needs_transfer_; }
    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision_; }

private:
    ResourcePlan(AdmissionCandidate&& admission, std::uint64_t revision, bool needs_transfer)
        noexcept;

    AdmissionCandidate admission_;
    std::uint64_t revision_     = 0;
    bool needs_transfer_        = false;

    friend class Program;
    friend class PressurePlanningSession;
};

class PersistentBackfillProof {
public:
    PersistentBackfillProof(PersistentBackfillProof&&) noexcept = default;
    PersistentBackfillProof& operator=(PersistentBackfillProof&&) noexcept = default;
    PersistentBackfillProof(const PersistentBackfillProof&)            = delete;
    PersistentBackfillProof& operator=(const PersistentBackfillProof&) = delete;

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision_; }

private:
    explicit PersistentBackfillProof(std::uint64_t revision) noexcept : revision_(revision) {}

    std::uint64_t revision_ = 0;

    friend class Program;
};

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

struct SequenceHandle {
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    bool operator==(const SequenceHandle&) const noexcept = default;

    friend struct detail::RuntimeContractAccess;
};

struct ContinuationHandle {
    ContinuationHandle() noexcept = default;
    ContinuationHandle(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}
    ContinuationHandle(ContinuationHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}
    ContinuationHandle& operator=(ContinuationHandle&&) = delete;
    ContinuationHandle(const ContinuationHandle&)            = delete;
    ContinuationHandle& operator=(const ContinuationHandle&) = delete;

    const void* owner_      = nullptr;
    std::uint32_t index_    = 0;
    std::uint64_t generation_ = 0;

    friend struct detail::RuntimeContractAccess;
};

struct SharedPrefixHandle {
    SharedPrefixHandle() noexcept = default;
    SharedPrefixHandle(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}
    SharedPrefixHandle(SharedPrefixHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}
    SharedPrefixHandle& operator=(SharedPrefixHandle&&) = delete;
    SharedPrefixHandle(const SharedPrefixHandle&)            = delete;
    SharedPrefixHandle& operator=(const SharedPrefixHandle&) = delete;

    const void* owner_      = nullptr;
    std::uint32_t index_    = 0;
    std::uint64_t generation_ = 0;

    friend struct detail::RuntimeContractAccess;
};

// ---------------------------------------------------------------------------
// Pressure planning (v1-unreachable; exact signatures, bodies throw)
// ---------------------------------------------------------------------------

struct PressureTargetHandle {
    const void* session_  = nullptr;
    std::uint32_t generation_ = 0;
    std::uint32_t index_      = 0;

    bool operator==(const PressureTargetHandle&) const = default;

    friend class Program;
    friend class PressurePlanningSession;
};

class PreparedPressureExpansion {
public:
    PreparedPressureExpansion(PreparedPressureExpansion&&) noexcept = default;
    PreparedPressureExpansion&
    operator=(PreparedPressureExpansion&&) noexcept = default;
    PreparedPressureExpansion(const PreparedPressureExpansion&) = delete;
    PreparedPressureExpansion& operator=(const PreparedPressureExpansion&) = delete;

    [[nodiscard]] std::uint32_t new_canonical_count() const noexcept {
        return new_canonical_count_;
    }

private:
    PreparedPressureExpansion(const void* session, std::uint32_t session_generation,
                              std::uint32_t scratch_generation, std::uint32_t parent_index,
                              std::uint32_t new_canonical_count) noexcept
        : session_(session), session_generation_(session_generation),
          scratch_generation_(scratch_generation), parent_index_(parent_index),
          new_canonical_count_(new_canonical_count) {}

    const void* session_        = nullptr;
    std::uint32_t session_generation_ = 0;
    std::uint32_t scratch_generation_ = 0;
    std::uint32_t parent_index_     = 0;
    std::uint32_t new_canonical_count_ = 0;

    friend class PressurePlanningSession;
};

struct PressureExpansionView {
    std::span<const PressureTargetHandle> children;
    std::uint32_t new_canonical_count = 0;
};

class PressurePlanningSession {
public:
    PressurePlanningSession() noexcept = default;
    PressurePlanningSession(PressurePlanningSession&&) noexcept = default;
    PressurePlanningSession& operator=(PressurePlanningSession&&) noexcept = default;
    PressurePlanningSession(const PressurePlanningSession&) = delete;
    PressurePlanningSession& operator=(const PressurePlanningSession&) = delete;
    ~PressurePlanningSession();

    [[nodiscard]] PressureTargetHandle
    identity_target(const AdmissionCandidate& candidate) const;
    [[nodiscard]] PressureTargetHandle root_maximal_target(const AdmissionCandidate& candidate);
    [[nodiscard]] runtime::PressureTargetAssessment assess(PressureTargetHandle target) const;
    [[nodiscard]] PreparedPressureExpansion prepare_expansion(PressureTargetHandle target);
    [[nodiscard]] PressureExpansionView commit_expansion(PreparedPressureExpansion&& prepared);
    void discard_expansion(PreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] std::optional<ResourcePlan>
    seal(PressureTargetHandle target, const PreparedPrompt& prompt);

private:
    explicit PressurePlanningSession(std::unique_ptr<detail::PressurePlanningSessionImpl> impl)
        noexcept;

    std::unique_ptr<detail::PressurePlanningSessionImpl> impl_;

    friend class Program;
};

// ---------------------------------------------------------------------------
// Capture / execution / commit results
// ---------------------------------------------------------------------------

struct CaptureOffer {
    CaptureOffer() noexcept = default;
    CaptureOffer(const void* owner, runtime::LaneId lane, std::uint64_t epoch,
                 std::uint64_t id) noexcept
        : owner_(owner), lane_(lane), epoch_(epoch), id_(id) {}
    CaptureOffer(CaptureOffer&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), lane_(other.lane_),
          epoch_(std::exchange(other.epoch_, 0)), id_(std::exchange(other.id_, 0)) {}
    CaptureOffer& operator=(CaptureOffer&&) = delete;
    CaptureOffer(const CaptureOffer&)            = delete;
    CaptureOffer& operator=(const CaptureOffer&) = delete;

    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;
    std::uint64_t id_ = 0;

    friend struct detail::RuntimeContractAccess;
};

class PendingBatch {
public:
    PendingBatch() noexcept = default;
    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          transaction_(std::exchange(other.transaction_, 0)), rows_(other.rows_),
          row_count_(std::exchange(other.row_count_, 0)), tokens_(other.tokens_),
          row_counts_(other.row_counts_), row_stride_(std::exchange(other.row_stride_, 0)),
          timing_(other.timing_) {
        other.tokens_     = {};
        other.row_counts_ = {};
        other.row_stride_ = 0;
        other.timing_     = {};
    }
    PendingBatch& operator=(PendingBatch&&) = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }
    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept {
        return row_counts_;
    }
    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }
    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept {
        return timing_;
    }

private:
    const void* owner_ = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_{};
    std::span<const std::int32_t> row_counts_{};
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_{};

    friend struct detail::RuntimeContractAccess;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing{};
    std::optional<PendingBatch> pending;
    std::optional<CaptureOffer> capture;
};

struct CaptureAssessment {
    PrefixShortlistKey shortlist_key;
    runtime::PrefillWork protected_rebuild_work;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::vector<runtime::CheckpointRef> private_replacement_candidates;
    std::uint32_t frontier                  = 0;
    bool publishes_private                  = false;
    bool publishes_shared                   = false;
    bool needs_transfer                     = false;
    bool recycles_private_state             = false;
    CaptureStatePlacement state_placement = CaptureStatePlacement::DeviceFork;
};

struct SharedPrefixPublication {
    SharedPrefixHandle handle;
    SharedPrefixSummary summary;
};

struct ActiveCaptureResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    bool capacity_preparation_committed = false;
    ContinuationSummary active_summary;
    std::optional<SharedPrefixPublication> shared;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

struct StartResult {
    SequenceHandle sequence;
};

struct MaterializationVictimResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    bool pressure_committed               = false;
    std::optional<ContinuationSummary> final_summary;
};

struct MaterializationSharedVictimResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    bool pressure_committed               = false;
    std::optional<SharedPrefixSummary> final_summary;
};

struct MaterializationSourceResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    std::optional<ContinuationSummary> final_summary;
};

struct MaterializationSharedSourceResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    std::optional<SharedPrefixSummary> final_summary;
};

struct MaterializationResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    std::optional<StartResult> published;
    std::optional<MaterializationSourceResult> source;
    std::optional<MaterializationSharedSourceResult> shared_source;
    std::vector<MaterializationVictimResult> victims;
    std::vector<MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

using ContextTransactionProgress =
    std::variant<runtime::ContextTransactionInProgress, MaterializationResult,
                 ActiveCaptureResult>;

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::array<std::optional<CaptureOffer>, kMaximumConcurrency> captures{};
    std::size_t row_count    = 0;
    runtime::ExecutionTiming timing{};
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status       = runtime::ConsumeStatus::InvariantMismatch;
    runtime::FinishDisposition disposition = runtime::FinishDisposition::Released;
    GenerationTimings timings;
    SpeculativeStats speculative;
    ContinuationSummary summary;
    std::optional<ContinuationHandle> continuation;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ReleaseResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
};

// ---------------------------------------------------------------------------
// Program (28 engine-facing methods; v1 bodies in impl/runtime/program.cpp)
// ---------------------------------------------------------------------------

struct ModelView;

class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    [[nodiscard]] RequestBasePlan
    plan_request(const PreparedPrompt& prompt, const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::optional<AdmissionCandidate>
    inspect_admission(const PreparedPrompt& prompt, const RequestBasePlan& base,
                      runtime::LaneId destination, const ContinuationHandle* source,
                      const SharedPrefixHandle* shared_source,
                      std::optional<runtime::CheckpointRef> checkpoint,
                      bool must_retain_private_source,
                      const runtime::ContextMachineCostModel& machine_cost);
    [[nodiscard]] std::optional<ResourcePlan>
    seal_identity(const AdmissionCandidate& candidate, const PreparedPrompt& prompt);
    [[nodiscard]] PressurePlanningSession
    begin_pressure_planning(const runtime::ContextMachineCostModel& machine_cost,
                            std::span<const AdmissionCandidate* const> candidates,
                            std::span<const ContinuationHandle* const> private_owners,
                            std::span<const std::uint32_t> private_owner_ordinals,
                            std::span<const SharedPrefixHandle* const> shared_owners,
                            std::span<const std::uint32_t> shared_owner_ordinals);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                               runtime::CancellationFlagView cancellation);
    [[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(const RequestBasePlan& blocked_head, const ResourcePlan& candidate,
                              std::span<const SequenceHandle> persistent_borrowers) const;
    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] PrefillProgress
    advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                    const SharedPrefixHandle* replacement,
                    std::optional<runtime::CheckpointRef> private_replacement) const;
    [[nodiscard]] bool
    shared_capture_matches(const CaptureOffer& offer, const SharedPrefixHandle& shared) const;
    void skip_capture(CaptureOffer&& offer);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                           const SharedPrefixHandle* replacement,
                           std::optional<runtime::CheckpointRef> private_replacement,
                           runtime::CancellationFlagView cancellation);
    [[nodiscard]] PendingBatch
    decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
           runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult
    commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    [[nodiscard]] ReleaseResult release_continuation(ContinuationHandle&& continuation) noexcept;
    [[nodiscard]] ReleaseResult release_shared_prefix(SharedPrefixHandle&& shared) noexcept;
    void fail_all_cleanup() noexcept;
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;
    [[nodiscard]] std::uint64_t resource_revision() const noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl> impl_;

    friend std::unique_ptr<Program>
    create_program(const ModelView& model, WeightsProfile weights_profile, SequencePlan&& plan,
                   DeviceContext& device);
    // M2 real-mode factory: wires the proven single-lane RealProgram into the same
    // Program facade (detail::ProgramImpl in real mode).
    friend std::unique_ptr<Program>
    create_real_program(const std::filesystem::path& artifact, const std::filesystem::path& ngram,
                        std::uint32_t max_context, DeviceContext& device,
                        ops::QsaIndexerKvDtype idx_dtype);
};

// ---------------------------------------------------------------------------
// detail::RuntimeContractAccess (handle/batch internals; all bodies inline)
// ---------------------------------------------------------------------------

namespace detail {

struct RuntimeContractAccess {
    static SequenceHandle make_sequence(const void* owner, runtime::LaneId lane, std::uint64_t epoch)
        noexcept {
        return SequenceHandle{owner, lane, epoch};
    }

    static const void* owner(const SequenceHandle& handle) noexcept { return handle.owner_; }
    static runtime::LaneId lane(const SequenceHandle& handle) noexcept { return handle.lane_; }
    static std::uint64_t epoch(const SequenceHandle& handle) noexcept { return handle.epoch_; }

    static ContinuationHandle
    make_continuation(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept {
        return ContinuationHandle{owner, index, generation};
    }

    static SharedPrefixHandle
    make_shared_prefix(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept {
        return SharedPrefixHandle{owner, index, generation};
    }

    static const void* owner(const ContinuationHandle& handle) noexcept { return handle.owner_; }
    static std::uint32_t index(const ContinuationHandle& handle) noexcept { return handle.index_; }
    static std::uint64_t generation(const ContinuationHandle& handle) noexcept {
        return handle.generation_;
    }
    static void consume(ContinuationHandle& handle) noexcept {
        handle.owner_ = nullptr;
        handle.generation_ = 0;
    }

    static const void* owner(const SharedPrefixHandle& handle) noexcept { return handle.owner_; }
    static std::uint32_t index(const SharedPrefixHandle& handle) noexcept { return handle.index_; }
    static std::uint64_t generation(const SharedPrefixHandle& handle) noexcept {
        return handle.generation_;
    }
    static void consume(SharedPrefixHandle& handle) noexcept {
        handle.owner_ = nullptr;
        handle.generation_ = 0;
    }

    static CaptureOffer
    make_capture_offer(const void* owner, runtime::LaneId lane, std::uint64_t epoch,
                       std::uint64_t id) noexcept {
        return CaptureOffer{owner, lane, epoch, id};
    }

    static const void* owner(const CaptureOffer& offer) noexcept { return offer.owner_; }
    static runtime::LaneId lane(const CaptureOffer& offer) noexcept { return offer.lane_; }
    static std::uint64_t epoch(const CaptureOffer& offer) noexcept { return offer.epoch_; }
    static std::uint64_t id(const CaptureOffer& offer) noexcept { return offer.id_; }
    static void consume(CaptureOffer& offer) noexcept {
        offer.owner_ = nullptr;
        offer.id_ = 0;
    }

    static PendingBatch
    make_pending(const void* owner, std::uint64_t transaction,
                 std::span<const SequenceHandle> rows, std::span<const TokenId> tokens,
                 std::span<const std::int32_t> row_counts, std::uint32_t row_stride,
                 runtime::ExecutionTiming timing) noexcept {
        PendingBatch pending;
        pending.owner_       = owner;
        pending.transaction_ = transaction;
        std::copy(rows.begin(), rows.end(), pending.rows_.begin());
        pending.row_count_  = rows.size();
        pending.tokens_     = tokens;
        pending.row_counts_ = row_counts;
        pending.row_stride_ = row_stride;
        pending.timing_     = timing;
        return pending;
    }

    static std::span<const SequenceHandle>
    rows(const PendingBatch& pending) noexcept {
        return std::span<const SequenceHandle>{pending.rows_.data(), pending.row_count_};
    }

    static void consume(PendingBatch& pending) noexcept {
        pending.owner_       = nullptr;
        pending.transaction_ = 0;
        pending.row_count_   = 0;
        pending.tokens_      = {};
        pending.row_counts_  = {};
        pending.row_stride_  = 0;
        pending.timing_      = {};
    }
};

} // namespace detail

// ---------------------------------------------------------------------------
// Family factories
// ---------------------------------------------------------------------------

[[nodiscard]] SequencePlanner make_sequence_planner(DeviceContext& device,
                                                    const EngineOptions& options,
                                                    WeightsProfile weights_profile);

[[nodiscard]] std::unique_ptr<Program>
create_program(const ModelView& model, WeightsProfile weights_profile, SequencePlan&& plan,
               DeviceContext& device);

// M2 real-mode factory: open the real 48x512 artifact + ngram, build the proven
// single-lane RealProgram (self-contained recurrent/KV/MoE state), and return it
// inside the family's Program facade (detail::ProgramImpl in real mode). `max_context`
// sizes the QSA paged-KV pool and the plan_request room; it must be >= the largest
// prompt the serve should accept.
[[nodiscard]] std::unique_ptr<Program>
create_real_program(const std::filesystem::path& artifact, const std::filesystem::path& ngram,
                    std::uint32_t max_context, DeviceContext& device,
                    ops::QsaIndexerKvDtype idx_dtype);

} // namespace ninfer::targets::qwen3_8_flash_next
