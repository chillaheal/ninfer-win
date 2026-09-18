#pragma once

// Human-readable request summaries and the optional full-precision JSONL event log used for
// measurement. The HTTP layer owns request ids; this module owns one stable JSON schema and
// serializes concurrent writes from non-streaming handlers and streaming workers.

#include "serve/generation_service.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>

namespace ninfer::serve {

inline constexpr int kRequestLogSchemaVersion        = 19;
inline constexpr const char* kRequestLogArtifactType = "ninfer_serve_request_log";
// Rotation: the base ledger rolls over at 10 MiB into at most two shifted files (`.1` newest,
// `.2` oldest), and rotated files older than 7 days are dropped at each rotation and at startup.
inline constexpr std::size_t kRequestLogRotateBytes   = 10ULL * 1024 * 1024;
inline constexpr int kRequestLogRetentionDays         = 7;

struct RequestLogContext {
    std::uint64_t id = 0;
    std::string protocol;
    std::string model;
    bool stream                             = false;
    std::size_t message_count               = 0;
    std::size_t media_item_count            = 0;
    int requested_output_tokens             = 0;
    bool requested_output_tokens_client_set = false;
    std::size_t tool_count                  = 0;
    ToolChoice tool_choice;
    bool has_tool_history = false;
    bool enable_thinking  = true;
    std::optional<std::uint32_t> thinking_budget;
    bool preserve_thinking                 = false;
    bool preserve_thinking_semantic_change = false;
    // Client's requested reasoning effort (e.g. "medium" from output_config.effort); absent when
    // the client sent none. Logged so the effective sampling (and any effort-coupled temperature)
    // can be correlated with the effort that drove it.
    std::optional<std::string> reasoning_effort;
    ninfer::ResolvedSamplingParameters sampling;
    double acquisition_seconds = 0.0;
    ninfer::PromptPreparationStats preparation;
};

// A parsed generation request that failed during synchronous preparation. It intentionally has a
// separate shape from RequestLogContext: sampler and prompt semantics are not guaranteed to have
// resolved when preparation rejects the request.
struct RequestRejectionLogContext {
    std::uint64_t id = 0;
    std::string protocol;
    std::string model;
    bool stream                             = false;
    std::size_t message_count               = 0;
    std::size_t media_item_count            = 0;
    int requested_output_tokens             = 0;
    bool requested_output_tokens_client_set = false;
    std::size_t tool_count                  = 0;
    ToolChoice tool_choice;
    bool has_tool_history = false;
    ApiError error;
};

struct ServerLogEnvironment {
    int device = 0;
    std::string gpu_name;
    std::string gpu_uuid;
    std::uint64_t total_device_memory_bytes = 0;
    int compute_capability_major            = 0;
    int compute_capability_minor            = 0;
    std::string cuda_compile_version;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
};

struct ThroughputReport {
    double interval_seconds               = 0.0;
    std::uint64_t computed_prefill_tokens = 0;
    std::uint64_t committed_decode_tokens = 0;
    std::uint64_t decode_rounds           = 0;
    std::uint64_t decode_row_rounds       = 0;
    ninfer::RuntimeStats previous;
    ninfer::RuntimeStats current;
};

RequestLogContext make_request_log_context(std::uint64_t id, std::string protocol,
                                           const GenerationRequest& request,
                                           const PreparedRequest& prepared);
RequestRejectionLogContext make_request_rejection_log_context(std::uint64_t id,
                                                              std::string protocol,
                                                              const GenerationRequest& request,
                                                              ApiError error);

// Compact console records retained for operator visibility.
std::string format_request_start(const RequestLogContext& context);
std::string format_request_rejected(const RequestRejectionLogContext& context);
std::string format_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
std::string format_request_error(const RequestLogContext& context, const std::string& message);
std::string format_throughput(const ThroughputReport& report);

// Pure JSON formatters are public to repository tests. Each return value is one complete JSON
// object without a trailing newline.
std::string format_server_start_json(
    const std::string& server_instance_id, std::uint64_t timestamp_unix_ms,
    const ServeOptions& options, const ninfer::EngineOptions& engine_options,
    const ninfer::ModelSamplingDefaults& sampling_defaults, const std::string& public_model_id,
    const ninfer::LoadSummary& load, const ninfer::MemorySummary& memory,
    const ServerLogEnvironment& environment, std::optional<std::uint64_t> artifact_size_bytes);
std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context);
std::string format_request_rejected_json(const std::string& server_instance_id,
                                         std::uint64_t timestamp_unix_ms,
                                         const RequestRejectionLogContext& context);
std::string format_request_done_json(const std::string& server_instance_id,
                                     std::uint64_t timestamp_unix_ms,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome);
std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context, const std::string& message);
std::string format_throughput_json(const std::string& server_instance_id,
                                   std::uint64_t timestamp_unix_ms, const ThroughputReport& report);

ServerLogEnvironment query_server_log_environment(int device);

// Opens in append mode so one campaign file can contain multiple independently started MTP/model
// blocks. Every line carries server_instance_id because request ids restart at one per process.
class JsonlRequestLog {
public:
    explicit JsonlRequestLog(const std::string& path,
                             const std::string& protected_artifact_path = {});

    JsonlRequestLog(const JsonlRequestLog&)            = delete;
    JsonlRequestLog& operator=(const JsonlRequestLog&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return output_.is_open(); }

    [[nodiscard]] const std::string& server_instance_id() const noexcept {
        return server_instance_id_;
    }

    void write_server_start(const ServeOptions& options,
                            const ninfer::EngineOptions& engine_options,
                            const ninfer::ModelSamplingDefaults& sampling_defaults,
                            const std::string& public_model_id, const ninfer::LoadSummary& load,
                            const ninfer::MemorySummary& memory);
    void write_request_start(const RequestLogContext& context);
    void write_request_rejected(const RequestRejectionLogContext& context);
    void write_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void write_request_error(const RequestLogContext& context, const std::string& message);
    void write_throughput(const ThroughputReport& report);

private:
    void append(std::string record);
    // Called with mutex_ held, before writing: rolls the base file over once it reaches
    // kRequestLogRotateBytes (base -> .1 -> .2, dropping the old .2) and prunes stale rotated
    // files; leaves output_ open on the fresh base file.
    void rotate_if_needed();
    // Called with mutex_ held: deletes `.1`/`.2` files whose last-write time is older than
    // kRequestLogRetentionDays.
    void prune_rotated();

    std::string path_;
    std::string server_instance_id_;
    // Per-process sequence for the file_epoch marker: a fresh base (at startup or after a
    // rotation) gets `seq`, `seq+1`, ... so its first line is unique per file even within one
    // process. The GUI fingerprints the first 256 bytes of each ledger file; a shared head
    // would collapse the three files into one consumed-offset entry and re-fold the tails.
    std::uint32_t file_epoch_seq_ = 0;
    std::ofstream output_;
    std::mutex mutex_;
    bool failed_ = false;
};

} // namespace ninfer::serve
