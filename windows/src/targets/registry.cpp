#include "targets/registry.h"
#include "targets/qwen3_6_27b/impl/config.h"
#include "targets/qwen3_6_35b_a3b/impl/config.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/context_cost.h"
#include "runtime/engine/options_normalize.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

artifact::LoadProgress artifact_progress(const LoadProgress& progress) {
    return artifact::LoadProgress{.callback = progress.callback};
}

std::size_t runtime_bytes_after_planned_weights(std::uint64_t weight_bytes) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (weight_bytes > free_bytes) {
        throw std::invalid_argument("model weights require " + std::to_string(weight_bytes) +
                                    " bytes of device memory, but only " +
                                    std::to_string(free_bytes) +
                                    " bytes are free before loading weights");
    }
    return free_bytes - static_cast<std::size_t>(weight_bytes);
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, DeviceContext& device,
                                       artifact::Reader& reader, Clock::time_point load_start,
                                       std::string_view target_key) {
    const auto& identity                          = reader.identity();
    const auto weights_profile                    = Target::resolve_weights(identity);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);
    const runtime::ContextCostIdentity context_cost_identity{
        .hardware_class = runtime::context_cost_hardware_class(
            device.props.name, device.props.major, device.props.minor),
        .model_id   = identity.model_id,
        .weights_id = identity.weights_id,
    };
    runtime::ResolvedContextMachineCost context_cost = runtime::resolve_context_machine_cost(
        context_cost_identity, options.context_cost.preset_path);

    artifact::Binder binder(reader);
    auto load_plan        = Target::plan_load(binder, options, weights_profile);
    auto sequence_planner = Target::make_sequence_planner(device, options, weights_profile);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    const std::size_t preflight_runtime_bytes =
        runtime_bytes_after_planned_weights(load_plan.materialization().device_capacity_bytes);
    (void)runtime::resolve_kv_capacity(options.kv_capacity, curve, preflight_runtime_bytes);

    auto progress     = artifact_progress(options.load_progress);
    auto materialized = artifact::materialize(reader, load_plan.materialization(), device,
                                              progress.callback ? &progress : nullptr);
    const artifact::MaterializationStats stats = materialized.stats();

    auto model = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    device.synchronize();
    runtime::KvCapacityResolution capacity_resolution =
        runtime::resolve_kv_capacity(options.kv_capacity, curve, current_free_device_bytes());
    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    auto loaded   = std::make_unique<Loaded>(std::move(model), options);
    auto instance = std::make_unique<Instance>(std::move(loaded), capacity_resolution,
                                               std::move(sequence_plan), device);
    device.synchronize();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();

    LoadSummary summary;
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    summary.context_cost         = context_cost.summary;
    return ConstructedTarget{.active            = ActiveTarget(std::move(instance)),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults,
                             .context_cost      = std::move(context_cost.model)};
}

// VRAM probe for one target type. Mirrors the preflight half of construct_registered (plan load
// -> free bytes after weights) but stops short of materializing weights, constructing the model,
// or capturing CUDA graphs, so it completes in seconds.
//
// It reports the largest single-sequence context window whose minimum device reservation fits the
// current free-after-weights VRAM. The sequence planner's capacity curve caps its pool at
// max_concurrency * page_count(max_context) — a ceiling derived from the *requested* window, not a
// model architectural limit — so resolve_kv_capacity cannot discover a larger fit through it.
// Instead we build the real planner at candidate windows and binary-search the largest one whose
// minimum reservation (KV cache plus the capacity-scaled attention workspace) fits the budget.
// Querying the planner per candidate accounts for the max_context-dependent overhead exactly, with
// no extrapolation across a change in window size.
//
// `maximum_context` is the target's native context (its architectural window ceiling): the fit can
// never exceed it, and the layout validator rejects windows past it, so the search is capped there
// too (a 1 GiB+ fp8 KV pool on a 32 GiB card would otherwise drive the exponential phase past the
// ceiling into invalid windows).
template <class Target>
KvProbeResult probe_registered(const EngineOptions& options, DeviceContext& device,
                               artifact::Reader& reader, std::uint32_t maximum_context) {
    // The probe builds a capacity curve without constructing an Engine, so it applies the same
    // context-cache normalization the Engine constructor would (the layout code requires non-null
    // cache capacities).
    const EngineOptions normalized = normalize_engine_options(options);
    const auto& identity      = reader.identity();
    const auto weights_profile = Target::resolve_weights(identity);

    artifact::Binder binder(reader);
    auto load_plan        = Target::plan_load(binder, normalized, weights_profile);

    const std::uint64_t weight_bytes     = load_plan.materialization().device_capacity_bytes;
    const std::size_t free_after_weights =
        runtime_bytes_after_planned_weights(weight_bytes);

    // ~97% utilization (unsloth convention): leave a fixed margin of the free-after-weights
    // bytes for driver reservations, other processes, and fragmentation.
    constexpr std::uint32_t kProbeHeadroomPercent = 3;
    const std::size_t headroom_bytes              = free_after_weights * kProbeHeadroomPercent / 100;
    const std::uint64_t budget                    = static_cast<std::uint64_t>(free_after_weights) -
                                                     static_cast<std::uint64_t>(headroom_bytes);

    // Minimum device reservation for one full window of `max_context` tokens at the engine's
    // default concurrency (1): the KV cache plus the capacity-scaled attention workspace. This is
    // exactly what the Engine reserves when launched with --max-context <window> and no explicit
    // concurrency, so it matches the GUI launch. Monotonic non-decreasing in max_context.
    auto reservation_for = [&](std::uint32_t max_context) -> std::uint64_t {
        EngineOptions candidate = normalized;
        candidate.max_context   = max_context;
        // The probe sizes the window against VRAM, so it must not be pinned to a fixed explicit KV
        // token count (the default is Explicit 2048), which validate_target_options rejects once
        // max_context grows past it. Automatic mode skips that range check and does not change the
        // minimum reservation read below.
        candidate.kv_capacity = KvCapacityPolicy::automatic(0);
        auto planner          = Target::make_sequence_planner(device, candidate, weights_profile);
        return static_cast<std::uint64_t>(planner.capacity_curve().minimum_device_reservation_bytes);
    };

    constexpr std::uint32_t kFloorWindow  = 1024;    // smallest window we probe for
    constexpr std::uint32_t kCeilingProbe = 1u << 20; // generous cap; VRAM binds long before this
    // The search runs against the smaller of the probe cap and the target's native context:
    // windows past the native context are rejected by the layout validator, and the fit can
    // never exceed it anyway.
    const std::uint32_t ceiling = std::min(kCeilingProbe, maximum_context);

    std::uint32_t fit_tokens = 0;
    if (reservation_for(kFloorWindow) <= budget) {
        // Exponential search for an upper bound that does not fit, then binary-search the largest
        // window that does. `lo` always fits, `hi` is the first candidate that does not.
        std::uint32_t lo = kFloorWindow;
        std::uint32_t hi = kFloorWindow * 2U;
        while (hi < ceiling && reservation_for(hi) <= budget) {
            lo = hi;
            hi *= 2U;
        }
        if (hi >= ceiling) {
            // The doubling overshot the ceiling without ever evaluating it (only a fitting hi
            // doubles), so retarget the search at the ceiling itself: `hi = ceiling + 1` is an
            // exclusive bound whose mids all stay within [floor, ceiling].
            hi = ceiling + 1;
        }
        std::uint32_t best = lo;
        while (lo + 1 < hi) {
            const std::uint32_t mid = lo + (hi - lo) / 2U;
            if (reservation_for(mid) <= budget) {
                best = mid;
                lo   = mid;
            } else {
                hi = mid;
            }
        }
        fit_tokens = best;
    }

    KvProbeResult result;
    result.model_id                      = identity.model_id;
    result.vram_total_bytes              = device.total_vram();
    result.weights_bytes                 = weight_bytes;
    result.vram_free_after_weights_bytes = free_after_weights;
    result.kv_headroom_percent           = kProbeHeadroomPercent;
    result.kv_fit_tokens                 = fit_tokens;
    return result;
}

} // namespace

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model, options)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model, options)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               runtime::KvCapacityResolution resolution,
                                               Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                               DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_35BA3B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            options, device, reader, load_start, Qwen3_6_35BA3B::target_key);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

KvProbeResult probe_kv_capacity(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }

    // The probe owns its device context for the call's duration: it only queries VRAM and
    // resolves a capacity curve, so no weights are materialized and no model is built.
    DeviceContext device(options.device);
    artifact::Reader reader(options.artifact_path);
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return probe_registered<Qwen3_6_27B>(options, device, reader,
                                             qwen3_6_27b::detail::kNativeContext);
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return probe_registered<Qwen3_6_27B>(options, device, reader,
                                             qwen3_6_27b::detail::kNativeContext);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return probe_registered<Qwen3_6_35BA3B>(options, device, reader,
                                                qwen3_6_35b_a3b::detail::kNativeContext);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets
