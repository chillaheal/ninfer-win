#include <stdexcept>
#include <variant>

#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/config.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"

namespace ninfer::targets::qwen3_8_flash_next {

namespace {

using ArtifactIdentity = artifact::ArtifactIdentity;

}  // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model != model_id) {
        throw std::runtime_error("model '" + std::string(model) +
                                 "' has no sampling defaults in target package '" +
                                 std::string(target_key) + "'");
    }
    constexpr ModelSamplingDefaults kFlashNextDefaults{
        .thinking =
            {.temperature        = 1.0F,
             .top_k              = 20,
             .top_p              = 0.95F,
             .min_p              = 0.0F,
             .presence_penalty   = 0.0F,
             .frequency_penalty  = 0.0F},
        .non_thinking =
            {.temperature        = 0.7F,
             .top_k              = 20,
             .top_p              = 0.80F,
             .min_p              = 0.0F,
             .presence_penalty   = 1.5F,
             .frequency_penalty  = 0.0F}};
    return kFlashNextDefaults;
}

std::unique_ptr<LoadedModel> Package::construct_loaded_model(
    const std::filesystem::path& artifact_path, const std::filesystem::path& ngram_path,
    const std::filesystem::path& archive_path, DeviceContext& device) {
    return qwen3_8_flash_next::LoadedModel::open(artifact_path, ngram_path, archive_path, device);
}

Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    (void)model;
    return qwen3_8_flash_next::make_frontend(FrontendOptions{options.max_context});
}

SequencePlanner Package::make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                                               WeightsProfile weights_profile) {
    return qwen3_8_flash_next::make_sequence_planner(device, options, weights_profile);
}

std::unique_ptr<Program> Package::create_program(const LoadedModel& model, SequencePlan&& plan,
                                                 DeviceContext& device) {
    return qwen3_8_flash_next::create_program(model.view(), WeightsProfile::Nvfp4, std::move(plan),
                                              device);
}

Package::WeightsProfile Package::resolve_weights(const ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == weights_id) {
        return WeightsProfile::Nvfp4;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" +
                             identity.weights_id + "' is not supported by target '" +
                             std::string(target_key) + "'");
}

std::uint64_t Package::kv_bytes_per_token(KvCacheStorage storage) {
    // BF16 stores 2 bytes/element; the group/row-compressed storages store 1
    // byte/element (scale-plane overhead is a P7 refinement, see header).
    const std::uint32_t element_bytes =
        storage == KvCacheStorage::BFloat16 ? 2u : 1u;
    return detail::kv_bytes_per_token(element_bytes);
}

std::uint64_t Package::gdn_state_bytes_per_layer() {
    return detail::gdn_state_bytes_per_layer();
}

std::uint64_t Package::gdn_state_bytes(std::uint32_t max_concurrency) {
    return detail::Geometry::kGdnLayers * detail::gdn_state_bytes_per_layer() *
           static_cast<std::uint64_t>(max_concurrency);
}

Package::Shell Package::plan(const std::filesystem::path& artifact_path,
                             const std::filesystem::path& ngram_path,
                             const EngineOptions& options,
                             std::uint64_t device_budget_bytes,
                             std::uint64_t system_ram_bytes,
                             std::uint64_t expert_window_bytes,
                             bool host_kv,
                             std::uint64_t workspace_bytes) {
    BindResult binder = bind_artifact(artifact_path, ngram_path);

    PlacementRequest request;
    request.device_budget_bytes  = device_budget_bytes;
    request.system_ram_bytes     = system_ram_bytes;
    request.backbone_weight_bytes = binder.gpu_weight_bytes;
    request.expert_host_bytes    = binder.host_expert_bytes;
    request.ple_bytes            = binder.ple.bytes();
    request.kv_bytes_per_token   = kv_bytes_per_token(options.kv_cache);
    // Indexer-K pool per-token footprint: 512-dim key (1 B/elem FP8 or 2 B/elem BF16)
    // + one 2 B half scale per token (the scale plane is allocated for both dtypes).
    request.indexer_kv_bytes_per_token =
        (options.indexer_kv_dtype == "bf16") ? 512u * 2u + 2u : 512u + 2u;
    request.max_context          = options.max_context;
    request.max_concurrency      = options.max_concurrency;
    request.expert_window_bytes  = expert_window_bytes;
    request.host_kv              = host_kv;
    request.workspace_bytes      = workspace_bytes;
    request.gdn_state_bytes      = gdn_state_bytes(options.max_concurrency);

    const auto outcome = plan_placement(request);
    if (const auto* error = std::get_if<PlacementError>(&outcome)) {
        throw std::runtime_error(error->message());
    }
    return Shell{std::move(binder), std::get<PlacementTable>(outcome)};
}

}  // namespace ninfer::targets::qwen3_8_flash_next
