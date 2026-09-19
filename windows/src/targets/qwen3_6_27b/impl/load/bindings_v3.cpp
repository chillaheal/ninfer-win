#include "targets/qwen3_6_27b/impl/load/bindings_v3.h"

#include "artifact/v3/reader.h"
#include "core/weight_view.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

// The fused quantized parents are exposed only as fine-grained sub-range views in v3; the
// logical slot binds the whole parent. Look the parent up through any representative sub-range.
const artifact::v3::ObjectHandle
parent_of(const artifact::v3::Reader& reader, const std::string& name) {
    const auto found = reader.directory().bindings.find(name);
    if (found == reader.directory().bindings.end()) {
        throw artifact::v3::ArtifactError("v3: missing logical parameter " + name);
    }
    const auto& binding = found->second;
    if (binding.parts.empty()) {
        throw artifact::v3::ArtifactError("v3: empty binding " + name);
    }
    return binding.parts.front().object;
}

V3Slot bind_device_slot(artifact::v3::Binder& binder, const std::string& name, float divisor) {
    const auto object = parent_of(binder.reader(), name);
    binder.require_device(object);
    return V3Slot{object, divisor};
}

// Host-residency variant: bind the parent into host memory (for CPU vision). Marks the object
// host-resident so the materializer reads it into host memory and the CPU vision kernels read
// host pointers instead of device pointers.
V3Slot bind_host_slot(artifact::v3::Binder& binder, const std::string& name, float divisor) {
    const auto object = parent_of(binder.reader(), name);
    (void)binder.host_object(object);
    V3Slot slot{object, divisor};
    slot.host = true;
    return slot;
}

// Reads the single-FP32 "activation_input_divisor" auxiliary attached to a USE, when present
// (NVFP4 slots only). Absent auxiliary or absent USE -> 0 (FP8/RowScale carry none).
float read_input_divisor(artifact::v3::Binder& binder, const std::string& parameter,
                         const std::string& input) {
    try {
        const auto& use = binder.use(parameter, input);
        const auto it   = use.auxiliaries.find("activation_input_divisor");
        if (it != use.auxiliaries.end()) {
            return binder.values(it->second).scalar_f32();
        }
    } catch (const artifact::v3::ArtifactError&) {
    }
    return 0.0F;
}

// Reads a named component resource straight from the artifact (the v3 equivalent of v2's raw
// frontend resources). Text lives under the "text" component, the vision preprocessor configs
// under "vision". The bytes are copied out immediately; the binder-side storage moves away at
// finish().
std::string read_resource_string(artifact::v3::Binder& binder, const std::string& component,
                                 const std::string& role) {
    const auto handle = binder.resource(component, role);
    const auto bytes  = binder.host_object(handle);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void bind_text_layers(artifact::v3::Binder& binder, V3BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        V3TextLayerPlan& target = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.is_full_attention = is_full_layer(layer);
        target.input_norm        = bind_device_slot(binder, prefix + "input_norm", 0.0F);

        if (target.is_full_attention) {
            target.attention.query_key_gate_value =
                bind_device_slot(binder, prefix + "attention/query", 0.0F);
            target.attention.query_norm =
                bind_device_slot(binder, prefix + "attention/query_norm", 0.0F);
            target.attention.key_norm =
                bind_device_slot(binder, prefix + "attention/key_norm", 0.0F);
            target.attention.output = bind_device_slot(binder, prefix + "attention/output", 0.0F);
        } else {
            target.gdn.a_log       = bind_device_slot(binder, prefix + "gdn/a_log", 0.0F);
            target.gdn.dt_bias     = bind_device_slot(binder, prefix + "gdn/dt_bias", 0.0F);
            target.gdn.convolution = bind_device_slot(binder, prefix + "gdn/convolution", 0.0F);
            target.gdn.control_projection =
                bind_device_slot(binder, prefix + "gdn/a_projection", 0.0F);
            target.gdn.input_projection =
                bind_device_slot(binder, prefix + "gdn/query", 0.0F);
            target.gdn.norm   = bind_device_slot(binder, prefix + "gdn/norm", 0.0F);
            target.gdn.output = bind_device_slot(binder, prefix + "gdn/output", 0.0F);
        }
        target.post_attention_norm =
            bind_device_slot(binder, prefix + "post_attention_norm", 0.0F);
        target.mlp.gate_up = bind_device_slot(
            binder, prefix + "mlp/gate",
            read_input_divisor(binder, prefix + "mlp/gate", prefix + "ffn_input"));
        target.mlp.down = bind_device_slot(
            binder, prefix + "mlp/down",
            read_input_divisor(binder, prefix + "mlp/down", prefix + "mlp/product"));
    }
}

// Row-slice a complete RowSplit-quantized Weight by row range (mirrors the v2 load-side helper;
// the q8_g32_fp16/row_split parents expose the same RowSplit ABI). Used for the dflash2
// context_key / context_value views over the fused query_key_value parent.
Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid v3 target row view");
    }
    const std::uint64_t groups     = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_group  = 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S   ? 8
                                 : block.qtype == QType::Q6G64_F16S ? 16
                                                                    : 0;
    const std::uint64_t low_row    = groups * low_group;
    const std::uint64_t high_row   = groups * high_group;
    const std::uint64_t scale_row  = groups * 2;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_group == 0 ? nullptr
                                 : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

V3VisionPlan bind_vision_v3(artifact::v3::Binder& binder, bool host) {
    V3VisionPlan out;
    // Route every vision slot to the requested residency: device (GPU vision) or host (CPU vision).
    const auto bind = [&](const std::string& name) {
        return host ? bind_host_slot(binder, name, 0.0F) : bind_device_slot(binder, name, 0.0F);
    };
    out.patch_embedding      = bind("vision/patch_embedding");
    out.patch_embedding_bias = bind("vision/patch_embedding_bias");
    out.position_embedding   = bind("vision/position_embedding");
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        V3VisionLayerPlan& target = out.layers[layer];
        const std::string prefix  = "vision/layers/" + std::to_string(layer) + "/";
        target.qkv          = bind(prefix + "attention/query");
        target.qkv_bias     = bind(prefix + "attention/query_bias");
        target.output       = bind(prefix + "attention/output");
        target.output_bias  = bind(prefix + "attention/output_bias");
        target.fc1          = bind(prefix + "mlp/fc1");
        target.fc1_bias     = bind(prefix + "mlp/fc1_bias");
        target.fc2          = bind(prefix + "mlp/fc2");
        target.fc2_bias     = bind(prefix + "mlp/fc2_bias");
        target.norm1_weight = bind(prefix + "norm1_weight");
        target.norm1_bias   = bind(prefix + "norm1_bias");
        target.norm2_weight = bind(prefix + "norm2_weight");
        target.norm2_bias   = bind(prefix + "norm2_bias");
    }
    out.merger_fc1         = bind("vision/merger/fc1");
    out.merger_fc1_bias    = bind("vision/merger/fc1_bias");
    out.merger_norm_weight = bind("vision/merger/norm_weight");
    out.merger_norm_bias   = bind("vision/merger/norm_bias");
    out.merger_fc2         = bind("vision/merger/fc2");
    out.merger_fc2_bias    = bind("vision/merger/fc2_bias");
    return out;
}

V3DFlash2Plan bind_dflash2_v3(artifact::v3::Binder& binder) {
    V3DFlash2Plan out;
    out.feature_projection = bind_device_slot(binder, "dflash2/feature_projection", 0.0F);
    out.context_norm       = bind_device_slot(binder, "dflash2/context_norm", 0.0F);
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        V3DFlash2LayerPlan& target = out.layers[layer];
        const std::string prefix   = "dflash2/layers/" + std::to_string(layer) + "/";
        target.input_norm = bind_device_slot(binder, prefix + "input_norm", 0.0F);
        target.attention_conv.base_kernel =
            bind_device_slot(binder, prefix + "attention_conv/base_kernel", 0.0F);
        target.attention_conv.kernel_projection =
            bind_device_slot(binder, prefix + "attention_conv/kernel_projection", 0.0F);
        target.query_key_value = bind_device_slot(binder, prefix + "attention/query", 0.0F);
        target.query_norm = bind_device_slot(binder, prefix + "attention/query_norm", 0.0F);
        target.key_norm   = bind_device_slot(binder, prefix + "attention/key_norm", 0.0F);
        target.attention_output = bind_device_slot(binder, prefix + "attention/output", 0.0F);
        target.post_attention_norm =
            bind_device_slot(binder, prefix + "post_attention_norm", 0.0F);
        target.mlp_conv.base_kernel =
            bind_device_slot(binder, prefix + "mlp_conv/base_kernel", 0.0F);
        target.mlp_conv.kernel_projection =
            bind_device_slot(binder, prefix + "mlp_conv/kernel_projection", 0.0F);
        target.gate_up = bind_device_slot(binder, prefix + "mlp/gate", 0.0F);
        target.down    = bind_device_slot(binder, prefix + "mlp/down", 0.0F);
    }
    out.final_norm = bind_device_slot(binder, "dflash2/final_norm", 0.0F);
    out.candidate_selector.hidden_projection =
        bind_device_slot(binder, "dflash2/candidate_selector/hidden_projection", 0.0F);
    out.candidate_selector.predecessor_codebook =
        bind_device_slot(binder, "dflash2/candidate_selector/predecessor_codebook", 0.0F);
    out.candidate_selector.successor_codebook =
        bind_device_slot(binder, "dflash2/candidate_selector/successor_codebook", 0.0F);
    return out;
}

// MTP is a single draft layer (layers/0). The fused attention parent is reached through the
// representative sub-range "attention/query" and the fused mlp parent through "mlp/gate";
// geometry/row order (q|k|gate|value, gate|up) are byte-identical to the v2 layout.
V3MtpPlan bind_mtp_v3(artifact::v3::Binder& binder) {
    V3MtpPlan out;
    out.input_projection = bind_device_slot(binder, "mtp/input_projection", 0.0F);
    out.embedding_norm   = bind_device_slot(binder, "mtp/embedding_norm", 0.0F);
    out.hidden_norm      = bind_device_slot(binder, "mtp/hidden_norm", 0.0F);
    const std::string prefix = "mtp/layers/0/";
    out.input_norm           = bind_device_slot(binder, prefix + "input_norm", 0.0F);
    out.query_key_gate_value = bind_device_slot(binder, prefix + "attention/query", 0.0F);
    out.query_norm           = bind_device_slot(binder, prefix + "attention/query_norm", 0.0F);
    out.key_norm             = bind_device_slot(binder, prefix + "attention/key_norm", 0.0F);
    out.output               = bind_device_slot(binder, prefix + "attention/output", 0.0F);
    out.post_attention_norm  = bind_device_slot(binder, prefix + "post_attention_norm", 0.0F);
    out.mlp.gate_up          = bind_device_slot(binder, prefix + "mlp/gate", 0.0F);
    out.mlp.down             = bind_device_slot(binder, prefix + "mlp/down", 0.0F);
    out.final_norm           = bind_device_slot(binder, "mtp/final_norm", 0.0F);
    return out;
}

} // namespace

V3LoadPlan bind_artifact_v3(artifact::v3::Binder& binder, WeightsProfile weights_profile,
                            qwen3_6::StartupFeatures features) {
    (void)weights_profile;
    V3LoadPlan load_plan;
    V3BindingPlan& out = load_plan.bindings;
    out.features       = features;

    out.frontend.tokenizer_json          = read_resource_string(binder, "text", "tokenizer.json");
    out.frontend.tokenizer_config_json   = read_resource_string(binder, "text", "tokenizer_config.json");
    out.frontend.chat_template_jinja     = read_resource_string(binder, "text", "chat_template.jinja");
    out.frontend.generation_config_json  = read_resource_string(binder, "text", "generation_config.json");
    out.frontend.preprocessor_config_json =
        read_resource_string(binder, "vision", "preprocessor_config.json");
    out.frontend.video_preprocessor_config_json =
        read_resource_string(binder, "vision", "video_preprocessor_config.json");

    out.token_embedding = bind_device_slot(binder, "text/token_embedding", 0.0F);
    bind_text_layers(binder, out);
    out.final_norm  = bind_device_slot(binder, "text/final_norm", 0.0F);
    out.output_head = bind_device_slot(binder, "text/output_head", 0.0F);

    // Optional weight groups are bound only when the startup features request them, so an
    // unselected group never reserves device memory (mirrors the v2 feature gating).
    if (features.vision) { out.vision = bind_vision_v3(binder, features.vision_cpu); }
    if (features.mtp()) { out.mtp = bind_mtp_v3(binder); }
    if (features.dflash2()) { out.dflash2 = bind_dflash2_v3(binder); }
    if (features.optimized_proposal()) {
        out.draft_head           = bind_device_slot(binder, "proposal/head", 0.0F);
        out.draft_head_token_ids = bind_device_slot(binder, "proposal/token_ids", 0.0F);
    }

    load_plan.materialization = std::move(binder).finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(V3BindingPlan plan,
                                 artifact::v3::MaterializedArtifact materialized)
    : backing(std::in_place_type<artifact::v3::MaterializedArtifact>, std::move(materialized)) {
    auto& backing = std::get<artifact::v3::MaterializedArtifact>(this->backing);
    frontend           = std::move(plan.frontend);
    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;

    // Whole-parent views: region [0, geometry.elements) over the resident parent. Host-resident
    // slots (CPU vision) resolve the host parent; device-resident slots resolve the device parent.
    const auto view_of = [&](const V3Slot& slot) {
        const auto& parent = slot.host ? backing.host_parent(slot.object)
                                       : backing.device_parent(slot.object);
        const auto& geometry = parent.geometry;
        WeightRegion region{&parent, 0, geometry.elements};
        return WeightView{geometry.shape, {region}};
    };
    const auto weight_of = [&](const V3Slot& slot) {
        return native_weight(view_of(slot), slot.input_divisor);
    };
    const auto tensor_of = [&](const V3Slot& slot, std::initializer_list<std::int32_t> shape) {
        return weight_tensor(view_of(slot), shape);
    };

    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding = weight_of(plan.token_embedding);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const V3TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm =
                tensor_of(source.input_norm, {5120});
            target.projection = FusedAttentionProjectionPayload{
                .query_key_gate_value = weight_of(source.attention.query_key_gate_value),
            };
            target.query_norm = tensor_of(source.attention.query_norm, {256});
            target.key_norm   = tensor_of(source.attention.key_norm, {256});
            target.output     = weight_of(source.attention.output);
            target.post_attention_norm = tensor_of(source.post_attention_norm, {5120});
            target.post_mixer          = DensePostMixerPayload{
                .gate_up = weight_of(source.mlp.gate_up),
                .down    = weight_of(source.mlp.down),
            };
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm = tensor_of(source.input_norm, {5120});
            target.projection.a_log =
                tensor_of(source.gdn.a_log, {48});
            target.projection.dt_bias =
                tensor_of(source.gdn.dt_bias, {48});
            target.convolution = tensor_of(source.gdn.convolution, {10240, 4});
            target.projection.control_projection = FusedGdnControlProjectionPayload{
                .a_b_projection = weight_of(source.gdn.control_projection),
            };
            target.projection.input_projection = FusedGdnInputProjectionPayload{
                .query_key_value_z = weight_of(source.gdn.input_projection),
            };
            target.norm = tensor_of(source.gdn.norm, {128});
            target.output = weight_of(source.gdn.output);
            target.post_attention_norm = tensor_of(source.post_attention_norm, {5120});
            target.post_mixer          = DensePostMixerPayload{
                .gate_up = weight_of(source.mlp.gate_up),
                .down    = weight_of(source.mlp.down),
            };
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("v3: text topology binding is incomplete");
    }
    final_norm  = tensor_of(plan.final_norm, {5120});
    output_head = weight_of(plan.output_head);

    if (plan.features.vision) {
        using VC = qwen3_6::VisionBackboneConfig;
        const V3VisionPlan& vp     = *plan.vision;
        auto& vision               = runtime.vision.emplace();
        vision.common.patch_embedding      = weight_of(vp.patch_embedding);
        vision.common.patch_embedding_bias = tensor_of(vp.patch_embedding_bias, {VC::hidden});
        vision.common.position_embedding   =
            tensor_of(vp.position_embedding, {VC::hidden, VC::position_embeddings});
        for (std::size_t layer = 0; layer < vision.common.layers.size(); ++layer) {
            const V3VisionLayerPlan& source = vp.layers[layer];
            qwen3_6::VisionLayerWeights& target = vision.common.layers[layer];
            target.qkv          = weight_of(source.qkv);
            target.qkv_bias     = tensor_of(source.qkv_bias, {3 * VC::hidden});
            target.output       = weight_of(source.output);
            target.output_bias  = tensor_of(source.output_bias, {VC::hidden});
            target.fc1          = weight_of(source.fc1);
            target.fc1_bias     = tensor_of(source.fc1_bias, {VC::intermediate});
            target.fc2          = weight_of(source.fc2);
            target.fc2_bias     = tensor_of(source.fc2_bias, {VC::hidden});
            target.norm1_weight = tensor_of(source.norm1_weight, {VC::hidden});
            target.norm1_bias   = tensor_of(source.norm1_bias, {VC::hidden});
            target.norm2_weight = tensor_of(source.norm2_weight, {VC::hidden});
            target.norm2_bias   = tensor_of(source.norm2_bias, {VC::hidden});
        }
        vision.common.merger_fc1         = weight_of(vp.merger_fc1);
        vision.common.merger_fc1_bias    = tensor_of(vp.merger_fc1_bias, {VC::merger_hidden});
        vision.common.merger_norm_weight = tensor_of(vp.merger_norm_weight, {VC::hidden});
        vision.common.merger_norm_bias   = tensor_of(vp.merger_norm_bias, {VC::hidden});
        vision.merger_fc2                = weight_of(vp.merger_fc2);
        vision.merger_fc2_bias           = tensor_of(vp.merger_fc2_bias, {5120});
    }

    if (plan.features.dflash2()) {
        const V3DFlash2Plan& dp = *plan.dflash2;
        auto& dflash            = runtime.dflash.emplace();
        dflash.feature_projection = weight_of(dp.feature_projection);
        dflash.context_norm       = tensor_of(dp.context_norm, {5120});
        for (std::size_t layer = 0; layer < dflash.layers.size(); ++layer) {
            const V3DFlash2LayerPlan& source = dp.layers[layer];
            qwen3_6::DFlash2LayerWeights& target = dflash.layers[layer];
            target.input_norm = tensor_of(source.input_norm, {5120});
            target.attention_conv.base_kernel =
                tensor_of(source.attention_conv.base_kernel, {5120, 2, 2});
            target.attention_conv.kernel_projection =
                weight_of(source.attention_conv.kernel_projection);
            target.query_key_value = weight_of(source.query_key_value);
            target.context_key   = row_view(target.query_key_value, 4096, 1024);
            target.context_value = row_view(target.query_key_value, 5120, 1024);
            target.query_norm    = tensor_of(source.query_norm, {128});
            target.key_norm      = tensor_of(source.key_norm, {128});
            target.attention_output = weight_of(source.attention_output);
            target.post_attention_norm = tensor_of(source.post_attention_norm, {5120});
            target.mlp_conv.base_kernel =
                tensor_of(source.mlp_conv.base_kernel, {5120, 2, 2});
            target.mlp_conv.kernel_projection =
                weight_of(source.mlp_conv.kernel_projection);
            target.gate_up = weight_of(source.gate_up);
            target.down    = weight_of(source.down);
        }
        dflash.final_norm = tensor_of(dp.final_norm, {5120});
        dflash.candidate_selector.hidden_projection =
            weight_of(dp.candidate_selector.hidden_projection);
        dflash.candidate_selector.predecessor_codebook =
            tensor_of(dp.candidate_selector.predecessor_codebook, {256, 248320});
        dflash.candidate_selector.successor_codebook =
            tensor_of(dp.candidate_selector.successor_codebook, {256, 248320});
    }

    if (plan.features.mtp()) {
        const V3MtpPlan& mp = *plan.mtp;
        auto& mtp           = runtime.mtp.emplace();
        mtp.input_projection  = weight_of(mp.input_projection);
        mtp.embedding_norm    = tensor_of(mp.embedding_norm, {5120});
        mtp.hidden_norm       = tensor_of(mp.hidden_norm, {5120});
        mtp.input_norm        = tensor_of(mp.input_norm, {5120});
        mtp.attention.packed  = weight_of(mp.query_key_gate_value);
        mtp.attention.query        = row_view(mtp.attention.packed, 0, 6144);
        mtp.attention.key          = row_view(mtp.attention.packed, 6144, 1024);
        mtp.attention.output_gate  = row_view(mtp.attention.packed, 7168, 6144);
        mtp.attention.value        = row_view(mtp.attention.packed, 13312, 1024);
        mtp.query_norm            = tensor_of(mp.query_norm, {256});
        mtp.key_norm              = tensor_of(mp.key_norm, {256});
        mtp.output                = weight_of(mp.output);
        mtp.post_attention_norm   = tensor_of(mp.post_attention_norm, {5120});
        mtp.post_mixer            = DensePostMixerPayload{
            .gate_up = weight_of(mp.mlp.gate_up),
            .down    = weight_of(mp.mlp.down),
        };
        mtp.final_norm            = tensor_of(mp.final_norm, {5120});
    }

    if (plan.features.optimized_proposal()) {
        auto& proposal   = runtime.optimized_proposal.emplace();
        proposal.head      = weight_of(plan.draft_head);
        proposal.token_ids = tensor_of(plan.draft_head_token_ids, {131072});
    }
}

LoadedModel::Impl::Impl(WeightsProfile weights_profile_in, V3BindingPlan plan,
                        artifact::v3::MaterializedArtifact materialized)
    : weights_profile(weights_profile_in), data(std::move(plan), std::move(materialized)) {}

} // namespace ninfer::targets::qwen3_6_27b::detail
