#pragma once

// Parallel v3 load-side for qwen3_6_27b. Consumes the v3 artifact taxonomy (fine-grained
// sub-range views over fused parents) and materializes the SAME RuntimeModelView the v2 path
// fills, via core::native_weight / core::weight_tensor over whole-parent WeightRegion views.
// The v2 binding in bindings.{h,cpp} is untouched; the registry selects by artifact magic.

#include "targets/qwen3_6_27b/impl/load/bindings.h"   // RuntimeModelView + payload aliases
#include "artifact/v3/binder.h"
#include "artifact/v3/materializer.h"

namespace ninfer::targets::qwen3_6_27b::detail {

// One logical slot: the whole parent object to bind plus an optional activation input divisor
// (the v3 USE auxiliary "activation_input_divisor", carried only on NVFP4 slots). The parent's
// shape/layout/format are read from the materialized geometry at construction time, so this
// plan stays agnostic to per-slot quantization.
struct V3Slot {
    artifact::v3::ObjectHandle object;
    float input_divisor = 0.0F;
    // True when the parent is materialized into host memory instead of the device arena.
    // Set only on vision slots when the runtime runs vision on the CPU.
    bool host = false;
};

struct V3MlpPlan {
    V3Slot gate_up;   // fused gate+up parent
    V3Slot down;      // whole parent
};

struct V3AttentionPlan {
    V3Slot query_key_gate_value;  // fused attention parent
    V3Slot query_norm;
    V3Slot key_norm;
    V3Slot output;                // whole parent
};

struct V3GdnPlan {
    V3Slot a_log;
    V3Slot dt_bias;
    V3Slot convolution;
    V3Slot control_projection;    // fused a+b parent {96,5120}
    V3Slot input_projection;      // fused q+k+v+z parent {16384,5120}
    V3Slot norm;
    V3Slot output;                // whole parent
};

struct V3TextLayerPlan {
    V3Slot input_norm;
    V3Slot post_attention_norm;
    V3MlpPlan mlp;
    V3AttentionPlan attention{};
    V3GdnPlan gdn{};
    bool is_full_attention = false;
};

// Vision (v3 exposes q/k/v + biases as contiguous row-sub-ranges of one fused parent; every
// logical slot therefore binds a representative sub-range and materializes the whole parent).
struct V3VisionLayerPlan {
    V3Slot qkv;           // bind attention/query -> parent [3456,1152]
    V3Slot qkv_bias;      // bind attention/query_bias -> parent [3456]
    V3Slot output;
    V3Slot output_bias;
    V3Slot fc1;
    V3Slot fc1_bias;
    V3Slot fc2;
    V3Slot fc2_bias;
    V3Slot norm1_weight;
    V3Slot norm1_bias;
    V3Slot norm2_weight;
    V3Slot norm2_bias;
};

struct V3VisionPlan {
    V3Slot patch_embedding;
    V3Slot patch_embedding_bias;
    V3Slot position_embedding;
    std::array<V3VisionLayerPlan, qwen3_6::VisionBackboneConfig::layers> layers;
    V3Slot merger_fc1;
    V3Slot merger_fc1_bias;
    V3Slot merger_norm_weight;
    V3Slot merger_norm_bias;
    V3Slot merger_fc2;
    V3Slot merger_fc2_bias;
};

struct V3DFlash2DynamicConvPlan {
    V3Slot base_kernel;
    V3Slot kernel_projection;
};

struct V3DFlash2LayerPlan {
    V3Slot input_norm;
    V3DFlash2DynamicConvPlan attention_conv;
    V3Slot query_key_value;  // bind attention/query -> parent [6144,5120]; context_key/value slice it
    V3Slot query_norm;
    V3Slot key_norm;
    V3Slot attention_output;
    V3Slot post_attention_norm;
    V3DFlash2DynamicConvPlan mlp_conv;
    V3Slot gate_up;          // bind mlp/gate -> parent [34816,5120]
    V3Slot down;
};

struct V3DFlash2CandidateSelectorPlan {
    V3Slot hidden_projection;
    V3Slot predecessor_codebook;
    V3Slot successor_codebook;
};

struct V3DFlash2Plan {
    V3Slot feature_projection;
    V3Slot context_norm;
    std::array<V3DFlash2LayerPlan, qwen3_6::DFlash2Weights::layer_count> layers;
    V3Slot final_norm;
    V3DFlash2CandidateSelectorPlan candidate_selector;
};

struct V3MtpPlan {
    V3Slot input_projection;
    V3Slot embedding_norm;
    V3Slot hidden_norm;
    V3Slot input_norm;
    V3Slot query_key_gate_value;  // bind attention/query -> parent [14336,5120]; q|k|gate|value slices it
    V3Slot query_norm;
    V3Slot key_norm;
    V3Slot output;                // whole parent [5120,6144]
    V3Slot post_attention_norm;
    V3MlpPlan mlp;                // bind mlp/gate -> parent [34816,5120] (gate|up)
    V3Slot final_norm;
};

struct V3BindingPlan {
    qwen3_6::StartupFeatures features;
    qwen3_6::FrontendResources frontend;
    V3Slot token_embedding;
    std::array<V3TextLayerPlan, kTextLayers> text_layers;
    V3Slot final_norm;
    V3Slot output_head;
    V3Slot draft_head;           // proposal/head whole parent Q4G64_F16S [131072,5120] (mtp+dflash2)
    V3Slot draft_head_token_ids; // proposal/token_ids whole parent int32 [131072]
    std::optional<V3MtpPlan> mtp;
    std::optional<V3VisionPlan> vision;
    std::optional<V3DFlash2Plan> dflash2;
};

struct V3LoadPlan {
    V3BindingPlan bindings;
    artifact::v3::MaterializationPlan materialization;
};

V3LoadPlan bind_artifact_v3(artifact::v3::Binder& binder, WeightsProfile weights_profile,
                            qwen3_6::StartupFeatures features);

} // namespace ninfer::targets::qwen3_6_27b::detail
