#pragma once
// P2: locked Flash-Next geometry + canonical residency rules.
//
// Source of truth: the converter's locked geometry (tools/convert/qwen3_8_flash_next/
// config.py) and the P0 real-inventory audit (out/flash_next_dev/audit_objects.csv:
// 6 frontend resources / 1504 gpu_resident / 98 host_experts objects).

#include <cstdint>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {

enum class ResidencyClass : std::uint8_t {
    GpuResident,   // device-resident weights (backbone, attention, GDN, shared expert, divisors)
    HostExperts,   // fused 512-expert matrices: paged from host pinned memory (P3)
    Resource,      // frontend resources (tokenizer etc.): validated only, never loaded
};

struct Geometry {
    static constexpr std::uint32_t kHidden               = 2560;
    static constexpr std::uint32_t kVocab                = 248320;
    static constexpr std::uint32_t kNumLayers            = 48;   // text layers
    static constexpr std::uint32_t kMtpLayers            = 1;    // MTP layer 0 = full attention + MoE
    static constexpr std::uint32_t kFullAttentionLayers  = 12;   // text layers L % 4 == 3 (3,7,...,47)
    static constexpr std::uint32_t kNumAttentionHeads    = 24;
    static constexpr std::uint32_t kNumKvHeads           = 2;    // QSA/GQA: 2 KV heads
    static constexpr std::uint32_t kHeadDim              = 256;
    static constexpr std::uint32_t kNumExperts           = 512;
    static constexpr std::uint32_t kExpertsPerToken      = 10;
    static constexpr std::uint32_t kMoeIntermediate      = 640;
    static constexpr std::uint32_t kSharedIntermediate   = 640;
    static constexpr std::uint32_t kHcCount              = 4;
    static constexpr std::uint32_t kHcLowRank            = 320;
    static constexpr std::uint32_t kPleLayer             = 1;
    static constexpr std::uint32_t kGdnQkHeads           = 16;  // GDN: 16 keys x 128
    static constexpr std::uint32_t kGdnValueHeads        = 48;  // GDN: 48 values x 128
    static constexpr std::uint32_t kGdnStateDim          = 128;
    static constexpr std::uint32_t kGdnConvWidth         = 4;   // causal conv width
    static constexpr std::uint32_t kGdnLayers = kNumLayers - kFullAttentionLayers;  // 36
};

// Native context ceiling: the P13 QSA paged-KV default window is 130 pages of 64
// tokens each. This is the DEFAULT max_context, not a hard cap (see below).
constexpr std::uint32_t kNativeContext = 8320;  // 130 x 64

// Full QSA paged-KV window (the T1-T5 262k reconstruction). RealProgram sizes the
// paged pool dynamically (ceil(max_context/64) pages), so the serve window is NOT
// capped at kNativeContext — it extends to the full 262k region the model is built
// to serve, matching the 27B/35B kNativeContext. The probe reports this as the KV
// fit so the GUI can launch the serve here (and honor any typed context up to it).
constexpr std::uint32_t kPagedMaxContext = 262144;

[[nodiscard]] constexpr bool is_full_attention_layer(std::uint32_t layer) {
    return layer % 4 == 3;
}

// Per-token KV cost of the 12 full/QSA attention layers: K+V, GQA
// (2 KV heads x 256 head dim). The QSA indexer's paged KV is a P7 refinement
// of this model.
[[nodiscard]] constexpr std::uint64_t kv_bytes_per_token(std::uint32_t dtype_element_bytes) {
    return static_cast<std::uint64_t>(Geometry::kFullAttentionLayers) * 2u *
           static_cast<std::uint64_t>(Geometry::kNumKvHeads) *
           static_cast<std::uint64_t>(Geometry::kHeadDim) * dtype_element_bytes;
}

// Per-layer, per-sequence GDN recurrent state bytes: FP32 [128, 128, 48 value
// heads] = 3,145,728 bytes (786,432 FP32 elements). Matches the shared
// gated_delta_net Op's state tensor
// [state_dim, state_dim, value_heads] at the production head counts (P6).
[[nodiscard]] constexpr std::uint64_t gdn_state_bytes_per_layer() {
    return static_cast<std::uint64_t>(Geometry::kGdnStateDim) *
           static_cast<std::uint64_t>(Geometry::kGdnStateDim) *
           static_cast<std::uint64_t>(Geometry::kGdnValueHeads) * 4u;
}

// Canonical residency rule (P0 audit):
//  - frontend/*            -> Resource (validated only)
//  - .../mlp/experts/{gate_up,down} (fused 512-expert matrices, text + MTP: 98
//    objects)                   -> HostExperts (paged from host pinned memory)
//  - everything else (shared_expert, routing, divisors, attention, GDN, norms,
//    embedding, output head)     -> GpuResident
[[nodiscard]] inline ResidencyClass residency_class(std::string_view object_name) {
    if (object_name.starts_with("frontend/")) return ResidencyClass::Resource;
    if (object_name.ends_with("/mlp/experts/gate_up") ||
        object_name.ends_with("/mlp/experts/down")) {
        return ResidencyClass::HostExperts;
    }
    return ResidencyClass::GpuResident;
}

}  // namespace ninfer::targets::qwen3_8_flash_next::detail
