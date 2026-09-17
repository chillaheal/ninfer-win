// Flash-Next S1 — real-geometry loader, gate (a).
//
// Binds + places EVERY object in the real 180B artifact through the Binder's
// finish() invariant (no unconsumed, no unplaced) WITHOUT materialization, so
// no GPU / VRAM is touched. The 1235 text objects bind at their Geometry-derived
// shapes (the meaningful shape check); the 367 non-text (vision / MTP) objects
// are consumed shape-agnostically from their own descriptors (shape-agnostic, but
// still placed by residency so the plan's device set is the complete backbone).
//
// Placement follows the canonical residency rule (detail::residency_class) for
// EVERY object, so the plan's device set is the full GpuResident backbone
// (text + mtp + vision = 1504 objects, ~6.02 GiB per the P0 audit / P11):
//   frontend/*                      -> Resource   (validate_only, never loaded)
//   .../mlp/experts/{gate_up,down}  -> HostExperts (validate_only; the pager
//                                           supplies them from the archive)
//   everything else                 -> GpuResident (materialize_on_device) --
//   includes the per-layer input_scale_divisor scalars (4 B each; the W4A4 route
//   reads 1/divisor) and the mtp/vision GpuResident weights.

#include "targets/qwen3_8_flash_next/impl/load/real_bindings.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/config.h"

#include <initializer_list>
#include <set>
#include <span>
#include <string>
#include <variant>

namespace ninfer::targets::qwen3_8_flash_next {

namespace {

using artifact::Binder;
using artifact::MaterializationPlan;
using artifact::NumericFormat;
using artifact::ObjectDescriptor;
using artifact::ObjectHandle;
using artifact::Reader;
using artifact::ResourceDescriptor;
using artifact::ResourceEncoding;
using artifact::StorageLayout;
using artifact::TensorDescriptor;
using artifact::TensorPlacement;

constexpr NumericFormat kBF16    = NumericFormat::BF16;
constexpr NumericFormat kFP32    = NumericFormat::FP32;
constexpr NumericFormat kNVFP4   = NumericFormat::NVFP4;
constexpr NumericFormat kFP8     = NumericFormat::FP8_E4M3FN_ROW_BF16S;
constexpr StorageLayout kContig     = StorageLayout::ContiguousLeV1;
constexpr StorageLayout kRowScale   = StorageLayout::RowScaleV1;
constexpr StorageLayout kBlockScale = StorageLayout::BlockScaleK16M128x4V1;

using detail::Geometry;
constexpr std::uint32_t kHidden      = Geometry::kHidden;            // 2560
constexpr std::uint32_t kVocab       = Geometry::kVocab;             // 248320
constexpr std::uint32_t kNumLayers   = Geometry::kNumLayers;         // 48
constexpr std::uint32_t kHcCount     = Geometry::kHcCount;           // 4
constexpr std::uint32_t kHcLowRank   = Geometry::kHcLowRank;         // 320
constexpr std::uint32_t kStreamWidth = kHcCount * kHidden;           // 10240
constexpr std::uint32_t kNumExperts  = Geometry::kNumExperts;        // 512
constexpr std::uint32_t kMoeInter    = Geometry::kMoeIntermediate;   // 640
constexpr std::uint32_t kSharedInter = Geometry::kSharedIntermediate;  // 640
constexpr std::uint32_t kNumAttHeads = Geometry::kNumAttentionHeads; // 24
constexpr std::uint32_t kNumKvHeads  = Geometry::kNumKvHeads;        // 2
constexpr std::uint32_t kHeadDim     = Geometry::kHeadDim;           // 256
constexpr std::uint32_t kGdnQkHeads  = Geometry::kGdnQkHeads;        // 16
constexpr std::uint32_t kGdnValHeads = Geometry::kGdnValueHeads;     // 48
constexpr std::uint32_t kGdnStateDim = Geometry::kGdnStateDim;       // 128
constexpr std::uint32_t kGdnConvW    = Geometry::kGdnConvWidth;      // 4
constexpr std::uint32_t kPleLayer    = Geometry::kPleLayer;          // 1

// Real-only derived dims (each verified against the P0 audit inventory):
constexpr std::uint32_t kAttnQueryRows = 12288;  // = 2*kNumAttHeads*kHeadDim (QSA query)
constexpr std::uint32_t kAttnKvRows    = kNumKvHeads * kHeadDim;      // 512
constexpr std::uint32_t kAttnOutCols   = kNumAttHeads * kHeadDim;     // 6144
constexpr std::uint32_t kAttnNormDim   = kHeadDim;                    // 256
constexpr std::uint32_t kIndexerQkRows = 640;                         // indexer/qk_proj rows
constexpr std::uint32_t kIndexerNorm   = 128;                         // indexer/{q,k}_norm
constexpr std::uint32_t kGdnAbRows     = 2 * kGdnValHeads;            // 96 (a + b, per value head)
constexpr std::uint32_t kGdnQkvzRows   =
    (2 * kGdnQkHeads + 2 * kGdnValHeads) * kGdnStateDim;              // 16384
constexpr std::uint32_t kGdnConvChan   = kStreamWidth;                // 10240
constexpr std::uint32_t kGdnNormDim    = kGdnStateDim;                // 128
constexpr std::uint32_t kGdnOutCols    = kGdnValHeads * kGdnStateDim; // 6144

}  // namespace

RealBindSummary real_bind_only(const std::filesystem::path& artifact_path) {
  Reader reader(artifact_path);
  const auto& ident = reader.identity();
  if (ident.model_id != "qwen3.8-flash-next" || ident.weights_id != "nvfp4") {
    throw std::runtime_error("flash_next real_bind: unexpected artifact identity " +
                             ident.model_id + ":" + ident.weights_id);
  }

  Binder b(reader);
  std::set<std::string> bound;

  // Bind helpers: consume + place one object, recording its name so the
  // non-text enumeration below does not re-bind it.
  auto bind = [&b, &bound](std::string_view name, NumericFormat fmt, StorageLayout lay,
                           std::initializer_list<std::uint64_t> shape, TensorPlacement place) {
    const std::span<const std::uint64_t> sp(shape.begin(), shape.size());
    const ObjectHandle h = b.require_tensor(name, fmt, lay, sp);
    bound.emplace(std::string(name));
    if (place == TensorPlacement::Device) {
      b.materialize_on_device(h);
    } else if (place == TensorPlacement::Host) {
      b.retain_on_host(h);
    } else {
      b.validate_only(h);
    }
    return h;
  };
  auto bind_res = [&b, &bound](std::string_view name) {
    const ObjectHandle h = b.require_resource(name, ResourceEncoding::RawBytesV1);
    bound.emplace(std::string(name));
    b.validate_only(h);
  };

  // Frontend resources (raw-bytes metadata) — bound + validated, never loaded.
  bind_res("frontend/tokenizer.json");
  bind_res("frontend/tokenizer_config.json");
  bind_res("frontend/chat_template.jinja");
  bind_res("frontend/generation_config.json");
  bind_res("frontend/preprocessor_config.json");
  bind_res("frontend/video_preprocessor_config.json");

  // Global embedding + output head (device FP8 row-scale).
  bind("text/token_embedding", kFP8, kRowScale, {kVocab, kHidden}, TensorPlacement::Device);
  bind("text/output_head", kFP8, kRowScale, {kVocab, kHidden}, TensorPlacement::Device);

  // Global hyper-connection mixer (device BF16).
  bind("text/hyper_connection_mixer/hc_norm", kBF16, kContig, {kStreamWidth}, TensorPlacement::Device);
  bind("text/hyper_connection_mixer/input_mix_weight_down", kBF16, kContig,
       {kHcLowRank, kStreamWidth}, TensorPlacement::Device);
  bind("text/hyper_connection_mixer/input_mix_weight_up", kBF16, kContig,
       {kStreamWidth, kHcLowRank}, TensorPlacement::Device);

  // Per-layer text tensors.
  for (std::uint32_t l = 0; l < kNumLayers; ++l) {
    const std::string pfx = "text/layers/" + std::to_string(l) + "/";

    // Hyper-connection sets (device BF16): attention + MLP.
    bind(pfx + "attn_hyper_connection/input_mix_weight_down", kBF16, kContig,
         {kHcLowRank, kStreamWidth}, TensorPlacement::Device);
    bind(pfx + "attn_hyper_connection/input_mix_weight_up", kBF16, kContig,
         {kStreamWidth, kHcLowRank}, TensorPlacement::Device);
    bind(pfx + "attn_hyper_connection/block_inject_weight", kBF16, kContig,
         {kHcCount, kStreamWidth}, TensorPlacement::Device);
    bind(pfx + "attn_hyper_connection/hc_norm", kBF16, kContig, {kStreamWidth},
         TensorPlacement::Device);
    bind(pfx + "mlp_hyper_connection/input_mix_weight_down", kBF16, kContig,
         {kHcLowRank, kStreamWidth}, TensorPlacement::Device);
    bind(pfx + "mlp_hyper_connection/input_mix_weight_up", kBF16, kContig,
         {kStreamWidth, kHcLowRank}, TensorPlacement::Device);
    bind(pfx + "mlp_hyper_connection/block_inject_weight", kBF16, kContig,
         {kHcCount, kStreamWidth}, TensorPlacement::Device);
    bind(pfx + "mlp_hyper_connection/hc_norm", kBF16, kContig, {kStreamWidth},
         TensorPlacement::Device);

    // MoE (device router / shared expert; fused routed experts via the pager).
    bind(pfx + "mlp/experts/routing", kBF16, kContig, {kNumExperts, kHidden},
         TensorPlacement::Device);
    bind(pfx + "mlp/shared_expert_gate", kBF16, kContig, {1, kHidden}, TensorPlacement::Device);
    bind(pfx + "mlp/shared_expert/gate_up", kNVFP4, kBlockScale, {2 * kSharedInter, kHidden},
         TensorPlacement::Device);
    bind(pfx + "mlp/shared_expert/down", kNVFP4, kBlockScale, {kHidden, kMoeInter},
         TensorPlacement::Device);
    bind(pfx + "mlp/experts/gate_up", kNVFP4, kBlockScale, {kNumExperts * 2 * kMoeInter, kHidden},
         TensorPlacement::ValidateOnly);
    bind(pfx + "mlp/experts/down", kNVFP4, kBlockScale, {kNumExperts * kHidden, kMoeInter},
         TensorPlacement::ValidateOnly);
    // Activation divisors (W4A4 route reads 1/divisor) — GpuResident (device).
    bind(pfx + "mlp/shared_expert/gate_up_projection/input_scale_divisor", kFP32, kContig, {},
         TensorPlacement::Device);
    bind(pfx + "mlp/shared_expert/down_projection/input_scale_divisor", kFP32, kContig, {},
         TensorPlacement::Device);
    bind(pfx + "mlp/experts/gate_up_projection/input_scale_divisor", kFP32, kContig, {},
         TensorPlacement::Device);
    bind(pfx + "mlp/experts/down_projection/input_scale_divisor", kFP32, kContig, {},
         TensorPlacement::Device);

    // Attention (full / QSA layers: l % 4 == 3) or GDN (the other 36).
    if (detail::is_full_attention_layer(l)) {
      const std::string at = pfx + "attention/";
      bind(at + "query", kFP8, kRowScale, {kAttnQueryRows, kHidden}, TensorPlacement::Device);
      bind(at + "key", kFP8, kRowScale, {kAttnKvRows, kHidden}, TensorPlacement::Device);
      bind(at + "value", kFP8, kRowScale, {kAttnKvRows, kHidden}, TensorPlacement::Device);
      bind(at + "output", kFP8, kRowScale, {kHidden, kAttnOutCols}, TensorPlacement::Device);
      bind(at + "query_norm", kBF16, kContig, {kAttnNormDim}, TensorPlacement::Device);
      bind(at + "key_norm", kBF16, kContig, {kAttnNormDim}, TensorPlacement::Device);
      bind(at + "indexer/qk_proj", kBF16, kContig, {kIndexerQkRows, kHidden},
           TensorPlacement::Device);
      bind(at + "indexer/q_norm", kBF16, kContig, {kIndexerNorm}, TensorPlacement::Device);
      bind(at + "indexer/k_norm", kBF16, kContig, {kIndexerNorm}, TensorPlacement::Device);
    } else {
      const std::string g = pfx + "gdn/";
      bind(g + "a_log", kFP32, kContig, {kGdnValHeads}, TensorPlacement::Device);
      bind(g + "dt_bias", kFP32, kContig, {kGdnValHeads}, TensorPlacement::Device);
      bind(g + "convolution", kBF16, kContig, {kGdnConvW, kGdnConvChan}, TensorPlacement::Device);
      bind(g + "a_b_projection", kBF16, kContig, {kGdnAbRows, kHidden}, TensorPlacement::Device);
      bind(g + "query_key_value_z", kFP8, kRowScale, {kGdnQkvzRows, kHidden},
           TensorPlacement::Device);
      bind(g + "norm", kBF16, kContig, {kGdnNormDim}, TensorPlacement::Device);
      bind(g + "output", kFP8, kRowScale, {kHidden, kGdnOutCols}, TensorPlacement::Device);
    }

    // PLE (layer 1 only).
    if (l == kPleLayer) {
      const std::string ple = pfx + "ple/";
      bind(ple + "convolution", kBF16, kContig, {kGdnConvW, kStreamWidth}, TensorPlacement::Device);
      bind(ple + "key_projection", kBF16, kContig, {kStreamWidth, kHidden}, TensorPlacement::Device);
      bind(ple + "value_projection", kBF16, kContig, {kHidden, kHidden}, TensorPlacement::Device);
      bind(ple + "norm_conv", kBF16, kContig, {kStreamWidth}, TensorPlacement::Device);
      bind(ple + "norm_key", kBF16, kContig, {kStreamWidth}, TensorPlacement::Device);
      bind(ple + "norm_query", kBF16, kContig, {kStreamWidth}, TensorPlacement::Device);
    }
  }

  // Non-text objects (vision / MTP): not in the P12 text forward. Consume each
  // shape-agnostically from its own descriptor (the meaningful shape check is on
  // the text path); place by residency so the mtp/vision GpuResident tensors
  // join the device set (completing the 1504-object backbone) while finish()
  // still sees every object placed.
  std::size_t nontext = 0;
  for (const ObjectDescriptor& obj : reader.objects()) {
    const std::string_view name = artifact::object_name(obj);
    if (bound.count(std::string(name))) continue;
    if (const TensorDescriptor* latched = std::get_if<TensorDescriptor>(&obj)) {
      const std::span<const std::uint64_t> sp(latched->shape.data(), latched->shape.size());
      const ObjectHandle h = b.require_tensor(name, latched->format, latched->layout, sp);
      if (detail::residency_class(name) == detail::ResidencyClass::GpuResident) {
        b.materialize_on_device(h);
      } else {
        b.validate_only(h);
      }
      ++nontext;
    } else {
      const ResourceDescriptor* r = std::get_if<ResourceDescriptor>(&obj);
      const ObjectHandle h = b.require_resource(name, r->encoding);
      b.validate_only(h);
      ++nontext;
    }
  }

  // Finish: every object is now consumed + placed (no materialization).
  const MaterializationPlan plan = b.finish();

  RealBindSummary s;
  s.object_count          = plan.object_count;
  s.resource_count        = 6;
  s.text_tensor_count     = 1235;
  s.nontext_tensor_count  = nontext;
  s.device_count          = plan.device_objects.size();
  s.validate_count        = plan.object_count - plan.device_objects.size() - plan.host_objects.size();
  s.device_capacity_bytes = plan.device_capacity_bytes;
  return s;
  }

}  // namespace ninfer::targets::qwen3_8_flash_next
