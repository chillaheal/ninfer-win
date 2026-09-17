// Flash-Next P9 — loader (see bindings.h for the lifetime contract).
//
// Binds + places EVERY object in the mini artifact (the Binder's finish()
// invariant: no unconsumed, no unplaced). Device tensors are materialized to
// the device arena; the hyper-connection / GDN-conv / PLE-conv BF16 tensors are
// retained on host and converted (FP32 / channel-first) into LoadedModel-owned
// DeviceBuffers; the routed-expert fused NVFP4 objects, the global mixer, the
// PLE projections, and the input_scale_divisor objects are validate_only (the
// pager supplies the routed experts from the archive; the rest are unused by the
// v1 W4A16 path). The four NVFP4 weight divisors are the per-object inline
// weight words (shared + fused-routed); the input_scale_divisor objects are
// activation divisors used only by the W4A4 route, so they are bound but not
// read.

#include "targets/qwen3_8_flash_next/impl/load/bindings.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "ninfer/ops/sparse_moe_nvfp4.h"

namespace ninfer::targets::qwen3_8_flash_next {

namespace {

using artifact::Binder;
using artifact::BlockScaleGeometry;
using artifact::MaterializationPlan;
using artifact::MaterializedArtifact;
using artifact::NumericFormat;
using artifact::ObjectHandle;
using artifact::Reader;
using artifact::ResourceEncoding;
using artifact::RowScaleGeometry;
using artifact::StorageLayout;

using mini::kHidden;
using mini::kHcCount;
using mini::kHcLowRank;
using mini::kLayers;
using mini::kMoEExperts;
using mini::kMoEIntermediate;
using mini::kPleConvKernel;
using mini::kQsaHeadDim;
using mini::kQsaKvHeads;
using mini::kStreamWidth;
using mini::kVocab;

constexpr NumericFormat kBF16 = NumericFormat::BF16;
constexpr NumericFormat kFP32 = NumericFormat::FP32;
constexpr NumericFormat kNVFP4 = NumericFormat::NVFP4;
constexpr NumericFormat kFP8   = NumericFormat::FP8_E4M3FN_ROW_BF16S;
constexpr StorageLayout kContig     = StorageLayout::ContiguousLeV1;
constexpr StorageLayout kRowScale   = StorageLayout::RowScaleV1;
constexpr StorageLayout kBlockScale = StorageLayout::BlockScaleK16M128x4V1;

// Exact BF16 -> FP32 on the host (BF16 is the top 16 bits of an IEEE FP32).
float bf16_to_f32(std::uint16_t bits) {
  const std::uint32_t f32 = static_cast<std::uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &f32, sizeof(out));
  return out;
}

ObjectHandle bind_device(Binder& b, std::string_view name, NumericFormat fmt, StorageLayout lay,
                         std::initializer_list<std::uint64_t> shape) {
  const std::span<const std::uint64_t> sp(shape.begin(), shape.size());
  const ObjectHandle h = b.require_tensor(name, fmt, lay, sp);
  b.materialize_on_device(h);
  return h;
}

ObjectHandle bind_host(Binder& b, std::string_view name, NumericFormat fmt, StorageLayout lay,
                       std::initializer_list<std::uint64_t> shape) {
  const std::span<const std::uint64_t> sp(shape.begin(), shape.size());
  const ObjectHandle h = b.require_tensor(name, fmt, lay, sp);
  b.retain_on_host(h);
  return h;
}

ObjectHandle bind_validate(Binder& b, std::string_view name, NumericFormat fmt, StorageLayout lay,
                           std::initializer_list<std::uint64_t> shape) {
  const std::span<const std::uint64_t> sp(shape.begin(), shape.size());
  const ObjectHandle h = b.require_tensor(name, fmt, lay, sp);
  b.validate_only(h);
  return h;
}

ObjectHandle bind_resource(Binder& b, std::string_view name) {
  const ObjectHandle h = b.require_resource(name, ResourceEncoding::RawBytesV1);
  b.validate_only(h);
  return h;
}

// NVFP4 per-object weight divisor (little-endian FP32 word at the geometry
// offset), read from the host-backed payload before materialization.
float read_divisor(Binder& b, ObjectHandle h, std::uint32_t rows, std::uint32_t cols,
                   const char* label) {
  const std::array<std::uint64_t, 2> shape{rows, cols};
  const BlockScaleGeometry g = artifact::block_scale_geometry(kNVFP4, shape);
  const auto payload = b.payload(h);
  if (payload.data.size() < g.weight_divisor_offset + 4) {
    throw artifact::ArtifactError(std::string("NVFP4 divisor word out of range: ") + label);
  }
  const std::byte* p = payload.data.data() + g.weight_divisor_offset;
  const std::uint32_t bits =
      static_cast<std::uint32_t>(static_cast<unsigned char>(p[0])) |
      (static_cast<std::uint32_t>(static_cast<unsigned char>(p[1])) << 8) |
      (static_cast<std::uint32_t>(static_cast<unsigned char>(p[2])) << 16) |
      (static_cast<std::uint32_t>(static_cast<unsigned char>(p[3])) << 24);
  float v;
  std::memcpy(&v, &bits, sizeof(v));
  if (!(v > 0.0F) || !std::isfinite(v)) {
    throw artifact::ArtifactError(std::string("non-positive or non-finite NVFP4 divisor: ") + label);
  }
  return v;
}

Fp8RowScaleWeight make_fp8(const MaterializedArtifact& backing, ObjectHandle h, std::uint32_t rows,
                           std::uint32_t cols) {
  const std::array<std::uint64_t, 2> shape{rows, cols};
  const RowScaleGeometry g = artifact::row_scale_geometry(kFP8, shape);
  const std::uint8_t* base = static_cast<const std::uint8_t*>(backing.device_data(h));
  return Fp8RowScaleWeight{
      .codes = base,
      .scale = reinterpret_cast<const std::uint16_t*>(base + g.scale_plane_offset),
      .rows = rows,
      .cols = cols,
  };
}

// Shared-expert NVFP4 code/scale planes (device-resident, canonical layout).
struct Nvfp4Planes {
  const std::uint8_t* codes;
  const std::uint8_t* scales;
};

Nvfp4Planes make_nvfp4(const MaterializedArtifact& backing, ObjectHandle h, std::uint32_t rows,
                       std::uint32_t cols) {
  const std::array<std::uint64_t, 2> shape{rows, cols};
  const BlockScaleGeometry g = artifact::block_scale_geometry(kNVFP4, shape);
  const std::uint8_t* base = static_cast<const std::uint8_t*>(backing.device_data(h));
  return Nvfp4Planes{.codes = base, .scales = base + g.scale_plane_offset};
}

Bf16Weight make_bf16(const MaterializedArtifact& backing, ObjectHandle h, std::uint32_t rows,
                     std::uint32_t cols) {
  return Bf16Weight{.data = static_cast<const std::uint16_t*>(backing.device_data(h)),
                    .rows = rows,
                    .cols = cols};
}

// Host-retained BF16 (row-major, `elements`) -> a device FP32 buffer appended
// to `derived`; returns the device pointer.
const float* host_bf16_to_device_f32(const MaterializedArtifact& backing, ObjectHandle h,
                                     std::size_t elements, std::vector<DeviceBuffer>& derived) {
  const auto src = backing.resource_bytes(h);
  const std::uint16_t* b = reinterpret_cast<const std::uint16_t*>(src.data());
  std::vector<float> f(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    f[i] = bf16_to_f32(b[i]);
  }
  derived.emplace_back(elements * sizeof(float));
  derived.back().copy_from_host(f.data(), elements * sizeof(float));
  return static_cast<const float*>(derived.back().p);
}

// GDN conv: host [width][channels] BF16 (tap-major) -> device [channels][width]
// (channel-first) BF16 buffer appended to `derived`; returns the device pointer.
const std::uint16_t* host_bf16_to_device_transpose(const MaterializedArtifact& backing,
                                                   ObjectHandle h, std::uint32_t width,
                                                   std::uint32_t channels,
                                                   std::vector<DeviceBuffer>& derived) {
  const auto src = backing.resource_bytes(h);
  const std::uint16_t* b = reinterpret_cast<const std::uint16_t*>(src.data());
  std::vector<std::uint16_t> t(static_cast<std::size_t>(channels) * width);
  for (std::uint32_t c = 0; c < channels; ++c) {
    for (std::uint32_t w = 0; w < width; ++w) {
      t[static_cast<std::size_t>(c) * width + w] = b[static_cast<std::size_t>(w) * channels + c];
    }
  }
  derived.emplace_back(t.size() * sizeof(std::uint16_t));
  derived.back().copy_from_host(t.data(), t.size() * sizeof(std::uint16_t));
  return static_cast<const std::uint16_t*>(derived.back().p);
}

// PLE conv: host [K][1024] BF16 -> first kPleEmbedDim channels of each tap as
// [K][128] FP32 (host, for NgramOpParams.conv_weights).
std::array<std::array<float, mini::kPleEmbedDim>, kPleConvKernel> ple_conv_host(
    const std::span<const std::byte>& src) {
  const std::uint16_t* b = reinterpret_cast<const std::uint16_t*>(src.data());
  std::array<std::array<float, mini::kPleEmbedDim>, kPleConvKernel> out;
  for (std::uint32_t tap = 0; tap < kPleConvKernel; ++tap) {
    for (std::uint32_t c = 0; c < mini::kPleEmbedDim; ++c) {
      out[tap][c] = bf16_to_f32(b[static_cast<std::size_t>(tap) * kStreamWidth + c]);
    }
  }
  return out;
}

}  // namespace

LoadedModel::~LoadedModel() = default;

std::unique_ptr<LoadedModel> LoadedModel::open(const std::filesystem::path& artifact_path,
                                               const std::filesystem::path& ngram_path,
                                               const std::filesystem::path& archive_path,
                                               DeviceContext& device) {
  Reader reader(artifact_path);
  const auto& ident = reader.identity();
  if (ident.model_id != "qwen3.8-flash-next" || ident.weights_id != "nvfp4") {
    throw std::runtime_error("flash_next: unexpected artifact identity " + ident.model_id + ":" +
                             ident.weights_id);
  }

  auto lm = std::unique_ptr<LoadedModel>(new LoadedModel());
  Binder b(reader);

  // Resources (raw-bytes front-end metadata) — bound + validated, never loaded.
  bind_resource(b, "frontend/tokenizer.json");
  bind_resource(b, "frontend/tokenizer_config.json");
  bind_resource(b, "frontend/chat_template.jinja");
  bind_resource(b, "frontend/generation_config.json");
  bind_resource(b, "frontend/preprocessor_config.json");
  bind_resource(b, "frontend/video_preprocessor_config.json");

  // Global embedding + output head (device FP8 row-scale).
  const ObjectHandle h_embed =
      bind_device(b, "text/token_embedding", kFP8, kRowScale, {kVocab, kHidden});
  const ObjectHandle h_out   = bind_device(b, "text/output_head", kFP8, kRowScale, {kVocab, kHidden});

  // Global hyper-connection mixer — v1 replicates the embedding into all
  // branches, so this is bound but unused.
  bind_validate(b, "text/hyper_connection_mixer/hc_norm", kBF16, kContig, {kStreamWidth});
  bind_validate(b, "text/hyper_connection_mixer/input_mix_weight_down", kBF16, kContig,
                {kHcLowRank, kStreamWidth});
  bind_validate(b, "text/hyper_connection_mixer/input_mix_weight_up", kBF16, kContig,
                {kStreamWidth, kHcLowRank});

  // Per-layer handles (device tensors resolved after materialize; host tensors
  // converted below).
  std::array<ObjectHandle, kLayers> h_moe_router, h_moe_shared_gate, h_moe_shared_gu, h_moe_shared_dn;
  std::array<ObjectHandle, kLayers> h_attn_down, h_attn_up, h_attn_inj, h_attn_norm;
  std::array<ObjectHandle, kLayers> h_mlp_down, h_mlp_up, h_mlp_inj, h_mlp_norm;
  std::array<ObjectHandle, mini::kGdnLayers> h_gdn_alog, h_gdn_dt, h_gdn_conv, h_gdn_ab, h_gdn_qkvz,
      h_gdn_norm, h_gdn_out;
  std::array<float, kLayers> shared_gu_div, shared_dn_div, routed_gu_div, routed_dn_div;

  for (std::uint32_t l = 0; l < kLayers; ++l) {
    const std::string prefix = "text/layers/" + std::to_string(l) + "/";

    // Hyper-connection sets (host BF16 -> device FP32).
    h_attn_down[l] = bind_host(b, prefix + "attn_hyper_connection/input_mix_weight_down", kBF16,
                               kContig, {kHcLowRank, kStreamWidth});
    h_attn_up[l]   = bind_host(b, prefix + "attn_hyper_connection/input_mix_weight_up", kBF16,
                               kContig, {kStreamWidth, kHcLowRank});
    h_attn_inj[l]  = bind_host(b, prefix + "attn_hyper_connection/block_inject_weight", kBF16,
                               kContig, {kHcCount, kStreamWidth});
    h_attn_norm[l] = bind_host(b, prefix + "attn_hyper_connection/hc_norm", kBF16, kContig,
                               {kStreamWidth});
    h_mlp_down[l]  = bind_host(b, prefix + "mlp_hyper_connection/input_mix_weight_down", kBF16,
                               kContig, {kHcLowRank, kStreamWidth});
    h_mlp_up[l]    = bind_host(b, prefix + "mlp_hyper_connection/input_mix_weight_up", kBF16, kContig,
                               {kStreamWidth, kHcLowRank});
    h_mlp_inj[l]   = bind_host(b, prefix + "mlp_hyper_connection/block_inject_weight", kBF16,
                               kContig, {kHcCount, kStreamWidth});
    h_mlp_norm[l]  = bind_host(b, prefix + "mlp_hyper_connection/hc_norm", kBF16, kContig,
                               {kStreamWidth});

    // MoE (device router/shared; routed fused NVFP4 via the pager).
    h_moe_router[l]      = bind_device(b, prefix + "mlp/experts/routing", kBF16, kContig,
                                       {kMoEExperts, kHidden});
    h_moe_shared_gate[l] = bind_device(b, prefix + "mlp/shared_expert_gate", kBF16, kContig,
                                       {1, kHidden});
    h_moe_shared_gu[l]   = bind_device(b, prefix + "mlp/shared_expert/gate_up", kNVFP4, kBlockScale,
                                       {2 * kMoEIntermediate, kHidden});
    h_moe_shared_dn[l]   = bind_device(b, prefix + "mlp/shared_expert/down", kNVFP4, kBlockScale,
                                       {kHidden, kMoEIntermediate});
    const ObjectHandle h_routed_gu =
        bind_validate(b, prefix + "mlp/experts/gate_up", kNVFP4, kBlockScale,
                      {kMoEExperts * 2 * kMoEIntermediate, kHidden});
    const ObjectHandle h_routed_dn =
        bind_validate(b, prefix + "mlp/experts/down", kNVFP4, kBlockScale,
                      {kMoEExperts * kHidden, kMoEIntermediate});
    bind_validate(b, prefix + "mlp/shared_expert/gate_up_projection/input_scale_divisor", kFP32,
                  kContig, {});
    bind_validate(b, prefix + "mlp/shared_expert/down_projection/input_scale_divisor", kFP32,
                  kContig, {});
    bind_validate(b, prefix + "mlp/experts/gate_up_projection/input_scale_divisor", kFP32, kContig,
                  {});
    bind_validate(b, prefix + "mlp/experts/down_projection/input_scale_divisor", kFP32, kContig,
                  {});

    shared_gu_div[l] = read_divisor(b, h_moe_shared_gu[l], 2 * kMoEIntermediate, kHidden,
                                    (prefix + "shared_expert/gate_up").c_str());
    shared_dn_div[l] = read_divisor(b, h_moe_shared_dn[l], kHidden, kMoEIntermediate,
                                    (prefix + "shared_expert/down").c_str());
    routed_gu_div[l] =
        read_divisor(b, h_routed_gu, kMoEExperts * 2 * kMoEIntermediate, kHidden,
                     (prefix + "experts/gate_up").c_str());
    routed_dn_div[l] =
        read_divisor(b, h_routed_dn, kMoEExperts * kHidden, kMoEIntermediate,
                     (prefix + "experts/down").c_str());
  }

  // GDN (layers 0..2).
  for (std::uint32_t l = 0; l < mini::kGdnLayers; ++l) {
    const std::string prefix = "text/layers/" + std::to_string(l) + "/";
    h_gdn_alog[l] = bind_device(b, prefix + "gdn/a_log", kFP32, kContig, {mini::kGdnValueHeads});
    h_gdn_dt[l]   = bind_device(b, prefix + "gdn/dt_bias", kFP32, kContig, {mini::kGdnValueHeads});
    h_gdn_conv[l] = bind_host(b, prefix + "gdn/convolution", kBF16, kContig,
                              {mini::kGdnConvWidth, mini::kGdnConvChannels});
    h_gdn_ab[l]   = bind_device(b, prefix + "gdn/a_b_projection", kBF16, kContig,
                                {mini::kGdnAbRows, kHidden});
    h_gdn_qkvz[l] = bind_device(b, prefix + "gdn/query_key_value_z", kFP8, kRowScale,
                                {mini::kGdnQkvzRows, kHidden});
    h_gdn_norm[l] = bind_device(b, prefix + "gdn/norm", kBF16, kContig, {32});
    h_gdn_out[l]  = bind_device(b, prefix + "gdn/output", kFP8, kRowScale,
                                {kHidden, mini::kGdnOutputCols});
  }

  // PLE (layer 1): conv is used; the projection/norm tensors are bound but
  // unused by the v1 gather+add.
  ObjectHandle h_ple_conv;
  {
    const std::string ple = "text/layers/" + std::to_string(mini::kPleLayer) + "/ple/";
    h_ple_conv = bind_host(b, ple + "convolution", kBF16, kContig, {kPleConvKernel, kStreamWidth});
    bind_validate(b, ple + "key_projection", kBF16, kContig, {kStreamWidth, kHidden});
    bind_validate(b, ple + "value_projection", kBF16, kContig, {kHidden, kHidden});
    bind_validate(b, ple + "norm_conv", kBF16, kContig, {kStreamWidth});
    bind_validate(b, ple + "norm_key", kBF16, kContig, {kStreamWidth});
    bind_validate(b, ple + "norm_query", kBF16, kContig, {kStreamWidth});
  }

  // QSA (layer 3).
  ObjectHandle h_q_query, h_q_key, h_q_value, h_q_out, h_q_qnorm, h_q_knorm, h_q_qkproj, h_q_qin,
      h_q_kin;
  {
    const std::string qsa = "text/layers/" + std::to_string(mini::kQsaLayer) + "/attention/";
    h_q_query = bind_device(b, qsa + "query", kFP8, kRowScale, {kHidden, kHidden});
    h_q_key   = bind_device(b, qsa + "key", kFP8, kRowScale,
                            {kQsaKvHeads * kQsaHeadDim, kHidden});
    h_q_value = bind_device(b, qsa + "value", kFP8, kRowScale,
                            {kQsaKvHeads * kQsaHeadDim, kHidden});
    h_q_out   = bind_device(b, qsa + "output", kFP8, kRowScale, {kHidden, 128});
    h_q_qnorm = bind_device(b, qsa + "query_norm", kBF16, kContig, {kQsaHeadDim});
    h_q_knorm = bind_device(b, qsa + "key_norm", kBF16, kContig, {kQsaHeadDim});
    h_q_qkproj = bind_device(b, qsa + "indexer/qk_proj", kBF16, kContig, {160, kHidden});
    h_q_qin   = bind_device(b, qsa + "indexer/q_norm", kBF16, kContig, {kQsaHeadDim});
    h_q_kin   = bind_device(b, qsa + "indexer/k_norm", kBF16, kContig, {kQsaHeadDim});
  }

  // Finish: every object is now consumed + placed.
  const MaterializationPlan plan = b.finish();
  lm->backing_ = artifact::materialize(reader, plan, device);
  MaterializedArtifact& back = lm->backing_;

  // Resolve device pointers into the ModelView.
  lm->view_.embedding   = make_fp8(back, h_embed, kVocab, kHidden);
  lm->view_.output_head = make_fp8(back, h_out, kVocab, kHidden);

  for (std::uint32_t l = 0; l < kLayers; ++l) {
    LayerWeights& lw = lm->view_.layers[l];

    // Hyper-connection sets: host BF16 -> device FP32.
    lw.attn_hc.w_down =
        host_bf16_to_device_f32(back, h_attn_down[l], kHcLowRank * kStreamWidth, lm->derived_);
    lw.attn_hc.w_up   = host_bf16_to_device_f32(back, h_attn_up[l], kStreamWidth * kHcLowRank,
                                                lm->derived_);
    lw.attn_hc.w_inj  =
        host_bf16_to_device_f32(back, h_attn_inj[l], kHcCount * kStreamWidth, lm->derived_);
    lw.attn_hc.norm_w = host_bf16_to_device_f32(back, h_attn_norm[l], kStreamWidth, lm->derived_);
    lw.mlp_hc.w_down  =
        host_bf16_to_device_f32(back, h_mlp_down[l], kHcLowRank * kStreamWidth, lm->derived_);
    lw.mlp_hc.w_up    = host_bf16_to_device_f32(back, h_mlp_up[l], kStreamWidth * kHcLowRank,
                                                lm->derived_);
    lw.mlp_hc.w_inj   =
        host_bf16_to_device_f32(back, h_mlp_inj[l], kHcCount * kStreamWidth, lm->derived_);
    lw.mlp_hc.norm_w  = host_bf16_to_device_f32(back, h_mlp_norm[l], kStreamWidth, lm->derived_);

    // MoE device weights.
    lw.mlp.router_bf16      = make_bf16(back, h_moe_router[l], kMoEExperts, kHidden).data;
    lw.mlp.shared_gate_bf16 = make_bf16(back, h_moe_shared_gate[l], 1, kHidden).data;
    const Nvfp4Planes gu    = make_nvfp4(back, h_moe_shared_gu[l], 2 * kMoEIntermediate, kHidden);
    lw.mlp.shared_gu_codes  = gu.codes;
    lw.mlp.shared_gu_scales = gu.scales;
    const Nvfp4Planes dn    = make_nvfp4(back, h_moe_shared_dn[l], kHidden, kMoEIntermediate);
    lw.mlp.shared_dn_codes  = dn.codes;
    lw.mlp.shared_dn_scales = dn.scales;
    lw.mlp.shared_gu_divisor = shared_gu_div[l];
    lw.mlp.shared_dn_divisor = shared_dn_div[l];
    lw.mlp.routed_gu_divisor = routed_gu_div[l];
    lw.mlp.routed_dn_divisor = routed_dn_div[l];
  }

  for (std::uint32_t l = 0; l < mini::kGdnLayers; ++l) {
    GdnLayerWeights g;
    g.qkvz             = make_fp8(back, h_gdn_qkvz[l], mini::kGdnQkvzRows, kHidden);
    g.output           = make_fp8(back, h_gdn_out[l], kHidden, mini::kGdnOutputCols);
    g.a_b_projection   = make_bf16(back, h_gdn_ab[l], mini::kGdnAbRows, kHidden);
    g.a_log            = static_cast<const float*>(back.device_data(h_gdn_alog[l]));
    g.dt_bias          = static_cast<const float*>(back.device_data(h_gdn_dt[l]));
    g.norm             = make_bf16(back, h_gdn_norm[l], 1, 32).data;
    g.conv             = host_bf16_to_device_transpose(back, h_gdn_conv[l], mini::kGdnConvWidth,
                                                       mini::kGdnConvChannels, lm->derived_);
    lm->view_.layers[l].gdn = std::move(g);
  }

  {
    QsaLayerWeights q;
    q.query             = make_fp8(back, h_q_query, kHidden, kHidden);
    q.key               = make_fp8(back, h_q_key, kQsaKvHeads * kQsaHeadDim, kHidden);
    q.value             = make_fp8(back, h_q_value, kQsaKvHeads * kQsaHeadDim, kHidden);
    q.output            = make_fp8(back, h_q_out, kHidden, 128);
    q.query_norm        = make_bf16(back, h_q_qnorm, 1, kQsaHeadDim).data;
    q.key_norm          = make_bf16(back, h_q_knorm, 1, kQsaHeadDim).data;
    q.indexer_qk_proj   = make_bf16(back, h_q_qkproj, 160, kHidden);
    q.indexer_q_norm    = make_bf16(back, h_q_qin, 1, kQsaHeadDim).data;
    q.indexer_k_norm    = make_bf16(back, h_q_kin, 1, kQsaHeadDim).data;
    lm->view_.layers[mini::kQsaLayer].qsa = std::move(q);
  }

  // PLE conv weights (host [4][128] FP32).
  lm->view_.ple_conv_weights = ple_conv_host(back.resource_bytes(h_ple_conv));

  // PLE table + expert archive + pager.
  lm->ple_table_ = std::unique_ptr<ops::NgramEmbeddingTable>(
      new ops::NgramEmbeddingTable(ops::NgramEmbeddingTable::open(ngram_path)));
  lm->archive_ = std::unique_ptr<paging::ExpertArchive>(
      new paging::ExpertArchive(paging::ExpertArchive::open(archive_path)));
  const auto& arch = *lm->archive_;
  if (arch.num_layers() != kLayers || arch.experts_per_layer() != kMoEExperts ||
      arch.bytes_per_expert() !=
          ops::sparse_moe_nvfp4_expert_bytes(ops::SparseMoeNvfp4Geometry{8, 2, 256, 64})) {
    throw std::runtime_error("flash_next: expert archive geometry mismatch");
  }
  const paging::PagerConfig pager_config{1, 32, 0};  // host 1 / device 32 / auto slot
  lm->pager_ = std::unique_ptr<paging::Pager>(new paging::Pager(arch, pager_config));

  lm->view_.ple_table     = lm->ple_table_.get();
  lm->view_.archive       = lm->archive_.get();
  lm->view_.pager         = lm->pager_.get();
  lm->view_.weights_arena = &back.device_arena();

  return lm;
}

}  // namespace ninfer::targets::qwen3_8_flash_next
