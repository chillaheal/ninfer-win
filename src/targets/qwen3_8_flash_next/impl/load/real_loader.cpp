// Flash-Next S1 gate (b) — real-geometry device loader (see real_loader.h).
//
// Mirrors the mini LoadedModel::open (bindings.cpp) at REAL geometry with the
// mini loader's materialization placement, and adds the gate-(a) non-text
// enumeration loop so the Binder's finish() invariant sees all 1608 objects
// placed (the real artifact carries 367 vision/MTP objects the mini lacked).

#include "targets/qwen3_8_flash_next/impl/load/real_loader.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/config.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {

namespace {

using artifact::Binder;
using artifact::BlockScaleGeometry;
using artifact::MaterializationPlan;
using artifact::MaterializedArtifact;
using artifact::NumericFormat;
using artifact::ObjectDescriptor;
using artifact::ObjectHandle;
using artifact::Reader;
using artifact::ResourceDescriptor;
using artifact::ResourceEncoding;
using artifact::RowScaleGeometry;
using artifact::StorageLayout;
using artifact::TensorDescriptor;

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
// Real-only derived text shapes (identical to real_bindings.cpp, gate a):
constexpr std::uint32_t kAttnQueryRows = 2 * kNumAttHeads * kHeadDim;  // 12288
constexpr std::uint32_t kAttnKvRows    = kNumKvHeads * kHeadDim;       // 512
constexpr std::uint32_t kAttnOutCols   = kNumAttHeads * kHeadDim;      // 6144
constexpr std::uint32_t kAttnNormDim   = kHeadDim;                     // 256
constexpr std::uint32_t kIndexerQkRows = 640;
constexpr std::uint32_t kIndexerNorm   = 128;
constexpr std::uint32_t kGdnAbRows     = 2 * kGdnValHeads;             // 96
constexpr std::uint32_t kGdnQkvzRows   =
    (2 * kGdnQkHeads + 2 * kGdnValHeads) * kGdnStateDim;               // 16384
constexpr std::uint32_t kGdnConvChan   = kStreamWidth;                 // 10240
constexpr std::uint32_t kGdnNormDim    = kGdnStateDim;                 // 128
constexpr std::uint32_t kGdnOutCols    = kGdnValHeads * kGdnStateDim;  // 6144

constexpr NumericFormat kBF16 = NumericFormat::BF16;
constexpr NumericFormat kFP32 = NumericFormat::FP32;
constexpr NumericFormat kNVFP4 = NumericFormat::NVFP4;
constexpr NumericFormat kFP8   = NumericFormat::FP8_E4M3FN_ROW_BF16S;
constexpr StorageLayout kContig     = StorageLayout::ContiguousLeV1;
constexpr StorageLayout kRowScale   = StorageLayout::RowScaleV1;
constexpr StorageLayout kBlockScale = StorageLayout::BlockScaleK16M128x4V1;

// One-shot gate (b) diagnostics: localize the 0xC0000409 fastfail to a stage.
// stderr is unbuffered by the driver, so a crash preserves every prior line.
void log_progress(const char* stage) { std::fprintf(stderr, "[real_loader] %s\n", stage); }

// Exact BF16 -> FP32 on the host (BF16 is the top 16 bits of an IEEE FP32).
float bf16_to_f32(std::uint16_t bits) {
  const std::uint32_t f32 = static_cast<std::uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &f32, sizeof(out));
  return out;
}

// Bind helpers (each records the name into `bound` so the non-text enumeration
// below does not re-bind it).
ObjectHandle bind_device(Binder& b, std::set<std::string>& bound, std::string_view name,
                         NumericFormat fmt, StorageLayout lay,
                         std::initializer_list<std::uint64_t> shape) {
  const std::span<const std::uint64_t> sp(shape.begin(), shape.size());
  const ObjectHandle h = b.require_tensor(name, fmt, lay, sp);
  bound.emplace(std::string(name));
  b.materialize_on_device(h);
  return h;
}

ObjectHandle bind_host(Binder& b, std::set<std::string>& bound, std::string_view name,
                       NumericFormat fmt, StorageLayout lay,
                       std::initializer_list<std::uint64_t> shape) {
  const std::span<const std::uint64_t> sp(shape.begin(), shape.size());
  const ObjectHandle h = b.require_tensor(name, fmt, lay, sp);
  bound.emplace(std::string(name));
  b.retain_on_host(h);
  return h;
}

ObjectHandle bind_validate(Binder& b, std::set<std::string>& bound, std::string_view name,
                           NumericFormat fmt, StorageLayout lay,
                           std::initializer_list<std::uint64_t> shape) {
  const std::span<const std::uint64_t> sp(shape.begin(), shape.size());
  const ObjectHandle h = b.require_tensor(name, fmt, lay, sp);
  bound.emplace(std::string(name));
  b.validate_only(h);
  return h;
}

ObjectHandle bind_resource(Binder& b, std::set<std::string>& bound, std::string_view name) {
  const ObjectHandle h = b.require_resource(name, ResourceEncoding::RawBytesV1);
  bound.emplace(std::string(name));
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
  return Fp8RowScaleWeight{.codes = base,
                           .scale = reinterpret_cast<const std::uint16_t*>(base + g.scale_plane_offset),
                           .rows = rows,
                           .cols = cols};
}

// NVFP4 code/scale planes (device-resident, canonical layout).
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

// Per-layer fused 512-expert routed host bases (the Reader's mmap, never H2D'd).
// `gu_name` / `dn_name` are the layer's fused gate_up / down objects; the code
// plane is at the object start, the NVFP4 scale plane at the block-scale
// geometry offset. The per-expert stride within each plane (512 experts fused)
// is:
//   gu: codes 1,638,400 B/expert, scales 204,800 B/expert   (rows 512*1280, cols 2560)
//   dn: codes 819,200 B/expert,   scales 102,400 B/expert   (rows 512*2560, cols 640)
// 128-row swizzle slabs align to expert boundaries (1280 = 10*128, 2560 = 20*128),
// so each expert's scale block is contiguous.
RoutedHostBases routed_host_bases(const Reader& reader, const char* gu_name, const char* dn_name) {
  RoutedHostBases out;
  const auto gu = reader.payload(gu_name);
  const auto dn = reader.payload(dn_name);
  const std::array<std::uint64_t, 2> gu_shape{kNumExperts * 2 * kMoeInter, kHidden};
  const std::array<std::uint64_t, 2> dn_shape{kNumExperts * kHidden, kMoeInter};
  const BlockScaleGeometry gu_g = artifact::block_scale_geometry(kNVFP4, gu_shape);
  const BlockScaleGeometry dn_g = artifact::block_scale_geometry(kNVFP4, dn_shape);
  const std::uint8_t* gu_codes = reinterpret_cast<const std::uint8_t*>(gu.data.data());
  const std::uint8_t* dn_codes = reinterpret_cast<const std::uint8_t*>(dn.data.data());
  out.gu_codes  = gu_codes;
  out.gu_scales = gu_codes + gu_g.scale_plane_offset;
  out.dn_codes  = dn_codes;
  out.dn_scales = dn_codes + dn_g.scale_plane_offset;
  return out;
}

// Host-retained BF16 (row-major, `elements`) -> a device FP32 buffer appended to
// `derived`; returns the device pointer.
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

}  // namespace

RealLoadedModel::~RealLoadedModel() = default;

std::unique_ptr<RealLoadedModel> RealLoadedModel::open(const std::filesystem::path& artifact_path,
                                                       const std::filesystem::path& ngram_path,
                                                       DeviceContext& device) {
  Reader reader(artifact_path);
  const auto& ident = reader.identity();
  if (ident.model_id != "qwen3.8-flash-next" || ident.weights_id != "nvfp4") {
    throw std::runtime_error("flash_next real load: unexpected artifact identity " +
                             ident.model_id + ":" + ident.weights_id);
  }
  log_progress("A: identity ok");

  auto lm = std::unique_ptr<RealLoadedModel>(new RealLoadedModel());
  Binder b(reader);
  std::set<std::string> bound;

  // Resources (raw-bytes front-end metadata) — bound + validated, never loaded.
  bind_resource(b, bound, "frontend/tokenizer.json");
  bind_resource(b, bound, "frontend/tokenizer_config.json");
  bind_resource(b, bound, "frontend/chat_template.jinja");
  bind_resource(b, bound, "frontend/generation_config.json");
  bind_resource(b, bound, "frontend/preprocessor_config.json");
  bind_resource(b, bound, "frontend/video_preprocessor_config.json");

  // Global embedding + output head (device FP8 row-scale).
  const ObjectHandle h_embed =
      bind_device(b, bound, "text/token_embedding", kFP8, kRowScale, {kVocab, kHidden});
  const ObjectHandle h_out =
      bind_device(b, bound, "text/output_head", kFP8, kRowScale, {kVocab, kHidden});

  // Global hyper-connection mixer — v1 replicates the embedding into all
  // branches, so this is bound but unused (validate_only).
  bind_validate(b, bound, "text/hyper_connection_mixer/hc_norm", kBF16, kContig, {kStreamWidth});
  bind_validate(b, bound, "text/hyper_connection_mixer/input_mix_weight_down", kBF16, kContig,
                {kHcLowRank, kStreamWidth});
  bind_validate(b, bound, "text/hyper_connection_mixer/input_mix_weight_up", kBF16, kContig,
                {kStreamWidth, kHcLowRank});

  // Per-layer handles (device tensors resolved after materialize; host tensors
  // converted below).
  std::array<ObjectHandle, kNumLayers> h_moe_router, h_moe_shared_gate, h_moe_shared_gu,
      h_moe_shared_dn;
  std::array<ObjectHandle, kNumLayers> h_attn_down, h_attn_up, h_attn_inj, h_attn_norm;
  std::array<ObjectHandle, kNumLayers> h_mlp_down, h_mlp_up, h_mlp_inj, h_mlp_norm;
  std::array<ObjectHandle, kNumLayers> h_gdn_alog, h_gdn_dt, h_gdn_conv, h_gdn_ab, h_gdn_qkvz,
      h_gdn_norm, h_gdn_out;
  std::array<ObjectHandle, kNumLayers> h_q_query, h_q_key, h_q_value, h_q_out, h_q_qnorm, h_q_knorm,
      h_q_qkproj, h_q_qin, h_q_kin;
  std::array<float, kNumLayers> shared_gu_div, shared_dn_div, routed_gu_div, routed_dn_div;
  ObjectHandle h_ple_conv;
  for (std::uint32_t l = 0; l < kNumLayers; ++l) {
    const std::string prefix = "text/layers/" + std::to_string(l) + "/";

    // Hyper-connection sets (host BF16 -> device FP32).
    h_attn_down[l] = bind_host(b, bound, prefix + "attn_hyper_connection/input_mix_weight_down",
                               kBF16, kContig, {kHcLowRank, kStreamWidth});
    h_attn_up[l] = bind_host(b, bound, prefix + "attn_hyper_connection/input_mix_weight_up", kBF16,
                             kContig, {kStreamWidth, kHcLowRank});
    h_attn_inj[l] = bind_host(b, bound, prefix + "attn_hyper_connection/block_inject_weight", kBF16,
                              kContig, {kHcCount, kStreamWidth});
    h_attn_norm[l] = bind_host(b, bound, prefix + "attn_hyper_connection/hc_norm", kBF16, kContig,
                               {kStreamWidth});
    h_mlp_down[l] = bind_host(b, bound, prefix + "mlp_hyper_connection/input_mix_weight_down",
                              kBF16, kContig, {kHcLowRank, kStreamWidth});
    h_mlp_up[l] = bind_host(b, bound, prefix + "mlp_hyper_connection/input_mix_weight_up", kBF16,
                            kContig, {kStreamWidth, kHcLowRank});
    h_mlp_inj[l] = bind_host(b, bound, prefix + "mlp_hyper_connection/block_inject_weight", kBF16,
                             kContig, {kHcCount, kStreamWidth});
    h_mlp_norm[l] = bind_host(b, bound, prefix + "mlp_hyper_connection/hc_norm", kBF16, kContig,
                              {kStreamWidth});

    // MoE (device router/shared; routed fused NVFP4 validate — the pager supplies
    // them in S3).
    h_moe_router[l] =
        bind_device(b, bound, prefix + "mlp/experts/routing", kBF16, kContig, {kNumExperts, kHidden});
    h_moe_shared_gate[l] =
        bind_device(b, bound, prefix + "mlp/shared_expert_gate", kBF16, kContig, {1, kHidden});
    h_moe_shared_gu[l] = bind_device(b, bound, prefix + "mlp/shared_expert/gate_up", kNVFP4,
                                     kBlockScale, {2 * kSharedInter, kHidden});
    h_moe_shared_dn[l] = bind_device(b, bound, prefix + "mlp/shared_expert/down", kNVFP4, kBlockScale,
                                     {kHidden, kMoeInter});
    const ObjectHandle h_routed_gu = bind_validate(
        b, bound, prefix + "mlp/experts/gate_up", kNVFP4, kBlockScale,
        {kNumExperts * 2 * kMoeInter, kHidden});
    const ObjectHandle h_routed_dn = bind_validate(
        b, bound, prefix + "mlp/experts/down", kNVFP4, kBlockScale,
        {kNumExperts * kHidden, kMoeInter});
    bind_validate(b, bound, prefix + "mlp/shared_expert/gate_up_projection/input_scale_divisor",
                  kFP32, kContig, {});
    bind_validate(b, bound, prefix + "mlp/shared_expert/down_projection/input_scale_divisor", kFP32,
                  kContig, {});
    bind_validate(b, bound, prefix + "mlp/experts/gate_up_projection/input_scale_divisor", kFP32,
                  kContig, {});
    bind_validate(b, bound, prefix + "mlp/experts/down_projection/input_scale_divisor", kFP32,
                  kContig, {});

    shared_gu_div[l] = read_divisor(b, h_moe_shared_gu[l], 2 * kSharedInter, kHidden,
                                    (prefix + "shared_expert/gate_up").c_str());
    shared_dn_div[l] = read_divisor(b, h_moe_shared_dn[l], kHidden, kMoeInter,
                                    (prefix + "shared_expert/down").c_str());
    routed_gu_div[l] =
        read_divisor(b, h_routed_gu, kNumExperts * 2 * kMoeInter, kHidden,
                     (prefix + "experts/gate_up").c_str());
    routed_dn_div[l] =
        read_divisor(b, h_routed_dn, kNumExperts * kHidden, kMoeInter,
                     (prefix + "experts/down").c_str());

    // Attention (full / QSA layers: l % 4 == 3) or GDN (the other 36).
    if (detail::is_full_attention_layer(l)) {
      const std::string at = prefix + "attention/";
      h_q_query[l] = bind_device(b, bound, at + "query", kFP8, kRowScale, {kAttnQueryRows, kHidden});
      h_q_key[l]   = bind_device(b, bound, at + "key", kFP8, kRowScale, {kAttnKvRows, kHidden});
      h_q_value[l] = bind_device(b, bound, at + "value", kFP8, kRowScale, {kAttnKvRows, kHidden});
      h_q_out[l]   = bind_device(b, bound, at + "output", kFP8, kRowScale, {kHidden, kAttnOutCols});
      h_q_qnorm[l] = bind_device(b, bound, at + "query_norm", kBF16, kContig, {kAttnNormDim});
      h_q_knorm[l] = bind_device(b, bound, at + "key_norm", kBF16, kContig, {kAttnNormDim});
      h_q_qkproj[l]= bind_device(b, bound, at + "indexer/qk_proj", kBF16, kContig,
                                 {kIndexerQkRows, kHidden});
      h_q_qin[l]   = bind_device(b, bound, at + "indexer/q_norm", kBF16, kContig, {kIndexerNorm});
      h_q_kin[l]   = bind_device(b, bound, at + "indexer/k_norm", kBF16, kContig, {kIndexerNorm});
    } else {
      const std::string g = prefix + "gdn/";
      h_gdn_alog[l] = bind_device(b, bound, g + "a_log", kFP32, kContig, {kGdnValHeads});
      h_gdn_dt[l]   = bind_device(b, bound, g + "dt_bias", kFP32, kContig, {kGdnValHeads});
      h_gdn_conv[l] = bind_host(b, bound, g + "convolution", kBF16, kContig,
                                {kGdnConvW, kGdnConvChan});
      h_gdn_ab[l]   = bind_device(b, bound, g + "a_b_projection", kBF16, kContig,
                                  {kGdnAbRows, kHidden});
      h_gdn_qkvz[l] = bind_device(b, bound, g + "query_key_value_z", kFP8, kRowScale,
                                  {kGdnQkvzRows, kHidden});
      h_gdn_norm[l] = bind_device(b, bound, g + "norm", kBF16, kContig, {kGdnNormDim});
      h_gdn_out[l]  = bind_device(b, bound, g + "output", kFP8, kRowScale,
                                  {kHidden, kGdnOutCols});
    }

    // PLE (layer 1): conv is host (used); the projection/norm tensors are bound
    // but unused by the v1 gather+add (validate_only).
    if (l == kPleLayer) {
      const std::string ple = prefix + "ple/";
      h_ple_conv = bind_host(b, bound, ple + "convolution", kBF16, kContig,
                             {kGdnConvW, kStreamWidth});
      bind_validate(b, bound, ple + "key_projection", kBF16, kContig, {kStreamWidth, kHidden});
      bind_validate(b, bound, ple + "value_projection", kBF16, kContig, {kHidden, kHidden});
      bind_validate(b, bound, ple + "norm_conv", kBF16, kContig, {kStreamWidth});
      bind_validate(b, bound, ple + "norm_key", kBF16, kContig, {kStreamWidth});
      bind_validate(b, bound, ple + "norm_query", kBF16, kContig, {kStreamWidth});
    }
  }
  log_progress("B: 48 layers bound");

  // Non-text objects (vision / MTP): not in the P12 text forward. Consume each
  // shape-agnostically from its own descriptor; place by residency so the
  // mtp/vision GpuResident tensors join the device set (completing the backbone)
  // while finish() still sees every object placed.
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

  // Finish: every object is now consumed + placed.
  log_progress("C: non-text bound");
  const MaterializationPlan plan = b.finish();
  log_progress("D: finish done");
  lm->backing_ = artifact::materialize(reader, plan, device);
  log_progress("E: materialize done");
  // Keep the Reader alive: its mmap backs the routed-expert host bases the S3
  // RealProgram scatters from. The fused routed objects are validate_only, so
  // their bytes live only in the Reader's mmap, never in the device arena. The
  // Binder (b) only references reader by const-ref and its implicit destructor
  // never dereferences it, so the move is safe once finish()/materialize are done.
  lm->reader_ = std::make_unique<artifact::Reader>(std::move(reader));
  MaterializedArtifact& back = lm->backing_;

  // PLE table (a read-only mmap of the 51.8 GB .ngram — not pinned).
  lm->ple_table_ = std::unique_ptr<ops::NgramEmbeddingTable>(
      new ops::NgramEmbeddingTable(ops::NgramEmbeddingTable::open(ngram_path)));
  const std::uint32_t ple_embed_dim = lm->ple_table_->info().embed_dim();
  log_progress("F: ngram opened");

  // The per-layer weight rows are a runtime vector (unlike the mini loader's
  // compile-time std::array<LayerWeights, 4>); size it before indexing.
  lm->view_.layers.resize(kNumLayers);
  lm->view_.routed_host.resize(kNumLayers);

  // Resolve device pointers into the RealModelView.
  lm->view_.embedding   = make_fp8(back, h_embed, kVocab, kHidden);
  lm->view_.output_head = make_fp8(back, h_out, kVocab, kHidden);

  for (std::uint32_t l = 0; l < kNumLayers; ++l) {
    LayerWeights& lw = lm->view_.layers[l];

    // Hyper-connection sets: host BF16 -> device FP32.
    lw.attn_hc.w_down =
        host_bf16_to_device_f32(back, h_attn_down[l], static_cast<std::size_t>(kHcLowRank) * kStreamWidth,
                                lm->derived_);
    lw.attn_hc.w_up =
        host_bf16_to_device_f32(back, h_attn_up[l],
                                static_cast<std::size_t>(kStreamWidth) * kHcLowRank, lm->derived_);
    lw.attn_hc.w_inj =
        host_bf16_to_device_f32(back, h_attn_inj[l], static_cast<std::size_t>(kHcCount) * kStreamWidth,
                                lm->derived_);
    lw.attn_hc.norm_w =
        host_bf16_to_device_f32(back, h_attn_norm[l], kStreamWidth, lm->derived_);
    lw.mlp_hc.w_down =
        host_bf16_to_device_f32(back, h_mlp_down[l], static_cast<std::size_t>(kHcLowRank) * kStreamWidth,
                                lm->derived_);
    lw.mlp_hc.w_up =
        host_bf16_to_device_f32(back, h_mlp_up[l],
                                static_cast<std::size_t>(kStreamWidth) * kHcLowRank, lm->derived_);
    lw.mlp_hc.w_inj =
        host_bf16_to_device_f32(back, h_mlp_inj[l], static_cast<std::size_t>(kHcCount) * kStreamWidth,
                                lm->derived_);
    lw.mlp_hc.norm_w =
        host_bf16_to_device_f32(back, h_mlp_norm[l], kStreamWidth, lm->derived_);

    // MoE device weights.
    lw.mlp.router_bf16      = make_bf16(back, h_moe_router[l], kNumExperts, kHidden).data;
    lw.mlp.shared_gate_bf16 = make_bf16(back, h_moe_shared_gate[l], 1, kHidden).data;
    const Nvfp4Planes gu = make_nvfp4(back, h_moe_shared_gu[l], 2 * kSharedInter, kHidden);
    lw.mlp.shared_gu_codes  = gu.codes;
    lw.mlp.shared_gu_scales = gu.scales;
    const Nvfp4Planes dn = make_nvfp4(back, h_moe_shared_dn[l], kHidden, kMoeInter);
    lw.mlp.shared_dn_codes  = dn.codes;
    lw.mlp.shared_dn_scales = dn.scales;
    lw.mlp.shared_gu_divisor = shared_gu_div[l];
    lw.mlp.shared_dn_divisor = shared_dn_div[l];
    lw.mlp.routed_gu_divisor = routed_gu_div[l];
    lw.mlp.routed_dn_divisor = routed_dn_div[l];
    // expert_window stays null: S3's RealProgram stages the device moe_window
    // and scatters the per-token selected experts from these host bases.
    lm->view_.routed_host[l] = routed_host_bases(
        *lm->reader_,
        ("text/layers/" + std::to_string(l) + "/mlp/experts/gate_up").c_str(),
        ("text/layers/" + std::to_string(l) + "/mlp/experts/down").c_str());
  }
  log_progress("G: embed/head + 48 MoE resolved");

  for (std::uint32_t l = 0; l < kNumLayers; ++l) {
    if (detail::is_full_attention_layer(l)) continue;
    GdnLayerWeights g;
    g.qkvz           = make_fp8(back, h_gdn_qkvz[l], kGdnQkvzRows, kHidden);
    g.output         = make_fp8(back, h_gdn_out[l], kHidden, kGdnOutCols);
    g.a_b_projection = make_bf16(back, h_gdn_ab[l], kGdnAbRows, kHidden);
    g.a_log          = static_cast<const float*>(back.device_data(h_gdn_alog[l]));
    g.dt_bias        = static_cast<const float*>(back.device_data(h_gdn_dt[l]));
    g.norm           = make_bf16(back, h_gdn_norm[l], 1, kGdnNormDim).data;
    g.conv           = host_bf16_to_device_transpose(back, h_gdn_conv[l], kGdnConvW, kGdnConvChan,
                                                     lm->derived_);
    lm->view_.layers[l].gdn = std::move(g);
  }
  log_progress("H: 36 GDN resolved");

  for (std::uint32_t l = 0; l < kNumLayers; ++l) {
    if (!detail::is_full_attention_layer(l)) continue;
    QsaLayerWeights q;
    q.query            = make_fp8(back, h_q_query[l], kAttnQueryRows, kHidden);
    q.key              = make_fp8(back, h_q_key[l], kAttnKvRows, kHidden);
    q.value            = make_fp8(back, h_q_value[l], kAttnKvRows, kHidden);
    q.output           = make_fp8(back, h_q_out[l], kHidden, kAttnOutCols);
    q.query_norm       = make_bf16(back, h_q_qnorm[l], 1, kAttnNormDim).data;
    q.key_norm         = make_bf16(back, h_q_knorm[l], 1, kAttnNormDim).data;
    q.indexer_qk_proj  = make_bf16(back, h_q_qkproj[l], kIndexerQkRows, kHidden);
    q.indexer_q_norm   = make_bf16(back, h_q_qin[l], 1, kIndexerNorm).data;
    q.indexer_k_norm   = make_bf16(back, h_q_kin[l], 1, kIndexerNorm).data;
    lm->view_.layers[l].qsa = std::move(q);
  }
  log_progress("I: 12 QSA resolved");

  // PLE conv weights (host [4][ple_embed_dim] FP32): the first ple_embed_dim
  // channels of each tap of ple/convolution [4, kStreamWidth] BF16.
  {
    const auto src = back.resource_bytes(h_ple_conv);
    const std::uint16_t* b = reinterpret_cast<const std::uint16_t*>(src.data());
    lm->view_.ple_conv_weights.assign(static_cast<std::size_t>(kGdnConvW) * ple_embed_dim, 0.0F);
    for (std::uint32_t tap = 0; tap < kGdnConvW; ++tap) {
      for (std::uint32_t c = 0; c < ple_embed_dim; ++c) {
        lm->view_.ple_conv_weights[static_cast<std::size_t>(tap) * ple_embed_dim + c] =
            bf16_to_f32(b[static_cast<std::size_t>(tap) * kStreamWidth + c]);
      }
    }
    lm->view_.ple_conv_channels = ple_embed_dim;
  }
  log_progress("J: PLE conv done");

  lm->view_.ple_table     = lm->ple_table_.get();
  lm->view_.weights_arena = &back.device_arena();
  // archive/pager stay null (S3 wires the real pager over the mmap'd experts).

  // Footprint summary (logged by the gate (b) driver).
  const auto& stats = back.stats();
  std::uint64_t derived_bytes = 0;
  for (const DeviceBuffer& db : lm->derived_) derived_bytes += db.bytes;
  lm->summary_.device_capacity_bytes = stats.device_capacity_bytes;
  lm->summary_.h2d_bytes             = stats.h2d_bytes;
  lm->summary_.arena_peak_bytes      = back.device_arena().peak_used();
  lm->summary_.derived_bytes         = derived_bytes;
  lm->summary_.total_device_bytes    = lm->summary_.arena_peak_bytes + derived_bytes;
  lm->summary_.object_count          = plan.object_count;
  lm->summary_.device_objects        = plan.device_objects.size();
  lm->summary_.host_objects          = plan.host_objects.size();
  lm->summary_.layer_count           = kNumLayers;
  lm->summary_.ple_embed_dim         = ple_embed_dim;
  lm->summary_.ngram_bytes           = std::filesystem::file_size(ngram_path);
  (void)nontext;
  log_progress("K: summary done, returning");

  return lm;
}

}  // namespace ninfer::targets::qwen3_8_flash_next
