// vision_encode_ab.cpp -- standalone Vision encode A/B oracle (no LLM, no serve).
//
// Loads ONLY the Vision weights + the six frontend resources from a .ninfer artifact
// (mmap; the LLM payload is never bound and never touched -- a full model load is
// neither a time nor a VRAM concern for this tool). It then runs the PRODUCTION
// frontend prepare() (image decode -> BF16 patch buffer + VisionItemControl) and the
// PRODUCTION VisionContext::encode() -- CPU leg (host FP32, BF16-snapped) or GPU leg
// (CUDA) -- with the env-gated NINFER_VISION_DUMP stage dumps enabled, and compares
// the dumped stages (vd_<mode>_{patches,x1,x2,blkNN,m1,m2,out}.bin) against the golden
// GPU set. Any CPU-vs-GPU divergence is thereby localized to the first diverging stage
// in ~1 minute per iteration -- no serve restart, no 21.5 GB model reload.
//
// Usage (from the project root):
//   ninfer_vision_encode_ab.exe <model.ninfer> <image.png>
//       [--mode cpu|gpu|both]        default: both (CPU leg, then GPU leg, one process)
//       [--dump-dir DIR]             default: .logs/vision_ab (created if missing)
//       [--golden-dir DIR]           default: .logs (the intact vd_gpu_* golden set)
//       [--general-capacity BYTES]   default: 4 GiB (device capacity hint for planning)
//
// Exit codes: 0 = every compared stage identical or BF16-noise, 1 = a stage diverged
// (or is missing), 2 = usage/environment error. Safe to run unattended: it only reads
// the model (mmap, PAGE_READONLY) and writes vd_*.bin files into --dump-dir.

#include "targets/qwen3_6_27b/impl/variant.h"

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime

#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"

#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>
#include <ninfer/targets/qwen3_6/vision_control.h>
#include <ninfer/types.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {

namespace rt  = ::ninfer::targets::qwen3_6::detail::qwen3_6_27b_runtime;
namespace q6  = ::ninfer::targets::qwen3_6;
namespace art = ::ninfer::artifact;
using DeviceContext = ::ninfer::DeviceContext;
using DeviceBuffer  = ::ninfer::DeviceBuffer;
using DeviceSpan    = ::ninfer::DeviceSpan;
using Tensor        = ::ninfer::Tensor;
using ::ninfer::ChatMessage;
using ::ninfer::ChatRole;
using ::ninfer::MediaKind;
using ::ninfer::MessagePart;
using ::ninfer::MessagePartKind;
using ::ninfer::PromptInput;

enum class LegOutcome { Pass, Fail, Skipped, NotRun };

constexpr std::size_t KiB = 1024;
constexpr std::size_t MiB = 1024 * 1024;

// BF16 noise threshold, ULP-relative to the stage magnitude. An absolute threshold is
// unsatisfiable for deep stages: each op boundary reintroduces sub-ULP noise (GEMM
// accumulation order, the GPU's BF16 rounding of the attention probabilities before P·V)
// that the 27-block residual stack amplifies ~1.13x per block, and cancellation on
// low-magnitude elements (small value = difference of large pre-terms) inflates the
// RELATIVE diff of a fixed ULP-count of pre-term noise -- measured clean-run peak is
// ~3 bf16 ULP of rms (blk25) with a ~0.3-0.5% fraction of small-value elements differing
// by >0.5x their magnitude (balanced signs, spread over all channels/patches = no layout
// structure). The three real layout bugs found so far (x1024 fp16-subnormal dequant,
// corner-major pos index, interleaved rope positions) showed 10-1000x the noise level.
// PASS = bit-identical, or meanabs < 8 ULP of the larger-leg rms. The big-outlier count
// (|d| > 0.5x the larger magnitude, magnitude > 0.1 to skip zero-boundary noise) is
// printed for inspection but does not gate the verdict -- cancellation produces it even
// for a correct implementation, and real layout bugs are caught by the ULP criterion.
constexpr double kBf16NoiseRel     = 8.0 * (1.0 / 256.0); // 8 ULP, relative to rms
constexpr double kBf16BigMagnitude = 0.1;

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "ERROR: " << message << "\n";
    std::exit(2);
}

std::string media_type_for(const fs::path& p) {
    std::string ext = p.extension().string();
    for (auto& c : ext) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
    if (ext == ".png") { return "image/png"; }
    if (ext == ".jpg" || ext == ".jpeg") { return "image/jpeg"; }
    if (ext == ".webp") { return "image/webp"; }
    if (ext == ".bmp") { return "image/bmp"; }
    return "image/png";
}

std::vector<std::uint8_t> read_bytes(const fs::path& p, const char* what) {
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    if (!in) { fail(std::string("cannot open ") + what + ": " + p.string()); }
    const std::streamsize size = in.tellg();
    if (size <= 0) { fail(std::string(what) + " is empty: " + p.string()); }
    in.seekg(0);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!in.read(reinterpret_cast<char*>(bytes.data()), size)) {
        fail(std::string("cannot read ") + what + ": " + p.string());
    }
    return bytes;
}

// ---------------------------------------------------------------------------
// Stage-dump comparison (C++ port of .logs/vision_dump_stages.py).
// ---------------------------------------------------------------------------

std::vector<std::uint16_t> load_u16(const fs::path& p) {
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    if (!in) { return {}; }
    const std::streamsize size = in.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) % 2 != 0) { return {}; }
    in.seekg(0);
    std::vector<std::uint16_t> out(static_cast<std::size_t>(size) / 2);
    if (!in.read(reinterpret_cast<char*>(out.data()), size)) { return {}; }
    return out;
}

float bf16_to_f32(std::uint16_t h) {
    std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

enum class Verdict { Ok, Noise, Diverges, Missing, SizeMismatch };

const char* verdict_name(Verdict v) {
    switch (v) {
    case Verdict::Ok:           return "IDENTICAL";
    case Verdict::Noise:        return "ok (bf16 noise)";
    case Verdict::Diverges:     return "DIVERGES";
    case Verdict::Missing:      return "MISSING";
    case Verdict::SizeMismatch: return "SIZE MISMATCH";
    }
    return "?";
}

struct StageResult {
    std::string tag;
    Verdict   verdict = Verdict::Missing;
    std::size_t elems = 0;
    double cpu_rms  = 0.0;
    double gpu_rms  = 0.0;
    double meanabs  = 0.0;
    double maxabs   = 0.0;
    double bigfrac  = 0.0;
};

bool stage_ok(Verdict v) { return v == Verdict::Ok || v == Verdict::Noise; }

// Detects the blkNN index set from both directories (union) and compares every
// stage: patches, x1, x2, blkNN..., m1, m2, out. `leg_mode` is "cpu" or "gpu"; the
// reference is always the golden `vd_gpu_<tag>.bin` in `golden_dir`.
std::vector<StageResult> compare_leg(const fs::path& dump_dir, const fs::path& golden_dir,
                                     const std::string& leg_mode) {
    std::set<int> blk;
    for (const auto& entry : fs::directory_iterator(dump_dir)) {
        if (!entry.is_regular_file()) { continue; }
        const std::string name = entry.path().filename().string();
        const std::string prefix = "vd_" + leg_mode + "_blk";
        if (name.size() == prefix.size() + 2 + 4 && name.compare(0, prefix.size(), prefix) == 0 &&
            name.compare(name.size() - 4, 4, ".bin") == 0) {
            blk.insert(std::atoi(name.substr(prefix.size()).c_str()));
        }
    }
    std::vector<std::string> tags = {"patches", "x1", "x2"};
    for (int i = 0; i < 99; ++i) {
        if (blk.count(i)) {
            char buf[16];
            std::snprintf(buf, sizeof buf, "blk%02d", i);
            tags.push_back(buf);
        }
    }
    tags.push_back("m1");
    tags.push_back("m2");
    tags.push_back("out");

    std::vector<StageResult> results;
    results.reserve(tags.size());
    for (const auto& tag : tags) {
        StageResult r;
        r.tag = tag;
        auto cpu = load_u16(dump_dir / ("vd_" + leg_mode + "_" + tag + ".bin"));
        auto gpu = load_u16(golden_dir / ("vd_gpu_" + tag + ".bin"));
        if (cpu.empty() || gpu.empty()) {
            r.verdict = Verdict::Missing;
            results.push_back(std::move(r));
            continue;
        }
        if (cpu.size() != gpu.size()) {
            r.verdict    = Verdict::SizeMismatch;
            r.elems      = cpu.size();
            results.push_back(std::move(r));
            continue;
        }
        double sum2_a = 0.0, sum2_b = 0.0, sumabs = 0.0;
        double maxabs = 0.0;
        std::size_t big = 0;
        for (std::size_t i = 0; i < cpu.size(); ++i) {
            const float a = bf16_to_f32(cpu[i]);
            const float b = bf16_to_f32(gpu[i]);
            const float d = a - b;
            sum2_a += static_cast<double>(a) * a;
            sum2_b += static_cast<double>(b) * b;
            sumabs += std::abs(static_cast<double>(d));
            maxabs = std::max(maxabs, std::abs(static_cast<double>(d)));
            if (std::abs(d) > 0.5 * std::max(std::abs(a), std::abs(b)) &&
                std::max(std::abs(a), std::abs(b)) > kBf16BigMagnitude) {
                ++big;
            }
        }
        const double n = static_cast<double>(cpu.size());
        r.elems   = cpu.size();
        r.cpu_rms = std::sqrt(sum2_a / n);
        r.gpu_rms = std::sqrt(sum2_b / n);
        r.meanabs = sumabs / n;
        r.maxabs  = maxabs;
        if (std::equal(cpu.begin(), cpu.end(), gpu.begin())) {
            r.verdict = Verdict::Ok;
        } else {
            const double rms = std::max({r.cpu_rms, r.gpu_rms, 1e-9});
            r.bigfrac        = static_cast<double>(big) / n;
            r.verdict = (r.meanabs / rms < kBf16NoiseRel) ? Verdict::Noise : Verdict::Diverges;
        }
        results.push_back(std::move(r));
    }
    return results;
}

void print_comparison(const std::string& leg_mode, const std::vector<StageResult>& results) {
    std::printf("\n[%s] stage comparison (vs golden vd_gpu_*):\n", leg_mode.c_str());
    std::printf("%-10s %-10s %-10s %-10s %-12s %s\n", "stage", "elems", "cpu_rms", "gpu_rms",
                "meanabs", "verdict");
    std::string first_bad;
    for (const auto& r : results) {
        if (r.verdict == Verdict::Missing || r.verdict == Verdict::SizeMismatch) {
            std::printf("%-10s %-10zu %-10s %-10s %-12s %s\n", r.tag.c_str(), r.elems, "-", "-",
                        "-", verdict_name(r.verdict));
        } else {
            std::printf("%-10s %-10zu %-10.4f %-10.4f %-12.6f %s", r.tag.c_str(), r.elems,
                        r.cpu_rms, r.gpu_rms, r.meanabs, verdict_name(r.verdict));
            if (r.verdict != Verdict::Ok) {
                const double rms = std::max({r.cpu_rms, r.gpu_rms, 1e-9});
                std::printf(" (maxabs %.6f, rel %.2f ULP, big %.3f%%)", r.maxabs,
                            r.meanabs / rms / (1.0 / 256.0), r.bigfrac * 100.0);
            }
            std::printf("\n");
        }
        if (first_bad.empty() &&
            (r.verdict == Verdict::Diverges || r.verdict == Verdict::SizeMismatch ||
             r.verdict == Verdict::Missing)) {
            first_bad = r.tag;
        }
    }
    std::printf("[%-8s] FIRST DIVERGENCE: %s\n", leg_mode.c_str(),
                first_bad.empty() ? "none -- all stages agree" : first_bad.c_str());
}

// ---------------------------------------------------------------------------
// One encode leg: bind + materialize the Vision-only view of the artifact, run the
// production frontend + VisionContext::encode, dump the stages, compare vs golden.
// ---------------------------------------------------------------------------

LegOutcome run_leg(const char* mode, const fs::path& model_path, const fs::path& image_path,
                   const fs::path& dump_dir, const fs::path& golden_dir,
                   std::size_t general_capacity) {
    const bool cpu_leg = std::strcmp(mode, "cpu") == 0;
    const std::string mode_name(cpu_leg ? "cpu" : "gpu");

    std::printf("\n=== %s leg ===\n", mode_name.c_str());
    DeviceContext device(0);
    if (!cpu_leg) {
        std::size_t free_bytes = 0, total_bytes = 0;
        const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess) {
            std::printf("[%s] cudaMemGetInfo failed: %s\n", mode_name.c_str(),
                        cudaGetErrorString(err));
            return LegOutcome::Skipped;
        }
        std::printf("[%s] device VRAM: %zu MiB free / %zu MiB total\n", mode_name.c_str(),
                    free_bytes / MiB, total_bytes / MiB);
        // The GPU leg H2D-uploads the Vision weights (~300 MB) plus encode workspace on top
        // of the CUDA context already held by this process. The live serve keeps ~28 GB of
        // the 32 GB card; refuse to fight it for the last slivers.
        if (free_bytes < 1500 * MiB) {
            std::printf("[%s] insufficient free VRAM (%zu MiB < 1500 MiB) -- skipping GPU leg\n",
                        mode_name.c_str(), free_bytes / MiB);
            return LegOutcome::Skipped;
        }
    }

    const auto image_bytes = read_bytes(image_path, "image");

    art::TensorPlacement placement = cpu_leg ? art::TensorPlacement::Host
                                             : art::TensorPlacement::Device;
    q6::VisionWeights weights;
    q6::FrontendResources resources;
    // MaterializedArtifact must outlive encode(): host weights point into its retained
    // resource bytes and device weights into its arena (production keeps it alive the same
    // way -- the Weight/Tensor views are raw pointers into this object).
    art::MaterializedArtifact materialized;
    {
        art::Reader reader(model_path);
        art::Binder binder(reader);
        const q6::FrontendResourcePlan fplan = q6::bind_frontend_resources(binder);
        const q6::VisionBackbonePlan backbone = q6::bind_vision_backbone(binder, placement);
        const q6::VisionMergerInputPlan merger_input =
            q6::bind_vision_merger_input(binder, placement);
        // Same names/shapes as production (qwen3_6_27b load bindings): the merger output
        // projection is a plain tensor bind, not part of the qwen3_6 bind_* helpers.
        const art::ObjectHandle merger_fc2 = art::bind_tensor(
            binder, "vision/merger/fc2", art::NumericFormat::W8G32_F16S, {5120, 4608}, placement);
        const art::ObjectHandle merger_fc2_bias = art::bind_tensor(
            binder, "vision/merger/fc2_bias", art::NumericFormat::BF16, {5120}, placement);
        const q6::VisionMergerNormPlan merger_norm = q6::bind_vision_merger_norm(binder, placement);
        // The artifact also holds the full LLM (~hundreds of objects) which this tool does
        // not need. finish() requires EVERY object to be both consumed and planned; the
        // bound objects above carry their real placements, so the rest are consumed with a
        // ValidateOnly placement -- the same mechanism production uses for disabled
        // features. require_* is called with each object's own descriptor values (so the
        // contract check is a no-op), and the "bound more than once" ArtifactError marks
        // the objects that were already bound above.
        for (const art::ObjectDescriptor& object : reader.objects()) {
            try {
                if (const auto* tensor = std::get_if<art::TensorDescriptor>(&object)) {
                    binder.validate_only(
                        binder.require_tensor(tensor->name, tensor->format, tensor->layout,
                                              tensor->shape));
                } else if (const auto* resource = std::get_if<art::ResourceDescriptor>(&object)) {
                    binder.validate_only(
                        binder.require_resource(resource->name, resource->encoding));
                }
            } catch (const art::ArtifactError&) {
                // already bound/planned above -- expected
            }
        }
        const art::MaterializationPlan plan = binder.finish();
        const auto t0 = std::chrono::steady_clock::now();
        materialized = art::materialize(reader, plan, device, /*progress=*/nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("[%s] materialized %zu host objects / %zu device objects in %.1f s "
                    "(h2d %zu B, device capacity %zu B)\n",
                    mode_name.c_str(), plan.host_objects.size(), plan.device_objects.size(),
                    std::chrono::duration<double>(t1 - t0).count(),
                    materialized.stats().h2d_bytes, plan.device_capacity_bytes);
        resources = q6::take_frontend_resources(materialized, fplan);
        weights.common          = q6::materialize_vision_common(materialized, backbone, merger_input,
                                                                merger_norm, cpu_leg);
        weights.merger_fc2      =
            cpu_leg ? art::materialized_host_weight(materialized, merger_fc2,
                                                    art::NumericFormat::W8G32_F16S, 5120, 4608)
                    : art::materialized_weight(materialized, merger_fc2,
                                               art::NumericFormat::W8G32_F16S, 5120, 4608);
        weights.merger_fc2_bias =
            cpu_leg ? art::materialized_host_tensor(materialized, merger_fc2_bias,
                                                    art::NumericFormat::BF16, {5120})
                    : art::materialized_tensor(materialized, merger_fc2_bias,
                                               art::NumericFormat::BF16, {5120});
    }

    q6::Frontend frontend = q6::FrontendTestAccess::create_component(resources, /*vision_enabled=*/true);

    PromptInput input;
    ChatMessage user;
    user.role = ChatRole::User;
    MessagePart media;
    media.kind = MessagePartKind::Media;
    media.media.kind         = MediaKind::Image;
    media.media.bytes        = std::move(image_bytes);
    media.media.media_type   = media_type_for(image_path);
    media.media.source_name  = image_path.filename().string();
    user.parts.push_back(std::move(media));
    MessagePart text;
    text.kind = MessagePartKind::Text;
    text.text = "Describe this image.";
    user.parts.push_back(std::move(text));
    input.messages.push_back(std::move(user));

    const auto p0 = std::chrono::steady_clock::now();
    q6::PreparedPrompt prepared = frontend.prepare(input);
    const auto p1 = std::chrono::steady_clock::now();
    if (!static_cast<bool>(prepared)) {
        std::printf("[%s] frontend prepare() produced an empty prompt\n", mode_name.c_str());
        return LegOutcome::Fail;
    }
    const q6::PreparedPromptData& data = q6::PreparedPromptAccess::view(prepared);
    if (data.vision_items.size() != 1 || data.media_payloads.size() != 1) {
        std::printf("[%s] expected exactly one Vision item, got %zu items / %zu payloads\n",
                    mode_name.c_str(), data.vision_items.size(), data.media_payloads.size());
        return LegOutcome::Fail;
    }
    std::printf("[%s] frontend prepare: %zu patches, %zu merged tokens, %.2f s\n",
                mode_name.c_str(), data.vision_items[0].patch_count,
                data.vision_items[0].patch_count / 4,
                std::chrono::duration<double>(p1 - p0).count());

    const q6::VisionControlPlan control_plan = q6::plan_vision_control(data);
    const q6::VisionControl control = q6::build_vision_control(data, control_plan, 0);
    if (control.items.size() != 1) {
        std::printf("[%s] expected one control item, got %zu\n", mode_name.c_str(),
                    control.items.size());
        return LegOutcome::Fail;
    }
    const q6::VisionItemControl& ctrl   = control.items[0];
    const auto& payload                = *data.media_payloads[0];
    const std::size_t patches          = ctrl.patch_count;
    const std::size_t merged_tokens    = ctrl.merged_count;
    if (payload.span().size() != patches * q6::kPreparedVisionPatchFeatures) {
        std::printf("[%s] payload size %zu != patches*patch_dim %zu\n", mode_name.c_str(),
                    payload.span().size(),
                    patches * q6::kPreparedVisionPatchFeatures);
        return LegOutcome::Fail;
    }

    rt::LoadedModelData model;
    model.features.vision     = true;
    model.features.vision_cpu = cpu_leg;
    model.vision              = std::move(weights);

    rt::schedule::VisionContext context(device, model);
    const rt::VisionWorkspacePlan workspace_plan = rt::schedule::VisionContext::plan_workspace(
        static_cast<std::uint32_t>(merged_tokens), general_capacity,
        /*include_encode_peak=*/!cpu_leg);
    DeviceBuffer backing(workspace_plan.capacity_bytes);
    const DeviceSpan span{backing.p, workspace_plan.capacity_bytes};
    Tensor output = rt::schedule::VisionContext::bind_output(span, workspace_plan, merged_tokens);

    // This toolchain's ucrt exposes putenv but not setenv/_setenv.
    // NINFER_VISION_AB_NODUMP=1 (2026-09-02, #19 STEP 12): measure the production (no-dump)
    // encode — skip the NINFER_VISION_DUMP putenv so dump_stage is a no-op (the 27 synchronous
    // per-stage fwrite dumps are a test-only cost absent from the serve) and report timing only.
    const bool nodump = std::getenv("NINFER_VISION_AB_NODUMP") != nullptr;
    if (!nodump) {
        char dump_env[512];
        std::snprintf(dump_env, sizeof dump_env, "NINFER_VISION_DUMP=%s", dump_dir.string().c_str());
        if (putenv(dump_env) != 0) {
            fail("cannot set NINFER_VISION_DUMP");
        }
    }

    rt::schedule::VisionItemView item{payload.span(), &ctrl};
    if (cpu_leg) {
        // One-time warmup: fill the process-lifetime dequant cache (and the pfor worker pool)
        // so the timed encode measures the steady-state per-image cost, not the first call's
        // one-time weight dequant (~75 ms). The warmup's output/dumps are overwritten by the
        // timed encode (the cached dequant is bit-identical to the per-call one).
        std::printf("[%s] warmup encode (one-time dequant-cache fill)\n", mode_name.c_str());
        context.encode(item, output, span, workspace_plan);
        device.synchronize();
        // NINFER_VISION_AB_IDLE_MS (2026-09-02, #19): sleep between the warmup and the timed
        // encode. The 9950X3D memory subsystem has a FAST state that decays after ~1-4 s of
        // sustained GEMM load (isolated fc2 GEMM: 5.7 -> 7.3 ms; a 5 s idle does NOT restore
        // it; the fresh state appears after process start / longer idles) and the back-to-back
        // warmup+timed sequence is the worst case. In the real serve the CPU idles between
        // encodes (the LLM decodes on the GPU), so a pre-timed idle simulates the production
        // regime. 0/unset (default) = the worst case.
        if (const char* e = std::getenv("NINFER_VISION_AB_IDLE_MS")) {
            const long ms = std::atol(e);
            if (ms > 0) {
                std::printf("[%s] idle %ld ms before timed encode\n", mode_name.c_str(), ms);
                std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            }
        }
    }
    std::printf("[%s] encode: %zu patches -> %zu tokens, workspace %zu B\n", mode_name.c_str(),
                patches, merged_tokens, workspace_plan.capacity_bytes);
    const auto e0 = std::chrono::steady_clock::now();
    context.encode(item, output, span, workspace_plan);
    device.synchronize();
    const auto e1 = std::chrono::steady_clock::now();
    const double encode_seconds = std::chrono::duration<double>(e1 - e0).count();
    std::printf("[%s] encode done in %.2f s\n", mode_name.c_str(), encode_seconds);

    if (nodump) {
        std::printf("[%s] NODUMP timing-only (no stages dumped/compared): encode %.2f s\n\n",
                    mode_name.c_str(), encode_seconds);
        return LegOutcome::Pass;
    }
    const auto results = compare_leg(dump_dir, golden_dir, mode_name);
    print_comparison(mode_name, results);
    const bool pass =
        std::all_of(results.begin(), results.end(), [](const StageResult& r) { return stage_ok(r.verdict); });
    std::printf("[%s] VERDICT: %s (encode %.2f s)\n\n", mode_name.c_str(), pass ? "PASS" : "FAIL",
                encode_seconds);
    return pass ? LegOutcome::Pass : LegOutcome::Fail;
}

} // namespace

int main(int argc, char** argv) {
    // Unbuffered stdout: progress lines must survive abort/fastfail for unattended diagnosis.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string model_path, image_path;
    std::string mode = "both";
    fs::path dump_dir = fs::path(".logs/vision_ab");
    fs::path golden_dir = fs::path(".logs");
    std::size_t general_capacity = 4 * 1024 * MiB;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (a == "--dump-dir" && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (a == "--golden-dir" && i + 1 < argc) {
            golden_dir = argv[++i];
        } else if (a == "--general-capacity" && i + 1 < argc) {
            general_capacity = std::stoul(argv[++i]);
        } else if (!a.starts_with("--") && model_path.empty()) {
            model_path = a;
        } else if (!a.starts_with("--") && image_path.empty()) {
            image_path = a;
        } else {
            fail("unknown argument: " + a);
        }
    }
    if (model_path.empty() || image_path.empty()) {
        fail("usage: ninfer_vision_encode_ab <model.ninfer> <image.png> "
             "[--mode cpu|gpu|both] [--dump-dir DIR] [--golden-dir DIR] [--general-capacity BYTES]");
    }
    if (mode != "cpu" && mode != "gpu" && mode != "both") {
        fail("invalid --mode: " + mode);
    }
    if (!fs::is_regular_file(model_path)) { fail("no such model: " + model_path); }
    if (!fs::is_regular_file(image_path)) { fail("no such image: " + image_path); }
    if (!fs::is_directory(golden_dir)) { fail("no such golden dir: " + golden_dir.string()); }
    fs::create_directories(dump_dir);

    std::printf("vision_encode_ab: model=%s image=%s mode=%s dump_dir=%s golden_dir=%s\n",
                model_path.c_str(), image_path.c_str(), mode.c_str(), dump_dir.string().c_str(),
                golden_dir.string().c_str());

    LegOutcome cpu = LegOutcome::NotRun;
    LegOutcome gpu = LegOutcome::NotRun;
    try {
        if (mode == "cpu" || mode == "both") {
            cpu = run_leg("cpu", model_path, image_path, dump_dir, golden_dir, general_capacity);
        }
        if (mode == "gpu" || mode == "both") {
            gpu = run_leg("gpu", model_path, image_path, dump_dir, golden_dir, general_capacity);
        }
    } catch (const std::exception& e) {
        // Unattended tool: a throw (artifact contract mismatch, CUDA error, ...) must end
        // in a clean diagnostic + exit 2, never a crash.
        std::printf("FATAL: %s\n", e.what());
        return 2;
    }

    std::printf("\nOVERALL: cpu=%s gpu=%s\n",
                cpu == LegOutcome::Pass ? "PASS" : cpu == LegOutcome::Fail ? "FAIL"
                                    : cpu == LegOutcome::Skipped ? "SKIPPED" : "not run",
                gpu == LegOutcome::Pass ? "PASS" : gpu == LegOutcome::Fail ? "FAIL"
                                      : gpu == LegOutcome::Skipped ? "SKIPPED" : "not run");
    if (cpu == LegOutcome::Fail || gpu == LegOutcome::Fail) { return 1; }
    if (mode == "gpu" && gpu == LegOutcome::Skipped) { return 2; }
    return 0;
}
