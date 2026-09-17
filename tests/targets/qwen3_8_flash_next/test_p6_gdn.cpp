// P6: GDN at Flash Next production geometry.
//
// Production GDN layer (P0 inventory; 36 of the 48 text layers, L % 4 != 3):
// 16 qk heads x 128, 48 value heads x 128, width-4 causal conv on the 10240
// qkv channels (BF16 [4,10240]), per-sequence FP32 [128,128,48] recurrent
// state, model hidden 2560. Per-layer GDN weight dtypes from the inventory:
// qkv_z [16384,2560] and output [2560,6144] are FP8_E4M3FN_ROW_BF16S;
// convolution [4,10240], a_b_projection [96,2560] and norm [128] are BF16
// (no NVFP4 in the GDN objects). The conv is the generic causal_conv1d_silu
// Op; the recurrence is the shared gated_delta_net Op, which already accepts
// this geometry (value_heads >= qk_heads and divisible; state dim 128; any T)
// -- no kernel forks, this test qualifies the acceptance.
//
// Cases, one batch, normalized q/k (the production decode form), against the
// same naive FP64 recurrence oracle as the shared suite (gdn_ref):
//   (a) T=1   recurrent decode, in-place state
//   (b) T=16  multi-token recurrent, distinct state
//   (c) T=256 chunked prefill (4 full 64-chunks), distinct state
//   (d) the published planner state-byte formula matches the Op's state
//       tensor: 48 x 128 x 128 x 4 per layer, 36 layers per concurrency slot.

#include <cuda_runtime.h>

#include <ninfer/ops/gated_delta_net.h>
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include "ops/gdn_ref.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using FlashNextPackage = ninfer::targets::qwen3_8_flash_next::Package;
using ReductionCriterion = ninfer::test::ReductionCriterion;

constexpr int kStateDim   = 128;
constexpr int kQkHeads    = 16;
constexpr int kValueHeads = 48;

// The Op's documented numerical criteria (same values as the shared suite):
// the BF16 output promotion and the FP32 state update of the recurrence.
constexpr ReductionCriterion out_criterion() {
    return {/*relative_l2=*/4.1e-3, /*gross_absolute=*/5.0e-6,
            /*gross_relative_to_max_reference=*/5.5e-3};
}

constexpr ReductionCriterion state_criterion() {
    return {/*relative_l2=*/2.7e-3, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/3.9e-3};
}

int failures = 0;

void fail_message(const std::string& what) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
}

void expect_true(bool condition, const std::string& what) {
    if (!condition) fail_message(what);
}

void fill_uniform(std::vector<float>& values, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = distribution(generator); }
}

gdn_ref::Inputs make_inputs(int tokens, std::uint32_t seed) {
    gdn_ref::Inputs in;
    in.head_dim    = kStateDim;
    in.qk_heads    = kQkHeads;
    in.value_heads = kValueHeads;
    in.tokens      = tokens;

    const std::size_t qk_size    = static_cast<std::size_t>(kStateDim * kQkHeads * tokens);
    const std::size_t value_size = static_cast<std::size_t>(kStateDim * kValueHeads * tokens);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * kValueHeads);
    in.q.resize(qk_size);
    in.k.resize(qk_size);
    in.v.resize(value_size);
    in.g.resize(static_cast<std::size_t>(kValueHeads * tokens));
    in.beta.resize(static_cast<std::size_t>(kValueHeads * tokens));
    in.state.resize(state_size);

    std::mt19937 generator(seed);
    fill_uniform(in.q, generator, -1.0f, 1.0f);
    fill_uniform(in.k, generator, -1.0f, 1.0f);
    fill_uniform(in.v, generator, -0.5f, 0.5f);
    fill_uniform(in.g, generator, -0.10f, -0.005f);
    fill_uniform(in.beta, generator, 0.05f, 0.95f);
    fill_uniform(in.state, generator, -0.02f, 0.02f);

    // normalize_qk = true: the Op applies the row normalization, so the raw
    // rows stay valid unnormalized inputs here.
    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    return in;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> doubles(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

int verify_recurrence(const std::string& label, const std::vector<double>& got,
                      const std::vector<double>& expected, const ReductionCriterion& criterion) {
    return ninfer::test::verify_reduction(label.c_str(), got, expected, criterion);
}

struct DeviceInputs {
    explicit DeviceInputs(const gdn_ref::Inputs& in)
        : q(to_device_bf16(in.q)), k(to_device_bf16(in.k)), v(to_device_bf16(in.v)),
          g(to_device_f32(in.g)), beta(to_device_f32(in.beta)) {}

    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer g;
    DeviceBuffer beta;
};

int inputs_unchanged(const std::string& label, const gdn_ref::Inputs& in,
                     const DeviceInputs& device) {
    int result = 0;
    result += ninfer::test::verify_exact((label + " q unchanged").c_str(),
                                         from_device<std::uint16_t>(device.q, in.q.size()),
                                         bf16_bits(in.q));
    result += ninfer::test::verify_exact((label + " k unchanged").c_str(),
                                         from_device<std::uint16_t>(device.k, in.k.size()),
                                         bf16_bits(in.k));
    result += ninfer::test::verify_exact((label + " v unchanged").c_str(),
                                         from_device<std::uint16_t>(device.v, in.v.size()),
                                         bf16_bits(in.v));
    result += ninfer::test::verify_exact((label + " g unchanged").c_str(),
                                         from_device<float>(device.g, in.g.size()), in.g);
    result += ninfer::test::verify_exact((label + " beta unchanged").c_str(),
                                         from_device<float>(device.beta, in.beta.size()),
                                         in.beta);
    return result;
}

int production_case(int tokens, bool in_place, std::uint32_t seed) {
    const gdn_ref::Inputs in        = make_inputs(tokens, seed);
    const float scale               = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref       = gdn_ref::evaluate(in, static_cast<double>(scale), true);
    DeviceInputs device(in);
    GuardedDeviceBuffer state_in(in.state.size() * sizeof(float));
    GuardedDeviceBuffer state_out(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state_in.copy_from_host(in.state.data(), state_in.bytes());
    state_out.fill(0xff);
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, kQkHeads, tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, kQkHeads, tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, kValueHeads, tokens});
    Tensor g(device.g.p, DType::FP32, {kValueHeads, tokens});
    Tensor beta(device.beta.p, DType::FP32, {kValueHeads, tokens});
    Tensor state_in_tensor(state_in.data(), DType::FP32,
                           {kStateDim, kStateDim, kValueHeads});
    Tensor state_out_tensor(state_out.data(), DType::FP32,
                            {kStateDim, kStateDim, kValueHeads});
    Tensor out_tensor(out.data(), DType::BF16, {kStateDim, kValueHeads, tokens});
    const std::size_t workspace_bytes =
        ops::gated_delta_net_workspace_capacity_bytes(kQkHeads, kValueHeads, true, 1, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    if (in_place) {
        ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace, state_in_tensor,
                             out_tensor, nullptr);
    } else {
        ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace, state_in_tensor,
                             state_out_tensor, out_tensor, nullptr);
    }
    cuda_synchronize();

    const std::string label = "production T=" + std::to_string(tokens);
    const void* state_data = in_place ? state_in.data() : state_out.data();
    failures += verify_recurrence(label + " out",
                                  from_device_bf16(out.data(), in.v.size()), ref.out,
                                  out_criterion());
    failures += verify_recurrence(label + " state", doubles(from_device<float>(state_data,
                                                                              in.state.size())),
                                  ref.final_state, state_criterion());
    failures += state_in.verify_guards((label + " state-in").c_str());
    failures += state_out.verify_guards((label + " state-out").c_str());
    failures += out.verify_guards((label + " out").c_str());
    if (!in_place) {
        failures += ninfer::test::verify_exact(
            (label + " state-in unchanged").c_str(),
            from_device<float>(state_in.data(), in.state.size()), in.state);
    }
    failures += inputs_unchanged(label, in, device);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        fail_message(label + ": workspace query/execution high-water mismatch");
    }
    return 0;
}

int state_formula() {
    const std::uint64_t per_layer = FlashNextPackage::gdn_state_bytes_per_layer();
    expect_true(per_layer == 48ull * 128ull * 128ull * 4ull,
                std::string("per-layer state bytes ") + std::to_string(per_layer) +
                    " != 48 x 128 x 128 x 4");
    expect_true(per_layer ==
                    static_cast<std::uint64_t>(kStateDim) * kStateDim * kValueHeads *
                        static_cast<std::uint64_t>(sizeof(float)),
                "planner formula disagrees with the Op's state tensor layout");
    expect_true(FlashNextPackage::gdn_state_bytes(1) == 36ull * per_layer,
                std::string("gdn_state_bytes(1) ") +
                    std::to_string(FlashNextPackage::gdn_state_bytes(1)) +
                    " != 36 GDN layers x per-layer");
    expect_true(FlashNextPackage::gdn_state_bytes(4) ==
                    4 * FlashNextPackage::gdn_state_bytes(1),
                "gdn_state_bytes not linear in max_concurrency");
    return 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    production_case(/*tokens=*/1, /*in_place=*/true, /*seed=*/14001u);
    production_case(/*tokens=*/16, /*in_place=*/false, /*seed=*/14002u);
    production_case(/*tokens=*/256, /*in_place=*/false, /*seed=*/14003u);
    std::cout << "case T=2048 start (32 chunks, production round size)\n";
    std::cout.flush();
    production_case(/*tokens=*/2048, /*in_place=*/false, /*seed=*/14004u);
    state_formula();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " flash next P6 gated delta net\n";
    return failures == 0 ? 0 : 1;
}
