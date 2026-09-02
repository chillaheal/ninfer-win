// Bit-identical check: legacy A16 small-T kernel vs the cp.async-staged kernel.
//
// For every registered NVFP4 shape and every small-T value in [2, 32], run both kernels on
// the SAME varied synthetic weight + activation and compare the outputs byte-for-byte. The
// weight codes are filled with a position-dependent pattern (not a uniform value) so that a
// staging bug (wrong row / wrong phase / wrong chunk) is caught: a uniform weight would pass
// even if the cp.async staging copied the wrong slice, because every slice would be identical.
//
// The two kernels differ only in how the weight codes reach the FMA: the legacy kernel issues
// ld.global.cg per phase; the cp.async kernel stages the same bytes to shared memory. The FMA
// order, scale decode, and finalization are identical by construction, so a byte-for-byte match
// proves the staging is correct and the math is bit-identical.

#include "ops/linear/nvfp4/nvfp4_launch.h"

#include "core/device.h"
#include "core/tensor.h"
#include "ninfer_bench_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace ninfer;

namespace {

// Varied deterministic fills. The code index i = row * kCodeBytesPerRow + value, so the
// pattern varies across both rows and phases (values), catching row- and phase-mapping bugs.
__global__ void fill_varied_codes(std::uint8_t* codes, std::uint64_t count) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { codes[i] = static_cast<std::uint8_t>((i * 131u + 7u) & 0xFFu); }
}

// e4m3: 0x00..0x7E are finite; 0x7F/0xFF are NaN. Map into [1, 126] to stay finite and nonzero.
__global__ void fill_varied_scales(std::uint8_t* scales, std::uint64_t count) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { scales[i] = static_cast<std::uint8_t>(((i * 17u + 3u) % 126u) + 1u); }
}

int launch_grid(std::uint64_t elements) {
    return static_cast<int>(
        std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (elements + 255) / 256)));
}

struct Shape {
    const char* name;
    std::int32_t n;
    std::int32_t k;
};

constexpr Shape kShapes[] = {
    {"AttnInput", 14336, 5120},
    {"GdnInput", 16384, 5120},
    {"MlpGateUp", 34816, 5120},
    {"Residual6144", 5120, 6144},
    {"Residual17408", 5120, 17408},
};

// Build a varied synthetic NVFP4 weight with the same layout as bench::make_nvfp4_weight.
Weight make_varied_nvfp4_weight(std::int32_t n, std::int32_t k, DeviceBuffer& storage,
                                cudaStream_t stream) {
    const std::uint64_t elements     = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k);
    const std::uint64_t code_bytes   = elements / 2;
    const std::uint64_t scale_offset = (code_bytes + 255) / 256 * 256;
    const std::uint64_t scale_bytes  = elements / 16;
    const std::uint64_t divisor_offset = scale_offset + scale_bytes;
    const std::uint64_t payload_bytes  = divisor_offset + sizeof(float);

    storage = DeviceBuffer(static_cast<std::size_t>(payload_bytes));
    CUDA_CHECK(cudaMemset(storage.p, 0, storage.bytes));
    fill_varied_codes<<<launch_grid(code_bytes), 256, 0, stream>>>(
        static_cast<std::uint8_t*>(storage.p), code_bytes);
    fill_varied_scales<<<launch_grid(scale_bytes), 256, 0, stream>>>(
        static_cast<std::uint8_t*>(storage.p) + scale_offset, scale_bytes);
    CUDA_CHECK(cudaGetLastError());
    constexpr float kDivisor = 0.125F;
    CUDA_CHECK(cudaMemcpy(static_cast<std::uint8_t*>(storage.p) + divisor_offset, &kDivisor,
                          sizeof(kDivisor), cudaMemcpyHostToDevice));

    Weight weight{};
    weight.payload            = storage.p;
    weight.payload_bytes      = payload_bytes;
    weight.qtype              = QType::NVFP4;
    weight.layout             = QuantLayout::BlockScaleK16M128x4;
    weight.scale_dtype        = DType::FP8_E4M3FN;
    weight.group_size         = 16;
    weight.group              = 16;
    weight.ndim               = 2;
    weight.shape[0]           = n;
    weight.shape[1]           = k;
    weight.padded_shape[0]    = n;
    weight.padded_shape[1]    = k;
    weight.qdata              = storage.p;
    weight.qhigh              = nullptr;
    weight.scales             = static_cast<const std::uint8_t*>(storage.p) + scale_offset;
    weight.n                  = n;
    weight.k                  = k;
    weight.weight_scale_divisor = kDivisor;
    weight.input_scale_divisor  = 3.5F;
    return weight;
}

bool check_point(const Shape& shape, std::int32_t t, cudaStream_t stream) {
    DeviceBuffer storage;
    const Weight weight = make_varied_nvfp4_weight(shape.n, shape.k, storage, stream);

    const std::uint64_t x_elements   = static_cast<std::uint64_t>(shape.k) * static_cast<std::uint64_t>(t);
    const std::uint64_t out_elements = static_cast<std::uint64_t>(shape.n) * static_cast<std::uint64_t>(t);
    DeviceBuffer x = bench::make_bf16(x_elements);
    DeviceBuffer out_a(out_elements * 2);
    DeviceBuffer out_b(out_elements * 2);

    Tensor act(x.p, DType::BF16, {shape.k, t});
    Tensor outA(out_a.p, DType::BF16, {shape.n, t});
    Tensor outB(out_b.p, DType::BF16, {shape.n, t});

    CUDA_CHECK(cudaMemsetAsync(out_a.p, 0, out_a.bytes, stream));
    ops::detail::launch_nvfp4_small_t(act, weight, outA, stream);
    CUDA_CHECK(cudaMemsetAsync(out_b.p, 0, out_b.bytes, stream));
    ops::detail::launch_nvfp4_small_t_cpasync(act, weight, outB, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<std::uint16_t> ha(out_elements), hb(out_elements);
    CUDA_CHECK(cudaMemcpy(ha.data(), out_a.p, out_a.bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), out_b.p, out_b.bytes, cudaMemcpyDeviceToHost));
    for (std::uint64_t i = 0; i < out_elements; ++i) {
        if (ha[i] != hb[i]) {
            std::printf("  MISMATCH %-12s n=%d k=%d t=%d elem %llu: legacy=0x%04x cpasync=0x%04x\n",
                        shape.name, shape.n, shape.k, t, static_cast<unsigned long long>(i),
                        ha[i], hb[i]);
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int failures = 0;
    for (const Shape& shape : kShapes) {
        int matches = 0;
        for (std::int32_t t = 2; t <= 32; ++t) {
            if (check_point(shape, t, stream)) { ++matches; } else { ++failures; }
        }
        std::printf("%-14s n=%-6d k=%-6d %d/31 MATCH\n", shape.name, shape.n, shape.k, matches);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("%s (%d failures)\n", failures == 0 ? "ALL BIT-IDENTICAL" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
