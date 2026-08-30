#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <string>

namespace ninfer {
namespace targets {

// Result of a VRAM probe: the largest KV capacity (in tokens) that fits the current free
// device memory after weights, minus a fixed percentage headroom. Computed without
// materializing weights or constructing the model, so it completes in seconds.
struct KvProbeResult {
    std::string model_id;
    std::uint64_t vram_total_bytes              = 0;
    std::uint64_t weights_bytes                 = 0;
    std::uint64_t vram_free_after_weights_bytes = 0;
    std::uint32_t kv_headroom_percent           = 0;
    std::uint32_t kv_fit_tokens                 = 0;
};

// Public, CUDA-free entry point for the VRAM probe. Declared here (rather than in
// targets/registry.h) so consumers that link the engine without its internal include paths —
// notably the CLI target — can call it without transitively pulling in CUDA toolkit headers.
[[nodiscard]] KvProbeResult probe_kv_capacity(const EngineOptions& options);

} // namespace targets
} // namespace ninfer
