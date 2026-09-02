#pragma once

// cp.async-staged A16 small-T kernel.
//
// Same math and FMA order as nvfp4_small_t_kernel (bit-identical), but the weight
// codes are staged into shared memory with cp.async in a multi-phase pipeline so the
// next phase's DRAM traffic overlaps the current phase's FMA. The legacy kernel issues
// ld.global.cg per phase and stalls on the load latency; this kernel hides it.
//
// The CTA's rows use the rmod pattern (spread across the 128-row m_tile), so a TMA 2D
// tile would over-fetch ~12x. cp.async copies each row's phase slice exactly, with no
// over-fetch, which is why this variant uses cp.async rather than TMA.

#include "ops/linear/nvfp4/nvfp4_small_t.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Geometry, int ActiveTokens, class Schedule, int Stages>
struct Nvfp4SmallTCpAsyncSharedStorage {
    static constexpr int kPhaseCodeBytes =
        Schedule::kWarpsPerRow * 32 * Schedule::kValuesPerLane / 2;
    static_assert((kPhaseCodeBytes % 16) == 0);

    Nvfp4SmallTSharedStorage<Geometry, ActiveTokens, Schedule> base;
    alignas(16) std::uint8_t codes[Stages][Schedule::kRowsPerCta][kPhaseCodeBytes];
};

// Stage one phase's weight codes (all kRowsPerCta rows) into shared memory stage `stage`.
// Each thread issues one 16-byte cp.async; kRowsPerCta * kPhaseCodeBytes / 16 == kThreads
// for the production schedules, so every thread does exactly one copy.
template <class Geometry, class Schedule, int Stages>
__device__ __forceinline__ void stage_nvfp4_small_t_codes(
    const std::uint8_t* __restrict__ codes,
    std::uint8_t* __restrict__ codes_smem, const int (&parent_rows)[Schedule::kRowsPerCta],
    int phase, int stage) {
    constexpr int kPhaseCodeBytes =
        Schedule::kWarpsPerRow * 32 * Schedule::kValuesPerLane / 2;
    constexpr int kChunksPerRow   = kPhaseCodeBytes / 16;
    constexpr int kTotalChunks    = Schedule::kRowsPerCta * kChunksPerRow;
    static_assert(kTotalChunks <= Schedule::kThreads);
    for (int chunk = static_cast<int>(threadIdx.x); chunk < kTotalChunks;
         chunk += Schedule::kThreads) {
        const int row        = chunk / kChunksPerRow;
        const int byte_chunk = chunk - row * kChunksPerRow;
        const std::int64_t source =
            static_cast<std::int64_t>(parent_rows[row]) * Geometry::kCodeBytesPerRow +
            static_cast<std::int64_t>(phase) * kPhaseCodeBytes + byte_chunk * 16;
        cp_async<16, Cache::cg>(
            codes_smem + stage * Schedule::kRowsPerCta * kPhaseCodeBytes +
                row * kPhaseCodeBytes + byte_chunk * 16,
            codes + source);
    }
}

// FMA mainloop, identical to compute_nvfp4_small_t_rows except the weight codes are read
// from the cp.async-staged shared memory (stage `stage`) instead of ld.global.cg.
template <class Geometry, int ActiveTokens, class Schedule, int Stages>
__device__ __forceinline__ void compute_nvfp4_small_t_rows_cpasync(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ scales,
    Nvfp4SmallTCpAsyncSharedStorage<Geometry, ActiveTokens, Schedule, Stages>& shared,
    float inverse_weight_divisor, const int (&parent_rows)[Schedule::kRowsPerWarp], int flat_row0,
    int token0, int warp_in_row, int lane, int phase, int stage,
    float (&accumulators)[Schedule::kRowsPerWarp][Schedule::kTokenTile]
                         [Schedule::kAccumulatorChains]) {
    constexpr int kValuesPerWarpPhase = 32 * Schedule::kValuesPerLane;
    constexpr int kValuesPerPhase     = Schedule::kWarpsPerRow * kValuesPerWarpPhase;
    constexpr int kPhaseCodeBytes     = kValuesPerPhase / 2;
    constexpr int kGroupsPerLane =
        Schedule::kValuesPerLane < 16 ? 1 : Schedule::kValuesPerLane / 16;
    static_assert((Geometry::kInputRows % kValuesPerPhase) == 0);

    auto* codes_smem = &shared.codes[stage][0][0];

        if constexpr (Schedule::kActivationAccess == Nvfp4SmallTActivationAccess::SharedPhase) {
            static_assert((kValuesPerPhase % 8) == 0);
            constexpr int kPacksPerToken = kValuesPerPhase / 8;
            constexpr int kStagePacks    = Schedule::kTokenTile * kPacksPerToken;
            auto* destination            = reinterpret_cast<uint4*>(shared.base.activation);
            for (int task = static_cast<int>(threadIdx.x); task < kStagePacks;
                 task += Schedule::kThreads) {
                const int local_token = task / kPacksPerToken;
                const int local_pack  = task - local_token * kPacksPerToken;
                const int token       = token0 + local_token;
                if (token < ActiveTokens) {
                    const __nv_bfloat16* source =
                        x + static_cast<std::int64_t>(token) * Geometry::kInputRows +
                        phase * kValuesPerPhase + local_pack * 8;
                    destination[task] = load_vec<uint4>(source);
                }
            }
            __syncthreads();
        }

        const int warp_phase = phase * Schedule::kWarpsPerRow + warp_in_row;
        float coefficients[Schedule::kRowsPerWarp][kGroupsPerLane];
        Nvfp4CodePack<Schedule::kValuesPerLane> row_codes[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            load_nvfp4_coefficients<Geometry, Schedule>(
                scales, shared.base.gemv, parent_rows[local_row], flat_row0 + local_row,
                warp_phase, lane, inverse_weight_divisor, coefficients[local_row]);
            const int smem_offset =
                warp_in_row * (kValuesPerWarpPhase / 2) + lane * Schedule::kPairsPerLane;
            const std::uint8_t* smem_codes =
                codes_smem + (flat_row0 + local_row) * kPhaseCodeBytes + smem_offset;
            row_codes[local_row] = load_vec<Nvfp4CodePack<Schedule::kValuesPerLane>>(smem_codes);
        }

        if constexpr (Schedule::kActivationAccess == Nvfp4SmallTActivationAccess::TokenPacked) {
            Nvfp4ActivationPack<Schedule::kValuesPerLane> activation[Schedule::kTokenTile];
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
                const int token = token0 + local_token;
                if (token < ActiveTokens) {
                    const int value_begin = phase * kValuesPerPhase +
                                            warp_in_row * kValuesPerWarpPhase +
                                            lane * Schedule::kValuesPerLane;
                    activation[local_token] = load_nvfp4_activation_pack<Schedule::kValuesPerLane>(
                        x + static_cast<std::int64_t>(token) * Geometry::kInputRows + value_begin);
                }
            }

#pragma unroll
            for (int pair = 0; pair < Schedule::kPairsPerLane; ++pair) {
                float2 row_weight[Schedule::kRowsPerWarp];
                const int group = ((lane * Schedule::kValuesPerLane & 15) + pair * 2) / 16;
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    const std::uint32_t word  = row_codes[local_row].words[pair / 4];
                    const std::uint8_t packed = static_cast<std::uint8_t>(word >> (8 * (pair & 3)));
                    const float2 code         = decode_nvfp4_e2m1x2(packed);
                    const float coefficient   = coefficients[local_row][group];
                    row_weight[local_row] = make_float2(code.x * coefficient, code.y * coefficient);
                }
#pragma unroll
                for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
                    const int token = token0 + local_token;
                    if (token < ActiveTokens) {
                        const float2 activation_value =
                            bf16x2_bits_to_float2(activation[local_token].words[pair]);
                        constexpr int kChainMask = Schedule::kAccumulatorChains - 1;
#pragma unroll
                        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                            accumulators[local_row][local_token][(2 * pair) & kChainMask] =
                                fmaf(row_weight[local_row].x, activation_value.x,
                                     accumulators[local_row][local_token][(2 * pair) & kChainMask]);
                            accumulators[local_row][local_token][(2 * pair + 1) & kChainMask] =
                                fmaf(row_weight[local_row].y, activation_value.y,
                                     accumulators[local_row][local_token]
                                                 [(2 * pair + 1) & kChainMask]);
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (int pair = 0; pair < Schedule::kPairsPerLane; ++pair) {
                float2 row_weight[Schedule::kRowsPerWarp];
                const int group = ((lane * Schedule::kValuesPerLane & 15) + pair * 2) / 16;
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    const std::uint32_t word  = row_codes[local_row].words[pair / 4];
                    const std::uint8_t packed = static_cast<std::uint8_t>(word >> (8 * (pair & 3)));
                    const float2 code         = decode_nvfp4_e2m1x2(packed);
                    const float coefficient   = coefficients[local_row][group];
                    row_weight[local_row] = make_float2(code.x * coefficient, code.y * coefficient);
                }
                const int pair_index = phase * (kValuesPerPhase / 2) +
                                       warp_in_row * (kValuesPerWarpPhase / 2) +
                                       lane * Schedule::kPairsPerLane + pair;
#pragma unroll
                for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
                    const int token = token0 + local_token;
                    if (token < ActiveTokens) {
                        float2 activation_value;
                        if constexpr (Schedule::kActivationAccess ==
                                      Nvfp4SmallTActivationAccess::SharedPhase) {
                            const auto* activation_pairs = reinterpret_cast<const std::uint32_t*>(
                                shared.base.activation + local_token * kValuesPerPhase);
                            const int local_pair = warp_in_row * (kValuesPerWarpPhase / 2) +
                                                   lane * Schedule::kPairsPerLane + pair;
                            activation_value = bf16x2_bits_to_float2(activation_pairs[local_pair]);
                        } else {
                            const auto* activation_pairs = reinterpret_cast<const std::uint32_t*>(
                                x + static_cast<std::int64_t>(token) * Geometry::kInputRows);
                            activation_value = bf16x2_bits_to_float2(activation_pairs[pair_index]);
                        }
                        constexpr int kChainMask = Schedule::kAccumulatorChains - 1;
#pragma unroll
                        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                            accumulators[local_row][local_token][(2 * pair) & kChainMask] =
                                fmaf(row_weight[local_row].x, activation_value.x,
                                     accumulators[local_row][local_token][(2 * pair) & kChainMask]);
                            accumulators[local_row][local_token][(2 * pair + 1) & kChainMask] =
                                fmaf(row_weight[local_row].y, activation_value.y,
                                     accumulators[local_row][local_token]
                                                 [(2 * pair + 1) & kChainMask]);
                        }
                    }
                }
            }
        }

        if constexpr (Schedule::kActivationAccess == Nvfp4SmallTActivationAccess::SharedPhase) {
            __syncthreads();
        }
}

template <class Geometry, int ActiveTokens, class Schedule, int Stages, class Epilogue,
          class OutputPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_small_t_cpasync_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse_weight_divisor, Epilogue epilogue,
    OutputPolicy output) {
    static_assert(ActiveTokens >= 2);
    static_assert(Schedule::kTokenTile <= ActiveTokens);
    static_assert((Geometry::kOutputRows % 128) == 0);
    static_assert((Schedule::kRowsPerCta % 4) == 0);
    static_assert((128 % Schedule::kRowsPerCta) == 0);
    static_assert(Stages >= 2);

    constexpr int kRowBlocks  = Geometry::kOutputRows / Schedule::kRowsPerCta;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kValuesPerPhase = Schedule::kWarpsPerRow * 32 * Schedule::kValuesPerLane;
    constexpr int kPhases         = Geometry::kInputRows / kValuesPerPhase;
    static_assert(kPhases >= Stages, "pipeline needs at least Stages phases");

    const int linear_block = static_cast<int>(blockIdx.x);
    int row_block;
    int token_tile;
    if constexpr (Schedule::kBlockOrder == Nvfp4SmallTBlockOrder::RowsContiguous) {
        token_tile = linear_block / kRowBlocks;
        row_block  = linear_block - token_tile * kRowBlocks;
    } else {
        row_block  = linear_block / kTokenTiles;
        token_tile = linear_block - row_block * kTokenTiles;
    }
    const int token0 = kTokenTiles == 1 ? 0 : token_tile * Schedule::kTokenTile;

    __shared__ Nvfp4SmallTCpAsyncSharedStorage<Geometry, ActiveTokens, Schedule, Stages> shared;
    constexpr int kCtasPerM128 = 128 / Schedule::kRowsPerCta;
    const int m_tile           = row_block / kCtasPerM128;
    const int cta_in_tile      = row_block - m_tile * kCtasPerM128;
    const int rmod_base        = cta_in_tile * (Schedule::kRowsPerCta / 4);
    stage_nvfp4_scales<Geometry, Schedule>(scales, shared.base.gemv, m_tile, rmod_base);

    // CTA-wide parent rows (rmod pattern) for the cp.async staging.
    int parent_rows_cta[Schedule::kRowsPerCta];
#pragma unroll
    for (int flat_row = 0; flat_row < Schedule::kRowsPerCta; ++flat_row) {
        const int rmod     = rmod_base + flat_row / 4;
        const int quartile = flat_row & 3;
        parent_rows_cta[flat_row] = m_tile * 128 + rmod + quartile * 32;
    }

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int row_group   = warp / Schedule::kWarpsPerRow;
    const int warp_in_row = warp - row_group * Schedule::kWarpsPerRow;
    const int flat_row0   = row_group * Schedule::kRowsPerWarp;
    int parent_rows[Schedule::kRowsPerWarp];
#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        parent_rows[local_row] = parent_rows_cta[flat_row0 + local_row];
    }

    // Prologue: stage the first (Stages-1) phases.
#pragma unroll
    for (int s = 0; s < Stages - 1; ++s) {
        stage_nvfp4_small_t_codes<Geometry, Schedule, Stages>(codes, &shared.codes[0][0][0],
                                                              parent_rows_cta, s, s);
        cp_commit();
    }

    float accumulators[Schedule::kRowsPerWarp][Schedule::kTokenTile][Schedule::kAccumulatorChains] =
        {};

    for (int phase = 0; phase < kPhases; ++phase) {
        cp_wait<Stages - 2>();
        __syncthreads();
        compute_nvfp4_small_t_rows_cpasync<Geometry, ActiveTokens, Schedule, Stages>(
            x, scales, shared, inverse_weight_divisor, parent_rows, flat_row0, token0,
            warp_in_row, lane, phase, phase % Stages, accumulators);
        const int next = phase + Stages - 1;
        if (next < kPhases) {
            stage_nvfp4_small_t_codes<Geometry, Schedule, Stages>(codes, &shared.codes[0][0][0],
                                                                  parent_rows_cta, next,
                                                                  next % Stages);
            cp_commit();
        }
    }

    // Elementwise finalization (matches the production A16 small-T path).
#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
#pragma unroll
        for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
            const int token = token0 + local_token;
            if (token < ActiveTokens) {
                float total = 0.0F;
#pragma unroll
                for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                    total += accumulators[local_row][local_token][chain];
                }
                total = warp_reduce_sum(total);
                if constexpr (Schedule::kWarpsPerRow == 1) {
                    if (lane == 0) {
                        const int parent_row = parent_rows[local_row];
                        output.store(parent_row, token,
                                     epilogue.apply(parent_row, token, total));
                    }
                } else if (lane == 0) {
                    shared.base.partials[row_group][local_row][local_token][warp_in_row] = total;
                }
            }
        }
    }

    if constexpr (Schedule::kWarpsPerRow > 1) {
        __syncthreads();
        if (warp_in_row == 0) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
#pragma unroll
                for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
                    const int token = token0 + local_token;
                    if (token < ActiveTokens) {
                        const float partial =
                            lane < Schedule::kWarpsPerRow
                                ? shared.base.partials[row_group][local_row][local_token][lane]
                                : 0.0F;
                        const float total = warp_reduce_sum(partial);
                        if (lane == 0) {
                            const int parent_row = parent_rows[local_row];
                            output.store(parent_row, token,
                                         epilogue.apply(parent_row, token, total));
                        }
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
