// Flash-Next S1 gate (b) + S3 driver — real-geometry device load + T = 1 forward.
//
// One-shot exe (not a ctest, so it can never fire the H2D by accident). Creates a
// DeviceContext, calls RealLoadedModel::open (H2D the ~7 GB backbone + mmap the
// 51.8 GB .ngram), logs the device-footprint summary, then constructs the real
// 48-layer RealProgram and runs ONE T = 1 forward through the 75.4 GB artifact,
// logging the sampled token + the peak device in-use. Run ONLY under the atomic
// stop/restore GPU script (the serve must be down so the VRAM is free).
//
//   usage: flash_next_real_load [artifact.ninfer] [ngram] [input_token]
// (defaults to the real models/ paths + input token 0; overridable by argv, then by
// env NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS / NINFER_QWEN3_8_FLASH_NEXT_NGRAM).

#include "targets/qwen3_8_flash_next/impl/load/real_loader.h"
#include "targets/qwen3_8_flash_next/impl/runtime/real_program.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using RoundTimingsT = ninfer::targets::qwen3_8_flash_next::RoundTimings;

// P13 M1: one measured round (driver wall + a copy of the program's per-round
// RoundTimings — last_timings() is overwritten by the next round, so it must be
// copied out of the program per call).
struct RoundStat {
  int t = 0;  // tokens in the round (T)
  float wall_ms = 0.0F;  // driver wall around run_sequence (incl. final D2H)
  RoundTimingsT tim;
  bool has_tim = false;
};
// RoundTimings bucket index of the MoE expert H2D scatter (real_program.cpp
// bucket order: 0 state_reset, 1 embed, 2 hc, 3 convert, 4 gdn, 5 qsa, 6 ple,
// 7 block_out, 8 moe_router, 9 moe_expert_h2d, 10 moe_gemm, 11 lm_head).
constexpr int kMoeScatterBucket = 9;
constexpr double kExpertBytes = 2764800.0;  // per scattered expert (P8 contract)

std::filesystem::path resolve_path(const char* env, const char* fallback, const char* argv,
                                   int argc, int idx) {
  if (argc > idx) return std::filesystem::path(argv);
  if (env && *env) return std::filesystem::path(env);
  return std::filesystem::path(fallback);
}

}  // namespace

int main(int argc, char** argv) {
  // Unbuffered so a crash preserves every progress line (a redirected stdout is
  // block-buffered and otherwise lost on a 0xC0000409 fastfail).
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
  const std::filesystem::path artifact =
      resolve_path(std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS"),
                   "C:/Users/Micke/Documents/Ninfer/models/qwen3_8_flash_next.ninfer", argv[1],
                   argc, 1);
  const std::filesystem::path ngram =
      resolve_path(std::getenv("NINFER_QWEN3_8_FLASH_NEXT_NGRAM"),
                   "C:/Users/Micke/Documents/Ninfer/models/qwen3_8_flash_next.ngram", argv[2], argc,
                   2);
#pragma warning(pop)

  // S3 single input token: argv[3] if present, else 0. It indexes the embedding
  // table (the gather reads e4m3(code[id*K + c])), so it must be in vocab range.
  constexpr std::uint32_t kVocab = 248320;  // = detail::Geometry::kVocab (config-free driver)
  std::uint32_t token = 0;
  if (argc > 3) {
    char* end = nullptr;
    const long v = std::strtol(argv[3], &end, 10);
    if (end == argv[3] || v < 0 || v >= static_cast<long>(kVocab)) {
      std::fprintf(stderr, "S3 FAIL: input token (argv[3]) must be an integer in [0, %u)\n", kVocab);
      return 1;
    }
    token = static_cast<std::uint32_t>(v);
  }

  // Dynamic QSA paged-KV context cap (2026-09-12 100k workstream). NINFER_P13_MAX_CTX
  // sizes the RealProgram's QSA K/V pool to ceil(max_context / 64) pages (the P13
  // default 8320 = 130 pages, byte-identical to the original hardcoded pool). 100000
  // -> 1563 pages. The GDN recurrent state (context-independent, no position limit) +
  // the 1024-token chunked QSA prefill (round-local indexer) are ALREADY
  // context-independent, so this only grows the K/V pool (~2.1 GiB more at 100k across
  // the 12 QSA layers). Minimum 64 (one page).
  std::uint32_t max_context = 8320;
  {
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
    const char* mc_env = std::getenv("NINFER_P13_MAX_CTX");
#pragma warning(pop)
    if (mc_env != nullptr && *mc_env != '\0') {
      char* end = nullptr;
      const long v = std::strtol(mc_env, &end, 10);
      if (end == mc_env || v < 64) {
        std::fprintf(stderr, "NINFER_P13_MAX_CTX must be an integer >= 64 (one page), got '%s'\n",
                     mc_env);
        return 1;
      }
      max_context = static_cast<std::uint32_t>(v);
    }
  }

  if (!std::filesystem::exists(artifact)) {
    std::fprintf(stderr, "S1 gate (b) FAIL: artifact not found: %s\n",
                 artifact.string().c_str());
    return 1;
  }
  if (!std::filesystem::exists(ngram)) {
    std::fprintf(stderr, "S1 gate (b) FAIL: ngram not found: %s\n", ngram.string().c_str());
    return 1;
  }

  const double gib = 1024.0 * 1024.0 * 1024.0;
  std::printf("S1 gate (b) real device load\n");
  std::printf("  artifact = %s\n", artifact.string().c_str());
  std::printf("  ngram    = %s\n", ngram.string().c_str());

  try {
    ninfer::DeviceContext device(0);
    std::printf("  device   = sm_%d, total VRAM %.2f GiB\n", device.sm(),
                static_cast<double>(device.total_vram()) / gib);

    auto lm = ninfer::targets::qwen3_8_flash_next::RealLoadedModel::open(artifact, ngram, device);
    const auto& s = lm->summary();

    std::printf("  load OK\n");
    std::printf("  layers              = %zu\n", s.layer_count);
    std::printf("  object_count        = %zu\n", s.object_count);
    std::printf("  device_objects      = %zu\n", s.device_objects);
    std::printf("  host_objects        = %zu\n", s.host_objects);
    std::printf("  ple_embed_dim       = %u\n", s.ple_embed_dim);
    std::printf("  ngram_bytes         = %llu (%.2f GiB, mmap not pinned)\n",
                static_cast<unsigned long long>(s.ngram_bytes), s.ngram_bytes / gib);
    std::printf("  device_capacity     = %llu (%.3f GiB, planned)\n",
                static_cast<unsigned long long>(s.device_capacity_bytes),
                s.device_capacity_bytes / gib);
    std::printf("  h2d_bytes           = %llu (%.3f GiB)\n",
                static_cast<unsigned long long>(s.h2d_bytes), s.h2d_bytes / gib);
    std::printf("  arena_peak          = %llu (%.3f GiB)\n",
                static_cast<unsigned long long>(s.arena_peak_bytes),
                s.arena_peak_bytes / gib);
    std::printf("  derived_bytes       = %llu (%.3f GiB, HC FP32 + GDN conv)\n",
                static_cast<unsigned long long>(s.derived_bytes), s.derived_bytes / gib);
    std::printf("  TOTAL device        = %llu (%.3f GiB)\n",
                static_cast<unsigned long long>(s.total_device_bytes), s.total_device_bytes / gib);

    device.synchronize();

    // Sanity gate: the total device footprint must fit the card with headroom
    // (the 32 GB card with the serve down). Bail loudly if it does not.
    if (s.total_device_bytes > device.total_vram()) {
      std::fprintf(stderr, "S1 gate (b) FAIL: total device %llu > VRAM %llu\n",
                   static_cast<unsigned long long>(s.total_device_bytes),
                   static_cast<unsigned long long>(device.total_vram()));
      return 1;
    }
    std::printf("S1 gate (b) PASS: real 180B backbone loaded + .ngram mmap'd.\n");

    // ---------------------------------------------------------------------
    // S3 — the real 48-layer Program: ONE T = 1 forward through the 75.4 GB
    // artifact. `program` is a shallow host-pointer view over `lm`'s device
    // bytes, constructed after `lm` and destroyed before it (reverse local
    // order), satisfying the lifetime contract. The round arena + moe_window +
    // recurrent state are all pre-allocated in the ctor, so the forward
    // allocates nothing new; the peak device in-use is read right after the
    // construction (the high point).
    // ---------------------------------------------------------------------
    std::size_t total = 0, free1 = 0, free2 = 0;
    std::printf("\nS3 real 48-layer Program (ONE T = 1 forward)\n");
    std::printf("  input token       = %u\n", token);
    std::printf("  max_context       = %u (QSA paged KV: %u pages x 64 tok; GDN recurrent "
                "has no position limit)\n",
                max_context, (max_context + 63u) / 64u);
    ninfer::targets::qwen3_8_flash_next::RealProgram program(lm->view(), device, max_context);
    // P13 M1: NINFER_P13_TIMING=1 turns on the per-op stream-event breakdown.
    // Events are stream annotations (no data or ordering change), so the greedy
    // stream must stay bit-identical to the P12 S5 constant (guarded below).
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
    const char* p13_timing_env = std::getenv("NINFER_P13_TIMING");
#pragma warning(pop)
    const bool p13_timing = (p13_timing_env != nullptr) && (p13_timing_env[0] == '1');
    if (p13_timing) {
      program.set_timings_enabled(true);
      std::printf("  P13 timing mode   = ON (per-op stream-event breakdown)\n");
    }
    std::vector<RoundStat> round_stats;
    if (cudaMemGetInfo(&free1, &total) != cudaSuccess) {
      std::fprintf(stderr, "S3 FAIL: cudaMemGetInfo (post-construction) failed\n");
      return 1;
    }
    std::printf("  GPU in-use peak   = %llu (%.3f GiB of %.2f GiB total, after Program ctor)\n",
                static_cast<unsigned long long>(total - free1),
                static_cast<double>(total - free1) / gib, static_cast<double>(total) / gib);

    std::vector<float> logits;
    const auto s3_t0 = std::chrono::steady_clock::now();
    const std::uint32_t sampled =
        program.run_sequence(std::vector<std::uint32_t>{token}, &logits);
    const auto s3_t1 = std::chrono::steady_clock::now();
    if (p13_timing) {
      if (const RoundTimingsT* lt = program.last_timings()) {
        RoundStat rs;
        rs.t = 1;
        rs.wall_ms = std::chrono::duration<float, std::milli>(s3_t1 - s3_t0).count();
        rs.tim = *lt;
        rs.has_tim = true;
        round_stats.push_back(rs);
      }
    }
    device.synchronize();
    if (cudaMemGetInfo(&free2, &total) != cudaSuccess) {
      std::fprintf(stderr, "S3 FAIL: cudaMemGetInfo (post-forward) failed\n");
      return 1;
    }
    std::printf("  sampled token     = %u\n", sampled);
    std::printf("  GPU in-use post   = %llu (%.3f GiB; forward is memory-neutral, arena pre-allocated)\n",
                static_cast<unsigned long long>(total - free2),
                static_cast<double>(total - free2) / gib);
    if (sampled >= kVocab) {
      std::fprintf(stderr, "S3 FAIL: sampled token %u out of vocab range [0, %u)\n", sampled, kVocab);
      return 1;
    }
    std::printf("S3 PASS: real 48-layer T = 1 forward completed; sampled token = %u\n", sampled);

    // ---------------------------------------------------------------------
    // S4 — MoE-scale confirm vs P8 oracle. The end-to-end lm_head logit scale
    // is the MoE-health signal: the S0 ~1e21 MoE epilogue defect (alpha was a
    // multiply instead of 1.0F/divisor) would propagate through the 48 real
    // MoE layers to O(1e21) logits if it were present at the real 512-expert
    // geometry; a healthy real stack reads ~O(10) (the P8-oracle scale).
    // Measure max/mean |logit|, the top-5 (value + token), a nan/inf count,
    // and a count of |logit| >= 1e6 as the defect detector.
    // ---------------------------------------------------------------------
    std::printf("\nS4 MoE-scale confirm (end-to-end lm_head logits, final position)\n");
    std::printf("  vocab size         = %zu\n", logits.size());
    double max_abs = 0.0, sum_abs = 0.0;
    long max_idx = -1, nan_inf = 0, big = 0;
    for (std::size_t n = 0; n < logits.size(); ++n) {
      const double v = static_cast<double>(logits[n]);
      if (std::isnan(v) || std::isinf(v)) {
        ++nan_inf;
        continue;
      }
      const double a = std::fabs(v);
      sum_abs += a;
      if (a >= 1e6) ++big;
      if (a > max_abs) {
        max_abs = a;
        max_idx = static_cast<long>(n);
      }
    }
    const double mean_abs = logits.empty() ? 0.0 : sum_abs / static_cast<double>(logits.size());
    std::vector<std::pair<float, std::size_t>> top;
    top.reserve(logits.size());
    for (std::size_t n = 0; n < logits.size(); ++n) top.emplace_back(logits[n], n);
    std::partial_sort(top.begin(),
                      top.begin() + std::min<std::size_t>(5, top.size()), top.end(),
                      [](const std::pair<float, std::size_t>& a,
                         const std::pair<float, std::size_t>& b) { return a.first > b.first; });
    std::printf("  max |logit|       = %.6g (token %ld)\n", max_abs, max_idx);
    std::printf("  mean |logit|      = %.6g\n", mean_abs);
    std::printf("  nan/inf count     = %ld\n", nan_inf);
    std::printf("  count |logit|>=1e6 = %ld\n", big);
    std::printf("  top-5 (value, token):\n");
    for (std::size_t i = 0; i < top.size() && i < 5; ++i)
      std::printf("    % .6g  (tok %zu)\n", static_cast<double>(top[i].first), top[i].second);
    // Sane = no non-finite, no O(1e21) tail (defect), and not a total collapse
    // (a healthy 180B max |logit| over 248k tokens is far above 1e-3).
    const bool sane = (nan_inf == 0) && (max_abs < 1e6) && (max_abs >= 1e-3);
    std::printf("S4 %s: real 512-expert MoE end-to-end logit scale = %.4g "
                "(%s the P8-oracle ~O(10); the S0 ~1e21 defect is %s)\n",
                sane ? "PASS" : "FAIL", max_abs, sane ? "consistent with" : "DEPARTS FROM",
                (max_abs >= 1e6 || nan_inf > 0) ? "PRESENT" : "absent");

    // ---------------------------------------------------------------------
    // S5 -- reproducible 16-token greedy over the real 75.4 GB artifact.
    //
    // The Program is a one-shot pos0 = 0 forward (the QSA indexer is
    // round-local), so greedy is a re-prefill: round k re-feeds the growing
    // prefix [seed, t1..t_{k-1}] (T = k) with fresh recurrent state
    // (run_sequence resets it) and takes the argmax at the final position ->
    // t_k. 16 rounds, T = 1..16 (the last prefix is exactly kMaxRoundTokens =
    // 16). Round 1 (T = 1, prefix = [seed]) reproduces the S3 sampled token.
    // Each round's full-vocab logit scale re-confirms the MoE health
    // end-to-end at T > 1 (the open S5 risk: the T>1 channel-major layout +
    // the recurrent GDN/QSA state carrying across positions in one round).
    // ---------------------------------------------------------------------
    constexpr int kGreedyLen = 16;
    std::printf("\nS5 reproducible 16-token greedy (re-prefill, T = 1..16)\n");
    std::printf("  seed token        = %u\n", token);
    std::vector<std::uint32_t> prefix = {token};
    std::vector<std::uint32_t> generated;
    generated.reserve(kGreedyLen);
    double s5_max_abs = 0.0;
    long s5_nan_inf = 0;
    bool s5_oob = false;
    std::vector<std::pair<int, double>> round_scale;
    round_scale.reserve(kGreedyLen);
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 1; k <= kGreedyLen; ++k) {
      std::vector<float> rlogits;
      const auto rk0 = std::chrono::steady_clock::now();
      const std::uint32_t next = program.run_sequence(prefix, &rlogits);
      const auto rk1 = std::chrono::steady_clock::now();
      if (p13_timing) {
        if (const RoundTimingsT* lt = program.last_timings()) {
          RoundStat rs;
          rs.t = k;
          rs.wall_ms = std::chrono::duration<float, std::milli>(rk1 - rk0).count();
          rs.tim = *lt;
          rs.has_tim = true;
          round_stats.push_back(rs);
        }
      }
      if (next >= kVocab) {
        std::fprintf(stderr, "S5 FAIL: round %d sampled token %u out of vocab [0, %u)\n",
                     k, next, kVocab);
        s5_oob = true;
        break;
      }
      double rmax = 0.0;
      long rnan = 0;
      for (std::size_t n = 0; n < rlogits.size(); ++n) {
        const double v = static_cast<double>(rlogits[n]);
        if (std::isnan(v) || std::isinf(v)) {
          ++rnan;
          continue;
        }
        const double a = std::fabs(v);
        if (a > rmax) rmax = a;
      }
      if (rmax > s5_max_abs) s5_max_abs = rmax;
      s5_nan_inf += rnan;
      round_scale.emplace_back(k, rmax);
      std::printf("  round %2d (T=%2d): argmax token = %-6u  max|logit| = %.6g\n", k, k, next,
                  rmax);
      prefix.push_back(next);
      generated.push_back(next);
    }
    device.synchronize();
    const double greedy_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                              t0)
            .count() /
        1e3;
    std::printf("  16-token stream   =");
    for (std::size_t i = 0; i < generated.size(); ++i) std::printf(" %u", generated[i]);
    std::printf("\n");
    std::printf("  global max|logit|= %.6g, nan/inf = %ld, 16-round wall = %.1f ms\n", s5_max_abs,
                s5_nan_inf, greedy_ms);
    const bool s5_ok =
        (generated.size() == static_cast<std::size_t>(kGreedyLen)) && !s5_oob &&
        (s5_nan_inf == 0) && (s5_max_abs < 1e6) && (s5_max_abs >= 1e-3);
    // Durability: write the token stream + metadata so two reloads can be
    // diffed for bit-determinism (cwd-independent absolute path).
    {
      const char* out =
          "C:/Users/Micke/Documents/Kodprojekt/Ninfer/ninfer-win/out/flash_next_dev/p12_greedy.txt";
      std::ofstream of(out, std::ios::trunc);
      if (!of) {
        std::fprintf(stderr, "S5 WARN: could not write %s\n", out);
      } else {
        of << "Flash-Next P12 S5 -- reproducible 16-token greedy (real 75.4 GB artifact, "
              "real 512-expert MoE geometry)\n";
        of << "seed_token            = " << token << "\n";
        of << "generated_token_count = " << generated.size() << "\n";
        of << "gpu_in_use_peak_gib   = " << static_cast<double>(total - free1) / gib << "\n";
        of << "global_max_abs_logit  = " << s5_max_abs << "\n";
        of << "nan_inf_count         = " << s5_nan_inf << "\n";
        of << "generated_stream      =";
        for (std::size_t i = 0; i < generated.size(); ++i) of << " " << generated[i];
        of << "\n";
        for (const auto& r : round_scale)
          of << "round_T_" << r.first << "  max_abs_logit = " << r.second << "\n";
        of << "S5_verdict            = " << (s5_ok ? "PASS" : "FAIL") << "\n";
      }
    }
    std::printf(
        "S5 %s: 16-token greedy (%s; global max|logit| = %.4g %s, %s)\n",
        s5_ok ? "PASS" : "FAIL",
        (generated.size() == static_cast<std::size_t>(kGreedyLen) && !s5_oob)
            ? "16/16 tokens in vocab"
            : "INCOMPLETE",
        s5_max_abs,
        (s5_max_abs >= 1e6 || s5_nan_inf > 0) ? "(~1e21 defect PRESENT)" : "(no ~1e21 defect)",
        s5_ok ? "healthy at T>1" : "UNHEALTHY");
    // Bit-determinism guard (runs on EVERY default-seed run, timing mode or not):
    // with seed 0 the greedy stream must equal the post-QSA-262k-reconstruction S5
    // constant. The FP8 global-paged indexer + T cap 2048 deliberately changed the
    // numerics vs the pre-reconstruction round-local BF16 v1, so the old P12 oracle
    // (256,389,...) is STALE. The verified stream below is from the P1 NaN-fix probe
    // .logs/nan_probe/probe_20260913_152854 (S3 sampled 3241; S5 16/16 in vocab,
    // finite + healthy-scaled, bit-determinism-vs-old-oracle FAIL = EXPECTED). This is
    // the invariant the M3 run_sequence->run_round refactor must preserve (S5 still
    // re-prefills a pos0 = 0 round through the single-round path).
    // RE-RECORDED 2026-09-13 after Fix B (MoE token-major layout fix: removed two
    // spurious cm transposes so the op reads xhat + writes y directly, making op
    // dids == host selected; verified dev_uniq=0 / max|dlogit|=0 on all VERIFY IDS
    // lines, window path). The stream changed from token 2 (the T=1 single-token
    // path is transpose-invariant, so token 1 = 3241 is unchanged). Provenance:
    // .logs/p13_gpu_verify/probe_20260913_231713 (GpuSlots=0, VERIFY=1).
    if (token == 0) {
      const std::uint32_t expected[16] = {3241U, 7U, 10435U, 72181U, 18532U, 224070U,
                                          1347U, 153926U, 198U, 28958U, 211173U, 381U,
                                          1317U, 17598U, 183870U, 220U};
      bool match = (generated.size() == 16);
      for (std::size_t i = 0; match && i < 16; ++i) match = (generated[i] == expected[i]);
      std::printf("  bit-determinism vs post-reconstruction S5 stream: %s\n",
                  match ? "PASS" : "FAIL");
      if (!match) {
        // CPU-offload (NINFER_MOE_CPU_LAYERS=1) deliberately breaks bit-exactness: the
        // CPU-routed layers use the documented host-logits approximation + CPU F32
        // dequant/GEMM, so the greedy stream legitimately diverges from the CPU-off
        // baseline. Relax the hard abort to a WARN so the P13-M1/M3 timing breakdowns
        // (incl. moe_expert_h2d) still print under CPU-on.
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
        const char* cpu_env = std::getenv("NINFER_MOE_CPU_LAYERS");
#pragma warning(pop)
        const bool cpu_on = (cpu_env != nullptr) && (cpu_env[0] != '\0') &&
                            !(cpu_env[0] == '0' && cpu_env[1] == '\0');
        if (cpu_on) {
          std::printf(
              "  (EXPECTED under NINFER_MOE_CPU_LAYERS=1; continuing to timing "
              "breakdown)\n");
        } else {
          std::fprintf(stderr,
                       "S5 FAIL: greedy stream diverged from the post-reconstruction S5 "
                       "constant\n");
          return 1;
        }
      }
    }
    if (p13_timing) {
      // -------------------------------------------------------------------
      // P13 M1 report: the per-op timing breakdown of the real forward.
      // round_stats[0] = the S3 T=1 forward; [1..16] = the S5 re-prefill
      // rounds (T = 1..16). Single stream -> the buckets partition the
      // stream-busy time; wall - gpu_total = total GPU-idle (CPU-gated).
      // -------------------------------------------------------------------
      std::printf("\nP13-M1 per-op timing breakdown (NINFER_P13_TIMING=1; %zu rounds)\n",
                  round_stats.size());
      std::printf("  round  T    wall_ms   gpu_ms  cpu_barrier_ms  barrier_gpu_ms "
                  "  union_exp  scatter_MiB  scatter_GB/s  lru_hit  lru_miss\n");
      double wall_sum = 0.0, gpu_sum = 0.0, barrier_sum = 0.0, barrier_gpu_sum = 0.0;
      unsigned long long lru_hit_sum = 0, lru_miss_sum = 0;
      for (std::size_t i = 0; i < round_stats.size(); ++i) {
        const RoundStat& r = round_stats[i];
        if (!r.has_tim) continue;
        double gpu = 0.0;
        for (int b = 0; b < RoundTimingsT::kBuckets; ++b) gpu += r.tim.gpu_ms[b];
        const double scatter_bytes =
            static_cast<double>(r.tim.moe_union_experts) * kExpertBytes;
        const double scatter_ms = r.tim.gpu_ms[kMoeScatterBucket];
        const double gbps =
            (scatter_ms > 0.0) ? scatter_bytes / (scatter_ms * 1e-3) / 1e9 : 0.0;
        wall_sum += r.wall_ms;
        gpu_sum += gpu;
        barrier_sum += r.tim.host_barrier_ms;
        barrier_gpu_sum += r.tim.barrier_gpu_ms;
        lru_hit_sum += static_cast<unsigned long long>(r.tim.lru_hits);
        lru_miss_sum += static_cast<unsigned long long>(r.tim.lru_misses);
        std::printf("  %4zu  %2d  %9.1f  %8.1f  %13.1f  %14.1f  %9lld  %11.2f  %13.2f  %8llu  %9llu\n",
                    i + 1, r.t, static_cast<double>(r.wall_ms), gpu,
                    static_cast<double>(r.tim.host_barrier_ms),
                    static_cast<double>(r.tim.barrier_gpu_ms), r.tim.moe_union_experts,
                    scatter_bytes / (1024.0 * 1024.0), gbps,
                    static_cast<unsigned long long>(r.tim.lru_hits),
                    static_cast<unsigned long long>(r.tim.lru_misses));
      }
      std::printf("  bucket totals (all %zu rounds):\n", round_stats.size());
      for (int b = 0; b < RoundTimingsT::kBuckets; ++b) {
        double t = 0.0;
        for (const RoundStat& r : round_stats) {
          if (r.has_tim) t += r.tim.gpu_ms[b];
        }
        std::printf("    %-16s %12.1f ms  %6.2f%% of GPU  %6.2f%% of wall\n",
                    RoundTimingsT::name(b), t, (gpu_sum > 0) ? 100.0 * t / gpu_sum : 0.0,
                    (wall_sum > 0) ? 100.0 * t / wall_sum : 0.0);
      }
      std::printf("  TOTALS: wall = %.1f ms, GPU-busy = %.1f ms, GPU-idle = %.1f ms "
                  "(%6.2f%% of wall)\n",
                  wall_sum, gpu_sum, (wall_sum - gpu_sum),
                  (wall_sum > 0) ? 100.0 * (wall_sum - gpu_sum) / wall_sum : 0.0);
      std::printf("  MoE router D2H gate: CPU-blocked = %.1f ms total, stream time "
                  "inside the window = %.1f ms (CPU-blocked minus that = pure stall "
                  "of the CPU while the GPU idles)\n",
                  barrier_sum, barrier_gpu_sum);
      if (lru_hit_sum == 0 && lru_miss_sum == 0) {
        std::printf("  Pinned expert LRU (M2): disabled (default = M1 pageable "
                    "path, the measured-best; opt in via NINFER_EXPERT_PINNED_SLOTS)\n");
      } else {
        std::printf("  Pinned expert LRU (M2): %llu hits, %llu misses (a hit = one "
                    "contiguous pinned->device DMA; a miss = M1 pageable DMAs + "
                    "post-staging promotion into a slot)\n",
                    lru_hit_sum, lru_miss_sum);
      }
      std::printf("  Note: wall includes the per-round final sample/logits D2H; "
                  "barrier_gpu overlaps the buckets (router-GEMM drain + scatter "
                  "DMA), so it is NOT added to the bucket totals.\n");

      // Durability: the measured report (cwd-independent absolute path).
      {
        const char* out =
            "C:/Users/Micke/Documents/Kodprojekt/Ninfer/ninfer-win/out/flash_next_dev/p13_timing.txt";
        std::ofstream of(out, std::ios::trunc);
        if (!of) {
          std::fprintf(stderr, "P13-M1 WARN: could not write %s\n", out);
        } else {
          of << "Flash-Next P13 M1 -- per-op timing breakdown (real 75.4 GB artifact, "
                "NINFER_P13_TIMING=1)\n";
          of << "seed_token        = " << token << "\n";
          of << "round_count       = " << round_stats.size() << "\n";
          of << "wall_total_ms     = " << wall_sum << "\n";
          of << "gpu_busy_total_ms = " << gpu_sum << "\n";
          of << "gpu_idle_total_ms = " << (wall_sum - gpu_sum) << "\n";
          of << "moe_barrier_wall_ms  = " << barrier_sum << "\n";
          of << "moe_barrier_gpu_ms   = " << barrier_gpu_sum << "\n";
          of << "buckets:\n";
          for (int b = 0; b < RoundTimingsT::kBuckets; ++b) {
            double t = 0.0;
            for (const RoundStat& r : round_stats) {
              if (r.has_tim) t += r.tim.gpu_ms[b];
            }
            of << "  " << RoundTimingsT::name(b) << "  " << t << " ms\n";
          }
          of << "per_round (t wall_ms gpu_ms barrier_ms barrier_gpu_ms union_experts "
                "scatter_bytes lru_hits lru_misses):\n";
          for (std::size_t i = 0; i < round_stats.size(); ++i) {
            const RoundStat& r = round_stats[i];
            if (!r.has_tim) continue;
            double gpu = 0.0;
            for (int b = 0; b < RoundTimingsT::kBuckets; ++b) gpu += r.tim.gpu_ms[b];
            of << "  T=" << r.t << "  " << r.wall_ms << "  " << gpu << "  "
               << r.tim.host_barrier_ms << "  " << r.tim.barrier_gpu_ms << "  "
               << r.tim.moe_union_experts << "  "
               << (static_cast<double>(r.tim.moe_union_experts) * kExpertBytes) << "  "
               << r.tim.lru_hits << "  " << r.tim.lru_misses << "\n";
          }
        }
      }
    }

    // ---------------------------------------------------------------------
    // P13 M3 -- benchmark matrix: C = 1, T_prompt in {512, 2048, 8192}, T_new = 128.
    // Runs only under NINFER_P13_M3=1. For each T_prompt: begin_sequence() (fresh
    // recurrent + PLE state), a chunked state-carrying prefill (T_prompt tokens in
    // <= kRoundCap = 1024-token rounds at their absolute positions), then T_new
    // single-token greedy decode rounds. Measures prefill and decode throughput
    // (tok/s) + the GPU in-use. The numbers are MEASURED on this machine (no
    // invented tok/s). Caveat (footnoted on the M4 page): the QSA indexer is
    // round-local, so in this chunked-prefill + T = 1-decode schedule the QSA
    // attention does not span the full prompt + decode history (the long-range
    // path is the GDN recurrent state); this is the engine's existing QSA
    // semantics, not a benchmark artifact, and it does not change the measured
    // throughput (the GEMMs / GDN / MoE all run at full T).
    // ---------------------------------------------------------------------
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
    const char* m3_env = std::getenv("NINFER_P13_M3");
#pragma warning(pop)
    const bool m3 = (m3_env != nullptr) && (m3_env[0] == '1');
    if (m3) {
      constexpr int kTnew = 128;
      // QSA indexer v1 max (qsa_indexer.h: "1 <= T <= 1024"; scores buffer is [T, T]).
      // The MoE sparse_moe_nvfp4 2048 cap is never the binding constraint (1024 <= 2048).
      // Chunking prefill at 1024 is the engine's real production schedule; T = 1
      // decode rounds are always within the limit.
      constexpr int kRoundCap = 1024;
      const int t_prompts[] = {512, 2048, 8192};
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
      const char* maxtp_env = std::getenv("NINFER_P13_M3_MAXTP");
#pragma warning(pop)
      // Bound the matrix (default 8192 = full). A re-measure sets NINFER_P13_M3_MAXTP=512
      // to skip the heavy 2048/8192 prefill rows (the 8192 case is what hung ~11 h),
      // keeping a decode tok/s re-measure to a few minutes.
      const int maxtp = (maxtp_env != nullptr) ? std::atoi(maxtp_env) : 8192;
      std::printf("\nP13-M3 benchmark matrix (C = 1, T_new = %d, kv = QSA bf16 / GDN fp32, maxtp=%d)\n",
                  kTnew, maxtp);
      std::printf(
          "  T_prompt  chunks  prefill_ms  prefill_tok_s  decode_ms  decode_tok_s  peak_gpu_GiB\n");
      size_t peak_inuse = 0;
      {
        size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) == cudaSuccess) peak_inuse = t - f;
      }
      auto make_prompt = [](int n, std::uint32_t seed) {
        std::vector<std::uint32_t> p(n);
        std::uint64_t s = seed * 0x9E3779B97F4A7C15ULL + 1;
        for (int i = 0; i < n; ++i) {
          s = s * 6364136223846793005ULL + 1442695040888963407ULL;  // splitmix64 mix
          p[i] = static_cast<std::uint32_t>((s >> 33) % kVocab);
        }
        return p;
      };
      // Per-T_prompt measured rows (also written durably for the M4 page).
      struct Row {
        int tp = 0;
        int chunks = 0;
        double pf_ms = 0.0, pf_tps = 0.0, dc_ms = 0.0, dc_tps = 0.0, peak_gib = 0.0;
      };
      std::vector<Row> rows;
      for (const int tp : t_prompts) {
        if (tp > maxtp) continue;  // NINFER_P13_M3_MAXTP bound (skip heavy prefill rows)
        program.begin_sequence();
        const std::vector<std::uint32_t> prompt = make_prompt(tp, static_cast<std::uint32_t>(tp));
        device.synchronize();
        const auto pf0 = std::chrono::steady_clock::now();
        int pos0 = 0;
        int n_chunks = 0;
        std::uint32_t last = 0;
        for (int off = 0; off < tp; off += kRoundCap) {
          const int Tc = std::min(kRoundCap, tp - off);
          const std::vector<std::uint32_t> chunk(prompt.begin() + off, prompt.begin() + off + Tc);
          last = program.run_round(chunk, pos0, nullptr);
          pos0 += Tc;
          ++n_chunks;
        }
        device.synchronize();
        const auto pf1 = std::chrono::steady_clock::now();
        const double pf_ms = std::chrono::duration<float, std::milli>(pf1 - pf0).count();
        const double pf_tps = (pf_ms > 0.0) ? (1000.0 * tp) / pf_ms : 0.0;
        device.synchronize();
        const auto dc0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kTnew; ++i) {
          std::vector<std::uint32_t> one(1, last);
          last = program.run_round(one, pos0, nullptr);
          pos0 += 1;
        }
        device.synchronize();
        const auto dc1 = std::chrono::steady_clock::now();
        const double dc_ms = std::chrono::duration<float, std::milli>(dc1 - dc0).count();
        const double dc_tps = (dc_ms > 0.0) ? (1000.0 * kTnew) / dc_ms : 0.0;
        // Peak in-use during this T_prompt (sample post-prefill, the largest arena
        // point; the arena is pre-allocated so it is representative).
        size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) == cudaSuccess) peak_inuse = std::max(peak_inuse, t - f);
        Row r;
        r.tp = tp;
        r.chunks = n_chunks;
        r.pf_ms = pf_ms;
        r.pf_tps = pf_tps;
        r.dc_ms = dc_ms;
        r.dc_tps = dc_tps;
        r.peak_gib = static_cast<double>(peak_inuse) / gib;
        rows.push_back(r);
        std::printf("  %8d  %6d  %11.1f  %13.1f  %10.1f  %12.1f  %12.3f\n", tp, n_chunks, pf_ms,
                    pf_tps, dc_ms, dc_tps, r.peak_gib);
      }
      std::printf("  GPU in-use (model + program + arena, pre-allocated) = %.3f GiB of %.2f GiB\n",
                  static_cast<double>(peak_inuse) / gib,
                  static_cast<double>(std::size_t(device.total_vram())) / gib);
      std::printf("P13-M3 note: QSA indexer is round-local (chunked prefill + T = 1 decode); "
                  "the long-range path is the GDN recurrent state. See the M4 page footnote.\n");
      // Durability: the measured matrix (cwd-independent absolute path).
      {
        const char* out =
            "C:/Users/Micke/Documents/Kodprojekt/Ninfer/ninfer-win/out/flash_next_dev/p13_m3_matrix.txt";
        std::ofstream of(out, std::ios::trunc);
        if (!of) {
          std::fprintf(stderr, "P13-M3 WARN: could not write %s\n", out);
        } else {
          of << "Flash-Next P13 M3 -- benchmark matrix (real 75.4 GB artifact, C = 1, "
                "T_new = "
             << kTnew << ")\n";
          of << "kv_dtype = QSA bf16, GDN state fp32\n";
          of << "max_context = " << max_context
             << " positions (QSA paged KV: " << ((max_context + 63u) / 64u)
             << " pages x 64; GDN recurrent has no position limit)\n";
          of << "paging_mode = file-pageable experts (mmap) + PLE .ngram mmap + QSA paged KV "
             << "(identity block table, " << ((max_context + 63u) / 64u) << " pages/layer)\n";
          of << "qsa_indexer = round-local, 1024-token causal window (v1 limit; prefill chunked "
                "at 1024; documented engine limitation, footnoted on the M4 page)\n";
          of << "gpu_inuse_gib = " << static_cast<double>(peak_inuse) / gib << "\n";
          of << "rows (T_prompt chunks prefill_ms prefill_tok_s decode_ms decode_tok_s):\n";
          for (const Row& r : rows) {
            of << "  T_prompt=" << r.tp << "  chunks=" << r.chunks << "  prefill_ms=" << r.pf_ms
               << "  prefill_tok_s=" << r.pf_tps << "  decode_ms=" << r.dc_ms
               << "  decode_tok_s=" << r.dc_tps << "\n";
          }
        }
      }
      std::printf("P13-M3 PASS: benchmark matrix measured (see p13_m3_matrix.txt).\n");
    }

    // ---------------------------------------------------------------------
    // P13 BWIN (2026-09-15 lever B, task #21): MEASURE the prefix-cache (B)
    // prefill win on a realistic shared-prefix prompt. M3 measures the FRESH
    // full-prefill baseline; this mode measures the CONTINUATION (B) cost -- the
    // steady-state serve cost for a request that reuses an already-cached shared
    // prefix and prefills ONLY its delta (chunk(s) at pos0 = L, NO reset), so the
    // delta's round(s) alone are the repeated-request cost. Reports
    // win = T_fresh(full L+D) - T_delta(delta-only) and cross-checks the
    // continuation's final-position logits vs the fresh full prefill (ULP-level:
    // NOT bit-identical -- the settled P13-CONT test artifact -- so a small
    // nonzero max_abs_diff is EXPECTED and OK; the argmax MUST match).
    //
    //   Warm-up : begin_sequence; full L+D prefill (page-cache warm; discarded).
    //   Run A   : begin_sequence; chunked prefill of the full (L+D) prompt
    //             (pos0 0..L+D), timed -> T_fresh + final logits la.
    //   Run B   : begin_sequence; chunked prefill of the L-token prefix (pos0
    //             0..L, "cache fill", timed as T_prefix for reference); then
    //             chunked prefill of the D-token delta at pos0 = L (NO reset),
    //             timed -> T_delta + final logits lb.
    //
    // L = 4096 (shared system prefix), D = 1024 (delta) -> total 5120 tokens.
    // Chunked at 1024 (the QSA indexer v1 round-local limit; the production
    // schedule). The warm-up removes the cold-disk confound so A and B compare at
    // equal expert-residency warmth. Runs only under NINFER_P13_BWIN=1; 27B serve
    // down (VRAM). A small ULP diff + matching argmax = the end-to-end confirmation
    // that the B continuation is numerically viable for serving (task #12).
    // ---------------------------------------------------------------------
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
    const char* bwin_env = std::getenv("NINFER_P13_BWIN");
#pragma warning(pop)
    const bool bwin = (bwin_env != nullptr) && (bwin_env[0] == '1');
    if (bwin) {
      constexpr int kL = 4096;        // shared prefix (system prompt) tokens
      constexpr int kD = 1024;        // delta (user turn) tokens
      constexpr int kT = kL + kD;     // total context = 5120
      constexpr int kRoundCap = 1024; // QSA indexer v1 round-local limit
      auto gen = [&](int n, std::uint32_t seed) {
        std::vector<std::uint32_t> p(n);
        std::uint64_t s = seed * 0x9E3779B97F4A7C15ULL + 1;
        for (int i = 0; i < n; ++i) {
          s = s * 6364136223846793005ULL + 1442695040888963407ULL;  // splitmix64 mix
          p[i] = static_cast<std::uint32_t>((s >> 33) % kVocab);
        }
        return p;
      };
      const std::vector<std::uint32_t> prompt = gen(kT, 0xB711U);
      // Chunk the token range [off, off+len) of `prompt` into <= kRoundCap run_round
      // rounds at absolute pos0 `pos0_start` (NO reset between chunks); capture the
      // final-position logits into `out_logits` on the LAST chunk only.
      auto run_chunks = [&](int off, int len, int pos0_start, std::vector<float>* out_logits) {
        std::uint32_t last = 0;
        for (int o = 0; o < len; o += kRoundCap) {
          const int Tc = std::min(kRoundCap, len - o);
          std::vector<std::uint32_t> chunk(prompt.begin() + off + o,
                                           prompt.begin() + off + o + Tc);
          last = program.run_round(chunk, pos0_start + o,
                                   (o + Tc == len && out_logits != nullptr) ? out_logits
                                                                           : nullptr);
        }
        return last;
      };
      std::printf("\nP13-BWIN prefix-cache (B) prefill win (L=%d shared, D=%d delta, total %d)\n",
                  kL, kD, kT);
      std::printf("  %-16s  %-14s  %-14s  %-14s\n", "phase", "tokens", "wall_ms", "tok_s");
      // ---- Warm-up: full L+D prefill to populate the expert page cache (discard) --
      {
        program.begin_sequence();
        run_chunks(0, kT, 0, nullptr);
        device.synchronize();
        std::printf("  [warm-up]       full L+D=%d prefill (page cache warm, discarded)\n", kT);
      }
      // ---- Run A (baseline/fresh): full L+D prefill, timed. ---------------------
      std::vector<float> la;
      double t_fresh_ms = 0.0;
      {
        program.begin_sequence();
        device.synchronize();
        const auto a0 = std::chrono::steady_clock::now();
        run_chunks(0, kT, 0, &la);
        device.synchronize();
        const auto a1 = std::chrono::steady_clock::now();
        t_fresh_ms = std::chrono::duration<float, std::milli>(a1 - a0).count();
      }
      // ---- Run B (continuation/B): L prefix (timed, reference) + D delta (timed). -
      std::vector<float> lb;
      double t_prefix_ms = 0.0, t_delta_ms = 0.0;
      {
        program.begin_sequence();
        device.synchronize();
        const auto p0 = std::chrono::steady_clock::now();
        run_chunks(0, kL, 0, nullptr);  // one-time first-request "cache fill"
        device.synchronize();
        const auto p1 = std::chrono::steady_clock::now();
        t_prefix_ms = std::chrono::duration<float, std::milli>(p1 - p0).count();
        device.synchronize();
        const auto d0 = std::chrono::steady_clock::now();
        run_chunks(kL, kD, kL, &lb);    // steady-state: delta only, NO reset
        device.synchronize();
        const auto d1 = std::chrono::steady_clock::now();
        t_delta_ms = std::chrono::duration<float, std::milli>(d1 - d0).count();
      }
      const double win_ms = t_fresh_ms - t_delta_ms;
      const double win_frac = (t_fresh_ms > 0.0) ? (100.0 * win_ms / t_fresh_ms) : 0.0;
      std::printf("  [A fresh]       %-14d  %12.1f  %12.1f   (full L+D prefill; cold/first request)\n",
                  kT, t_fresh_ms, (t_fresh_ms > 0.0) ? (1000.0 * kT / t_fresh_ms) : 0.0);
      std::printf("  [B prefix fill] %-14d  %12.1f  %12.1f   (one-time; first request only)\n",
                  kL, t_prefix_ms, (t_prefix_ms > 0.0) ? (1000.0 * kL / t_prefix_ms) : 0.0);
      std::printf("  [B delta cont.] %-14d  %12.1f  %12.1f   (steady-state repeated request)\n",
                  kD, t_delta_ms, (t_delta_ms > 0.0) ? (1000.0 * kD / t_delta_ms) : 0.0);
      std::printf("  -> B win (repeated request) = %.1f ms (%.1f%% of the fresh %d-token prefill saved)\n",
                  win_ms, win_frac, kT);
      // ---- Correctness: final-position logits A vs B (ULP-level; argmax MUST match) --
      double max_ad = 0.0;
      int arg_a = 0, arg_b = 0;
      std::size_t nan = 0;
      for (std::size_t i = 0; i < la.size(); ++i) {
        const double d0 = la[i];
        const double d1 = (i < lb.size()) ? lb[i] : 0.0;
        if (!(d0 == d0 && d1 == d1)) {
          ++nan;
          continue;
        }
        double ad = d0 - d1;
        if (ad < 0.0) ad = -ad;
        if (ad > max_ad) max_ad = ad;
        if (la[i] > la[arg_a]) arg_a = static_cast<int>(i);
        if (i < lb.size() && lb[i] > lb[arg_b]) arg_b = static_cast<int>(i);
      }
      const bool argmax_match = (arg_a == arg_b);
      const char* verdict = argmax_match ? "PASS" : "FAIL";
      std::printf("  correctness: final_logits size=%zu  max_abs_diff=%.3e  argmax A=%d B=%d  nan=%zu  %s\n",
                  la.size(), max_ad, arg_a, arg_b, nan,
                  argmax_match ? "(argmax match; ULP-level continuation OK)"
                               : "(argmax MISMATCH)");
      {
        const char* out =
            "C:/Users/Micke/Documents/Kodprojekt/Ninfer/ninfer-win/out/flash_next_dev/p13_bwin.txt";
        std::ofstream of(out, std::ios::trunc);
        if (!of) {
          std::fprintf(stderr, "P13-BWIN WARN: could not write %s\n", out);
        } else {
          of << "Flash-Next P13 BWIN -- prefix-cache (B) prefill win (real 75.4 GB artifact)\n";
          of << "L = " << kL << "  D = " << kD << "  total = " << kT
             << "  chunk_cap = " << kRoundCap << "\n";
          of << "max_context = " << max_context << " positions\n";
          of << "fresh_ms = " << t_fresh_ms << "  fresh_tok_s = "
             << ((t_fresh_ms > 0.0) ? (1000.0 * kT / t_fresh_ms) : 0.0) << "\n";
          of << "prefix_fill_ms = " << t_prefix_ms << "  prefix_tok_s = "
             << ((t_prefix_ms > 0.0) ? (1000.0 * kL / t_prefix_ms) : 0.0) << "\n";
          of << "delta_ms = " << t_delta_ms << "  delta_tok_s = "
             << ((t_delta_ms > 0.0) ? (1000.0 * kD / t_delta_ms) : 0.0) << "\n";
          of << "win_ms = " << win_ms << "  win_frac_pct = " << win_frac << "\n";
          of << "final_logits_max_abs_diff = " << max_ad << "  argmax_A = " << arg_a
             << "  argmax_B = " << arg_b << "  nan = " << nan << "\n";
          of << "verdict = " << verdict << "\n";
        }
      }
      std::printf("P13-BWIN %s: B prefill win measured (see p13_bwin.txt).\n", verdict);
    }

    // ---------------------------------------------------------------------
    // P13 CONT (2026-09-15 lever B): on-device prefix-continuation self-consistency.
    // Runs only under NINFER_P13_CONT=1. Verifies the load-bearing property of the
    // program.cpp continuation (skip begin_sequence on a shared prefix + delta
    // prefill at pos0=L): splitting the prefill at L with NO reset is BIT-IDENTICAL
    // to a fresh full prefill. Both paths drive the SAME RealProgram directly.
    //
    //   Path A (fresh): begin_sequence; run_round(Q[0..P], pos0=0)
    //   Path B (split): begin_sequence; run_round(Q[0..L], pos0=0);
    //                                   run_round(Q[L..P], pos0=L)   // NO reset
    //
    // run_round's sampling is deterministic (argmax, real_program.cpp:1976) and it
    // fills final_logits at the final position (real_program.cpp:1999-2011), so the
    // test compares (a) the two prefills' final logits (max abs diff + argmax) and
    // (b) a short greedy decode after each (must be identical). P and L are
    // single-round (<= 2048 = RealProgram::kMaxRoundTokens), so the ONLY variable
    // is the split at L -- no chunk-size confound. A nonzero max_abs_diff or a
    // decode divergence localizes a real continuation bug (e.g. a round-2 pool
    // write clobbering round-1's absolute-position K/V or indexer-K entries).
    // ---------------------------------------------------------------------
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
    const char* cont_env = std::getenv("NINFER_P13_CONT");
#pragma warning(pop)
    const bool cont = (cont_env != nullptr) && (cont_env[0] == '1');
    if (cont) {
      constexpr int kP = 512;   // full prompt length (single run_round round)
      constexpr int kL = 256;   // split point (the continuation frontier)
      constexpr int kN = 8;     // greedy decode tokens after the prefill
      auto gen = [](int n, std::uint32_t seed) {
        std::vector<std::uint32_t> p(n);
        std::uint64_t s = seed * 0x9E3779B97F4A7C15ULL + 1;
        for (int i = 0; i < n; ++i) {
          s = s * 6364136223846793005ULL + 1442695040888963407ULL;  // splitmix64 mix
          p[i] = static_cast<std::uint32_t>((s >> 33) % kVocab);
        }
        return p;
      };
      const std::vector<std::uint32_t> q = gen(kP, 0x51E7U);
      // split == 0 -> fresh full prefill (Path A); split == kL -> split prefill (Path B).
      auto run_prefill_and_decode = [&](std::vector<float>& out_logits,
                                        std::vector<std::uint32_t>& out_tokens, int split) {
        program.begin_sequence();
        std::vector<std::uint32_t> last_tok(1, 0);
        if (split == 0) {
          last_tok[0] = program.run_round(q, 0, &out_logits);
        } else {
          const std::vector<std::uint32_t> first(q.begin(), q.begin() + split);
          last_tok[0] = program.run_round(first, 0, nullptr);  // token at pos L-1 (discarded)
          const std::vector<std::uint32_t> rest(q.begin() + split, q.end());
          last_tok[0] = program.run_round(rest, split, &out_logits);
        }
        int pos0 = kP;
        for (int i = 0; i < kN; ++i) {
          last_tok[0] = program.run_round(last_tok, pos0, nullptr);
          pos0 += 1;
          out_tokens.push_back(last_tok[0]);
        }
      };
      std::vector<float> la, lb;
      std::vector<std::uint32_t> ta, tb;
      // Per-layer value-dump probe (real_program.cpp NINFER_P13_LAYPROBE): point it
      // at a distinct file per path so the driver can compare the global-position
      // (P-1) attention/MoE outputs of the two paths and localize the first
      // diverging layer. Set right before each run (the probe reads the env per
      // call and reopens "w" when the path changes).
      const char* lp_base =
          "C:/Users/Micke/Documents/Kodprojekt/Ninfer/ninfer-win/out/flash_next_dev/layprobe_";
      // Extra-local-index 255 on the fresh run only: it makes the fresh T=512 round
      // also dump local 255 (global 255) so the split boundary (round-1 T-1 = global
      // 255) can be compared in the same run. The split path needs no extra dump --
      // its round 1 (T=256) T-1 is global 255 and round 2 (T=256, pos0=256) T-1 is
      // global 511, so the T-1 default already covers both.
      // Single-pass mode (NINFER_P13_CONT_PATH=fresh|split): run ONLY that path in
      // this process -- one begin_sequence + its prefill + kN decodes -- then persist
      // the probe rows (fresh.txt/split.txt), the final logits, and the decode tokens
      // to disk. fresh and split are compared across TWO separate process invocations.
      // Rationale: the probe-instrumented build crashes (0xC0000409) at the SECOND
      // begin_sequence (the fresh->split transition); running each path in its own
      // process makes each path a single begin_sequence and avoids that transition.
      // Unset -> the original both-in-one-process path + in-process compare.
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
      const char* cpath_env = std::getenv("NINFER_P13_CONT_PATH");
#pragma warning(pop)
      const std::string cpath = (cpath_env != nullptr) ? cpath_env : "";
      const bool want_fresh = (cpath == "fresh");
      const bool want_split = (cpath == "split");
      if (!cpath.empty() && !want_fresh && !want_split) {
        std::fprintf(stderr,
                     "P13-CONT: bad NINFER_P13_CONT_PATH '%s' (want fresh|split)\n",
                     cpath_env);
        std::fflush(stderr);
      }
      // Single-pass persistence: dump the path's final logits + decode tokens to disk
      // so fresh and split (run in separate processes) can be compared host-side.
      auto dump_p13_single =
          [&](const char* name, const std::vector<float>& logits,
              const std::vector<std::uint32_t>& tokens) {
            const std::string base =
                "C:/Users/Micke/Documents/Kodprojekt/Ninfer/ninfer-win/out/flash_next_dev/";
            {
              std::ofstream of(base + std::string(name) + "_logits.txt", std::ios::trunc);
              if (of) {
                for (std::size_t i = 0; i < logits.size(); ++i) of << logits[i] << " ";
                of << "\n";
              }
            }
            {
              std::ofstream of(base + std::string(name) + "_tokens.txt", std::ios::trunc);
              if (of) {
                for (std::uint32_t t : tokens) of << t << " ";
                of << "\n";
              }
            }
          };
#pragma warning(push)
#pragma warning(disable : 4996)  // _putenv_s
      if (want_fresh) {
        _putenv_s("NINFER_P13_HB",
                  "C:/Users/Micke/Documents/Kodprojekt/Ninfer/ninfer-win/out/flash_next_dev/"
                  "layprobe_hb_fresh.txt");
        _putenv_s("NINFER_P13_LAYPROBE", (std::string(lp_base) + "fresh.txt").c_str());
        _putenv_s("NINFER_P13_LAYPROBE_EXTRA_LOCAL", "255");
        run_prefill_and_decode(la, ta, 0);   // Path A: fresh (single begin_sequence)
      } else if (want_split) {
        _putenv_s("NINFER_P13_HB",
                  "C:/Users/Micke/Documents/Kodprojekt/Ninfer/ninfer-win/out/flash_next_dev/"
                  "layprobe_hb_split.txt");
        _putenv_s("NINFER_P13_LAYPROBE", (std::string(lp_base) + "split.txt").c_str());
        run_prefill_and_decode(lb, tb, kL);  // Path B: split (single begin_sequence)
      } else {
        _putenv_s("NINFER_P13_HB",
                  "C:/Users/Micke/Documents/Kodprojekt/Ninfer/ninfer-win/out/flash_next_dev/"
                  "layprobe_hb.txt");
        _putenv_s("NINFER_P13_LAYPROBE", (std::string(lp_base) + "fresh.txt").c_str());
        _putenv_s("NINFER_P13_LAYPROBE_EXTRA_LOCAL", "255");
        run_prefill_and_decode(la, ta, 0);   // Path A: fresh
        _putenv_s("NINFER_P13_LAYPROBE_EXTRA_LOCAL", nullptr);
        _putenv_s("NINFER_P13_LAYPROBE", (std::string(lp_base) + "split.txt").c_str());
        run_prefill_and_decode(lb, tb, kL);  // Path B: split at kL, no reset
      }
#pragma warning(pop)
      if (want_fresh || want_split) {
        const char* nm = want_fresh ? "fresh" : "split";
        const std::vector<float>& lg = want_fresh ? la : lb;
        const std::vector<std::uint32_t>& tk = want_fresh ? ta : tb;
        dump_p13_single(nm, lg, tk);
        std::fprintf(stderr, "P13-CONT single-pass %s done (logits=%zu tokens=%zu)\n", nm,
                     lg.size(), tk.size());
        std::fflush(stderr);
        return 0;
      }
      device.synchronize();
      // (a) final-logits max abs diff + argmax (over the full 248320-vocab vector).
      double max_ad = 0.0;
      int arg_a = 0, arg_b = 0, bad = 0;
      for (std::size_t i = 0; i < la.size(); ++i) {
        const double d0 = static_cast<double>(la[i]);
        const double d1 = static_cast<double>(lb[i]);
        if (!(d0 == d0 && d1 == d1)) {
          ++bad;
          continue;
        }
        double ad = d0 - d1;
        if (ad < 0.0) ad = -ad;
        if (ad > max_ad) max_ad = ad;
        if (la[i] > la[arg_a]) arg_a = static_cast<int>(i);
        if (lb[i] > lb[arg_b]) arg_b = static_cast<int>(i);
      }
      // (b) decode sequence exact match + first divergence index.
      int div = (ta.size() != tb.size()) ? 0 : -1;
      for (std::size_t i = 0; div < 0 && i < ta.size(); ++i) {
        if (ta[i] != tb[i]) div = static_cast<int>(i);
      }
      const bool identical = (max_ad == 0.0) && (arg_a == arg_b) && (div < 0) && (bad == 0);
      const char* verdict = identical ? "PASS" : "FAIL";
      std::printf("\nP13-CONT prefix-continuation self-consistency (fresh P=%d vs split L=%d)\n",
                  kP, kL);
      std::printf("  final_logits size=%zu  max_abs_diff=%.3e  argmax fresh=%d split=%d  nan=%d\n",
                  la.size(), max_ad, arg_a, arg_b, bad);
      std::printf("  decode N=%d  identical=%s  first_divergence=%d\n", kN, (div < 0) ? "yes" : "no",
                  div);
      {
        const char* out =
            "C:/Users/Micke/Documents/Kodprojekt/Ninfer/ninfer-win/out/flash_next_dev/"
            "p13_cont_selfconsistency.txt";
        std::ofstream of(out, std::ios::trunc);
        if (!of) {
          std::fprintf(stderr, "P13-CONT WARN: could not write %s\n", out);
        } else {
          of << "Flash-Next P13 CONT -- on-device prefix-continuation self-consistency "
               "(real 75.4 GB artifact)\n";
          of << "property: split prefill at L with NO reset == fresh full prefill "
               "(bit-identical)\n";
          of << "P = " << kP << "  L = " << kL << "  N_decode = " << kN
             << "  seed = 0x51E7\n";
          of << "max_context = " << max_context << " positions\n";
          of << "final_logits_max_abs_diff = " << max_ad << "\n";
          of << "argmax_fresh = " << arg_a << "  argmax_split = " << arg_b << "\n";
          of << "decode_identical = " << ((div < 0) ? "yes" : "no")
             << "  first_divergence = " << div << "\n";
          of << "nan_logits = " << bad << "\n";
          of << "verdict = " << verdict << "\n";
        }
      }
      std::printf("P13-CONT %s: %s (see p13_cont_selfconsistency.txt).\n", verdict,
                  identical ? "continuation is bit-identical to fresh prefill"
                            : "MISMATCH -- continuation diverges from fresh prefill");
    }

    // ---------------------------------------------------------------------
    // P13 100k prefill probe (2026-09-12 dynamic-context workstream). Runs only
    // under NINFER_P13_MAXCTX=1 and requires NINFER_P13_MAX_CTX >= 100000 (else the
    // QSA pool is smaller than the prompt and the absolute-position K/V gather
    // reads past the identity block table). Verifies a 100k-token prefill runs
    // END-TO-END through the real model: begin_sequence() (fresh recurrent + PLE
    // state), a chunked state-carrying prefill of 100000 tokens at the 1024-token
    // round cap (98 chunks), then 16 single-token greedy decode rounds at absolute
    // positions 100000..100015. Measures prefill wall / tok/s + the GPU in-use and
    // re-checks the MoE-health logit scale at the long context. MEASURED on this
    // machine (no invented tok/s); run ONLY with the 27B serve down (VRAM free).
    // ---------------------------------------------------------------------
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
    const char* maxctx_env = std::getenv("NINFER_P13_MAXCTX");
#pragma warning(pop)
    const bool maxctx = (maxctx_env != nullptr) && (maxctx_env[0] == '1');
    if (maxctx) {
      constexpr int kPrompt = 100000;
      constexpr int kDecLen = 16;
      constexpr int kRoundCap = 1024;  // QSA indexer v1 max (round-local)
      if (max_context < static_cast<std::uint32_t>(kPrompt)) {
        std::fprintf(stderr,
                     "P13 100k probe FAIL: max_context %u < %d -- set NINFER_P13_MAX_CTX=%d\n",
                     max_context, kPrompt, kPrompt);
        return 1;
      }
      std::printf("\nP13 100k prefill probe (NINFER_P13_MAXCTX=1, max_context = %u, %u pages)\n",
                  max_context, (max_context + 63u) / 64u);
      program.begin_sequence();
      std::vector<std::uint32_t> prompt(kPrompt);
      std::uint64_t s = static_cast<std::uint64_t>(kPrompt) * 0x9E3779B97F4A7C15ULL + 1;
      for (int i = 0; i < kPrompt; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;  // splitmix64 mix
        prompt[i] = static_cast<std::uint32_t>((s >> 33) % kVocab);
      }
      device.synchronize();
      const auto pf0 = std::chrono::steady_clock::now();
      int pos0 = 0;
      int n_chunks = 0;
      std::uint32_t last = 0;
      for (int off = 0; off < kPrompt; off += kRoundCap) {
        const int Tc = std::min(kRoundCap, kPrompt - off);
        const std::vector<std::uint32_t> chunk(prompt.begin() + off, prompt.begin() + off + Tc);
        last = program.run_round(chunk, pos0, nullptr);
        pos0 += Tc;
        ++n_chunks;
      }
      device.synchronize();
      const auto pf1 = std::chrono::steady_clock::now();
      const double pf_ms = std::chrono::duration<float, std::milli>(pf1 - pf0).count();
      const double pf_tps = (pf_ms > 0.0) ? (1000.0 * kPrompt) / pf_ms : 0.0;
      std::size_t pfpeak = 0;
      {
        std::size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) == cudaSuccess) pfpeak = t - f;
      }
      std::printf("  prefill %d tokens in %d chunks of <= %d: wall = %.1f ms (%.1f tok/s), "
                  "peak GPU = %.3f GiB\n",
                  kPrompt, n_chunks, kRoundCap, pf_ms, pf_tps, static_cast<double>(pfpeak) / gib);
      // 16 single-token greedy decode rounds at absolute positions 100000..100015,
      // measuring the long-context decode rate + re-checking the logit scale.
      const auto dc0 = std::chrono::steady_clock::now();
      std::vector<std::uint32_t> gen;
      gen.reserve(kDecLen);
      double dc_max_abs = 0.0;
      long dc_nan_inf = 0;
      bool dc_oob = false;
      for (int i = 0; i < kDecLen; ++i) {
        std::vector<float> l;
        std::vector<std::uint32_t> one(1, last);
        last = program.run_round(one, pos0, &l);
        pos0 += 1;
        if (last >= kVocab) {
          std::fprintf(stderr, "P13 100k probe FAIL: decode %d token %u out of vocab\n", i, last);
          dc_oob = true;
          break;
        }
        gen.push_back(last);
        for (std::size_t n = 0; n < l.size(); ++n) {
          const double v = static_cast<double>(l[n]);
          if (std::isnan(v) || std::isinf(v)) {
            ++dc_nan_inf;
            continue;
          }
          const double a = std::fabs(v);
          if (a > dc_max_abs) dc_max_abs = a;
        }
      }
      device.synchronize();
      const auto dc1 = std::chrono::steady_clock::now();
      const double dc_ms = std::chrono::duration<float, std::milli>(dc1 - dc0).count();
      const double dc_tps = (dc_ms > 0.0) ? (1000.0 * kDecLen) / dc_ms : 0.0;
      std::printf("  decode %d tokens: wall = %.1f ms (%.2f tok/s), max|logit| = %.6g, "
                  "nan/inf = %ld, stream =",
                  static_cast<int>(gen.size()), dc_ms, dc_tps, dc_max_abs, dc_nan_inf);
      for (std::size_t i = 0; i < gen.size(); ++i) std::printf(" %u", gen[i]);
      std::printf("\n");
      const bool ok = !dc_oob && (gen.size() == static_cast<std::size_t>(kDecLen)) &&
                      (dc_nan_inf == 0) && (dc_max_abs < 1e6) && (dc_max_abs >= 1e-3);
      {
        const char* out =
            "C:/Users/Micke/Documents/Kodprojekt/Ninfer/ninfer-win/out/flash_next_dev/p13_maxctx100k.txt";
        std::ofstream of(out, std::ios::trunc);
        if (!of) {
          std::fprintf(stderr, "P13 100k probe WARN: could not write %s\n", out);
        } else {
          of << "Flash-Next P13 100k prefill probe (real 75.4 GB artifact, "
                "NINFER_P13_MAXCTX=1)\n";
          of << "max_context     = " << max_context << " positions ("
             << ((max_context + 63u) / 64u) << " QSA pages x 64)\n";
          of << "prompt_tokens   = " << kPrompt << "\n";
          of << "chunks          = " << n_chunks << "\n";
          of << "round_cap       = " << kRoundCap << "\n";
          of << "prefill_ms      = " << pf_ms << "\n";
          of << "prefill_tok_s   = " << pf_tps << "\n";
          of << "peak_gpu_gib    = " << static_cast<double>(pfpeak) / gib << "\n";
          of << "decode_tokens   = " << gen.size() << "\n";
          of << "decode_ms       = " << dc_ms << "\n";
          of << "decode_tok_s    = " << dc_tps << "\n";
          of << "max_abs_logit   = " << dc_max_abs << "\n";
          of << "nan_inf_count   = " << dc_nan_inf << "\n";
          of << "decode_stream   =";
          for (std::size_t i = 0; i < gen.size(); ++i) of << " " << gen[i];
          of << "\n";
          of << "verdict         = " << (ok ? "PASS" : "FAIL") << "\n";
        }
      }
      std::printf("P13 100k probe %s: %d-token prefill + %d-token decode through the real model "
                  "(see p13_maxctx100k.txt).\n",
                  ok ? "PASS" : "FAIL", kPrompt, static_cast<int>(gen.size()));
      if (!ok) return 1;
    }

    if (!sane || !s5_ok) return 1;
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "S1 gate (b) FAIL (threw): %s\n", e.what());
    return 1;
  }
}
