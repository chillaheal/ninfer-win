// P3.2 — CPU MoE forward (W4A16 NVFP4) for cold MoE layers offloaded from
// the GPU (Unsloth -ncmoe style). Host-only C++; the math contract is in
// cpu_moe.h. The dequant cache holds F32 weight planes (one F32 round per
// code*scale, the exact value the device GEMM dequantizes per element), so
// the hot loop is a pure F32 dot over the same k order as the device's
// contracted FMA chain.
//
// /arch:AVX2 (per-source, set in CMakeLists.txt) gives the F32 dots
// 256-bit FMAs; without it the same 4-lane scalar fallback runs.

#include "targets/qwen3_8_flash_next/impl/runtime/cpu_moe.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <list>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace ninfer::targets::qwen3_8_flash_next {
namespace {

constexpr int kMaxTokens = 2048;  // op domain: T in [1, 2048]

inline float bf16_to_f32(std::uint16_t u) {
  const std::uint32_t bits = static_cast<std::uint32_t>(u) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// IEEE RNE to BF16 (== __float2bfloat16_rn; inputs are finite).
inline std::uint16_t f32_to_bf16_rn(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const std::uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<std::uint16_t>(u >> 16);
}

// OCP E2M1 (NVFP4): bit3 sign, magnitudes {0,.5,1,1.5,2,3,4,6}.
constexpr float kE2m1[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
inline float e2m1_decode(std::uint8_t nib) {
  const float m = kE2m1[nib & 7u];
  return (nib & 8u) ? -m : m;
}

// OCP E4M3FN: bit7 sign, bits6-3 exp (bias 7), bits2-0 mant. exp==0 ->
// subnormal (m/8)*2^-6 = m*2^-9; else (1 + m/8)*2^(e-7). Every value is
// exactly representable in F32 (no rounding), so this is bit-identical to
// the device's hardware e4m3->f32 conversion.
inline float e4m3_decode(std::uint8_t b) {
  const float sign = (b & 0x80u) ? -1.0F : 1.0F;
  const int e = (b >> 3) & 0xF;
  const int m = b & 0x7;
  const float v = (e == 0) ? (static_cast<float>(m) * 0.001953125F)
                           : ((1.0F + 0.125F * static_cast<float>(m)) * std::ldexp(1.0F, e - 7));
  return sign * v;
}

// 256-entry E4M3FN F32 decode table. Plain array + call_once: the previous
// make_unique + magic-static-pointer form materialized as an EMPTY vector at
// runtime under MSVC /O2 /arch:AVX2 (segfault isolation, p10, 2026-09-25).
const float* e4m3_lut() {
  static float table[256];
  static std::once_flag flag;
  std::call_once(flag, [] {
    for (int i = 0; i < 256; ++i) { table[i] = e4m3_decode(static_cast<std::uint8_t>(i)); }
  });
  return table;
}

// Exact replica of kernel 2: softmax-max shift, K rounds of argmax with
// (value DESC, id ASC) tie-break, insertion sort by expert id ASC, ids out
// ascending. top-k is deterministic given the logits, so the CPU result is
// bit-identical to the device kernel over the same F32 logits.
void topk_row(const float* row, int E, int K, std::int32_t* ids, float* alpha) {
  float m = -INFINITY;
  for (int e = 0; e < E; ++e) {
    m = std::fmax(m, row[e]);
  }

  std::int32_t sel_e[32];
  float sel_v[32];
  for (int s = 0; s < K; ++s) {
    float best_v = -INFINITY;
    int best_e   = -1;
    for (int e = 0; e < E; ++e) {
      bool taken = false;
      for (int q = 0; q < s; ++q) {
        if (sel_e[q] == e) { taken = true; break; }
      }
      if (taken) { continue; }
      const float v = row[e];
      if (v > best_v || (v == best_v && (best_e < 0 || e < best_e))) {
        best_v = v;
        best_e = e;
      }
    }
    sel_e[s] = best_e;
    sel_v[s] = best_v;
  }

  for (int a = 1; a < K; ++a) {
    const int ee = sel_e[a];
    const float vv = sel_v[a];
    int b = a;
    while (b > 0 && sel_e[b - 1] > ee) {
      sel_e[b] = sel_e[b - 1];
      sel_v[b] = sel_v[b - 1];
      --b;
    }
    sel_e[b] = ee;
    sel_v[b] = vv;
  }

  float ssum = 0.0F;
  float exps[32];
  for (int j = 0; j < K; ++j) {
    exps[j] = std::expf(sel_v[j] - m);
    ssum += exps[j];
  }
  for (int j = 0; j < K; ++j) {
    ids[j] = sel_e[j];
    alpha[j] = (ssum != 0.0F) ? exps[j] / ssum : 0.0F;
  }
}

// Dequantize one plane to F32: out[n*K + k] = e2m1(nib) * e4m3(scale), the
// swizzle exactly as moe_dequant_weight:
//   slab = ((n/128)*(K/64) + (k/64))*512
//   off  = ((n&127)&31)*16 + ((n&127)>>5)*4 + ((k&63)>>4)
// The scale index is constant over a 16-wide k group (sg) and 64-wide
// (kg), so each scale is decoded once per (n, kg, sg) via the F32 LUT.
void dequant_plane(const std::uint8_t* codes, const std::uint8_t* scales,
                   int N, int K, const float* e4m3, float* out) {
  const int k_groups = K / 64;
  for (int n = 0; n < N; ++n) {
    float* orow = out + static_cast<std::size_t>(n) * K;
    const std::uint8_t* crow = codes + static_cast<std::size_t>(n) * (K / 2);
    const int slab_base =
        ((n / 128) * k_groups) * 512 + ((n & 127) & 31) * 16 + ((n & 127) >> 5) * 4;
    for (int kg = 0; kg < k_groups; ++kg) {
      const int slab = slab_base + kg * 512;
      for (int sg = 0; sg < 4; ++sg) {
        const float scale = e4m3[scales[slab + sg]];
        const int k0 = kg * 64 + sg * 16;
        for (int k = k0; k < k0 + 16; ++k) {
          const std::uint8_t cb = crow[k >> 1];
          const float code =
              e2m1_decode((k & 1) ? static_cast<std::uint8_t>(cb >> 4)
                                  : static_cast<std::uint8_t>(cb & 0xFu));
          orow[k] = code * scale;
        }
      }
    }
  }
}

// F32 dot, FMA accumulation in the device's k order. The 4-lane reduction is
// a fixed tree (rel error ~1e-6 over K <= 4096 terms) — far inside the p8
// tolerance the CPU path is checked against.
#ifdef __AVX2__
inline float dot_f32(const float* a, const float* b, int K) {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  __m256 s2 = _mm256_setzero_ps();
  __m256 s3 = _mm256_setzero_ps();
  int k = 0;
  for (; k + 32 <= K; k += 32) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8), s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 16), _mm256_loadu_ps(b + k + 16), s2);
    s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 24), _mm256_loadu_ps(b + k + 24), s3);
  }
  float tail = 0.0F;
  for (; k < K; ++k) {
    tail = std::fmaf(a[k], b[k], tail);
  }
  float r[32];
  _mm256_storeu_ps(r, s0);
  _mm256_storeu_ps(r + 8, s1);
  _mm256_storeu_ps(r + 16, s2);
  _mm256_storeu_ps(r + 24, s3);
  for (int i = 0; i < 8; ++i) {
    r[i] = (r[i] + r[8 + i]) + (r[16 + i] + r[24 + i]);
  }
  float acc = r[0] + r[1];
  acc = (acc + r[2]) + r[3];
  acc = (acc + r[4]) + r[5];
  acc = acc + r[6] + r[7];
  acc += tail;
  return acc;
}
#else
inline float dot_f32(const float* a, const float* b, int K) {
  float a0 = 0.0F, a1 = 0.0F, a2 = 0.0F, a3 = 0.0F;
  int k = 0;
  for (; k + 4 <= K; k += 4) {
    a0 = std::fmaf(a[k + 0], b[k + 0], a0);
    a1 = std::fmaf(a[k + 1], b[k + 1], a1);
    a2 = std::fmaf(a[k + 2], b[k + 2], a2);
    a3 = std::fmaf(a[k + 3], b[k + 3], a3);
  }
  for (; k < K; ++k) {
    a0 = std::fmaf(a[k], b[k], a0);
  }
  return (a0 + a1) + (a2 + a3);
}
#endif

// Chunk-steal parallel for over [0, n).
void parallel_for(int n, int threads, const std::function<void(int, int)>& fn) {
  if (n <= 0) { return; }
  const int chunk = std::max(1, n / (std::max(1, threads) * 4));
  std::atomic<int> next{0};
  auto worker = [&] {
    for (int c0 = next.fetch_add(chunk); c0 < n; c0 = next.fetch_add(chunk)) {
      fn(c0, std::min(n, c0 + chunk));
    }
  };
  if (threads <= 1) {
    worker();
    return;
  }
  std::vector<std::thread> th;
  th.reserve(threads);
  for (int i = 0; i < threads; ++i) {
    th.emplace_back(worker);
  }
  for (auto& t : th) {
    t.join();
  }
}

}  // namespace

struct CpuMoe::Impl {
  int E = 0, K = 0, K_in = 0, I = 0;
  int C_max = 0;  // kMaxTokens * (K + 1)

  // Per-call scratch (sized once at construction for T = 2048).
  std::vector<float> x_f32;             // [kMaxTokens][K_in]
  std::vector<std::uint16_t> h_bf16;    // [C_max][2I]  gu out (BF16, as stored)
  std::vector<float> h2_f32;            // [C_max][I]   silu_mul out (exact bf16 round-trip)
  std::vector<float> y_f32;             // [C_max][K_in] dn out (FP32, as stored)
  std::vector<std::int32_t> ids;        // [kMaxTokens][K] ascending
  std::vector<float> alpha;             // [kMaxTokens][K]
  std::vector<float> sg;                // [kMaxTokens] shared-gate sigmoid

  // Dequant cache. One slot = both planes of one expert in F32.
  struct Slot {
    int id = -1;
    const std::uint8_t* gu_src = nullptr;  // gu codes base at dequant time
    const std::uint8_t* dn_src = nullptr;  // dn codes base at dequant time
    float* gu = nullptr;  // [2I][K_in]
    float* dn = nullptr;  // [K_in][I]
  };
  std::vector<Slot> slots;                 // routed LRU slots
  std::unordered_map<int, int> slot_of;    // expert id -> slot index
  std::list<int> lru;                      // slot indices, front = MRU
  std::vector<float> pool;                 // gu F32 backing for all slots
  std::vector<float> dpool;                // dn F32 backing for all slots
  std::mutex mu;

  // Dedicated non-evictable shared-expert slot.
  Slot shared{};
  std::vector<float> shared_gu_store, shared_dn_store;

  std::size_t gu_elems() const { return static_cast<std::size_t>(2 * I) * K_in; }
  std::size_t dn_elems() const { return static_cast<std::size_t>(K_in) * I; }

  void dequant_slot(Slot& s, const CpuMoePlane& gu_c, const CpuMoePlane& gu_s,
                    const CpuMoePlane& dn_c, const CpuMoePlane& dn_s, int e) {
    const float* e4m3 = e4m3_lut();
    const std::size_t off = static_cast<std::size_t>(e);
    dequant_plane(gu_c.base + off * gu_c.expert_stride,
                  gu_s.base + off * gu_s.expert_stride, 2 * I, K_in, e4m3, s.gu);
    dequant_plane(dn_c.base + off * dn_c.expert_stride,
                  dn_s.base + off * dn_s.expert_stride, K_in, I, e4m3, s.dn);
    s.gu_src = gu_c.base + off * gu_c.expert_stride;
    s.dn_src = dn_c.base + off * dn_c.expert_stride;
  }

  // Acquire the dequant F32 planes for routed expert e (LRU). Safe to call
  // from multiple threads.
  Slot& acquire(int e, const CpuMoeWeights& w) {
    std::lock_guard<std::mutex> lk(mu);
    auto it = slot_of.find(e);
    Slot* s;
    if (it != slot_of.end()) {
      s = &slots[it->second];
      const auto li = find_iter(it->second);
      if (li != lru.end()) {
        lru.splice(lru.begin(), lru, li);
      }
    } else {
      int si;
      if (slot_of.size() < slots.size()) {
        for (si = 0; si < static_cast<int>(slots.size()); ++si) {
          if (slots[si].id == -1) { break; }
        }
      } else {
        const int victim = static_cast<int>(lru.back());
        lru.pop_back();
        slot_of.erase(slots[victim].id);
        slots[victim].id = -1;
        si = victim;
      }
      s = &slots[si];
      s->id = e;
      slot_of[e] = si;
      lru.push_front(si);
      dequant_slot(*s, w.routed_gu_codes, w.routed_gu_scales, w.routed_dn_codes,
                   w.routed_dn_scales, e);
      return *s;
    }
    // Pointer-changed invalidation (weights re-bound to different memory).
    const std::uint8_t* want_gu = w.routed_gu_codes.base + static_cast<std::size_t>(e) * w.routed_gu_codes.expert_stride;
    const std::uint8_t* want_dn = w.routed_dn_codes.base + static_cast<std::size_t>(e) * w.routed_dn_codes.expert_stride;
    if (s->gu_src != want_gu || s->dn_src != want_dn) {
      dequant_slot(*s, w.routed_gu_codes, w.routed_gu_scales, w.routed_dn_codes,
                   w.routed_dn_scales, e);
    }
    return *s;
  }

  std::list<int>::iterator find_iter(int slot) {
    for (auto it = lru.begin(); it != lru.end(); ++it) {
      if (*it == slot) { return it; }
    }
    return lru.end();
  }

  // Shared expert: dedicated slot, invalidated by pointer change.
  const Slot& acquire_shared(const CpuMoeWeights& w) {
    const std::uint8_t* want_gu = w.shared_gu_codes;
    const std::uint8_t* want_dn = w.shared_dn_codes;
    if (shared.id != -2 || shared.gu_src != want_gu || shared.dn_src != want_dn) {
      shared.id = -2;
      const float* e4m3 = e4m3_lut();
      dequant_plane(w.shared_gu_codes, w.shared_gu_scales, 2 * I, K_in, e4m3,
                    shared.gu);
      dequant_plane(w.shared_dn_codes, w.shared_dn_scales, K_in, I, e4m3,
                    shared.dn);
      shared.gu_src = want_gu;
      shared.dn_src = want_dn;
    }
    return shared;
  }
};

CpuMoe::CpuMoe(int experts, int topk, int input_dim, int intermediate,
               int expert_cache)
    : impl_(std::make_unique<Impl>()) {
  if (!(input_dim % 64 == 0 && 128 <= input_dim && input_dim <= 4096) ||
      !(intermediate % 64 == 0 && 64 <= intermediate && intermediate <= 4096) ||
      !(1 <= topk && topk <= 32) || experts < 1 || expert_cache < 1) {
    throw std::invalid_argument("CpuMoe: geometry outside the op domain "
                                "(K_in %64==0 in [128,4096], I %64==0 in [64,4096], "
                                "1<=K<=32, E>=1, cache>=1)");
  }
  Impl& im = *impl_;
  im.E = experts;
  im.K = topk;
  im.K_in = input_dim;
  im.I = intermediate;
  im.C_max = kMaxTokens * (topk + 1);

  im.x_f32.resize(static_cast<std::size_t>(kMaxTokens) * input_dim);
  im.h_bf16.resize(static_cast<std::size_t>(im.C_max) * (2 * intermediate));
  im.h2_f32.resize(static_cast<std::size_t>(im.C_max) * intermediate);
  im.y_f32.resize(static_cast<std::size_t>(im.C_max) * input_dim);
  im.ids.resize(static_cast<std::size_t>(kMaxTokens) * topk);
  im.alpha.resize(static_cast<std::size_t>(kMaxTokens) * topk);
  im.sg.resize(kMaxTokens);

  im.shared_gu_store.resize(im.gu_elems());
  im.shared_dn_store.resize(im.dn_elems());
  im.shared.gu = im.shared_gu_store.data();
  im.shared.dn = im.shared_dn_store.data();

  im.slots.resize(expert_cache);
  for (auto& s : im.slots) {
    s.gu = nullptr;
    s.dn = nullptr;
  }
  std::vector<float> pool(im.gu_elems() * static_cast<std::size_t>(expert_cache), 0.0F);
  std::vector<float> dpool(im.dn_elems() * static_cast<std::size_t>(expert_cache), 0.0F);
  for (int i = 0; i < expert_cache; ++i) {
    im.slots[i].gu = pool.data() + im.gu_elems() * static_cast<std::size_t>(i);
    im.slots[i].dn = dpool.data() + im.dn_elems() * static_cast<std::size_t>(i);
    im.slots[i].id = -1;
  }
  im.pool = std::move(pool);
  im.dpool = std::move(dpool);
}

CpuMoe::~CpuMoe() = default;

void CpuMoe::run(const float* logits, int tokens, const std::uint16_t* x,
                 const std::uint16_t* shared_gate, const CpuMoeWeights& w,
                 std::uint16_t* y, std::int32_t* ids_out, float* alpha_out,
                 int threads) {
  Impl& im = *impl_;
  const int E = im.E, K = im.K, K_in = im.K_in, I = im.I;
  const int T = tokens;
  const int C = T * (K + 1);
  if (T < 1 || T > kMaxTokens) {
    throw std::invalid_argument("CpuMoe::run: T outside [1, 2048]");
  }
  if (threads <= 0) {
    const unsigned hw = std::thread::hardware_concurrency();
    threads = static_cast<int>(std::max(1u, hw / 2));
    threads = std::max(1, std::min(threads, 16));
  }

  // Phase A: top-k + shared-gate score per token. The sg dot is No-FMA
  // (plain mul+add, == the device's __fmul_rn/__fadd_rn router kernel).
  for (int t = 0; t < T; ++t) {
    topk_row(logits + static_cast<std::size_t>(t) * E, E, K,
             im.ids.data() + static_cast<std::size_t>(t) * K,
             im.alpha.data() + static_cast<std::size_t>(t) * K);
    float acc = 0.0F;
    for (int k = 0; k < K_in; ++k) {
      acc += bf16_to_f32(x[static_cast<std::size_t>(t) * K_in + k]) *
             bf16_to_f32(shared_gate[k]);
    }
    im.sg[t] = 1.0F / (1.0F + std::expf(-acc));
  }

  // x to F32 (exact bf16 -> f32) for the gu GEMM.
  for (int t = 0; t < T; ++t) {
    for (int k = 0; k < K_in; ++k) {
      im.x_f32[static_cast<std::size_t>(t) * K_in + k] =
          bf16_to_f32(x[static_cast<std::size_t>(t) * K_in + k]);
    }
  }

  // Phase B: shared expert slot + dequant the distinct routed experts.
  const Impl::Slot& sh = im.acquire_shared(w);
  {
    std::vector<char> seen(static_cast<std::size_t>(E), 0);
    std::vector<int> distinct;
    for (int t = 0; t < T; ++t) {
      for (int j = 0; j < K; ++j) {
        const int e = im.ids[static_cast<std::size_t>(t) * K + j];
        if (!seen[e]) {
          seen[e] = 1;
          distinct.push_back(e);
        }
      }
    }
    parallel_for(static_cast<int>(distinct.size()), threads,
                 [&](int c0, int c1) {
                   for (int i = c0; i < c1; ++i) {
                     im.acquire(distinct[i], w);
                   }
                 });
  }

  // Phase C: per-column gu GEMM -> silu_mul -> dn GEMM, parallel over
  // contiguous column chunks. Compact columns: c < T*K routed (token c/K,
  // expert ids[token*K + c%K]); c >= T*K shared expert (token c - T*K).
  const float inv_rgu = 1.0F / w.routed_gu_divisor;
  const float inv_sgu = 1.0F / w.shared_gu_divisor;
  const float inv_rdn = 1.0F / w.routed_dn_divisor;
  const float inv_sdn = 1.0F / w.shared_dn_divisor;
  parallel_for(C, threads, [&](int c0, int c1) {
    for (int c = c0; c < c1; ++c) {
      int token, expert;
      const Impl::Slot* slot = &sh;
      float inv_gu = inv_sgu, inv_dn = inv_sdn;
      if (c < T * K) {
        token = c / K;
        const int j = c - token * K;
        expert = im.ids[static_cast<std::size_t>(token) * K + j];
        slot = &im.acquire(expert, w);
        inv_gu = inv_rgu;
        inv_dn = inv_rdn;
      } else {
        token = c - T * K;
      }
      const float* xrow = im.x_f32.data() + static_cast<std::size_t>(token) * K_in;
      std::uint16_t* hcol = im.h_bf16.data() + static_cast<std::size_t>(c) * (2 * I);
      const float* Wgu = slot->gu;
      for (int n = 0; n < 2 * I; ++n) {
        const float acc = dot_f32(xrow, Wgu + static_cast<std::size_t>(n) * K_in, K_in);
        hcol[n] = f32_to_bf16_rn(acc * inv_gu);
      }
      // silu_mul: exact round-trip through BF16 (the device stores h2 as
      // BF16 and the dn GEMM reads it back as bf16 -> f32).
      float* h2row = im.h2_f32.data() + static_cast<std::size_t>(c) * I;
      for (int i = 0; i < I; ++i) {
        const float g = bf16_to_f32(hcol[i]);
        const float up = bf16_to_f32(hcol[i + I]);
        h2row[i] = bf16_to_f32(f32_to_bf16_rn(g / (1.0F + std::expf(-g)) * up));
      }
      float* yrow = im.y_f32.data() + static_cast<std::size_t>(c) * K_in;
      const float* Wdn = slot->dn;
      for (int n = 0; n < K_in; ++n) {
        yrow[n] = dot_f32(h2row, Wdn + static_cast<std::size_t>(n) * I, I) * inv_dn;
      }
    }
  });

  // Phase D: merge. FMA chain in j order then the shared term, == the
  // device's contracted merge kernel (Identity epilogue; no residual).
  parallel_for(T, threads, [&](int t0, int t1) {
    for (int t = t0; t < t1; ++t) {
      const float* alpha_row = im.alpha.data() + static_cast<std::size_t>(t) * K;
      const float sg_t = im.sg[t];
      const int shared_col = T * K + t;
      for (int k = 0; k < K_in; ++k) {
        float acc = 0.0F;
        for (int j = 0; j < K; ++j) {
          acc = std::fmaf(alpha_row[j],
                          im.y_f32[(static_cast<std::size_t>(t) * K + j) * K_in + k], acc);
        }
        acc = std::fmaf(sg_t, im.y_f32[static_cast<std::size_t>(shared_col) * K_in + k], acc);
        y[static_cast<std::size_t>(t) * K_in + k] = f32_to_bf16_rn(acc);
      }
    }
  });

  if (ids_out != nullptr) {
    std::memcpy(ids_out, im.ids.data(), static_cast<std::size_t>(T) * K * sizeof(std::int32_t));
  }
  if (alpha_out != nullptr) {
    std::memcpy(alpha_out, im.alpha.data(), static_cast<std::size_t>(T) * K * sizeof(float));
  }
}

}  // namespace ninfer::targets::qwen3_8_flash_next
