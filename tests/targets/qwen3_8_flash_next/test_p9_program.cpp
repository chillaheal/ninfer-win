// P9: Mini Program / first forward — the first full flash_next Engine forward
// over the deterministic mini fixture (tests a-f, ctest label flash_next_P9).
//
//   (a) build the per-expert .exarch from the artifact (deterministic slices),
//       validate the archive header, stage every layer through PrefillWindow,
//       confirm DecodeFetch hits after a prefill stage, and prove the paged
//       device bytes equal the artifact slice.
//   (b) greedy golden: Engine over the mini fixture (eager, context_cache on,
//       max_context 64), prepare_tokens(8 fixed ids), requested_output_tokens 4.
//       Assert success + finish_reason==OutputLimit + 4 tokens + captured logits
//       match the checked-in golden (cross-checked against the emitted argmax).
//   (c) bit-identical second run (a fresh Engine, same inputs).
//   (d) prefix reuse (behavioral, one Engine, same 8-token prompt twice).
//   (e) peak GPU memory (weights + workspace + KV) under an in-test cap.
//   (f) CLI determinism: build/apps/ninfer <mini> --prompt hi twice, byte-identical.
//
// --write-golden regenerates p9_golden_logits.bin from this run and prints the
// measured (e) peak for the cap; the default run verifies against it.
//
// HONESTY BLOCKER (carried from P9_design.md): the production modeling function
// is unrecoverable — all block math here is a documented v1 contract and the
// real weights are not loaded until P12. Every P9 assertion is on the
// deterministic mini fixture; composition is validated by (i) emitted tokens ==
// argmax(captured logits), (ii) bit-identical second run, (iii) the per-op
// oracles of P5-P8. There is NO standalone from-scratch FP32 forward in this test.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <ninfer/engine.h>
#include <ninfer/ops/sparse_moe_nvfp4.h>
#include <targets/qwen3_8_flash_next/paging/pager.h>

#include "artifact/reader.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next::paging;

using ninfer::artifact::NumericFormat;
using ninfer::artifact::Reader;
using ninfer::artifact::TensorDescriptor;
using ninfer::artifact::block_scale_geometry;
using ninfer::ops::SparseMoeNvfp4Geometry;
using ninfer::ops::SparseMoeNvfp4Plane;
using ninfer::ops::sparse_moe_nvfp4_planes;

int failures = 0;

void fail_or_count(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++failures;
}

[[noreturn]] void die(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  std::exit(1);
}

void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

constexpr const char* kLogitsEnv = "NINFER_FLASH_NEXT_LOGITS";
constexpr std::size_t kVocab = 256;
// Peak-GPU cap for (e): measured mini total (weights.peak + workspace.peak +
// kv_payload) = 3,780,416 bytes on 2026-09-11, rounded up to the next MiB.
constexpr std::size_t kPeakCapBytes = 4194304;

std::filesystem::path fixture_dir() {
  return std::filesystem::path(NINFER_SOURCE_DIR) / "out" / "flash_next_dev";
}
std::filesystem::path mini_artifact() { return fixture_dir() / "qwen3_8_flash_next_mini.ninfer"; }
std::filesystem::path mini_exarch() { return fixture_dir() / "qwen3_8_flash_next_mini.exarch"; }
std::filesystem::path golden_logits() { return fixture_dir() / "p9_golden_logits.bin"; }
std::filesystem::path temp_cap(const std::string& tag) {
  return std::filesystem::temp_directory_path() / ("ninfer_p9_" + tag + ".bin");
}
std::filesystem::path cli_exe() {
  return std::filesystem::path(NINFER_SOURCE_DIR) / "build" / "apps" / "ninfer-cli.exe";
}

// ---- captured-logits file format: [8B "P9LOGITS"][u32 rows LE][rows*256 f32 LE] ----

void write_logits(const std::filesystem::path& path, const std::vector<float>& values) {
  const auto rows = values.size() / kVocab;
  std::ofstream os(path, std::ios::binary | std::ios::trunc);
  if (!os) die("cannot write logits " + path.string());
  const std::uint8_t magic[8] = {'P', '9', 'L', 'O', 'G', 'I', 'T', 'S'};
  const std::uint32_t n = static_cast<std::uint32_t>(rows);
  os.write(reinterpret_cast<const char*>(magic), sizeof(magic));
  os.write(reinterpret_cast<const char*>(&n), sizeof(n));
  os.write(reinterpret_cast<const char*>(values.data()),
           static_cast<std::streamsize>(values.size() * sizeof(float)));
  os.flush();
  if (!os) die("logits write failed " + path.string());
}

std::vector<float> read_logits(const std::filesystem::path& path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) die("cannot read logits " + path.string());
  std::uint8_t magic[8] = {};
  is.read(reinterpret_cast<char*>(magic), sizeof(magic));
  const std::uint8_t want[8] = {'P', '9', 'L', 'O', 'G', 'I', 'T', 'S'};
  if (!is || std::memcmp(magic, want, sizeof(want)) != 0) die("bad logits magic " + path.string());
  std::uint32_t rows = 0;
  is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  if (!is) die("bad logits header " + path.string());
  std::vector<float> values(static_cast<std::size_t>(rows) * kVocab);
  if (rows > 0) {
    is.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
  }
  if (!is) die("short logits body " + path.string());
  return values;
}

// ---- per-expert exarch slice (deterministic row slices from the fused objects) ----

// One routed-expert blob = 27,648 B = four contiguous row slices of the fused
// gate_up [1024,256] and down [2048,64] NVFP4 objects (8 experts each,
// row-major, expert-major). The code plane is natural row-major; the scale
// plane starts at the geometry's scale_plane_offset.
std::vector<std::uint8_t> expert_slice(const Reader& reader, std::uint32_t layer, std::uint32_t expert,
                                       const SparseMoeNvfp4Plane& planes) {
  const std::string gu_name = "text/layers/" + std::to_string(layer) + "/mlp/experts/gate_up";
  const std::string dn_name = "text/layers/" + std::to_string(layer) + "/mlp/experts/down";
  const auto* gu_desc = std::get_if<TensorDescriptor>(reader.find(gu_name));
  const auto* dn_desc = std::get_if<TensorDescriptor>(reader.find(dn_name));
  if (!gu_desc || !dn_desc) die("missing gate_up/down object @ layer " + std::to_string(layer));
  if (gu_desc->shape != std::vector<std::uint64_t>{1024, 256})
    die("gate_up shape != [1024,256] @ layer " + std::to_string(layer));
  if (dn_desc->shape != std::vector<std::uint64_t>{2048, 64})
    die("down shape != [2048,64] @ layer " + std::to_string(layer));

  const auto gu_geo = block_scale_geometry(NumericFormat::NVFP4, gu_desc->shape);
  const auto dn_geo = block_scale_geometry(NumericFormat::NVFP4, dn_desc->shape);

  const std::uint64_t gu_codes_e = gu_geo.code_plane_bytes / 8;   // 16384
  const std::uint64_t gu_scales_e = gu_geo.scale_plane_bytes / 8; // 2048
  const std::uint64_t dn_codes_e = dn_geo.code_plane_bytes / 8;   // 8192
  const std::uint64_t dn_scales_e = dn_geo.scale_plane_bytes / 8; // 1024

  // Self-validate the source slice sizes against the per-expert plane slots.
  if (gu_codes_e != planes.gu_scales - planes.gu_codes) die("gu_codes slice != plane slot");
  if (gu_scales_e != planes.dn_codes - planes.gu_scales) die("gu_scales slice != plane slot");
  if (dn_codes_e != planes.dn_scales - planes.dn_codes) die("dn_codes slice != plane slot");
  if (dn_scales_e != planes.total - planes.dn_scales) die("dn_scales slice != plane slot");

  const auto gu = reader.payload(gu_name);
  const auto dn = reader.payload(dn_name);
  const std::uint8_t* gub = reinterpret_cast<const std::uint8_t*>(gu.data.data());
  const std::uint8_t* dnb = reinterpret_cast<const std::uint8_t*>(dn.data.data());

  std::vector<std::uint8_t> blob(planes.total);
  std::memcpy(blob.data() + planes.gu_codes, gub + static_cast<std::size_t>(expert) * gu_codes_e,
              static_cast<std::size_t>(gu_codes_e));
  std::memcpy(blob.data() + planes.gu_scales,
              gub + gu_geo.scale_plane_offset + static_cast<std::size_t>(expert) * gu_scales_e,
              static_cast<std::size_t>(gu_scales_e));
  std::memcpy(blob.data() + planes.dn_codes, dnb + static_cast<std::size_t>(expert) * dn_codes_e,
              static_cast<std::size_t>(dn_codes_e));
  std::memcpy(blob.data() + planes.dn_scales,
              dnb + dn_geo.scale_plane_offset + static_cast<std::size_t>(expert) * dn_scales_e,
              static_cast<std::size_t>(dn_scales_e));
  return blob;
}

void build_exarch(const Reader& reader, const std::filesystem::path& out,
                  const SparseMoeNvfp4Plane& planes) {
  constexpr std::uint32_t kLayers = 4;
  constexpr std::uint32_t kExperts = 8;
  const std::size_t header = kArchiveHeaderBytes;
  std::vector<std::uint8_t> file(header + std::uint64_t(kLayers) * kExperts * planes.total);
  std::uint8_t* f = file.data();
  std::memcpy(f, kArchiveMagic, sizeof(kArchiveMagic));
  const std::uint32_t nl = kLayers;
  const std::uint32_t ep = kExperts;
  const std::uint64_t bpe = planes.total;
  std::memcpy(f + 8, &nl, sizeof(nl));
  std::memcpy(f + 12, &ep, sizeof(ep));
  std::memcpy(f + 16, &bpe, sizeof(bpe));
  // [24,40) reserved zero (the vector 0-inits).
  for (std::uint32_t l = 0; l < kLayers; ++l) {
    for (std::uint32_t e = 0; e < kExperts; ++e) {
      const auto slice = expert_slice(reader, l, e, planes);
      std::memcpy(f + header + (std::uint64_t(l) * kExperts + e) * planes.total, slice.data(),
                  planes.total);
    }
  }
  std::ofstream os(out, std::ios::binary | std::ios::trunc);
  if (!os) die("cannot write exarch " + out.string());
  os.write(reinterpret_cast<const char*>(f), static_cast<std::streamsize>(file.size()));
  os.flush();
  if (!os) die("exarch write failed " + out.string());
}

// emitted tokens must equal argmax(captured logits), ties -> lowest index
// (the v1 final head is always ops::argmax; BF16->f32 is order-preserving).
void check_argmax(const std::vector<float>& logits, const std::vector<TokenId>& tokens) {
  const auto rows = logits.size() / kVocab;
  if (rows != tokens.size()) {
    fail_or_count("(b) logit rows (" + std::to_string(rows) + ") != tokens (" +
                  std::to_string(tokens.size()) + ")");
    return;
  }
  for (auto i = 0u; i < rows; ++i) {
    int best = 0;
    float bestv = logits[static_cast<std::size_t>(i) * kVocab];
    for (int v = 1; v < static_cast<int>(kVocab); ++v) {
      const float val = logits[static_cast<std::size_t>(i) * kVocab + v];
      if (val > bestv) {
        bestv = val;
        best = v;
      }
    }
    if (static_cast<int>(tokens[i]) != best) {
      fail_or_count("(b) argmax row " + std::to_string(i) + " = " + std::to_string(best) +
                    " != emitted token " + std::to_string(tokens[i]));
    }
  }
}

EngineOptions mini_engine_options() {
  EngineOptions opts;
  opts.artifact_path = mini_artifact();
  opts.device = 0;
  opts.max_context = 64;
  opts.kv_capacity = KvCapacityPolicy::explicit_capacity(64);
  opts.max_concurrency = 1;
  opts.prefill_chunk = 8;
  opts.use_cuda_graph = false;
  opts.context_cache.enabled = true;
  return opts;
}

struct RunResult {
  GenerationResult gen;
  MemorySummary mem;
};

// One Engine over the mini fixture: fixed 8-token prompt, requested_output_tokens
// decode steps. Logits are captured (via NINFER_FLASH_NEXT_LOGITS) and flushed at
// Engine destruction; the caller reads cap_path after the Engine is gone.
RunResult run_forward(const std::filesystem::path& cap_path, const std::vector<TokenId>& prompt,
                      std::uint32_t output_tokens) {
  const EngineOptions opts = mini_engine_options();
  set_env(kLogitsEnv, cap_path.string());
  RunResult out{};
  {
    Engine engine(opts);
    auto prepared = engine.prepare_tokens(prompt);
    if (!prepared) die("prepare_tokens failed");
    RequestOptions req;
    req.execution.requested_output_tokens = output_tokens;
    req.execution.sampling.temperature = 0.0F;
    out.gen = engine.generate(std::move(prepared), req);
    out.mem = engine.memory_summary();
  }
  return out;
}

// ---- (f) CLI double-run determinism ----

std::string run_cli_to_file(const std::filesystem::path& exe, const std::filesystem::path& artifact,
                            int* exit_code) {
  *exit_code = -1;
#if defined(_WIN32)
  // Capture stdout (the deterministic generated-token stream) via an anonymous pipe:
  // STARTF_USESTDHANDLES with hStdOutput = the pipe's write end, then ReadFile to EOF
  // before waiting on the child. stderr is left NULL so the child inherits the
  // parent's terminal (it carries non-deterministic metrics/progress and must NOT be
  // captured). A pipe is used rather than a CreateFileW file handle: for this CLI a
  // file handle on hStdOutput captured 0 bytes while a pipe captured the full 10-byte
  // stream (A/B-verified 2026-09-12 via PowerShell Start-Process -RedirectStandardOutput).
  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  HANDLE read_pipe = nullptr, write_pipe = nullptr;
  if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) return "";
  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdOutput = write_pipe;
  PROCESS_INFORMATION pi{};
  const std::wstring cmd = L"\"" + exe.wstring() + L"\" \"" + artifact.wstring() +
                           L"\" --prompt hi --max-new 8 --max-context 64";
  const BOOL ok = CreateProcessW(nullptr, const_cast<wchar_t*>(cmd.c_str()), nullptr, nullptr,
                                 /*bInheritHandles=*/TRUE, 0, nullptr, nullptr, &si, &pi);
  CloseHandle(write_pipe);  // parent keeps only the read end
  if (!ok) {
    CloseHandle(read_pipe);
    return "";
  }
  std::string out;
  std::vector<char> buf(65536);
  for (;;) {
    DWORD got = 0;
    if (!ReadFile(read_pipe, buf.data(), static_cast<DWORD>(buf.size()), &got, nullptr) ||
        got == 0)
      break;
    out.append(buf.data(), static_cast<std::size_t>(got));
  }
  CloseHandle(read_pipe);
  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD code = 0;
  GetExitCodeProcess(pi.hProcess, &code);
  *exit_code = static_cast<int>(code);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  return out;
#else
  (void)exe;
  (void)artifact;
  return "";
#endif
}

void check_cli_determinism() {
  const std::filesystem::path exe = cli_exe();
  const std::filesystem::path artifact = mini_artifact();
  if (!std::filesystem::exists(exe)) {
    fail_or_count("(f) CLI exe missing: " + exe.string());
    return;
  }
  int rc1 = -1, rc2 = -1;
  const std::string out1 = run_cli_to_file(exe, artifact, &rc1);
  const std::string out2 = run_cli_to_file(exe, artifact, &rc2);
  if (rc1 != 0) fail_or_count("(f) CLI run 1 exit " + std::to_string(rc1));
  if (rc2 != 0) fail_or_count("(f) CLI run 2 exit " + std::to_string(rc2));
  if (rc1 == 0 && rc2 == 0 && out1 != out2)
    fail_or_count("(f) CLI token stream differs between runs (not deterministic)");
  if (rc1 == 0 && rc2 == 0 && out1 == out2)
    std::printf("(f) CLI determinism OK (%zu bytes, exit 0 both runs)\n", out1.size());
}

}  // namespace

int main(int argc, char** argv) {
  const bool write_golden = argc > 1 && std::string(argv[1]) == "--write-golden";
  const std::filesystem::path artifact = mini_artifact();
  if (!std::filesystem::exists(artifact)) {
    std::fprintf(stderr,
                 "SKIP: mini fixture missing at %s; generate with "
                 "python -m tools.convert.qwen3_8_flash_next.make_mini_artifact\n",
                 artifact.string().c_str());
    return 77;
  }

  try {
    Reader reader(artifact);
    const SparseMoeNvfp4Plane planes =
        sparse_moe_nvfp4_planes(SparseMoeNvfp4Geometry{8, 2, 256, 64});

    // ---- (a) exarch build + pager staging ----
    build_exarch(reader, mini_exarch(), planes);
    const ExpertArchive archive = ExpertArchive::open(mini_exarch());
    if (archive.num_layers() != 4 || archive.experts_per_layer() != 8 ||
        archive.bytes_per_expert() != 27648) {
      fail_or_count("(a) exarch header mismatch: layers=" + std::to_string(archive.num_layers()) +
                    " experts=" + std::to_string(archive.experts_per_layer()) + " bytes=" +
                    std::to_string(archive.bytes_per_expert()));
    }
    // Independent re-slice: the on-disk blob for (l=2, e=5) must equal a fresh
    // artifact-derived slice (catches a wrong-but-consistent slice).
    {
      const auto fresh = expert_slice(reader, 2, 5, planes);
      const auto stored = archive.expert(2, 5);
      if (stored.size() != fresh.size() ||
          std::memcmp(stored.data(), fresh.data(), fresh.size()) != 0) {
        fail_or_count("(a) stored expert (2,5) != artifact re-slice");
      }
    }
    {
      PagerConfig cfg;
      cfg.host_slots = 32;
      cfg.device_slots = 32;
      Pager pager(archive, cfg);
      PrefillWindow prefill(pager);
      for (int l = 0; l < 4; ++l) {
        prefill.stage(l);
        prefill.wait();
      }
      const std::size_t h2d_after_prefill = pager.counters().h2d_bytes;
      // All 32 experts resident after the 4-layer prefill stage.
      if (h2d_after_prefill != 4u * 8u * 27648u)
        fail_or_count("(a) h2d_bytes after prefill = " + std::to_string(h2d_after_prefill) +
                      " (want " + std::to_string(4u * 8u * 27648u) + ")");
      DecodeFetch fetch(pager);
      const std::uint32_t ids[2] = {0, 1};
      const std::size_t misses = fetch.stage(0, ids);
      fetch.wait();
      if (misses != 0)
        fail_or_count("(a) decode fetch after prefill issued misses = " + std::to_string(misses));
      const Counters c = pager.counters();
      if (c.hits < 2) fail_or_count("(a) decode hits < 2 after prefill stage");
      if (c.h2d_bytes != h2d_after_prefill)
        fail_or_count("(a) h2d_bytes changed on a resident decode fetch");
      for (const std::uint32_t e : {0u, 1u}) {
        const auto dev = pager.device_bytes(0, e);
        const auto arc = archive.expert(0, e);
        if (dev.size() != arc.size() || std::memcmp(dev.data(), arc.data(), arc.size()) != 0) {
          fail_or_count("(a) device_bytes(layer0, expert" + std::to_string(e) +
                        ") != archive bytes");
        }
      }
    }

    // ---- (b) greedy golden ----
    const std::vector<TokenId> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    const RunResult rb = run_forward(temp_cap("b"), prompt, 4);
    const std::vector<float> logits_b = read_logits(temp_cap("b"));
    if (rb.gen.finish_reason != FinishReason::OutputLimit)
      fail_or_count("(b) finish_reason != OutputLimit (" +
                    std::to_string(static_cast<int>(rb.gen.finish_reason)) + ")");
    if (rb.gen.generated_token_ids.size() != 4)
      fail_or_count("(b) generated " + std::to_string(rb.gen.generated_token_ids.size()) +
                    " tokens (want 4)");
    check_argmax(logits_b, rb.gen.generated_token_ids);
    if (write_golden) {
      write_logits(golden_logits(), logits_b);
      std::printf("(b) WROTE golden %s (%zu logits = %zu rows)\n", golden_logits().string().c_str(),
                  logits_b.size(), logits_b.size() / kVocab);
    } else {
      const std::vector<float> golden = read_logits(golden_logits());
      if (golden.size() != logits_b.size() ||
          (golden.size() > 0 && std::memcmp(golden.data(), logits_b.data(), golden.size() * sizeof(float)) != 0)) {
        fail_or_count("(b) logits differ from golden (" + std::to_string(golden.size()) + " vs " +
                      std::to_string(logits_b.size()) + " values)");
      }
    }
    // ---- (e) peak GPU memory cap ----
    {
      const std::size_t peak = rb.mem.weights.peak_used_bytes + rb.mem.workspace.peak_used_bytes +
                               rb.mem.kv_payload_bytes;
      std::printf("(e) peak weights+workspace+kv = %zu bytes (cap %zu)\n", peak, kPeakCapBytes);
      if (kPeakCapBytes != 0 && peak > kPeakCapBytes)
        fail_or_count("(e) peak " + std::to_string(peak) + " > cap " +
                      std::to_string(kPeakCapBytes));
    }

    // ---- (c) bit-identical second run ----
    {
      const RunResult rc = run_forward(temp_cap("c"), prompt, 4);
      const std::vector<float> logits_c = read_logits(temp_cap("c"));
      const bool logits_same =
          logits_c.size() == logits_b.size() &&
          (logits_c.size() == 0 ||
           std::memcmp(logits_c.data(), logits_b.data(), logits_c.size() * sizeof(float)) == 0);
      if (!logits_same) fail_or_count("(c) second-run logits not bit-identical");
      if (rc.gen.generated_token_ids != rb.gen.generated_token_ids)
        fail_or_count("(c) second-run tokens differ from first");
    }

    // ---- (d) prefix reuse (behavioral) ----
    {
      const EngineOptions opts = mini_engine_options();
      set_env(kLogitsEnv, temp_cap("d").string());
      Engine engine(opts);
      auto p1 = engine.prepare_tokens(prompt);
      if (!p1) die("(d) prepare_tokens 1 failed");
      RequestOptions req;
      req.execution.requested_output_tokens = 4;
      req.execution.sampling.temperature = 0.0F;
      const GenerationResult r1 = engine.generate(std::move(p1), req);
      const std::uint64_t computed1 = engine.runtime_stats().computed_prefill_tokens;
      if (computed1 != 8)
        fail_or_count("(d) first computed_prefill_tokens = " + std::to_string(computed1) +
                      " (want 8)");
      auto p2 = engine.prepare_tokens(prompt);
      if (!p2) die("(d) prepare_tokens 2 failed");
      const GenerationResult r2 = engine.generate(std::move(p2), req);
      const std::uint64_t computed2 = engine.runtime_stats().computed_prefill_tokens;
      if (r2.reused_prompt_tokens == 0)
        fail_or_count("(d) second reused_prompt_tokens == 0 (expected reuse)");
      const std::uint64_t delta = computed2 - computed1;
      if (delta >= 8)
        fail_or_count("(d) second computed_prefill delta = " + std::to_string(delta) +
                      " (want < 8)");
      std::printf("(d) prefix reuse OK (reused=%u, prefill delta=%llu)\n",
                  r2.reused_prompt_tokens, static_cast<unsigned long long>(delta));
      (void)r1;
    }

    // ---- (f) CLI determinism ----
    check_cli_determinism();
  } catch (const std::exception& e) {
    die(std::string("P9 threw: ") + e.what());
  }

  if (failures == 0) {
    std::printf("OK flash_next P9 (mini program / first forward)\n");
  }
  return failures == 0 ? 0 : 1;
}
