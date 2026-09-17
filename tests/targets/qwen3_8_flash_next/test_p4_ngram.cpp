// P4: PLE mmap gather (ngram_embedding Op).
//
// * (a) Open the mini .ngram store: geometry checks (1024 rows x 32 cols,
//   4 heads, ngram_size 3, 256 rows/head, 1 shard); dequant of rows 0..127
//   is BIT-EXACT against the converter's .ple_golden.bin; hash ids stay in
//   range.
// * (b) Exact oracle vs Op on the mini table: an independent naive path
//   (ifstream file reads, FP64 e4m3/bf16 decode, FP64 conv) over T=64
//   deterministic tokens; the Op output (H2D, copied back) matches within
//   fp32-sum tolerance. The dequant product is exact in fp32 (dyadic x
//   dyadic, <=24-bit mantissa), so only the conv accumulation differs.
// * (c) A synthetic ~68 MB store (2,097,152 rows x 32 cols written by this
//   test) with a 4 MiB staging cap -> multi-chunk staging; numerically
//   correct vs the same naive path; device-memory delta around the Op is
//   bounded (no GPU allocation in the size of the table, peak << file size).
// * (d) T=64 prefill + T=1 + T=1 decode with request state carry == one
//   T=66 run: BIT-EXACT fp32 (chunking must not change any number).
// * (f) Binder integration: bind_artifact (P2) opens the lazy PLE handle,
//   view() maps it, NgramEmbeddingTable::adopt() reads the same rows
//   bit-exactly as the open(path) path.
//
// Skips (exit 77) when the mini fixture is missing; generate it with
// `python -m tools.convert.qwen3_8_flash_next.make_mini_artifact`.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <ninfer/ops/ngram_embedding.h>
#include <ninfer/targets/qwen3_8_flash_next/binder.h>

namespace {

using namespace ninfer::ops;

int failures = 0;

void fail_or_count(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++failures;
}

[[noreturn]] void die(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  std::exit(1);
}

std::uint32_t xorshift(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

// --- independent naive codec decodes (FP64 path, oracle side) ---------------

double naive_e4m3(std::uint8_t code) {
  const int s = code >> 7;
  const int e = (code >> 3) & 0xF;
  const int m = code & 7;
  const double v = (e == 0) ? std::ldexp(static_cast<double>(m), -9)
                            : std::ldexp(static_cast<double>(8 + m), e - 10);
  return s ? -v : v;
}

double naive_bf16(std::uint16_t bits) {
  const std::uint32_t f = static_cast<std::uint32_t>(bits) << 16;
  float fv;
  std::memcpy(&fv, &f, sizeof(fv));
  return static_cast<double>(fv);
}

// Naive row dequant straight from the raw file bytes (block-major layout,
// same addressing the Op uses -- the layout is part of the store contract).
double naive_dequant(const std::vector<std::uint8_t>& file, std::uint64_t codes_base,
                     std::uint64_t scales_base, std::uint64_t rows_per_shard,
                     std::uint32_t block_rows, std::uint32_t cols, std::uint64_t row,
                     std::uint32_t c) {
  const std::uint64_t r_in_shard = row % rows_per_shard;
  const std::uint64_t block = r_in_shard / block_rows;
  const std::uint64_t r_in_block = r_in_shard % block_rows;
  const std::uint8_t code =
      file[codes_base + block * static_cast<std::uint64_t>(block_rows) * cols +
                    r_in_block * cols + c];
  std::uint16_t sbits = 0;
  std::memcpy(&sbits,
              file.data() + scales_base + block * static_cast<std::uint64_t>(block_rows) * 2 +
                  r_in_block * 2,
              sizeof(sbits));
  return naive_e4m3(code) * naive_bf16(sbits);
}

// Naive oracle of the whole per-token contract (fresh request): hash -> row
// -> dequant -> K-tap depthwise conv with dilation, FP64 accumulation.
void naive_reference(const std::vector<std::uint8_t>& file, std::uint64_t codes_base,
                     std::uint64_t scales_base, std::uint64_t rows_per_shard,
                     std::uint32_t block_rows, std::uint32_t cols, std::uint32_t heads,
                     std::uint64_t rph, const std::vector<std::uint32_t>& ids,
                     const std::vector<float>& W, std::uint32_t K, std::uint32_t dil,
                     std::vector<double>& out) {
  const std::uint64_t T = ids.size();
  const std::uint32_t dim = heads * cols;
  std::vector<double> emb(T * dim);
  out.resize(T * dim);
  for (std::uint64_t t = 0; t < T; ++t) {
    const std::uint32_t x = ids[t];
    const std::uint32_t prev1 = (t == 0) ? 0 : ids[t - 1];  // missing context = 0
    const std::uint32_t prev2 = (t < 2) ? 0 : ids[t - 2];
    const std::uint32_t bi[2] = {prev1, x};
    const std::uint32_t tri[3] = {prev2, prev1, x};
    for (std::uint32_t s = 0; s < heads; ++s) {
      const std::span<const std::uint32_t> ng =
          (s < heads / 2) ? std::span<const std::uint32_t>(bi, 2)
                          : std::span<const std::uint32_t>(tri, 3);
      const std::uint64_t row =
          static_cast<std::uint64_t>(s) * rph + ngram_hash_id(s, ng, rph);
      for (std::uint32_t c = 0; c < cols; ++c) {
        emb[t * dim + s * cols + c] =
            naive_dequant(file, codes_base, scales_base, rows_per_shard, block_rows, cols, row, c);
      }
    }
    for (std::uint32_t c = 0; c < dim; ++c) {
      double acc = 0.0;
      for (std::uint32_t i = 0; i < K; ++i) {
        if (static_cast<std::uint64_t>(i) * dil <= t) {
          acc += static_cast<double>(W[static_cast<std::uint64_t>(i) * dim + c]) *
                 emb[(t - static_cast<std::uint64_t>(i) * dil) * dim + c];
        }
      }
      out[t * dim + c] = acc;
    }
  }
}

// --- synthetic store writer (same NNGRAM format as the converter) -----------

std::uint64_t align_up(std::uint64_t v, std::uint64_t a) { return (v + a - 1) / a * a; }

void write_synthetic_store(const std::filesystem::path& path, std::uint64_t rows,
                           std::uint32_t cols, std::uint32_t heads, std::uint32_t block_rows,
                           std::uint32_t seed) {
  const std::uint64_t code_plane = rows * cols;
  const std::uint64_t scale_off = align_up(code_plane, 4096);
  const std::uint64_t scale_plane = rows * 2;
  const std::uint64_t payload = code_plane + scale_plane;
  const std::uint64_t rph = rows / heads;

  std::string directory;
  std::uint64_t payload_offset = 0;
  for (int attempt = 0; attempt < 16; ++attempt) {
    const std::uint64_t shard_stride = align_up(payload, 4096);
    directory = std::string("{\"block\":{\"block_codes_bytes\":") +
        std::to_string(static_cast<std::uint64_t>(block_rows) * cols) +
        ",\"block_scale_bytes\":" + std::to_string(static_cast<std::uint64_t>(block_rows) * 2) +
        ",\"code_row_bytes\":" + std::to_string(cols) +
        ",\"cols\":" + std::to_string(cols) + ",\"rows\":" + std::to_string(block_rows) +
        ",\"scale_row_bytes\":2}"
        ",\"block_rows\":" + std::to_string(block_rows) +
        ",\"cols_per_shard\":" + std::to_string(cols) +
        ",\"model_id\":\"p4-synthetic\""
        ",\"n_shards\":1"
        ",\"ngram\":{\"heads\":" + std::to_string(heads) +
        ",\"heads_per_ngram\":2"
        ",\"size\":3"
        ",\"vocab_base\":" + std::to_string(rph) + "}" +
        ",\"payload_offset\":" + std::to_string(payload_offset) +
        ",\"planes\":{\"code_plane_bytes\":" + std::to_string(code_plane) +
        ",\"plane_alignment\":4096"
        ",\"scale_plane_bytes\":" + std::to_string(scale_plane) +
        ",\"scale_plane_offset\":" + std::to_string(scale_off) +
        ",\"payload_bytes\":" + std::to_string(payload) + "}" +
        ",\"quant_id\":\"fp8e4m3fn_row_bf16s-v1\""
        ",\"rows_per_shard\":" + std::to_string(rows) +
        ",\"shard_stride\":" + std::to_string(shard_stride) +
        ",\"shards\":[{\"bytes\":" + std::to_string(payload) +
        ",\"id\":0"
        ",\"offset\":" + std::to_string(payload_offset) +
        ",\"scale_offset\":" + std::to_string(payload_offset + scale_off) + "}]"
        ",\"version\":1}";
    const std::uint64_t candidate = align_up(16 + directory.size(), 4096);
    if (candidate == payload_offset) break;
    payload_offset = candidate;
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) die("cannot write synthetic store: " + path.string());
  std::uint8_t header[16] = {};
  std::memcpy(header, "NNGRAM", 6);
  header[6] = 0;
  header[7] = 1;
  std::uint64_t json_bytes = directory.size();
  std::memcpy(header + 8, &json_bytes, sizeof(json_bytes));
  out.write(reinterpret_cast<const char*>(header), sizeof(header));
  out.write(directory.data(), static_cast<std::streamsize>(directory.size()));
  std::vector<std::uint8_t> pad(payload_offset - (16 + directory.size()), 0);
  out.write(reinterpret_cast<const char*>(pad.data()),
            static_cast<std::streamsize>(pad.size()));

  std::uint32_t state = seed;
  std::vector<std::uint8_t> chunk(1u << 20);
  std::uint64_t written = 0;
  while (written < code_plane) {
    const std::uint64_t n = std::min(chunk.size(), code_plane - written);
    for (std::uint64_t i = 0; i < n; ++i) chunk[i] = static_cast<std::uint8_t>(xorshift(state));
    out.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(n));
    written += n;
  }
  written = 0;
  while (written < scale_plane) {
    const std::uint64_t n = std::min(chunk.size(), scale_plane - written);
    for (std::uint64_t i = 0; i < n / 2; ++i) {
      const std::uint16_t bits =
          static_cast<std::uint16_t>(0x3C00 | (static_cast<std::uint16_t>(xorshift(state)) & 0xFF));
      chunk[2 * i] = static_cast<std::uint8_t>(bits & 0xFF);
      chunk[2 * i + 1] = static_cast<std::uint8_t>(bits >> 8);
    }
    out.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(n));
    written += n;
  }
  if (!out) die("synthetic store write failed: " + path.string());
}

// --- helpers ------------------------------------------------------------------

// Runs the Op and (when host_out is non-null) copies the device result back.
// delta_bytes_out receives the device-memory delta around the call.
void run_op(const NgramEmbeddingTable& table, const std::vector<std::uint32_t>& ids,
            const std::vector<float>& W, NgramRequestState& state, float* device_out,
            std::vector<float>* host_out, std::uint64_t staging_cap,
            size_t* delta_bytes_out) {
  NgramOpParams params;
  params.conv_weights = W.data();
  params.staging_cap_bytes = staging_cap;

  size_t free_before = 0, total = 0;
  if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) die("cudaMemGetInfo (before)");
  ngram_embedding(table, std::span<const std::uint32_t>(ids), params, state, device_out, 0);
  if (cudaStreamSynchronize(0) != cudaSuccess) die("stream sync");
  if (host_out != nullptr) {
    host_out->resize(ids.size() * table.info().embed_dim());
    if (cudaMemcpy(host_out->data(), device_out, host_out->size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      die("D2H copy");
    }
  }
  size_t free_after = 0;
  if (cudaMemGetInfo(&free_after, &total) != cudaSuccess) die("cudaMemGetInfo (after)");
  if (delta_bytes_out != nullptr) {
    *delta_bytes_out = (free_before > free_after) ? free_before - free_after : 0;
  }
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("cannot read " + path.string());
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
}

void check_close_enough(const std::vector<float>& got, const std::vector<double>& ref,
                        const char* what) {
  double max_ref = 0.0;
  for (const double v : ref) max_ref = std::max(max_ref, std::fabs(v));
  const double tol = 1e-6 + 1e-4 * max_ref;
  double worst = 0.0;
  std::size_t where = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = std::fabs(static_cast<double>(got[i]) - ref[i]);
    if (d > worst) {
      worst = d;
      where = i;
    }
  }
  if (worst > tol) {
    fail_or_count(std::string(what) + ": max abs diff " + std::to_string(worst) +
                  " at " + std::to_string(where) + " (tolerance " + std::to_string(tol) +
                  ", max ref " + std::to_string(max_ref) + ")");
  }
}

}  // namespace

int main() {
  const std::filesystem::path out_dir =
      std::filesystem::path(NINFER_SOURCE_DIR) / "out" / "flash_next_dev";
  const std::filesystem::path ninfer_path = out_dir / "qwen3_8_flash_next_mini.ninfer";
  const std::filesystem::path ngram_path = out_dir / "qwen3_8_flash_next_mini.ngram";
  const std::filesystem::path golden_path = out_dir / "qwen3_8_flash_next_mini.ple_golden.bin";
  if (!std::filesystem::exists(ngram_path)) return 77;  // fixture not built

  try {
    // --- (a) open the mini store: geometry + golden dequant + hash sanity ----
    const NgramEmbeddingTable mini = NgramEmbeddingTable::open(ngram_path);
    const NgramStoreInfo& g = mini.info();
    if (g.version != 1) fail_or_count("mini store version != 1");
    if (g.cols != 32 || g.heads != 4 || g.ngram_size != 3 || g.block_rows != 128) {
      fail_or_count("mini store geometry mismatch (cols/heads/size/block_rows)");
    }
    if (g.rows != 1024 || g.rows_per_shard() != 1024 || g.rows_per_head() != 256 ||
        g.shards.size() != 1) {
      fail_or_count("mini store row/shard geometry mismatch");
    }
    if (g.embed_dim() != 128) fail_or_count("mini store embed_dim != 128");
    if (g.quant_id != "fp8e4m3fn_row_bf16s-v1") fail_or_count("mini store quant_id mismatch");
    if (g.file_bytes != std::filesystem::file_size(ngram_path)) {
      fail_or_count("store file_bytes != actual file size");
    }

    {
      // Golden = 16-byte header ("PLEGOLDN" + u32 rows + u32 cols) + row-major
      // f32 body of block 0.
      const std::vector<std::uint8_t> golden = read_file(golden_path);
      if (golden.size() < 16) {
        fail_or_count("golden smaller than its 16-byte header");
      } else {
        std::uint32_t grow = 0, gcol = 0;
        std::memcpy(&grow, golden.data() + 8, 4);
        std::memcpy(&gcol, golden.data() + 12, 4);
        if (std::memcmp(golden.data(), "PLEGOLDN", 8) != 0 || grow != 128 || gcol != 32 ||
            golden.size() != 16u + 128u * 32u * 4u) {
          fail_or_count("golden header/size mismatch (want PLEGOLDN, 128 x 32 x f32)");
        } else {
          std::vector<float> row(32);
          int bad = 0;
          for (std::uint64_t r = 0; r < 128 && bad < 3; ++r) {
            mini.dequant_row(r, row.data());
            if (std::memcmp(row.data(), golden.data() + 16 + r * 32 * 4, 32 * 4) != 0) ++bad;
          }
          if (bad != 0) fail_or_count("dequant rows 0..127 not bit-exact vs golden");
        }
      }
    }

    {
      std::uint32_t state = 42;
      for (std::uint32_t s = 0; s < 4; ++s) {
        const std::uint32_t a = xorshift(state), b = xorshift(state), c = xorshift(state);
        const std::uint32_t ids[3] = {a, b, c};
        const std::uint32_t id = ngram_hash_id(s, std::span<const std::uint32_t>(ids, 3), 256);
        if (id >= 256) fail_or_count("hash id out of range");
        const std::uint32_t again =
            ngram_hash_id(s, std::span<const std::uint32_t>(ids, 3), 256);
        if (again != id) fail_or_count("hash not deterministic");
      }
    }

    const std::vector<std::uint8_t> file = read_file(ngram_path);
    const std::uint64_t codes_base = g.shards[0].offset;
    const std::uint64_t scales_base = g.shards[0].scale_offset;

    // --- (b) exact oracle vs Op on the mini table (T=64) ---------------------
    {
      std::vector<std::uint32_t> ids(64);
      std::vector<float> W(4 * 128);
      std::uint32_t si = 1, wi = 2;
      for (auto& id : ids) id = xorshift(si) % 256;
      for (auto& w : W) w = static_cast<float>(xorshift(wi) >> 8) / 16777216.0f - 1.0f;

      NgramRequestState state;
      float* dev = nullptr;
      if (cudaMalloc(&dev, 64u * 128u * sizeof(float)) != cudaSuccess) die("cudaMalloc (b)");
      std::vector<float> got;
      size_t delta = 0;
      run_op(mini, ids, W, state, dev, &got, 0, &delta);
      cudaFree(dev);

      std::vector<double> ref;
      naive_reference(file, codes_base, scales_base, g.rows_per_shard(), g.block_rows, g.cols,
                      g.heads, g.rows_per_head(), ids, W, 4, 2, ref);
      check_close_enough(got, ref, "mini oracle (b)");
      if (delta > 16u * 1024u * 1024u) {
        fail_or_count("mini: device delta " + std::to_string(delta) + " B around the Op");
      }
    }

    // --- (c) ~68 MB synthetic store, 4 MiB staging cap (multi-chunk) ---------
    {
      const std::filesystem::path synth_path = out_dir / "p4_synthetic_store.ngram";
      const std::uint64_t rows = 2097152;  // 2^21
      write_synthetic_store(synth_path, rows, 32, 4, 128, 3);

      const NgramEmbeddingTable synth = NgramEmbeddingTable::open(synth_path);
      const NgramStoreInfo& s = synth.info();
      if (s.rows != rows || s.rows_per_head() != rows / 4 || s.embed_dim() != 128 ||
          s.file_bytes != std::filesystem::file_size(synth_path)) {
        fail_or_count("synthetic store geometry mismatch after open");
      }

      const std::vector<std::uint8_t> sfile = read_file(synth_path);
      const std::uint64_t scode = s.shards[0].offset;
      const std::uint64_t sscale = s.shards[0].scale_offset;

      std::vector<std::uint32_t> ids(16384);
      std::vector<float> W(4 * 128);
      std::uint32_t si = 5, wi = 6;
      for (auto& id : ids) id = xorshift(si) % 256;
      for (auto& w : W) w = static_cast<float>(xorshift(wi) >> 8) / 16777216.0f - 1.0f;

      NgramRequestState state;
      const std::size_t bytes = 16384u * 128u * sizeof(float);
      float* dev = nullptr;
      if (cudaMalloc(&dev, bytes) != cudaSuccess) die("cudaMalloc (c)");
      std::vector<float> got;
      size_t delta = 0;
      run_op(synth, ids, W, state, dev, &got, 4u * 1024u * 1024u, &delta);  // 4 MiB cap
      cudaFree(dev);

      std::vector<double> ref;
      naive_reference(sfile, scode, sscale, s.rows_per_shard(), s.block_rows, s.cols, s.heads,
                      s.rows_per_head(), ids, W, 4, 2, ref);
      check_close_enough(got, ref, "synthetic oracle (c)");
      if (delta > 16u * 1024u * 1024u) {
        fail_or_count("synthetic: device delta " + std::to_string(delta) +
                      " B around the Op (file " + std::to_string(s.file_bytes) + " B)");
      }
      std::printf("P4: 68 MB store, 4 MiB staging cap: device delta %zu B\n", delta);
    }

    // --- (d) prefill 64 + decode 1 + decode 1 == single 66 (bit-exact) -------
    {
      std::vector<std::uint32_t> ids(66);
      std::vector<float> W(4 * 128);
      std::uint32_t si = 7, wi = 8;
      for (auto& id : ids) id = xorshift(si) % 256;
      for (auto& w : W) w = static_cast<float>(xorshift(wi) >> 8) / 16777216.0f - 1.0f;

      const std::size_t bytes = 66u * 128u * sizeof(float);
      float* devA = nullptr;
      float* devB = nullptr;
      if (cudaMalloc(&devA, bytes) != cudaSuccess || cudaMalloc(&devB, bytes) != cudaSuccess) {
        die("cudaMalloc (d)");
      }
      NgramRequestState sA;
      std::vector<float> a(66u * 128u), b(66u * 128u);
      run_op(mini, std::vector<std::uint32_t>(ids.begin(), ids.begin() + 64), W, sA, devA,
             nullptr, 0, nullptr);
      // Per-call contract: each run writes its T rows at offset 0 of devA; the
      // test accumulates them into the request-ordered buffer `a`.
      if (cudaMemcpy(a.data(), devA, 64u * 128u * sizeof(float), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        die("D2H copy (d, 1)");
      }
      run_op(mini, std::vector<std::uint32_t>(ids.begin() + 64, ids.begin() + 65), W, sA, devA,
             nullptr, 0, nullptr);
      if (cudaMemcpy(a.data() + 64u * 128u, devA, 128u * sizeof(float), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        die("D2H copy (d, 2)");
      }
      run_op(mini, std::vector<std::uint32_t>(ids.begin() + 65, ids.begin() + 66), W, sA, devA,
             nullptr, 0, nullptr);
      if (cudaMemcpy(a.data() + 65u * 128u, devA, 128u * sizeof(float), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        die("D2H copy (d, 3)");
      }
      NgramRequestState sB;
      run_op(mini, ids, W, sB, devB, nullptr, 0, nullptr);
      if (cudaMemcpy(b.data(), devB, b.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        die("D2H copy (d, 4)");
      }
      cudaFree(devA);
      cudaFree(devB);
      if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
        fail_or_count("chunked 64+1+1 != single 66 (bit-exact expected)");
      }
    }

    // --- (f) binder integration: P2 handle -> view() -> adopt() --------------
    {
      using namespace ninfer::targets::qwen3_8_flash_next;
      BindResult br = bind_artifact(ninfer_path, ngram_path);
      if (!br.ple.opened()) fail_or_count("binder: PLE handle not opened");
      const std::uint8_t* view = br.ple.view();
      if (view == nullptr) fail_or_count("binder: PLE view() is null");
      const NgramEmbeddingTable via_binder = NgramEmbeddingTable::adopt(view, br.ple.bytes());
      if (via_binder.info().rows != mini.info().rows) {
        fail_or_count("binder-adopted store rows mismatch");
      }
      std::vector<float> r1(128), r2(128);
      std::uint32_t st = 99;
      int bad = 0;
      for (int i = 0; i < 32 && bad < 3; ++i) {
        const std::uint64_t row = xorshift(st) % 1024;
        mini.dequant_row(row, r1.data());
        via_binder.dequant_row(row, r2.data());
        if (std::memcmp(r1.data(), r2.data(), 128 * sizeof(float)) != 0) ++bad;
      }
      if (bad != 0) fail_or_count("binder-adopted rows differ from open(path) rows");
    }
  } catch (const std::exception& e) {
    die(e.what());
  }

  if (failures != 0) {
    std::fprintf(stderr, "%d ngram check(s) failed\n", failures);
    return 1;
  }
  std::printf("P4 ngram: all checks passed\n");
  return 0;
}
