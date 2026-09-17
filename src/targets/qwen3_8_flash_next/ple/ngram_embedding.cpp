#include "ninfer/ops/ngram_embedding.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <windows.h>

namespace {

const std::uint8_t kNgramMagic[8] = {'N', 'N', 'G', 'R', 'A', 'M', 0, 1};
constexpr std::uint64_t kPrefixBytes = 16;

[[noreturn]] void fail(std::string message) {
  throw std::runtime_error("ngram table: " + std::move(message));
}

void cuda_or_throw(cudaError_t error, const char* what) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
  }
}

// --- E4M3FN / BF16 exact decodes ------------------------------------------------

// 256-entry exact E4M3FN -> float table (4-bit exponent, bias 7, no inf:
// 0x7F/0xFF are NaN, max finite 448.0 = 0x7E).
const std::array<float, 256>& e4m3fn_table() {
  static const std::array<float, 256> table = [] {
    std::array<float, 256> t{};
    for (int code = 0; code < 256; ++code) {
      const int s = code >> 7;
      const int e = (code >> 3) & 0xF;
      const int m = code & 7;
      float v;
      if (e == 0) {
        v = std::ldexpf(static_cast<float>(m), -9);  // m * 2^-9
      } else if (e == 15 && m == 7) {
        v = std::nanf("");
      } else {
        v = std::ldexpf(static_cast<float>(8 + m), e - 10);  // (8+m) * 2^(e-10)
      }
      t[static_cast<std::size_t>(code)] = s ? -v : v;
    }
    return t;
  }();
  return table;
}

float bf16_to_f32(std::uint16_t bits) {
  std::uint32_t f = static_cast<std::uint32_t>(bits) << 16;
  float v;
  std::memcpy(&v, &f, sizeof(v));
  return v;
}

// --- directory extraction ---------------------------------------------------------
// The directory is machine-generated (json.dumps sort_keys, separators
// (",",":")) — a strict, whitespace-free extractor over the known schema.

struct DirScan {
  const std::uint8_t* p;
  std::size_t n;
};

const std::uint8_t* find_key(const DirScan& d, const char* key) {
  const std::size_t klen = std::strlen(key);
  const std::size_t pat_len = klen + 3;  // "key":
  for (std::size_t i = 0; i + pat_len <= d.n; ++i) {
    if (d.p[i] == '"' && std::memcmp(d.p + i + 1, key, klen) == 0 &&
        d.p[i + klen + 1] == '"' && d.p[i + klen + 2] == ':') {
      return d.p + i + klen + 3;
    }
  }
  return nullptr;
}

bool scan_uint(const DirScan& d, const char* key, std::uint64_t* out) {
  const std::uint8_t* q = find_key(d, key);
  if (q == nullptr) return false;
  std::uint64_t v = 0;
  std::size_t digits = 0;
  while (q < d.p + d.n && *q >= '0' && *q <= '9') {
    v = v * 10 + static_cast<std::uint64_t>(*q - '0');
    ++q;
    ++digits;
  }
  if (digits == 0) return false;
  *out = v;
  return true;
}

bool scan_string(const DirScan& d, const char* key, std::string* out) {
  const std::uint8_t* q = find_key(d, key);
  if (q == nullptr || q >= d.p + d.n || *q != '"') return false;
  ++q;
  const std::uint8_t* start = q;
  while (q < d.p + d.n && *q != '"') ++q;
  if (q >= d.p + d.n) return false;
  *out = std::string(reinterpret_cast<const char*>(start), q - start);
  return true;
}

}  // namespace

namespace ninfer::ops {

std::uint64_t NgramStoreInfo::rows_per_shard() const {
  return shards.empty() ? 0 : rows / shards.size();
}

std::uint64_t NgramStoreInfo::rows_per_head() const {
  return heads == 0 ? 0 : rows / heads;
}

std::uint32_t ngram_hash_id(std::uint32_t slot, std::span<const std::uint32_t> ids,
                            std::uint64_t rows_per_head) {
  if (rows_per_head == 0) throw std::invalid_argument("ngram_hash_id: rows_per_head is 0");
  constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ull;
  constexpr std::uint64_t kFnvPrime = 0x00000100000001B3ull;
  std::uint64_t h = kFnvOffset ^ (static_cast<std::uint64_t>(slot) * 0x9E3779B97F4A7C15ull);
  for (const std::uint32_t id : ids) {
    h = (h ^ id) * kFnvPrime;
  }
  return static_cast<std::uint32_t>(h % rows_per_head);
}

NgramEmbeddingTable::NgramEmbeddingTable() = default;

NgramEmbeddingTable::NgramEmbeddingTable(NgramEmbeddingTable&& other) noexcept
    : base_(other.base_), owns_view_(other.owns_view_), info_(std::move(other.info_)) {
  other.base_ = nullptr;
  other.owns_view_ = false;
}

NgramEmbeddingTable& NgramEmbeddingTable::operator=(NgramEmbeddingTable&& other) noexcept {
  if (this != &other) {
    if (owns_view_ && base_ != nullptr) UnmapViewOfFile(base_);
    base_ = other.base_;
    owns_view_ = other.owns_view_;
    info_ = std::move(other.info_);
    other.base_ = nullptr;
    other.owns_view_ = false;
  }
  return *this;
}

NgramEmbeddingTable::~NgramEmbeddingTable() {
  if (owns_view_ && base_ != nullptr) UnmapViewOfFile(base_);
}

void NgramEmbeddingTable::init(const std::uint8_t* base, std::uint64_t bytes, bool owns_view) {
  if (bytes < kPrefixBytes + 8) fail("file too small for prefix + magic");
  if (std::memcmp(base, kNgramMagic, 8) != 0) fail("bad ngram magic");
  std::uint64_t json_bytes = 0;
  std::memcpy(&json_bytes, base + 8, sizeof(json_bytes));
  if (json_bytes == 0 || kPrefixBytes + json_bytes > bytes) {
    fail("directory out of range");
  }

  DirScan d{base + kPrefixBytes, static_cast<std::size_t>(json_bytes)};
  NgramStoreInfo info;
  std::uint64_t n_shards = 0, rows_per_shard = 0, cols = 0, block_rows = 0;
  std::uint64_t code_row_bytes = 0, block_scale_bytes = 0, block_codes_bytes = 0;
  std::uint64_t scale_plane_offset = 0;
  std::uint64_t version = 0;
  if (!scan_uint(d, "version", &version) || version != 1) fail("unsupported version");
  info.version = static_cast<std::uint32_t>(version);
  if (!scan_string(d, "model_id", &info.model_id)) fail("missing model_id");
  if (!scan_string(d, "quant_id", &info.quant_id)) fail("missing quant_id");
  if (info.quant_id != "fp8e4m3fn_row_bf16s-v1") {
    fail("unsupported quant_id '" + info.quant_id + "'");
  }
  if (!scan_uint(d, "n_shards", &n_shards) || n_shards == 0) fail("bad n_shards");
  if (!scan_uint(d, "rows_per_shard", &rows_per_shard) || rows_per_shard == 0) {
    fail("bad rows_per_shard");
  }
  if (!scan_uint(d, "cols_per_shard", &cols) || cols == 0) fail("bad cols_per_shard");
  if (!scan_uint(d, "block_rows", &block_rows) || block_rows == 0) fail("bad block_rows");
  if (!scan_uint(d, "code_row_bytes", &code_row_bytes) || code_row_bytes != cols) {
    fail("code_row_bytes != cols_per_shard");
  }
  if (!scan_uint(d, "block_scale_bytes", &block_scale_bytes) ||
      block_scale_bytes != static_cast<std::uint64_t>(block_rows) * 2) {
    fail("block_scale_bytes != block_rows*2");
  }
  if (!scan_uint(d, "block_codes_bytes", &block_codes_bytes) ||
      block_codes_bytes != static_cast<std::uint64_t>(block_rows) * cols) {
    fail("block_codes_bytes != block_rows*cols");
  }
  if (!scan_uint(d, "scale_plane_offset", &scale_plane_offset)) fail("missing scale_plane_offset");

  const std::uint8_t* ngram_key = find_key(d, "ngram");
  if (ngram_key == nullptr) fail("missing ngram metadata");
  // The ngram object is a few integer keys; scan within a bounded window.
  DirScan ng{
      ngram_key,
      static_cast<std::size_t>(std::min<std::uint64_t>(256, d.n - (ngram_key - d.p)))};
  std::uint64_t ngram_size = 0, heads = 0;
  if (!scan_uint(ng, "size", &ngram_size) || ngram_size < 2) fail("bad ngram size");
  info.ngram_size = static_cast<std::uint32_t>(ngram_size);
  if (!scan_uint(ng, "heads", &heads) || heads < 2 || heads % 2 != 0) {
    fail("bad ngram heads");
  }
  info.heads = static_cast<std::uint32_t>(heads);

  // shards array: n_shards flat objects of "id"/"offset"/"scale_offset"/"bytes".
  const std::uint8_t* cursor = find_key(d, "shards");
  if (cursor == nullptr) fail("missing shards");
  for (std::uint64_t s = 0; s < n_shards; ++s) {
    while (cursor < d.p + d.n && *cursor != '{') ++cursor;
    if (cursor >= d.p + d.n) fail("truncated shards array");
    const std::uint8_t* obj_start = cursor;
    while (cursor < d.p + d.n && *cursor != '}') ++cursor;
    if (cursor >= d.p + d.n) fail("truncated shard " + std::to_string(s));
    DirScan obj{obj_start, static_cast<std::size_t>(cursor - obj_start + 1)};
    NgramStoreInfo::Shard shard;
    std::uint64_t id = 0;
    if (!scan_uint(obj, "id", &id) || !scan_uint(obj, "offset", &shard.offset) ||
        !scan_uint(obj, "scale_offset", &shard.scale_offset) ||
        !scan_uint(obj, "bytes", &shard.bytes)) {
      fail("truncated shard " + std::to_string(s));
    }
    if (id != s) fail("shard id out of order");
    if (shard.scale_offset < shard.offset) fail("shard scale_offset before offset");
    if (shard.offset + shard.bytes > bytes) fail("shard " + std::to_string(s) + " out of range");
    if (shard.scale_offset + static_cast<std::uint64_t>(block_rows) * 2 >
        shard.offset + shard.bytes) {
      fail("shard " + std::to_string(s) + " scale plane out of range");
    }
    info.shards.push_back(shard);
    ++cursor;
  }

  info.rows = rows_per_shard * n_shards;
  info.cols = static_cast<std::uint32_t>(cols);
  info.block_rows = static_cast<std::uint32_t>(block_rows);
  info.file_bytes = bytes;
  if (info.rows % info.heads != 0) fail("rows not divisible by heads");

  base_ = base;
  owns_view_ = owns_view;
  info_ = std::move(info);
}

NgramEmbeddingTable NgramEmbeddingTable::open(const std::filesystem::path& path) {
  const HANDLE file =
      CreateFileW(path.wstring().c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                  FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    fail("cannot open '" + path.string() + "' (error " + std::to_string(GetLastError()) + ")");
  }
  LARGE_INTEGER file_size{};
  if (!GetFileSizeEx(file, &file_size)) {
    CloseHandle(file);
    fail("GetFileSizeEx failed for '" + path.string() + "'");
  }
  const HANDLE mapping =
      CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    CloseHandle(file);
    fail("CreateFileMappingW failed for '" + path.string() + "' (error " +
         std::to_string(GetLastError()) + ")");
  }
  CloseHandle(file);
  const void* base = MapViewOfFileEx(mapping, FILE_MAP_READ, 0, 0, 0, 0);
  if (base == nullptr) {
    CloseHandle(mapping);
    fail("MapViewOfFileEx failed for '" + path.string() + "' (error " +
         std::to_string(GetLastError()) + ")");
  }
  CloseHandle(mapping);  // the view keeps the mapping alive
  NgramEmbeddingTable table;
  table.init(static_cast<const std::uint8_t*>(base),
             static_cast<std::uint64_t>(file_size.QuadPart), /*owns_view=*/true);
  return table;
}

NgramEmbeddingTable NgramEmbeddingTable::adopt(const std::uint8_t* base, std::uint64_t bytes) {
  NgramEmbeddingTable table;
  table.init(base, bytes, /*owns_view=*/false);
  return table;
}

std::span<const std::uint8_t> NgramEmbeddingTable::code_row(std::uint64_t row) const {
  const std::uint64_t rps = info_.rows_per_shard();
  const std::uint64_t shard_idx = row / rps;
  const std::uint64_t row_in_shard = row % rps;
  if (shard_idx >= info_.shards.size()) throw std::invalid_argument("ngram row out of range");
  const std::uint64_t block = row_in_shard / info_.block_rows;
  const std::uint64_t row_in_block = row_in_shard % info_.block_rows;
  const std::uint64_t code_bytes = static_cast<std::uint64_t>(info_.block_rows) * info_.cols;
  const std::uint8_t* p =
      base_ + info_.shards[shard_idx].offset + block * code_bytes + row_in_block * info_.cols;
  return std::span<const std::uint8_t>(p, info_.cols);
}

std::uint16_t NgramEmbeddingTable::scale_bits(std::uint64_t row) const {
  const std::uint64_t rps = info_.rows_per_shard();
  const std::uint64_t shard_idx = row / rps;
  const std::uint64_t row_in_shard = row % rps;
  if (shard_idx >= info_.shards.size()) throw std::invalid_argument("ngram row out of range");
  const std::uint64_t block = row_in_shard / info_.block_rows;
  const std::uint64_t row_in_block = row_in_shard % info_.block_rows;
  const std::uint8_t* p = base_ + info_.shards[shard_idx].scale_offset +
                          block * static_cast<std::uint64_t>(info_.block_rows) * 2 +
                          row_in_block * 2;
  std::uint16_t bits = 0;
  std::memcpy(&bits, p, sizeof(bits));
  return bits;
}

void NgramEmbeddingTable::dequant_row(std::uint64_t row, float* out) const {
  const std::span<const std::uint8_t> codes = code_row(row);
  const float scale = bf16_to_f32(scale_bits(row));
  const auto& table = e4m3fn_table();
  for (std::uint32_t c = 0; c < info_.cols; ++c) {
    out[c] = table[codes[c]] * scale;
  }
}

void ngram_embedding(const NgramEmbeddingTable& table,
                     std::span<const std::uint32_t> token_ids,
                     const NgramOpParams& params,
                     NgramRequestState& state,
                     float* device_out,
                     cudaStream_t stream) {
  const NgramStoreInfo& info = table.info();
  const std::uint32_t heads = info.heads;
  const std::uint64_t rph = info.rows_per_head();
  const std::uint32_t cols = info.cols;
  const std::uint32_t dim = info.embed_dim();
  const std::uint32_t K = params.conv_kernel;
  const std::uint32_t dil = params.dilation != 0 ? params.dilation : info.ngram_size - 1;
  if (K == 0 || dil == 0) throw std::invalid_argument("ngram_embedding: bad kernel/dilation");
  if (params.conv_weights == nullptr) {
    throw std::invalid_argument("ngram_embedding: missing conv weights");
  }
  const std::uint64_t T = token_ids.size();
  if (T == 0) return;

  const std::uint64_t history_len = static_cast<std::uint64_t>(K - 1) * dil;
  const std::uint64_t bytes_per_token = static_cast<std::uint64_t>(dim) * 4;
  const std::uint64_t chunk = params.staging_cap_bytes != 0
                                  ? std::max<std::uint64_t>(
                                        1, params.staging_cap_bytes / (2 * bytes_per_token))
                                  : T;

  for (std::uint64_t c0 = 0; c0 < T; c0 += chunk) {
    const std::uint64_t n = std::min(chunk, T - c0);
    const std::uint64_t bytes = n * bytes_per_token;
    float* e_stage = nullptr;
    float* o_stage = nullptr;
    cuda_or_throw(cudaHostAlloc(&e_stage, bytes, cudaHostAllocDefault), "cudaHostAlloc emb");
    cuda_or_throw(cudaHostAlloc(&o_stage, bytes, cudaHostAllocDefault), "cudaHostAlloc out");

    // Gather + dequant: per token, per slot, hash -> row -> exact dequant.
    for (std::uint64_t t = 0; t < n; ++t) {
      const std::uint32_t x = token_ids[c0 + t];
      const std::uint32_t prev1 =
          (t == 0 && !state.has_context) ? 0 : (t == 0 ? state.prev1_id : token_ids[c0 + t - 1]);
      const std::uint32_t prev2 = (t == 0)
                                      ? (state.has_context ? state.prev2_id : 0)
                                      : (t == 1 ? (state.has_context ? state.prev1_id : 0)
                                                : token_ids[c0 + t - 2]);
      const std::uint32_t bi[2] = {prev1, x};
      const std::uint32_t tri[3] = {prev2, prev1, x};
      for (std::uint32_t s = 0; s < heads; ++s) {
        const std::span<const std::uint32_t> ng =
            (s < heads / 2) ? std::span<const std::uint32_t>(bi, 2)
                            : std::span<const std::uint32_t>(tri, 3);
        const std::uint32_t id = ngram_hash_id(s, ng, rph);
        table.dequant_row(static_cast<std::uint64_t>(s) * rph + id,
                          e_stage + (t * heads + s) * cols);
      }
    }

    // Conv: out[t][c] = sum_i W[i][c] * emb[t - i*dil][c], call-local
    // positions. A tap landing inside this chunk reads e_stage; a tap
    // landing earlier (previous chunk of this call, or tokens of the
    // previous call, distance from the newest history row = back - t)
    // reads the request conv history; a tap before the first token of the
    // request is 0.
    for (std::uint64_t t = 0; t < n; ++t) {
      const std::uint64_t p = c0 + t;
      for (std::uint32_t c = 0; c < dim; ++c) {
        float acc = 0.0f;
        for (std::uint32_t i = 0; i < K; ++i) {
          const std::uint64_t back = static_cast<std::uint64_t>(i) * dil;
          const float e = [&] {
            const std::uint64_t have = state.conv_history.size() / dim;
            if (p < back) {
              // Tap target is before this call: only in history when the
              // request has at least `back - p` tokens of history.
              const std::uint64_t dist = back - p;
              return (dist <= have) ? state.conv_history[(have - dist) * dim + c] : 0.0f;
            }
            const std::uint64_t q = p - back;
            if (q >= c0) return e_stage[(q - c0) * dim + c];
            // Previous chunk of this call: distance from the newest history
            // row is c0 - q = back - t.
            const std::uint64_t dist = back - t;
            return (dist <= have) ? state.conv_history[(have - dist) * dim + c] : 0.0f;
          }();
          acc += params.conv_weights[static_cast<std::uint64_t>(i) * dim + c] * e;
        }
        o_stage[t * dim + c] = acc;
      }
    }

    cuda_or_throw(cudaMemcpyAsync(device_out + c0 * dim, o_stage, bytes,
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync H2D");

    // Request state: last two ids + conv history (last history_len emb rows,
    // oldest first). Runs while e_stage is still mapped (it is freed below).
    state.prev2_id = (n >= 2) ? token_ids[c0 + n - 2] : state.prev1_id;
    state.prev1_id = token_ids[c0 + n - 1];
    state.has_context = true;
    state.tokens_done += n;
    // Oldest first: one ascending insert of the whole chunk (a reverse-order
    // append would break the (have - dist) tap indexing).
    state.conv_history.insert(state.conv_history.end(), e_stage,
                              e_stage + n * dim);
    if (state.conv_history.size() > history_len * dim) {
      state.conv_history.erase(state.conv_history.begin(),
                               state.conv_history.begin() +
                                   (state.conv_history.size() - history_len * dim));
    }
    cuda_or_throw(cudaFreeHost(e_stage), "cudaFreeHost emb");
    cuda_or_throw(cudaFreeHost(o_stage), "cudaFreeHost out");
  }
}

}  // namespace ninfer::ops
