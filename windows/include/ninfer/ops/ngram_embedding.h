#pragma once
// Op: ngram_embedding — hashed n-gram embedding (PLE) gather from the
// mmap'd .ngram store.
//
// The PLE table lives in the out-of-artifact `.ngram` sidecar (codec
// `fp8e4m3fn_row_bf16s-v1`: one E4M3FN code per value + one BF16 scale per
// row) and is 51.8 GB in the production artifact. It is never device-resident
// and never repacked at runtime: v1 I/O is mmap + OS page cache + pinned
// staging. Not io_uring, not O_DIRECT.
//
// Store layout (self-describing JSON directory after the 16-byte
// NNGRAM\x00\x01 prefix): rows are addressed `row = head * rows_per_head +
// ngram_id`; row content is `cols` E4M3FN codes with one BF16 scale per row.
// Codes are laid out block-major (`block_rows` contiguous rows per block) at
// the shard `offset`, scales in a separate plane at the shard `scale_offset`.
//
// Per-token math (request position p, 0-based over the whole request):
//   For each slot s in [0, heads):
//     bigram slot (s < heads/2):   ngram = (tok[p-1], tok[p])
//     trigram slot (s >= heads/2): ngram = (tok[p-2], tok[p-1], tok[p])
//     missing context (request start) = token id 0
//     id_s = ngram_hash_id(s, ngram, rows_per_head)
//     row_s = s * rows_per_head + id_s
//   emb[p][s * cols + c] = dequant(row_s)[c]          (dim = heads * cols)
//   out[p][c] = sum_{i=0..K-1} W[i][c] * emb[p - i*dil][c]
//   where emb[q] = 0 for q < 0.
//
// Geometry: production heads=16 (8 bigram + 8 trigram), rows_per_head =
// 20,000,096, cols=160 -> emb dim 2560 = hidden. Mini heads=4 (2+2),
// rows_per_head=256, cols=32 -> emb dim 128.
//
// Hash contract v1 (deterministic): FNV-1a 64 over the ngram token ids in
// oldest-first order with the per-slot offset basis
// `FNV64_OFFSET ^ slot * 0x9E3779B97F4A7C15`; id = digest % rows_per_head.
// The production hash is NOT recoverable (the checkpoint ships weights +
// config only, no modeling code) — this is the documented v1 contract
// (audit BLOCKER #2: RUNTIME KERNEL REQUIRED for the production row
// selection). rows_per_head comes from the store geometry, never hardcoded.
//
// Conv v1 contract: K=4 taps, dilation = ngram_size - 1 (= 2). Production
// `ple/convolution` is BF16 (4, 10240) = [tap][channel] with 10240 =
// 4 (hc_count) x 2560; the caller passes fp32 weights shaped [K][C] where
// C is the emb dim the Op produces (the HC-stream replication is a P9
// wiring decision).
//
// Consumer rule (P9 program wiring): layer 0 has no PLE; the PLE layer
// (kPleLayer) adds `out` to its input; the MTP layer disables PLE.
//
// Effects / memory:
//   - Hash, gather, dequant, and conv all run on the host over the page
//     cache; the only transfer is one H2D per staging chunk of the conv
//     output ([chunk * emb_dim] fp32) into the caller's device buffer.
//   - The Op performs NO device allocation (the table is never on the GPU);
//     it allocates one pinned staging buffer per call, sized to the chunk.
//   - `device_out` is [token_ids.size() * emb_dim] fp32, caller-owned,
//     stream-ordered on `stream`. Single concurrent request per table.
//
// NgramRequestState is request-local: the last two token ids (for hash
// context across chunks) and the conv history — the last (K-1)*dilation
// per-token dequantized embeddings (oldest first). A fresh state has no
// context: ids 0 and empty history (conv taps at q < 0 are 0).

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::ops {

struct NgramStoreInfo {
  std::uint32_t version = 0;
  std::string model_id;
  std::string quant_id;
  std::uint64_t rows = 0;          // rows_per_shard * n_shards
  std::uint32_t cols = 0;          // E4M3FN values per row
  std::uint32_t block_rows = 0;
  std::uint32_t ngram_size = 0;
  std::uint32_t heads = 0;         // ngram slots (production 16, mini 4)
  std::uint64_t file_bytes = 0;
  struct Shard {
    std::uint64_t offset = 0;         // code plane base (file offset)
    std::uint64_t scale_offset = 0;   // scale plane base (file offset)
    std::uint64_t bytes = 0;
  };
  std::vector<Shard> shards;
  [[nodiscard]] std::uint64_t rows_per_shard() const;
  [[nodiscard]] std::uint64_t rows_per_head() const;
  [[nodiscard]] std::uint32_t embed_dim() const { return heads * cols; }
};

// v1 hash contract (see header). `ids` is the ngram oldest-first (2 ids for
// bigram slots, 3 for trigram slots). rows_per_head must be > 0.
[[nodiscard]] std::uint32_t ngram_hash_id(std::uint32_t slot,
                                          std::span<const std::uint32_t> ids,
                                          std::uint64_t rows_per_head);

// RAII over the whole-file PAGE_READONLY mmap of a .ngram store.
class NgramEmbeddingTable {
public:
  // Opens, maps, and validates the store (magic, version 1, quant id,
  // directory geometry cross-checks). Throws std::runtime_error.
  static NgramEmbeddingTable open(const std::filesystem::path& path);
  // Adopts an already-mapped view (e.g. PleMmapHandle::view()); validates the
  // directory against `bytes`. Throws std::runtime_error. Does not unmap.
  static NgramEmbeddingTable adopt(const std::uint8_t* base, std::uint64_t bytes);

  NgramEmbeddingTable(NgramEmbeddingTable&& other) noexcept;
  NgramEmbeddingTable& operator=(NgramEmbeddingTable&& other) noexcept;
  NgramEmbeddingTable(const NgramEmbeddingTable&) = delete;
  NgramEmbeddingTable& operator=(const NgramEmbeddingTable&) = delete;
  ~NgramEmbeddingTable();

  [[nodiscard]] const NgramStoreInfo& info() const noexcept { return info_; }
  [[nodiscard]] const std::uint8_t* base() const noexcept { return base_; }

  // Packed row access (row in [0, rows)): the cols code bytes and the 2-byte
  // BF16 scale. Spans are valid until the table is destroyed.
  [[nodiscard]] std::span<const std::uint8_t> code_row(std::uint64_t row) const;
  [[nodiscard]] std::uint16_t scale_bits(std::uint64_t row) const;
  // Exact dequant into fp32 (E4M3FN -> fp32 is exact, the fp32 RN multiply
  // matches the converter's golden): out[c] = e4m3fn(code[c]) * f32(scale).
  void dequant_row(std::uint64_t row, float* out) const;

private:
  NgramEmbeddingTable();
  void init(const std::uint8_t* base, std::uint64_t bytes, bool owns_view);

  const std::uint8_t* base_ = nullptr;
  bool owns_view_ = false;
  NgramStoreInfo info_;
};

struct NgramRequestState {
  std::uint32_t prev2_id = 0;  // token two back (oldest); 0 = no context
  std::uint32_t prev1_id = 0;  // last token of the previous chunk
  bool has_context = false;    // set once any token has been processed
  std::uint64_t tokens_done = 0;  // absolute request position of the next token
  // Last (K-1)*dilation per-token dequantized embeddings, oldest first.
  std::vector<float> conv_history;
};

struct NgramOpParams {
  std::uint32_t conv_kernel = 4;
  std::uint32_t dilation = 0;                 // 0 = derived: ngram_size - 1
  const float* conv_weights = nullptr;        // [conv_kernel][emb_dim] fp32, host
  std::uint64_t staging_cap_bytes = 0;        // 0 = single unbounded chunk
};

// Gathers + dequants + convs one token chunk (see header for the full math).
// Updates `state` (ids + conv history) and writes the conv output
// ([token_ids.size() * emb_dim] fp32) to `device_out` via H2D on `stream`.
void ngram_embedding(const NgramEmbeddingTable& table,
                     std::span<const std::uint32_t> token_ids,
                     const NgramOpParams& params,
                     NgramRequestState& state,
                     float* device_out,
                     cudaStream_t stream);

}  // namespace ninfer::ops
