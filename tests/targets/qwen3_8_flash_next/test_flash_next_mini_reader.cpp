// P1: mini Flash-Next fixture reader.
//
// * Opens the mini .ninfer with the production artifact Reader: identity,
//   object count/names/order, every payload readable at its planned size,
//   NVFP4 payload geometry for the mini expert block.
// * Mmaps the mini .ngram PLE store (Win32, lazy commit) and decodes block 0
//   bit-for-bit against the checked-in golden -- proving the (layer, row,
//   col) gather addressing the P4 Op will use, without committing the table.
//
// Skips (exit 77) when the fixture files are missing; generate them with
// `python -m tools.convert.qwen3_8_flash_next.make_mini_artifact`.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <windows.h>
#include <psapi.h>

#include <nlohmann/json.hpp>

#include "artifact/reader.h"

namespace {

using ninfer::artifact::BlockScaleGeometry;
using ninfer::artifact::NumericFormat;
using ninfer::artifact::Reader;
using ninfer::artifact::TensorDescriptor;
using ninfer::artifact::block_scale_geometry;
using ninfer::artifact::object_bytes;
using ninfer::artifact::object_name;

[[noreturn]] void fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  std::exit(1);
}

// Exact E4M3FN -> f32 (the codec's decode contract; NaN pattern is an error).
// Exponent field is the 4 bits at [6:3] -- masking with &7 (3 bits) misflags
// every exp-7 mant-7 byte (e.g. 0xBF = -1.875) as NaN.
float decode_e4m3_fn(std::uint8_t b, std::uint64_t row, std::uint64_t col) {
  const int sign = b >> 7;
  const int exp = (b >> 3) & 0xF;
  const int mant = b & 7;
  if (exp == 15 && mant == 7) {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "E4M3FN NaN pattern in PLE codes at row %llu col %llu (byte 0x%02X)",
                  static_cast<unsigned long long>(row),
                  static_cast<unsigned long long>(col), static_cast<unsigned>(b));
    fail(buf);
  }
  float value;
  if (exp == 0) {
    value = static_cast<float>(mant) * 0.001953125f;  // subnormal: m * 2^-9
  } else {
    value = static_cast<float>(8 + mant) * std::ldexpf(1.0f, exp - 10);
  }
  return sign ? -value : value;
}

// Exact bf16 -> f32 (bit replicate; no rounding, no special cases).
float bf16_to_f32(std::uint16_t u) {
  const std::uint32_t bits = static_cast<std::uint32_t>(u) << 16;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint64_t working_set_bytes() {
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    fail("GetProcessMemoryInfo");
  }
  return counters.WorkingSetSize;
}

void check_artifact(const std::filesystem::path& path) {
  Reader reader(path);
  const auto& identity = reader.identity();
  if (identity.model_id != "qwen3.8-flash-next") {
    fail("mini identity model_id = " + identity.model_id);
  }
  if (identity.weights_id != "nvfp4") {
    fail("mini identity weights_id = " + identity.weights_id);
  }
  const auto& objects = reader.objects();
  if (objects.size() != 119) {
    fail("mini object count = " + std::to_string(objects.size()) + " (want 119)");
  }
  if (object_name(objects.front()) != "frontend/tokenizer.json") {
    fail("first object = " + std::string(object_name(objects.front())));
  }
  if (object_name(objects.back()) != "text/output_head") {
    fail("last object = " + std::string(object_name(objects.back())));
  }
  for (const auto& object : objects) {
    const auto span = reader.payload(object);
    if (span.data.size() != object_bytes(object)) {
      fail(std::string("payload size mismatch for ") +
           std::string(object_name(object)));
    }
  }

  const auto* gdn = reader.find("text/layers/0/gdn/query_key_value_z");
  const auto* gdn_tensor = gdn ? std::get_if<TensorDescriptor>(gdn) : nullptr;
  if (!gdn_tensor || gdn_tensor->shape != std::vector<std::uint64_t>{384, 256} ||
      gdn_tensor->format != NumericFormat::FP8_E4M3FN_ROW_BF16S) {
    fail("gdn/query_key_value_z descriptor mismatch");
  }

  const auto* gate_up = reader.find("text/layers/0/mlp/experts/gate_up");
  const auto* gate_up_tensor = gate_up ? std::get_if<TensorDescriptor>(gate_up) : nullptr;
  if (!gate_up_tensor || gate_up_tensor->shape != std::vector<std::uint64_t>{1024, 256} ||
      gate_up_tensor->format != NumericFormat::NVFP4) {
    fail("mlp/experts/gate_up descriptor mismatch");
  }
  const auto shape = gate_up_tensor->shape;
  const BlockScaleGeometry geometry = block_scale_geometry(NumericFormat::NVFP4, shape);
  if (object_bytes(*gate_up) != geometry.encoded_bytes) {
    fail("NVFP4 payload size != blockscale geometry");
  }
  // (layer, expert_id) slicing contract: per-expert stride is a 128-row tile
  const std::uint64_t rows_per_expert = shape[0] / 8;  // mini: 8 experts
  if (rows_per_expert % 128 != 0) {
    fail("per-expert stride is not a 128-row blockscale N-tile multiple");
  }
}

void check_ngram_block0(const std::filesystem::path& ngram_path,
                        const std::filesystem::path& golden_path) {
  const std::uint64_t before = working_set_bytes();

  HANDLE file_handle =
      CreateFileW(ngram_path.wstring().c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                  FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file_handle == INVALID_HANDLE_VALUE) fail("CreateFileW(.ngram)");
  LARGE_INTEGER file_size{};
  if (!GetFileSizeEx(file_handle, &file_size)) fail("GetFileSizeEx(.ngram)");
  HANDLE mapping =
      CreateFileMappingW(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) fail("CreateFileMappingW(.ngram)");
  const auto* base = static_cast<const std::uint8_t*>(
      MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
  if (base == nullptr) fail("MapViewOfFile(.ngram)");

  std::vector<float> decoded;
  std::uint64_t rows = 0, cols = 0;
  {
    if (std::memcmp(base, "NNGRAM\0\1", 8) != 0) fail("ngram magic");
    std::uint64_t json_bytes = 0;
    std::memcpy(&json_bytes, base + 8, sizeof(json_bytes));  // x86/x64 LE
    if (16 + json_bytes > static_cast<std::uint64_t>(file_size.QuadPart)) {
      fail("ngram directory exceeds file");
    }
    const auto dir = nlohmann::json::parse(reinterpret_cast<const char*>(base + 16),
                                           reinterpret_cast<const char*>(base + 16 + json_bytes));
    const auto& shard = dir["shards"][0];
    const auto& block = dir["block"];
    const std::uint64_t code_offset = shard["offset"].get<std::uint64_t>();
    const std::uint64_t scale_offset = shard["scale_offset"].get<std::uint64_t>();
    rows = block["rows"].get<std::uint64_t>();
    cols = block["cols"].get<std::uint64_t>();
    const std::uint64_t block_codes = block["block_codes_bytes"].get<std::uint64_t>();
    const std::uint64_t block_scales = block["block_scale_bytes"].get<std::uint64_t>();
    if (block_codes % 4096 != 0) fail("block codes not 4096-aligned sized");
    if (code_offset + block_codes > static_cast<std::uint64_t>(file_size.QuadPart) ||
        scale_offset + block_scales > static_cast<std::uint64_t>(file_size.QuadPart)) {
      fail("block 0 out of range");
    }
    // Touch ONLY block 0: codes + scales (lazy-commit gather contract).
    const auto* codes = base + code_offset;
    const auto* scales = base + scale_offset;
    decoded.resize(rows * cols);
    for (std::uint64_t row = 0; row < rows; ++row) {
      const float scale = bf16_to_f32(
          *reinterpret_cast<const std::uint16_t*>(scales + row * 2));
      for (std::uint64_t col = 0; col < cols; ++col) {
        decoded[row * cols + col] =
            decode_e4m3_fn(codes[row * cols + col], row, col) * scale;
      }
    }
  }
  UnmapViewOfFile(base);
  CloseHandle(mapping);
  CloseHandle(file_handle);

  const std::uint64_t after = working_set_bytes();
  if (after > before && after - before > 1024 * 1024) {
    fail("PLE mmap committed more than 1 MiB for a single block read");
  }

  std::ifstream golden(golden_path, std::ios::binary);
  if (!golden) fail("cannot open golden");
  std::uint8_t magic[8];
  int rows_i = 0, cols_i = 0;
  golden.read(reinterpret_cast<char*>(magic), 8);
  golden.read(reinterpret_cast<char*>(&rows_i), 4);
  golden.read(reinterpret_cast<char*>(&cols_i), 4);
  if (std::memcmp(magic, "PLEGOLDN", 8) != 0) fail("golden magic");
  if (static_cast<std::uint64_t>(rows_i) != rows || static_cast<std::uint64_t>(cols_i) != cols) {
    fail("golden shape mismatch");
  }
  std::vector<float> expected(rows * cols);
  golden.read(reinterpret_cast<char*>(expected.data()),
              static_cast<std::streamsize>(expected.size() * sizeof(float)));
  if (!golden) fail("golden body short");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::memcmp(&decoded[i], &expected[i], sizeof(float)) != 0) {
      fail("PLE block 0 decode diverges from golden at element " + std::to_string(i));
    }
  }
}

}  // namespace

int main() {
  const std::filesystem::path root = NINFER_SOURCE_DIR;
  const std::filesystem::path out = root / "out" / "flash_next_dev";
  const std::filesystem::path ninfer_path = out / "qwen3_8_flash_next_mini.ninfer";
  const std::filesystem::path ngram_path = out / "qwen3_8_flash_next_mini.ngram";
  const std::filesystem::path golden_path = out / "qwen3_8_flash_next_mini.ple_golden.bin";
  if (!std::filesystem::exists(ninfer_path) || !std::filesystem::exists(ngram_path) ||
      !std::filesystem::exists(golden_path)) {
    return 77;  // fixture not built; generate with make_mini_artifact
  }
  try {
    check_artifact(ninfer_path);
    check_ngram_block0(ngram_path, golden_path);
  } catch (const std::exception& e) {
    fail(e.what());
  }
  return 0;
}
