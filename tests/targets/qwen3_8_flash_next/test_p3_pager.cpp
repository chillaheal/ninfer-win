// P3: expert pager (no GEMM).
//
// * PrefillWindow stages a full layer of a 4x8 random-byte archive; device
//   bytes == host (archive) bytes for every expert.
// * DecodeFetch {0,3,7} three times: the H2D counter increases once (first
//   miss), the following two rounds are device hits.
// * Shape: 1 layer x 512 experts at the REAL packed per-expert size
//   (gate_up + down NVFP4 code + block scales), host window for all 512,
//   device window of 2 slots -- LRU eviction, checksums after eviction.
// * Failures: truncated archive (open throws), device slot OOM (stage refuses
//   and moves no bytes), overlapping stage (second in-flight stage detected).
//
// The archive is synthetic and self-contained (written by this test), so the
// pager contract is exercised without the mini fixture and without any MoE
// math.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <targets/qwen3_8_flash_next/paging/pager.h>

namespace {

using namespace ninfer::targets::qwen3_8_flash_next::paging;

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

// Writes an ExpertArchive file: header + contiguous blobs, PRNG-filled.
void write_archive(const std::filesystem::path& path, std::uint32_t layers,
                   std::uint32_t experts, std::uint64_t bytes_per_expert,
                   std::uint32_t seed, std::uint64_t payload_trim_bytes = 0) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) die("cannot write archive: " + path.string());

  std::uint8_t header[kArchiveHeaderBytes] = {};
  std::memcpy(header, kArchiveMagic, sizeof(kArchiveMagic));
  std::memcpy(header + 8, &layers, sizeof(layers));
  std::memcpy(header + 12, &experts, sizeof(experts));
  std::memcpy(header + 16, &bytes_per_expert, sizeof(bytes_per_expert));
  out.write(reinterpret_cast<const char*>(header), sizeof(header));

  const std::uint64_t total =
      static_cast<std::uint64_t>(layers) * experts * bytes_per_expert;
  const std::uint64_t payload = payload_trim_bytes > 0 && payload_trim_bytes < total
                                    ? total - payload_trim_bytes
                                    : total;
  std::uint32_t state = seed;
  const std::uint64_t chunk = 1 << 20;
  std::vector<std::uint8_t> buffer(chunk);
  for (std::uint64_t written = 0; written < payload;) {
    const std::uint64_t n = std::min(chunk, payload - written);
    for (std::uint64_t i = 0; i < n; i += 4) {
      const std::uint32_t word = xorshift(state);
      buffer[i]     = static_cast<std::uint8_t>(word);
      buffer[i + 1] = static_cast<std::uint8_t>(word >> 8);
      buffer[i + 2] = static_cast<std::uint8_t>(word >> 16);
      buffer[i + 3] = static_cast<std::uint8_t>(word >> 24);
    }
    out.write(reinterpret_cast<const char*>(buffer.data()),
              static_cast<std::streamsize>(n));
    written += n;
  }
  if (!out) die("archive write failed: " + path.string());
}

void check_bytes_match(const std::string& what, const std::span<const std::uint8_t>& device,
                       const std::span<const std::uint8_t>& expected) {
  if (device.size() != expected.size() ||
      std::memcmp(device.data(), expected.data(), device.size()) != 0) {
    fail_or_count(what + ": device bytes != host (archive) bytes");
  }
}

}  // namespace

int main() {
  const std::filesystem::path out_dir =
      std::filesystem::path(NINFER_SOURCE_DIR) / "out" / "flash_next_dev";
  if (!std::filesystem::exists(out_dir)) die("missing out/flash_next_dev");

  const std::filesystem::path mini_archive = out_dir / "p3_archive_4x8.bin";
  const std::filesystem::path shape_archive = out_dir / "p3_archive_1x512.bin";
  const std::filesystem::path truncated_archive = out_dir / "p3_archive_truncated.bin";

  // Real packed per-expert bytes: the production fused expert objects are
  // gate_up (655360 x 2560) + down (1310720 x 640), NVFP4 blockscale k16 --
  // per expert that is 1280 x 2560 and 2560 x 640, i.e. N*K/2 packed codes +
  // N*K/16 block-scale bytes each.
  const std::uint64_t gate_up_packed = 1280ull * 2560 / 2 + 1280ull * 2560 / 16;  // 1,843,200
  const std::uint64_t down_packed = 2560ull * 640 / 2 + 2560ull * 640 / 16;       // 921,600
  const std::uint64_t real_bytes_per_expert = gate_up_packed + down_packed;       // 2,764,800

  try {
    // 1. Mini 4x8 random archive: PrefillWindow stages layer 0 then 1.
    {
      write_archive(mini_archive, 4, 8, 4096, 0xC0FFEE);
      const ExpertArchive archive = ExpertArchive::open(mini_archive);
      if (archive.num_layers() != 4 || archive.experts_per_layer() != 8 ||
          archive.bytes_per_expert() != 4096) {
        fail_or_count("mini archive header mismatch");
      }

      PagerConfig config;
      config.host_slots = 32;
      config.device_slots = 32;
      Pager pager(archive, config);
      PrefillWindow prefill(pager);

      const std::size_t misses_0 = prefill.stage(0);
      prefill.wait();
      if (misses_0 != 8) {
        fail_or_count("prefill layer 0 misses = " + std::to_string(misses_0) + " (want 8)");
      }
      const std::size_t misses_1 = prefill.stage(1);
      prefill.wait();
      if (misses_1 != 8) {
        fail_or_count("prefill layer 1 misses = " + std::to_string(misses_1) + " (want 8)");
      }

      for (std::uint32_t layer = 0; layer < 2; ++layer) {
        for (std::uint32_t expert = 0; expert < 8; ++expert) {
          check_bytes_match("mini (layer " + std::to_string(layer) + ", expert " +
                                std::to_string(expert) + ")",
                            pager.device_bytes(layer, expert), archive.expert(layer, expert));
        }
      }
      const Counters c = pager.counters();
      if (c.misses != 16 || c.hits != 0 || c.h2d_bytes != 16ull * 4096 ||
          c.stall_ms < 0.0) {
        fail_or_count("mini counters mismatch (misses " + std::to_string(c.misses) +
                      ", hits " + std::to_string(c.hits) + ", bytes " +
                      std::to_string(c.h2d_bytes) + ")");
      }
      if (pager.pending() != 0) fail_or_count("mini: pending not drained after wait()");
    }

    // 2. DecodeFetch {0,3,7} three times: one H2D round, then two hit rounds.
    {
      const ExpertArchive archive = ExpertArchive::open(mini_archive);
      PagerConfig config;
      config.host_slots = 8;
      config.device_slots = 8;
      Pager pager(archive, config);
      DecodeFetch fetch(pager);
      const std::uint32_t ids[] = {0, 3, 7};

      const std::size_t first = fetch.stage(0, ids);
      fetch.wait();
      const std::size_t second = fetch.stage(0, ids);
      fetch.wait();
      const std::size_t third = fetch.stage(0, ids);
      fetch.wait();
      if (first != 3 || second != 0 || third != 0) {
        fail_or_count("decode misses per round = " + std::to_string(first) + "/" +
                      std::to_string(second) + "/" + std::to_string(third) +
                      " (want 3/0/0)");
      }
      const Counters c = pager.counters();
      if (c.misses != 3 || c.hits != 6 || c.h2d_bytes != 3ull * 4096) {
        fail_or_count("decode counters mismatch (misses " + std::to_string(c.misses) +
                      ", hits " + std::to_string(c.hits) + ", bytes " +
                      std::to_string(c.h2d_bytes) + ")");
      }
      for (const std::uint32_t id : ids) {
        check_bytes_match("decode expert " + std::to_string(id),
                          pager.device_bytes(0, id), archive.expert(0, id));
      }
    }

    // 3. Shape: 1 layer x 512 experts at the real packed size; host fits all
    //    512, the device window is 2 slots -> LRU churn with checksums.
    {
      write_archive(shape_archive, 1, 512, real_bytes_per_expert, 0xBADDCAFE);
      const ExpertArchive archive = ExpertArchive::open(shape_archive);
      if (archive.bytes_per_expert() != real_bytes_per_expert) {
        fail_or_count("shape archive bytes_per_expert mismatch");
      }
      PagerConfig config;
      config.host_slots = 512;
      config.device_slots = 2;
      Pager pager(archive, config);

      // Stage 0, 1, 2, 3 one at a time: 2, 3 evict 0, 1 (LRU).
      const std::vector<std::uint32_t> sequence = {0, 1, 2, 3};
      for (const std::uint32_t expert : sequence) {
        const StageStatus status = pager.stage(0, expert);
        if (status != StageStatus::Miss) {
          fail_or_count("shape stage expert " + std::to_string(expert) +
                        " is not a miss");
        }
        pager.wait();
      }
      check_bytes_match("shape expert 2 (resident)", pager.device_bytes(0, 2),
                        archive.expert(0, 2));
      check_bytes_match("shape expert 3 (resident)", pager.device_bytes(0, 3),
                        archive.expert(0, 3));
      bool expert_0_evicted = false;
      try {
        (void)pager.device_bytes(0, 0);
      } catch (const std::runtime_error&) {
        expert_0_evicted = true;
      }
      if (!expert_0_evicted) {
        fail_or_count("shape: expert 0 should have been evicted by experts 2, 3");
      }
      // Re-stage 0: it evicts the LRU (expert 2); then 0 hits on re-stage.
      if (pager.stage(0, 0) != StageStatus::Miss) {
        fail_or_count("shape: re-stage of evicted expert 0 is not a miss");
      }
      pager.wait();
      if (pager.stage(0, 0) != StageStatus::Hit) {
        fail_or_count("shape: re-staged expert 0 is not a device hit");
      }
      check_bytes_match("shape expert 0 (re-staged)", pager.device_bytes(0, 0),
                        archive.expert(0, 0));
      const Counters c = pager.counters();
      if (c.misses != 5 || c.hits != 1 || c.h2d_bytes != 5ull * real_bytes_per_expert) {
        fail_or_count("shape counters mismatch (misses " + std::to_string(c.misses) +
                      ", hits " + std::to_string(c.hits) + ", bytes " +
                      std::to_string(c.h2d_bytes) + ")");
      }
    }

    // 4. Failures: truncated archive, device slot OOM, overlapping stage.
    {
      // 4a. Truncated payload: the header claims 4x8x4096, the file is short.
      write_archive(truncated_archive, 4, 8, 4096, 0x1234,
                    /*payload_trim_bytes=*/8 * 1024);
      bool truncated_rejected = false;
      try {
        (void)ExpertArchive::open(truncated_archive);
      } catch (const std::runtime_error&) {
        truncated_rejected = true;
      }
      if (!truncated_rejected) {
        fail_or_count("truncated archive accepted");
      }

      // 4b. Header shorter than the fixed 40-byte prefix.
      const std::filesystem::path tiny = out_dir / "p3_archive_tiny.bin";
      {
        std::ofstream out(tiny, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(kArchiveMagic), 8);
      }
      bool tiny_rejected = false;
      try {
        (void)ExpertArchive::open(tiny);
      } catch (const std::runtime_error&) {
        tiny_rejected = true;
      }
      if (!tiny_rejected) {
        fail_or_count("sub-header archive accepted");
      }

      // 4c. Device slot OOM: the window slot is one byte short of the blob.
      const ExpertArchive archive = ExpertArchive::open(mini_archive);
      PagerConfig config;
      config.host_slots = 8;
      config.device_slots = 4;
      config.device_slot_bytes = archive.bytes_per_expert() - 1;
      Pager pager(archive, config);
      for (std::uint32_t expert = 0; expert < 4; ++expert) {
        if (pager.stage(0, expert) != StageStatus::DeviceSlotOom) {
          fail_or_count("device slot OOM not reported for expert " +
                        std::to_string(expert));
        }
      }
      const Counters c = pager.counters();
      if (c.misses != 0 || c.hits != 0 || c.h2d_bytes != 0) {
        fail_or_count("device slot OOM moved bytes (misses " + std::to_string(c.misses) +
                      ", bytes " + std::to_string(c.h2d_bytes) + ")");
      }

      // 4d. Overlapping stage: a second stage for a key still in flight.
      const ExpertArchive archive2 = ExpertArchive::open(mini_archive);
      PagerConfig config2;
      config2.host_slots = 8;
      config2.device_slots = 8;
      Pager pager2(archive2, config2);
      if (pager2.stage(0, 0) != StageStatus::Miss) {
        fail_or_count("overlap setup: first stage is not a miss");
      }
      if (pager2.pending() != 1) {
        fail_or_count("overlap setup: pending != 1 before wait()");
      }
      if (pager2.stage(0, 0) != StageStatus::OverlappingStage) {
        fail_or_count("overlapping stage not detected");
      }
      pager2.wait();
      if (pager2.stage(0, 0) != StageStatus::Hit) {
        fail_or_count("after wait(), the staged expert is not a hit");
      }
    }
  } catch (const std::exception& e) {
    die(e.what());
  }

  if (failures != 0) {
    std::fprintf(stderr, "%d pager check(s) failed\n", failures);
    return 1;
  }
  std::printf("P3 pager: all checks passed\n");
  return 0;
}
