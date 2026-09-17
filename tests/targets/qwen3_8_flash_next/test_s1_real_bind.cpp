// Flash-Next S1 gate (a) — real-geometry header-only bind of the real 180B
// artifact. Runs the real loader's bind phase (all 1608 objects at their
// Geometry-derived shapes) + the Binder's finish() invariant WITHOUT
// materialization (no GPU / VRAM). Asserts the explicit object counts and a
// sane backbone device footprint. Self-skips (77) if the real artifact is absent.

#include "targets/qwen3_8_flash_next/impl/load/real_bindings.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace {

std::filesystem::path real_artifact_path() {
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
  const char* env = std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS");
#pragma warning(pop)
  if (env && *env) return std::filesystem::path(env);
  return std::filesystem::path("C:/Users/Micke/Documents/Ninfer/models/qwen3_8_flash_next.ninfer");
}

}  // namespace

int main() {
  const std::filesystem::path path = real_artifact_path();
  if (!std::filesystem::exists(path)) {
    std::printf("S1 gate (a) SKIP: real artifact not found at %s\n", path.string().c_str());
    return 77;  // SKIP_RETURN_CODE
  }

  ninfer::targets::qwen3_8_flash_next::RealBindSummary s;
  try {
    s = ninfer::targets::qwen3_8_flash_next::real_bind_only(path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "S1 gate (a) FAIL (threw): %s\n", e.what());
    return 1;
  }

  const double gib = static_cast<double>(s.device_capacity_bytes) / (1024.0 * 1024.0 * 1024.0);
  std::printf("S1 gate (a) real_bind_only:\n");
  std::printf("  object_count          = %zu\n", s.object_count);
  std::printf("  resource_count        = %zu\n", s.resource_count);
  std::printf("  text_tensor_count     = %zu\n", s.text_tensor_count);
  std::printf("  nontext_tensor_count  = %zu\n", s.nontext_tensor_count);
  std::printf("  device_count          = %zu\n", s.device_count);
  std::printf("  validate_count        = %zu\n", s.validate_count);
  std::printf("  device_capacity_bytes = %llu (%.2f GiB)\n",
              static_cast<unsigned long long>(s.device_capacity_bytes), gib);

  bool ok = true;
  const auto check = [&ok](bool cond, const char* what) {
    if (!cond) {
      std::fprintf(stderr, "S1 gate (a) ASSERT FAIL: %s\n", what);
      ok = false;
    }
  };
  // finish() did not throw, so the consumed+placed invariant already holds.
  check(s.object_count == 1608, "object_count == 1608");
  check(s.resource_count == 6, "resource_count == 6");
  check(s.text_tensor_count == 1235, "text_tensor_count == 1235");
  check(s.nontext_tensor_count == 367, "nontext_tensor_count == 367");
  check(s.device_count == 1504, "device_count == 1504 (full GpuResident backbone)");
  check(s.validate_count == 104, "validate_count == 104 (6 resources + 98 HostExperts)");
  const unsigned long long kBackbone = 6021195832ull;  // P0 audit / P11 backbone_gpu (1504 objs)
  check(s.device_capacity_bytes >= kBackbone * 99 / 100 &&
            s.device_capacity_bytes <= kBackbone * 101 / 100,
        "device_capacity_bytes within 1% of 6,021,195,832 (6.02 GiB backbone)");

  if (!ok) return 1;
  std::printf(
      "S1 gate (a) PASS: real 180B artifact bound at real geometry "
      "(1608 objects, invariant holds, no GPU).\n");
  return 0;
}
