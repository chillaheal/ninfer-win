// P2: identity resolution.
//
// * The six registered (model_id, weights_id) pairs resolve to their target
//   keys via the GPU-free registry resolver (5 existing + qwen3.8-flash-next).
// * Unregistered pairs resolve to std::nullopt.
// * The mini fixture's on-disk identity resolves to qwen3_8_flash_next.
//
// Skips (exit 77) when the mini fixture is missing; generate it with
// `python -m tools.convert.qwen3_8_flash_next.make_mini_artifact`.

#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include "artifact/reader.h"
#include "targets/registry.h"

namespace {

using ArtifactIdentity = ninfer::artifact::ArtifactIdentity;
using FlashNextPackage = ninfer::targets::qwen3_8_flash_next::Package;
using ninfer::targets::resolve_target_key;

int failures = 0;

void check(std::string_view model_id, std::string_view weights_id,
           std::string_view expected_key) {
    const auto key =
        resolve_target_key(ArtifactIdentity{std::string(model_id), std::string(weights_id)});
    if (expected_key.empty()) {
        if (key.has_value()) {
            std::fprintf(stderr, "FAIL: %.*s/%.*s resolved to '%.*s' (want unregistered)\n",
                         (int)model_id.size(), model_id.data(), (int)weights_id.size(),
                         weights_id.data(), (int)key->size(), key->data());
            ++failures;
        }
        return;
    }
    if (!key.has_value() || *key != expected_key) {
        std::fprintf(stderr, "FAIL: %.*s/%.*s -> %s (want '%.*s')\n", (int)model_id.size(),
                     model_id.data(), (int)weights_id.size(), weights_id.data(),
                     key.has_value() ? "registered" : "unregistered", (int)expected_key.size(),
                     expected_key.data());
        ++failures;
    }
}

}  // namespace

int main() {
    // The five pre-existing identities (regression set).
    check("qwen3.6-27b", "groupwise-int", "qwen3_6_27b");
    check("qwen3.6-27b", "nvfp4", "qwen3_6_27b");
    check("qwen3.8-27b", "groupwise-int", "qwen3_8_27b");
    check("qwen3.8-27b", "nvfp4", "qwen3_8_27b");
    check("qwen3.6-35b-a3b", "groupwise-int", "qwen3_6_35b_a3b");
    // The new identity.
    check("qwen3.8-flash-next", "nvfp4", "qwen3_8_flash_next");
    // Unregistered pairs.
    check("qwen3.8-flash-next", "bf16", "");
    check("qwen3.6-35b-a3b", "nvfp4", "");
    check("bogus-model", "nvfp4", "");

    // Package constants.
    if (FlashNextPackage::model_id != "qwen3.8-flash-next" ||
        FlashNextPackage::weights_id != "nvfp4" ||
        FlashNextPackage::target_key != "qwen3_8_flash_next") {
        std::fprintf(stderr, "FAIL: package identity constants\n");
        ++failures;
    }

    const std::filesystem::path fixture =
        std::filesystem::path(NINFER_SOURCE_DIR) / "out" / "flash_next_dev" /
        "qwen3_8_flash_next_mini.ninfer";
    if (!std::filesystem::exists(fixture)) {
        return 77;  // fixture not built; the resolver checks above already ran
    }
    try {
        ninfer::artifact::Reader reader(fixture);
        const auto key = resolve_target_key(reader.identity());
        if (!key.has_value() || *key != "qwen3_8_flash_next") {
            std::fprintf(stderr, "FAIL: mini artifact identity does not resolve to "
                                 "qwen3_8_flash_next\n");
            ++failures;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: mini artifact: %s\n", e.what());
        ++failures;
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d identity check(s) failed\n", failures);
        return 1;
    }
    return 0;
}
