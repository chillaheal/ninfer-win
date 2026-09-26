import io

path = r"C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\core\src\targets\qwen3_8_flash_next\impl\runtime\family.h"
with io.open(path, "r", encoding="utf-8") as f:
    src = f.read()

# 1) Forward-declare RealLoadedModel at the top of the namespace.
old_ns = "namespace ninfer::targets::qwen3_8_flash_next {\n\nenum class WeightsProfile : std::uint8_t {\n    Nvfp4,\n};"
new_ns = (
    "namespace ninfer::targets::qwen3_8_flash_next {\n"
    "\n"
    "class RealLoadedModel;  // impl/load/real_loader.h (real-geometry device loader)\n"
    "\n"
    "enum class WeightsProfile : std::uint8_t {\n"
    "    Nvfp4,\n"
    "};"
)
assert src.count(old_ns) == 1, "ns open: %d" % src.count(old_ns)
src = src.replace(old_ns, new_ns, 1)

# 2) Friend declaration inside the Program class (after the create_real_program friend).
old_friend = (
    "    friend std::unique_ptr<Program>\n"
    "    create_real_program(const std::filesystem::path& artifact, const std::filesystem::path& ngram,\n"
    "                        std::uint32_t max_context, DeviceContext& device,\n"
    "                        ops::QsaIndexerKvDtype idx_dtype);"
)
new_friend = (
    "    friend std::unique_ptr<Program>\n"
    "    create_real_program(const std::filesystem::path& artifact, const std::filesystem::path& ngram,\n"
    "                        std::uint32_t max_context, DeviceContext& device,\n"
    "                        ops::QsaIndexerKvDtype idx_dtype);\n"
    "    // M2 real-mode factory from an already-loaded model (no second open).\n"
    "    friend std::unique_ptr<Program>\n"
    "    create_real_program_from_loaded(std::unique_ptr<RealLoadedModel> model,\n"
    "                         std::uint32_t max_context, DeviceContext& device,\n"
    "                         ops::QsaIndexerKvDtype idx_dtype);"
)
assert src.count(old_friend) == 1, "friend: %d" % src.count(old_friend)
src = src.replace(old_friend, new_friend, 1)

# 3) Declaration after the create_real_program declaration.
old_decl = (
    "[[nodiscard]] std::unique_ptr<Program>\n"
    "create_real_program(const std::filesystem::path& artifact, const std::filesystem::path& ngram,\n"
    "                    std::uint32_t max_context, DeviceContext& device,\n"
    "                    ops::QsaIndexerKvDtype idx_dtype);"
)
new_decl = (
    "[[nodiscard]] std::unique_ptr<Program>\n"
    "create_real_program(const std::filesystem::path& artifact, const std::filesystem::path& ngram,\n"
    "                    std::uint32_t max_context, DeviceContext& device,\n"
    "                    ops::QsaIndexerKvDtype idx_dtype);\n"
    "\n"
    "// M2 real-mode factory from an already-loaded model (no second open): the 75 GB\n"
    "// backbone is H2D'd exactly once across construct_loaded_model + create_program.\n"
    "[[nodiscard]] std::unique_ptr<Program>\n"
    "create_real_program_from_loaded(std::unique_ptr<RealLoadedModel> model,\n"
    "                    std::uint32_t max_context, DeviceContext& device,\n"
    "                    ops::QsaIndexerKvDtype idx_dtype);"
)
assert src.count(old_decl) == 1, "decl: %d" % src.count(old_decl)
src = src.replace(old_decl, new_decl, 1)

with io.open(path, "w", encoding="utf-8", newline="") as f:
    f.write(src)
print("OK family.h")
