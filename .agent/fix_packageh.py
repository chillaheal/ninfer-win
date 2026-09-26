import io

path = r"C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\core\src\targets\qwen3_8_flash_next\export\ninfer\targets\qwen3_8_flash_next\package.h"
with io.open(path, "r", encoding="utf-8") as f:
    src = f.read()

old_fwd = "class LoadedModel;  // impl/load/bindings.h (self-materializes the mini .ninfer)"
new_fwd = "class RealLoadedModel;  // impl/load/real_loader.h (real-geometry device loader)"
assert src.count(old_fwd) == 1, "fwd decl: %d" % src.count(old_fwd)
src = src.replace(old_fwd, new_fwd, 1)

old_alias = "using LoadedModel   = qwen3_8_flash_next::LoadedModel;"
new_alias = "using LoadedModel   = qwen3_8_flash_next::RealLoadedModel;"
assert src.count(old_alias) == 1, "alias: %d" % src.count(old_alias)
src = src.replace(old_alias, new_alias, 1)

old_cp = (
    "    [[nodiscard]] static std::unique_ptr<Program> create_program(const LoadedModel& model,\n"
    "                                                                 SequencePlan&& plan,\n"
    "                                                                 DeviceContext& device);"
)
new_cp = (
    "    [[nodiscard]] static std::unique_ptr<Program> create_program(std::unique_ptr<LoadedModel> model,\n"
    "                                                                 SequencePlan&& plan,\n"
    "                                                                 DeviceContext& device);"
)
assert src.count(old_cp) == 1, "cp: %d" % src.count(old_cp)
src = src.replace(old_cp, new_cp, 1)

with io.open(path, "w", encoding="utf-8", newline="") as f:
    f.write(src)
print("OK package.h")
