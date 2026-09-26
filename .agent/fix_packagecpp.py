import io

path = r"C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\core\src\targets\qwen3_8_flash_next\impl\package.cpp"
with io.open(path, "r", encoding="utf-8") as f:
    src = f.read()

# 1) includes
old_inc = (
    '#include "artifact/reader.h"\n'
    '#include "targets/qwen3_8_flash_next/impl/config.h"\n'
    '#include "targets/qwen3_8_flash_next/impl/load/bindings.h"\n'
)
new_inc = (
    '#include "artifact/reader.h"\n'
    '#include "targets/qwen3_8_flash_next/impl/config.h"\n'
    '#include "targets/qwen3_8_flash_next/impl/load/bindings.h"\n'
    '#include "targets/qwen3_8_flash_next/impl/load/real_loader.h"\n'
    '#include "targets/qwen3_8_flash_next/impl/runtime/family.h"\n'
)
assert src.count(old_inc) == 1, "inc: %d" % src.count(old_inc)
src = src.replace(old_inc, new_inc, 1)

# 2) construct_loaded_model -> real loader (load the real model once)
old_clm = (
    "std::unique_ptr<LoadedModel> Package::construct_loaded_model(\n"
    "    const std::filesystem::path& artifact_path, const std::filesystem::path& ngram_path,\n"
    "    const std::filesystem::path& archive_path, DeviceContext& device) {\n"
    "    return qwen3_8_flash_next::LoadedModel::open(artifact_path, ngram_path, archive_path, device);\n"
    "}"
)
new_clm = (
    "std::unique_ptr<LoadedModel> Package::construct_loaded_model(\n"
    "    const std::filesystem::path& artifact_path, const std::filesystem::path& ngram_path,\n"
    "    const std::filesystem::path& archive_path, DeviceContext& device) {\n"
    "    (void)archive_path;  // the real path mmaps the .ninfer/.ngram directly\n"
    "    return qwen3_8_flash_next::RealLoadedModel::open(artifact_path, ngram_path, device);\n"
    "}"
)
assert src.count(old_clm) == 1, "clm: %d" % src.count(old_clm)
src = src.replace(old_clm, new_clm, 1)

# 3) create_program -> real factory from the already-loaded model (moved in, no second open)
old_cp = (
    "std::unique_ptr<Program> Package::create_program(const LoadedModel& model, SequencePlan&& plan,\n"
    "                                                 DeviceContext& device) {\n"
    "    return qwen3_8_flash_next::create_program(model.view(), WeightsProfile::Nvfp4, std::move(plan),\n"
    "                                              device);\n"
    "}"
)
new_cp = (
    "std::unique_ptr<Program> Package::create_program(std::unique_ptr<LoadedModel> model,\n"
    "                                                 SequencePlan&& plan,\n"
    "                                                 DeviceContext& device) {\n"
    "    return qwen3_8_flash_next::create_real_program_from_loaded(std::move(model),\n"
    "                                                    plan.capacity(), device,\n"
    "                                                    ops::QsaIndexerKvDtype::Fp8);\n"
    "}"
)
assert src.count(old_cp) == 1, "cp: %d" % src.count(old_cp)
src = src.replace(old_cp, new_cp, 1)

with io.open(path, "w", encoding="utf-8", newline="") as f:
    f.write(src)
print("OK package.cpp")
