import io

path = r"C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\core\src\targets\qwen3_8_flash_next\impl\load\real_loader.cpp"
with io.open(path, "r", encoding="utf-8") as f:
    src = f.read()

old = "  auto lm = std::unique_ptr<RealLoadedModel>(new RealLoadedModel());\n  Binder b(reader);"
new = (
    "  auto lm = std::unique_ptr<RealLoadedModel>(new RealLoadedModel());\n"
    "  lm->artifact_path_ = artifact_path;\n"
    "  lm->ngram_path_ = ngram_path;\n"
    "  Binder b(reader);"
)
assert src.count(old) == 1, "anchor not found: %d" % src.count(old)
src = src.replace(old, new, 1)
with io.open(path, "w", encoding="utf-8", newline="") as f:
    f.write(src)
print("OK real_loader.cpp")
