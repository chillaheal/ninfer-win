import io

path = r"C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\core\src\targets\registry.cpp"
with io.open(path, "r", encoding="utf-8") as f:
    src = f.read()

old = "      program(Qwen3_8_FlashNext::create_program(*loaded->model, std::move(sequence_plan), device)) {}"
new = (
    "      program(Qwen3_8_FlashNext::create_program(std::move(loaded->model),\n"
    "                                             std::move(sequence_plan), device)) {}"
)
assert src.count(old) == 1, "cp call site: %d" % src.count(old)
src = src.replace(old, new, 1)

with io.open(path, "w", encoding="utf-8", newline="") as f:
    f.write(src)
print("OK registry.cpp")
