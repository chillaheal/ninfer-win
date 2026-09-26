import io, sys

path = r"C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\core\src\targets\qwen3_8_flash_next\impl\load\real_loader.h"
with io.open(path, "r", encoding="utf-8") as f:
    src = f.read()

old_accessors = "  const artifact::Reader& reader() const noexcept { return *reader_; }"
new_accessors = (
    "  const artifact::Reader& reader() const noexcept { return *reader_; }\n"
    "  // The artifact/ngram paths this model was opened from (used by the Package to\n"
    "  // hand them to the real-mode Program factory without a second open).\n"
    "  const std::filesystem::path& artifact_path() const noexcept { return artifact_path_; }\n"
    "  const std::filesystem::path& ngram_path() const noexcept { return ngram_path_; }"
)
assert src.count(old_accessors) == 1, "accessors anchor not unique/found: %d" % src.count(old_accessors)
src = src.replace(old_accessors, new_accessors, 1)

old_member = "  std::unique_ptr<artifact::Reader> reader_;"
new_member = (
    "  std::unique_ptr<artifact::Reader> reader_;\n"
    "  std::filesystem::path artifact_path_;\n"
    "  std::filesystem::path ngram_path_;"
)
assert src.count(old_member) == 1, "member anchor not unique/found: %d" % src.count(old_member)
src = src.replace(old_member, new_member, 1)

with io.open(path, "w", encoding="utf-8", newline="") as f:
    f.write(src)
print("OK real_loader.h")
