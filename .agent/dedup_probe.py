import io

path = r"C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\core\src\targets\registry.cpp"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

marker = "KvProbeResult probe_flash_next"
first = text.index(marker)
second = text.index(marker, first + 1)

# The first block begins with the "// Flash-Next probe." comment immediately above it.
comment_start = text.rindex("// Flash-Next probe.", 0, first)

removed = second - comment_start
new_text = text[:comment_start] + text[second:]

with open(path, "w", encoding="utf-8", newline="") as f:
    f.write(new_text)

count = new_text.count(marker)
print("removed_chars=%d remaining_markers=%d" % (removed, count))
