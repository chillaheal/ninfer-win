import io

path = "tools/flash_next_dev/p12_s1b_build.bat"
with io.open(path, "r", encoding="utf-8") as fh:
    text = fh.read()

old = "Documents\\\\Ninner\\\\ninner-win\\\\build"
new = "Documents\\\\Kodprojekt\\\\N" + chr(102) + "ner\\\\ninner-win\\\\build"
if old in text:
    text = text.replace(old, new)
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("FIXED")
else:
    print("PATTERN NOT FOUND")
    for line in text.splitlines():
        if "build" in line:
            print(repr(line))
