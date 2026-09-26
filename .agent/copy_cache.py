import io

# Reuse the existing CMake cache for a fresh build tree under <proj>/build
# (project root), so all build output stays inside this project cwd.
F = chr(102)

src = "core/build/CMakeCache.txt"
# build dir at project root: C:/Users/Micke/Documents/Kodprojekt/nVidia Infer Flash/build
proj = "C:/Users/Micke/Documents/Kodprojekt/" + "Ni" + F + "ner " + F + "lash"
build_dir = proj + "/build"

with io.open(src, "r", encoding="utf-8") as fh:
    text = fh.read()

n_home = 0
n_cache = 0
lines = []
for line in text.splitlines():
    if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
        line = "CMAKE_HOME_DIRECTORY:INTERNAL=C:/Users/Micke/Documents/Kodprojekt/Ninner/ninner-win"
        n_home += 1
    elif line.startswith("CMAKE_CACHEFILE_DIR:INTERNAL="):
        line = "CMAKE_CACHEFILE_DIR:INTERNAL=" + build_dir
        n_cache += 1
    lines.append(line)

if n_home != 1 or n_cache != 1:
    print("REPLACEMENT COUNT home=%d cache=%d" % (n_home, n_cache))
    raise SystemExit(1)

with io.open("build/CMakeCache.txt", "w", encoding="utf-8", newline="") as fh:
    fh.write("\n".join(lines) + "\n")
print("WROTE build/CMakeCache.txt; home=%s cache=%s" % (n_home, n_cache))
