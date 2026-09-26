import os, re, glob

root = os.getcwd()
core = os.path.realpath('core')
root_nc = os.path.normcase(root)
core_nc = os.path.normcase(core)

# Mangled (all-n) on-disk byte patterns = wrong version of the project name
# Real "Ninfer" = 4e 69 6e 66 65 72  (4th char f=0x66)
# Mangled     = 4e 69 6e 6e 65 72  (4th char n=0x6e)
MC = bytes([0x4e, 0x69, 0x6e, 0x6e, 0x65, 0x72])
ML = bytes([0x6e, 0x69, 0x6e, 0x6e, 0x65, 0x72])

exts = (".bat", ".ps1", ".psm1", ".psd1", ".py", ".sh", ".cmd",
         ".cmake", ".cmake.in", ".txt", ".md", ".ini", ".cfg",
         ".json", ".jsonc", ".yaml", ".yml")
files = set()
for ext in exts:
    files.update(glob.glob(os.path.join("**", "*" + ext), recursive=True))
files = sorted(set(f for f in files if os.path.isfile(f)))
print("files scanned:", len(files))

# --- Part A: mangled byte count ---
mf = 0
mb = 0
for fp in files:
    raw = open(fp, "rb").read()
    c = raw.count(MC) + raw.count(ML)
    if c:
        mf += 1
        mb += c
        for i, ln in enumerate(raw.splitlines(), 1):
            if MC in ln or ML in ln:
                print("MANGLE", fp, i, ln.decode("utf-8", "replace").strip())
print("mangled files:", mf, "bytes:", mb)

# --- Part B: classify absolute Kodprojekt paths ---
# normpath collapses YAML/JSON double-backslash escaping before normcase
q_re = re.compile(rb'"([A-Za-z]:\\[^"]*)"')
nr = 0
nc = 0
fl = []
for fp in files:
    raw = open(fp, "rb").read()
    for i, ln in enumerate(raw.splitlines(), 1):
        if b"Kodprojekt" not in ln:
            continue
        for m in q_re.finditer(ln):
            b = m.group(1)
            p = os.path.normpath(b.decode("utf-8", "replace"))
            pc = os.path.normcase(p)
            if pc == root_nc or pc.startswith(root_nc + os.sep):
                nr += 1
            elif pc == core_nc or pc.startswith(core_nc + os.sep):
                nc += 1
            else:
                fl.append((fp, i, p))
print("ROOT-OK:", nr, "CORE-OK:", nc, "FLAGGED:", len(fl))
for fp, i, p in fl[:20]:
    print("FLAG", fp, i, p[:80])
