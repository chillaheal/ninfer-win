import os, glob, re

real_root = os.getcwd()
core = os.path.realpath("core")
root_base = os.path.basename(real_root)
kodproj = os.path.basename(os.path.dirname(real_root))
core_parent = os.path.basename(os.path.dirname(core))

# byte patterns (mangled = all-n, no f)
MANGLED = [
    bytes([0x4e,0x69,0x6e,0x6e,0x65,0x72]),
    bytes([0x6e,0x69,0x6e,0x6e,0x65,0x72]),
]

files = set()
for ext in EXTS:
    files.update(glob.glob(os.path.join("**", "*"+ext), recursive=True))
files = sorted(set(f for f in files if os.path.isfile(f)))
print("files scanned:", len(files))
print("real root:", real_root.encode().hex())
print("core:", core.encode().hex())
print()
print("=== MANGLED patterns (all-n, no f): ===")
n_mangled_files = 0
for fp in files:
    raw = open(fp,'rb').read()
    for pat in MANGLED:
        c = raw.count(pat)
        if c:
            n_mangled_files += 1
            for i, ln in enumerate(raw.splitlines(),1):
                if pat in ln:
                    print(f"  MANGLE {fp}:{i}  pat={pat.hex()}")
                    break
print("mangled files:", n_mangled_files)
print()
print("=== wd/BUILDDIR/Set-Location lines mentioning Kodprojekt: ===")
nwd = 0
for fp in files:
    raw = open(fp,'rb').read()
    for i, ln in enumerate(raw.splitlines(),1):
        if b'Kodprojekt' not in ln: continue
        if not re.search(rb'(?i)(^\s*(cd|pushd|Set-Location|Set-Location|pushd)\b|BUILDDIR)', ln):
            continue
        nwd += 1
        m = re.search(rb'[A-Za-z]:\[^\x27"\s]+', ln)
        if m:
            p = m.group(0).rstrip(b"'\"").decode('utf-8','replace')
            pc = os.path.normcase(p)
            if pc == os.path.normcase(real_root): tag = "ROOT-OK"
            elif pc == os.path.normcase(core) or pc.startswith(os.path.normcase(core)): tag = "CORE-OK"
            else: tag = "?? OTHER"
        else:
            tag = "rel/other"
        print(f"  {tag:<10} {fp}:{i}: {ln.decode('utf-8','replace').strip()[:75]}")
print("wd/BUILDDIR lines:", nwd)
