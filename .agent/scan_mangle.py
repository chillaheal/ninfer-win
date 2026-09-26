import os, glob

# mangle: f (0x66) -> n (0x6e). The 4-char trigger is the "ninf" sequence.
# Correct dir names:  Ninner=4e696e666572 , ninfer=6e696e666572
# Mangled (broken) bytes:
MANGLED_CAP = bytes([0x4e,0x69,0x6e,0x6e,0x65,0x72])   # Ninner  (N i n n e r)
MANGLED_LC  = bytes([0x6e,0x69,0x6e,0x6e,0x65,0x72])   # ninner

files = set()
for ext in (".bat",".ps1",".py",".sh",".cmake",".cmake.in",".txt",".md",".mdx",".mdown",".ini",".ini",".cfg",".ini",".json",".jsonc",".yaml",".yml",".yml"):
    files.update(glob.glob(os.path.join("**","*"+ext), recursive=True))
files = sorted(set(f for f in files if os.path.isfile(f)))

total = 0
for fp in files:
    raw = open(fp,"rb").read()
    n_cap = raw.count(MANGLED_CAP)
    n_lc  = raw.count(MANGLED_LC)
    if n_cap or n_lc:
        total += n_cap + n_lc
        print(f"{n_cap:2d}/{n_lc:<2d}  {fp}")
        for i,ln in enumerate(raw.splitlines(),1):
        if MANGLED_CAP in ln or MANGLED_LC in ln:
            print(f"        L{i}: {ln.decode('utf-8','replace').strip()}")
print("TOTAL files with mangle:", total)
print("scanned files:", len(files))
