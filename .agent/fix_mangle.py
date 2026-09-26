import os, glob, sys

APPLY = "--apply" in sys.argv

# Byte patterns built from codes, never typed.
# real: Ninner=4e696e666572  ninfer=6e696e666572
# mangled (all-n, no f): 4e696e6e6572 (Ninner)  6e696e6e6572 (ninner)
# full project name: "nVidia Infer Flash" = N,i,n,f,e,r,space,F,l,a,s,h
# mangled full name: N,i,n,n,e,r,space,F,l,a,s,h
CORRECT_FULL  = bytes([78,105,110,102,101,114,32,70,108,97,115,104])   # nVidia Infer Flash
MANGLED_FULL  = bytes([78,105,110,110,101,114,32,70,108,97,115,104])  # nVidia Infer Flash
CORRECT_CAP   = bytes([78,105,110,102,101,114])   # Ninner
MANGLED_CAP   = bytes([78,105,110,110,101,114])   # Ninner (N i n n e r)
CORRECT_LC    = bytes([110,105,110,102,101,114]) # ninfer
MANGLED_LC    = bytes([110,105,110,110,101,114]) # ninner

# longest/most-specific first to avoid partial overlap
REPLS = [
    (MANGLED_FULL, CORRECT_FULL),
    (MANGLED_CAP, CORRECT_CAP),
    (MANGLED_LC, CORRECT_LC),
]

APPLY = "--apply" in sys.argv
total_files = 0
total_reps = 0
for fp in files:
    try:
        data = open(fp, "rb").read()
    except Exception:
        continue
    orig = data
    for m, c in REPLS:
        c2 = data.count(m)
        if c2:
            if APPLY:
                data = data.replace(m, c)
                total_reps += c2
            print(f"  {'FIX' if APPLY else 'dry'} {fp}: {m.hex()} -> {c.hex()}  x{c2}")
    if APPLY and data != orig:
        open(fp, "wb").write(data)
        total_files += 1
print("APPLY" if APPLY else "DRY-RUN")
print("files changed:", total_files, "  total replacements:", total_reps)
