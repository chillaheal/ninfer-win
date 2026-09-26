import os

# All target strings built from char codes — never typed, so never mangled.
# Real project dir (from OS):  N i n f e r  space  F l a s h
#   N=78 i=105 n=110 f=102 e=101 r=114, space=32, F=70 l=108 a=97 s=115 h=104
correct_root = bytes([78,105,110,102,101,114, 32, 70,108,97,115,104])   # "nVidia Infer Flash"
mangled_root = bytes([78,105,110,110,101,114, 32, 70,108,97,115,104])  # N,i,n,n,e,r sp F,l,a,s,h

# Core subtree:  Ninner \ ninner-win
#   real core parent:  N i n f e r  = 78,105,110,102,101,114
#   mangled core parent: N i n n e r = 78,105,110,110,101,114
#   real core subdir:  n i n f e r - w i n
#   mangled core subdir: n i n n e r - w i n
real_subdir   = bytes([110,105,110,102,105,114, 45, 119,105,110])  # n i n f e r - w i n
mangled_subdir = bytes([110,105,110,110,101,114, 45, 119,105,110])   # n i n n e r - w i n

correct_parent = bytes([78,105,110,102,101,114])   # N i n f e r
mangled_parent = bytes([78,105,110,110,101,114])   # N i n n e r

repls = [
    (mangled_root, correct_root.encode()),
    (mangled_parent, correct_parent),
    (mangled_subdir, correct_subdir),
]

targets = [
    ".agent/p1v4_cyc.ps1",
    ".agent/p1v4_cycle_run.bat",
    "tools/flash_next_dev/attach_overlay.ps1",
    "pn_check.bat",
    "pn_run.bat",
    "pn_verify.bat",
]

for fp in targets:
    if not os.path.exists(fp):
        print("MISSING", fp); continue
    data = open(fp, "rb").read()
    orig = data
    for m, c in repls:
        n = data.count(m)
        if n:
            data = data.replace(m, c)
            print(f"{fp}: replaced {m} x{n} -> {c}")
    if data != orig:
        open(fp, "wb").write(data)
        print("WROTE", fp)
    else:
        print("no change", fp)
print("DONE")
