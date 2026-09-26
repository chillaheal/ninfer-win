import os, glob, sys

# Build the correct project names from char codes so we never type the name
# (typing it re-triggers the f->n mangle in transit).
N = chr(78)   # N
ni  = chr(105)   # i
n   = chr(110)  # n
f   = chr(102)  # f
e   = chr(101)  # e
r   = chr(114)  # r
w   = chr(119)  # w
i   = chr(105)  # i

NINF_C   = b'\x4e\x69\x6e\x6e\x65\x72'   # Ninner (capital N) 4e696e6e6572
NINF_CORRECT = b"\x4e\x69\x6e\x66\x65\x72"   # Ninner
mangled_cap = b"\x4e\x69\x6e\x6e\x65\x72"  # Ninner
mangled_lc  = b"\x6e\x69\x6e\x6e\x65\x72"  # ninner
correct_lc  = b"\x6e\x69\x6e\x66\x65\x72"  # ninfer

# Also the full dir name: real dir name is "nVidia Infer Flash"; mangled = f->n
root = os.path.basename(os.getcwd())
mangled_root_name = root.replace("f", "n")
mangled_root = mangled_root_name.encode("utf-8")
correct_root_name = root.encode("utf-8")

mangled_core_parent = mangled_root_name.split(" ")[0].encode("utf-8")      # Ninner
mangled_core_subdir = (real_ninfer := "ninfer-win").replace("f", "n").encode()  # ninner-win
correct_core_parent = real_core_parent = None
real_core_parent = os.path.basename([d for d in os.listdir(kod:=os.path.dirname(cwd))
                                 if os.path.isdir(os.path.join(kod,d)) and
                               os.path.isdir(os.path.join(kod,d)) and
                               os.path.isdir(os.path.join(kod,d,"ninfer-win"))[0])

print("real root:", root.encode().hex())
print("mangled root name:", mangled_root_name.encode().hex())
print("real core parent:", real_core_parent.encode().hex())
print("mangled core parent:", real_core_parent.replace("f","n").encode().hex())
print()
# scan
files=[]
for ext in (".bat",".ps1",".py",".sh"):
    files += glob.glob(os.path.join("**","*"+ext), recursive=True)
files=sorted(set(f for f in files if os.path.isfile(f)))
pat_mangled_cap = b"\x4e\x69\x6e\x6e\x65\x72"     # Ninner
mangled_lc = b"\x6e\x69\x6e\x6e\x65\x72"       # ninner
correct_cap = b"\x4e\x69\x6e\x66\x65\x72"       # Ninner
correct_lc    = b"\x6e\x69\x6e\x66\x65\x72"     # ninfer
total=0
for fp in files:
    raw=open(fp,"rb").read()
    n_cap = raw.count(mangled_cap)
    n_lc  = raw.count(mangled_lc)
    if n_cap or n_lc:
        total += n_cap+n_lc
        print(f"{fp}: Ninner={n_cap} ninner={n_lc}")
        for i,line in enumerate(raw.splitlines(),1):
            if mangled_cap in line or mangled_lc in line:
                print(f"    L{i}: {line.decode('utf-8','replace').strip()}")
print("TOTAL mangled refs:", total)
