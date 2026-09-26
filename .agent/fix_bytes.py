import os

def B(*cs):
    return bytes(cs)

p = os.path.join(os.getcwd(), "p1v4_full_cycle.ps1")
b = open(p, "rb").read()

# 7-char mangled stem "n i n f n e r"  ->  "n i n f e r"
bad_stem = B(110,105,110,102,110,101,114)
good_stem = B(110,105,110,102,101,114)
c1 = b.count(bad_stem)
b = b.replace(bad_stem, good_stem)

# model ext ".n n e r" (n,n) -> ".n i n f e r"
bad_ext = B(46,110,105,110,110,101,114)
good_ext = B(46,110,105,110,102,101,114)
c2 = b.count(bad_ext)
b = b.replace(bad_ext, good_ext)

open(p, "wb").write(b)
print("stem replaced:", c1, " ext replaced:", c2)
