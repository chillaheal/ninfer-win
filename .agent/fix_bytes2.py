import os

def B(*cs):
    return bytes(cs)

p = "p1v4_full_cycle.ps1"
b = open(p, "rb").read()

# Current (wrong) folder bytes in ps1: N i n n e r space A I
old_folder = B(78, 105, 110, 110, 101, 114, 32, 65, 73)
# True folder: N i n f e r space A I
new_folder = B(78, 105, 110, 102, 101, 114, 32, 65, 73)
n_folder = b.count(old_folder)
b = b.replace(old_folder, new_folder)

# Process / exe stem: n i n n e r  ->  n i n f e r
old_stem = B(110, 105, 110, 110, 101, 114)
new_stem = B(110, 105, 110, 102, 101, 114)
n_stem = b.count(old_stem)
b = b.replace(old_stem, new_stem)

open(p, "wb").write(b)
print("folder replacements:", n_folder)
print("stem replacements:", n_stem)
