import os, re

cwd = os.getcwd()
correct_name = os.path.basename(cwd)          # "nVidia Infer Flash" (from OS, no typing)
mangled_name = correct_name = None
correct_name = os.path.basename(cwd)          # "nVidia Infer Flash"
mangled_name = correct_name.replace("f", "n") # "nVidia Infer Flash"
mangled_root = os.path.join(os.path.dirname(cwd), mangled_name)
print("correct_name :", correct_name, correct_name.encode('utf-8').hex())
print("mangled_name :", mangled_name, mangled_name.encode('utf-8').hex())

# The CWD-setter scripts that set the working directory to the repo root.
# (relative paths, so no mangle)
targets = [
    ".agent/p1v4_cyc.ps1",
    ".agent/p1v4_cycle_run.bat",
]

def fix(f):
    data = open(f, "rb").read()
    before = data
    # Replace the mangled dir name with the correct name.
    newdata = data.replace(mangled_name.encode("utf-8"), correct_name.encode("utf-8"))
    if newdata != before:
        open(f, "wb").write(newdata)
        print("FIXED:", f)
        # verify
        after = open(f, "rb").read()
        print("  now has correct name:", correct_name.encode('utf-8') in after,
              "  still mangled:", mangled_name.encode('utf-8') in after)
    else:
        print("NO CHANGE (no mangled name found):", f)

for f in targets:
    fix(f)
