import io

# Fix 1: package.cpp — out-of-class return/param types must name RealLoadedModel
# directly. The class-scope alias `LoadedModel` resolves to the namespace-scope
# mini LoadedModel outside the class body, so name the real type explicitly.
pcpp = r"C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\core\src\targets\qwen3_8_flash_next\impl\package.cpp"
with io.open(pcpp, "r", encoding="utf-8") as f:
    s = f.read()

old_clm = "std::unique_ptr<LoadedModel> Package::construct_loaded_model("
new_clm = "std::unique_ptr<RealLoadedModel> Package::construct_loaded_model("
old_cp = "std::unique_ptr<Program> Package::create_program(std::unique_ptr<LoadedModel> model,"
new_cp = "std::unique_ptr<Program> Package::create_program(std::unique_ptr<RealLoadedModel> model,"

n1 = s.count(old_clm)
n2 = s.count(old_cp)
print("package.cpp: clm=%d cp=%d" % (n1, n2))
s = s.replace(old_clm, new_clm)
s = s.replace(old_cp, new_cp)

with io.open(pcpp, "w", encoding="utf-8", newline="") as f:
    f.write(s)
print("OK package.cpp")

# Fix 2: registry.cpp — include real_loader.h so RealLoadedModel is a complete
# type when the unique_ptr<RealLoadedModel> member's destructor is instantiated.
rcpp = r"C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\core\src\targets\registry.cpp"
with io.open(rcpp, "r", encoding="utf-8") as f:
    r = f.read()

old_inc = '#include "targets/qwen3_8_flash_next/impl/load/real_bindings.h"\n'
new_inc = ('#include "targets/qwen3_8_flash_next/impl/load/real_bindings.h"\n'
           '#include "targets/qwen3_8_flash_next/impl/load/real_loader.h"\n')
ninc = r.count(old_inc)
ninc2 = r.count(new_inc)
print("registry.cpp: old_inc=%d new_inc=%d" % (ninc, ninc2))
if ninc == 1 and ninc2 == 0:
    r = r.replace(old_inc, new_inc, 1)
    with io.open(rcpp, "w", encoding="utf-8", newline="") as f:
        f.write(r)
    print("OK registry.cpp")
elif ninc2 >= 1:
    print("registry.cpp: include already present, no change")
else:
    print("registry.cpp: UNEXPECTED, no change made")
