import os, re, glob

real_root = os.getcwd()
real_nc = os.path.normcase(real_root)
core = os.path.realpath("core")
core_nc = os.path.normcase(core)

files=set()
for ext in (".bat",".ps1",".psm1",".psd1",".psd1",".py",".sh",".sh",".cmd",".cmake",".cmake.in",".cmake.in"):
    files.update(glob.glob(os.path.join("**","*"+ext), recursive=True))
files=sorted(set(f for f in files if os.path.isfile(f)))

root_re = re.compile(r'C:\Users\Micke\Documents\Kodprojekt\[^"\x27\x20]')
n_root=0; n_core=0; n_other=0; n_other_paths=0
other=[]
for fp in files:
    try: raw=open(fp,'rb').read()
    except: continue
    for m in root_re.finditer(raw):
        line_no = raw[:m.start()].count(b'\n')+1
        path = m.group(0).decode('utf-8','replace')
        pc = os.path.normcase(path)
        if pc == real_nc or pc.startswith(real_nc+os.sep) or pc==real_nc:
            n_root+=1; continue
        if pc == core_nc or pc.startswith(core_nc+os.sep) or real_nc.startswith(pc) or pc.startswith(core_nc):
            n_core+=1; continue
        n_other+=1
        other.append((fp,line_no,path))

print("absolute Kodprojekt paths found:", n_root+n_core+n_other)
print("  -> real project root (or subdir):", n_root)
print("  -> core tree (or subdir):", n_core)
print("  -> OTHER (flag):", n_other)
for fp,l,p in other:
    print("   FLAG", fp, l, p)
