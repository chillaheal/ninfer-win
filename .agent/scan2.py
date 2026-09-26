import os, glob, re
MANGLED=[bytes([0x4e,0x69,0x6e,0x6e,0x65,0x72]),bytes([0x6e,0x69,0x6e,0x6e,0x65,0x72]),os.path.basename(os.getcwd()).replace("f","n").encode()]
EXTS=(".bat",".ps1",".py",".sh",".cmd",".cmake",".cmake.in")
files=set()
for ext in EXTS:
    files.update(glob.glob(os.path.join("**","*"+ext),recursive=True))
files=sorted(set(f for f in files if os.path.isfile(f)))
real_root=os.getcwd()
real_nc=os.path.normcase(real_root)
core=os.path.realpath("core")
core_nc=os.path.normcase(core)
core_parent_nc=os.path.normcase(os.path.dirname(core))
print("files scanned:",len(files))
print()
print("=== NON-COMMENT mangled matches: ===")
n_func=0
for fp in files:
    try: raw=open(fp,"rb").read()
    except: continue
    for i,ln in enumerate(raw.splitlines(),1):
        if not any(p in ln for p in MANGLED): continue
        if is_comment_ln(ln): continue
        n_func+=1
        print("  FUNC",fp,i)
print("  non-comment mangled:",n_func)
print()
print("=== wd/BUILDDIR lines mentioning Kodprojekt: ===")
wdre=re.compile(rb'(?i)^\s*(cd|pushd|Set-Location|Push-Location)\b')
nwd=0
for fp in files:
    try: raw=open(fp,"rb").read()
    except: continue
    for i,ln in enumerate(raw.splitlines(),1):
        if b"Kodprojekt" not in ln: continue
        if not (wdre.match(ln) or b"BUILDDIR" in ln): continue
        nwd+=1
        s=ln.decode("utf-8","replace").strip()
        m=re.search(rb'[A-Za-z]:\[^\x27"\s]+',ln)
        if m:
            p=m.group(0).rstrip(b"'\''").decode("utf-8","replace")
            pc=os.path.normcase(p)
            if pc==real_nc: tag="ROOT-OK"
            elif pc.startswith(core_nc) or pc.startswith(core_parent_nc): tag="CORE-OK"
            else: tag="?? OTHER"
        else:
            tag="REL/OTHER"
        print("  %-10s %s:%d: %s"%(tag,fp,i,s[:72]))
print("  wd/BUILDDIR lines:",nwd)
