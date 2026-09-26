import os, glob, re

MANGLED = [
    bytes([0x4e,0x69,0x6e,0x6e,0x65,0x72]),
    bytes([0x6e,0x69,0x6e,0x6e,0x65,0x72]),
    os.path.basename(os.getcwd()).replace("f","n").encode(),
]
EXTS = (".bat",".ps1",".psm1",".py",".sh",".cmd",".cmake",".cmake.in",".json",".jsonc",".txt")
files=set()
for ext in EXTS:
    files.update(glob.glob(os.path.join("**","*"+ext), recursive=True))
files=sorted(set(f for f in files if os.path.isfile(f)))
real_root = os.getcwd()

def is_comment(ln):
    s = ln.lstrip()
    return s.startswith(b'#') or s.startswith(b'//') or s.startswith(b';')

print("=== NON-COMMENT mangled matches (would affect behavior): ===")
n_func=0
for fp in files:
    try: raw=open(fp,'rb').read()
    except: continue
    for i,ln in enumerate(raw.splitlines(),1):
        if any(p in ln for p in MANGLED) and not is_comment(ln):
            n_func+=1
            print("  FUNC", fp, i, ln.decode('utf-8','replace').strip())
print("  non-comment mangled matches:", n_func)
print()
print("=== Working-dir / BUILDDIR lines mentioning Kodprojekt: ===")
wdre=re.compile(rb'(?i)^\s*(cd\b|pushd\b|Set-Location\b|set\s+"[^"]*BUILDDIR)')
nwd=0
for fp in files:
    try: raw=open(fp,'rb').read()
    except: continue
    for i,ln in enumerate(raw.splitlines(),1):
        if b'Kodprojekt' not in ln: continue
        if not (wdre.match(ln) or b'BUILDDIR' in ln): continue
        nwd+=1
        s=ln.decode('utf-8','replace').strip()
        m=re.search(rb'[A-Za-z]:\[^\x27"\s]+', ln)
        if m:
            p=m.group(0).rstrip(b"'\"").decode('utf-8','replace')
            ok = os.path.normcase(p).startswith(os.path.normcase(real_root)) or os.path.normcase(real_root).startswith(os.path.normcase(p))
            tag = "root-OK" if ok else "!! NOT-root"
        else:
            tag="rel/other"
        print("  %-10s %s:%d: %s" % (tag, fp, i, s[:75]))
print("  total wd/BUILDDIR lines:", nwd)
