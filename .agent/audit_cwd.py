import os, re, glob

cwd = os.getcwd()
real_dirname = os.path.basename(cwd)
print("REAL CWD     :", cwd.encode('utf-8').hex())
print("REAL DIRNAME :", real_dirname.encode('utf-8').hex(), "len", len(real_dirname))

# mangled sibling: same basename but f->n (the known mangle)
parent = os.path.dirname(cwd)
mangled_name = real_dirname.replace('f', 'n')
stray = os.path.join(parent, mangled_name)
print("STRAY exists? : ", os.path.exists(stray), "->", stray.encode('utf-8').hex())
print()

exts = ('.bat', '.ps1', '.py', '.sh')
files = set()
for ext in exts:
    files.update(glob.glob(os.path.join('**', '*' + ext), recursive=True))
files = sorted(f for f in files if os.path.isfile(f) and f.lower().endswith(exts))
print("script files found:", len(files))
print()

def show(fp):
    raw = open(fp, 'rb').read()
    for enc in ('utf-8', 'latin-1'):
        try:
            text = raw.decode(enc); break
        except Exception: continue
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if ('Kodprojekt' in s) or ('cd /d' in s) or ('Set-Location' in s) or re.match(r'cd\s', s) or re.match(r'cd\s+', s, re.I):
            has_real = real_dirname in line
            has_mangled = (mangled_name in line) and (real_dirname not in line)
            tag = 'OK    ' if has_real else ('MANGLE' if has_mangled else 'OTHER ')
            print(f"[{tag}] {fp}:{i}: {s}")
            print("        hex:", s.encode('utf-8').hex())
    print()

for fp in files:
    show(fp)
