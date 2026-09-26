import os
NL = chr(92)
desk = 'C:' + NL + 'Users' + NL + 'Micke' + NL + 'Desktop'
dirs = [d for d in os.listdir(desk) if os.path.isdir(os.path.join(desk, d)) and d.endswith(' AI')]
print('num AI dirs:', len(dirs))
d = dirs[0]
print('folder:', [ord(c) for c in d])
base = os.path.join(desk, d)
# build candidate filenames with both spellings of the suffix and stem
f = chr(102); n = chr(110)
stems = [ninf := (n + chr(105) + n + chr(102) + chr(105) + chr(101) + chr(114))]
# enumerate all variants of the stem+ext for serve exe
for a in (chr(102), chr(110)):
    for b in (chr(102), chr(110)):
        stem = n + chr(105) + a + b + chr(101) + chr(114)
        for ext in ('.exe',):
            p = os.path.join(base, stem + ext)
            if os.path.exists(p):
            print('EXE FOUND:', [ord(c) for c in stem + ext])
# models
mods = [x for x in os.listdir(base + NL + 'models') if x.startswith('qwen3_8_27b_nvfp4')]
for m in mods:
    print('model:', [ord(c) for c in m])
# system prompt
for x in os.listdir(base):
    if x.endswith('.md'):
        print('md:', [ord(c) for c in x])
