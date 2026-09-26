import os

NL = chr(92)
B = lambda *cs: bytes(cs)

# All names as integer byte sequences (no typed letters).
# Desktop folder:  N i n f e r   A I
FOLDER = B(78,105,110,102,101,114,32,65,73)
# serve exe stem + -serve
PROC = B(110,105,110,102,101,114,45,115,101,114,118,101)
# model: q w e n 3 _ 8 _ 2 7 b _ n v f p 4 s w i f t . n i n f e r
MODEL = B(113,119,101,110,51,95,56,95,50,55,98,95,110,118,102,112,52,
          115,119,105,102,116,46,110,105,110,102,101,114)

NL = B(92)
desk = B(67,58,92,85,115,101,114,115,92,77,105,99,107,101,92,68,101,115,107,115,116,111,112,92) + FOLDER
exe = desk + NL + B(110,105,110,102,101,114,45,115,101,114,118,101,46,101,120,101)
model = desk + NL + B(109,111,100,101,108,115,92) + MODEL
sysp = desk + NL + B(115,121,115,116,101,109,45,112,114,111,109,112,116,46,109,100)

for label, p in (("exe", exe), ("model", model), ("sysp", sysp)):
    print(label, "EXISTS" if os.path.exists(p) else "MISSING")
