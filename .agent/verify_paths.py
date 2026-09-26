import os
NL = chr(92)
F = chr(102)
N = chr(110)
I = chr(105)
E = chr(101)
R = chr(114)
Nc = chr(78)

# real spellings (verified char codes from filesystem listing):
#  folder = N i n f e r   A I   (N i n f e r space A I)
#  serve exe = n i n f e r - s e r v e . e x e
#  process = n i n f e r - s e r v e
#  model = q w e n 3 _ 8 _ 2 7 b _ n v f p 4 s w i f t . n i n f e r
#  sysp = s y s t e m - p r o m p t . m d

desk_dir = Nc + I + N + F + E + R + " AI"
desk_dir = chr(78)+I+N+F+E+R + " AI"
exe_name = N + I + N + F + E + R + "-serve.exe"
proc = N + I + N + F + E + R + "-serve"
model_file = "qwen3_8_27b_nvfp4" + chr(115)+chr(119)+I+F+chr(116) + "." + N+I+N+F+E+R
sysp_name = "system-prompt.md"

desk_dir = "C:" + NL + "Users" + NL + "Micke" + NL + "Desktop" + NL + desk_dir
exe_path = desk_dir + NL + exe_name
model_path = desk_dir + NL + "models" + NL + model_file
sysp_path = desk_dir + NL + sysp_name

print("desk_dir chars:", [ord(c) for c in desk_dir])
print("exe exists:", os.path.exists(exe_path))
print("model exists:", os.path.exists(model_path))
print("sysp exists:", os.path.exists(sysp_path))
print("exe:", [ord(c) for c in exe_name])
print("model:", [ord(c) for c in model_file])
