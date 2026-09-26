import os
f = chr(102)
n = chr(110)
N = chr(78)
NL = chr(92)

deskname = N + chr(105) + n + n + chr(101) + n      # desktop folder
srvname = n + chr(105) + n + n + chr(101) + n       # serve exe stem
modelname = "qwen3_8_27b_nvfp4swift." + n + chr(105) + n + n + chr(101) + n

desk = "C:" + NL + "Users" + NL + "Micke" + NL + "Desktop" + NL + deskname
print("desk folder:", [ord(c) for c in deskname])
exe = desk + NL + srvname + ".exe"
model = desk + NL + "models" + NL + modelname
sysp = desk + NL + "system-prompt.md"
print("exe exists:", os.path.exists(exe))
print("model exists:", os.path.exists(model))
print("sysprompt exists:", os.path.exists(sysp))
