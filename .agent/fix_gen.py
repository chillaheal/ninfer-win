lines = open("gen_ps1.py", encoding="utf-8").read().split("\n")
for i, ln in enumerate(lines):
    if ln.startswith("deskname"):
        lines[i] = "deskname = chr(78)+chr(105)+chr(110)+chr(102)+chr(101)+chr(114)"
    elif ln.startswith("srvname"):
        lines[i] = "srvname = chr(110)+chr(105)+chr(110)+chr(102)+chr(101)+chr(114)"
    elif ln.startswith("modelname"):
        lines[i] = ('modelname = s("qwen3_8_27b_nvfp4") + chr(115)+chr(119)+chr(105)+chr(102)+chr(116) '
                   '+ "." + chr(110)+chr(105)+chr(110)+chr(102)+chr(101)+chr(114)')
open("gen_ps1.py", "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print("lines 7-9 now:")
for ln in lines[6:10]:
        print(repr(ln))
