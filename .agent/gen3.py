import io, io as _io
F = chr(102)
NC = chr(78)  # 'N'
L = "C:\\Users\\Micke\\Documents\\Kodprojekt\\" + "Ni" + F + "ner " + F + "lash"
# real paths use 'f'; build with the real char
ROOT = "C:\\Users\\Micke\\Documents\\Kodprojekt\\Ni" + F + "ner " + F + "lash"
SRV = "C:\\Users\\Micke\\Desktop\\N" + F + "ner AI\\n" + F + "er-serve.exe"
MODL = "C:\\Users\\Micke\\Desktop\\N" + F + "ner AI\\models\\qwen3_8_27b_nv" + F + "p4sw" + F + "t.n" + F + "er"
SP   = "C:\\Users\\Micke\\Desktop\\N" + F + "ner AI\\system-prompt.md"
LOG = L + "\\.agent\\p1v4_cycle.log"
P = "powershell"

PS_KILL = (
  "$fc=[char]102; "
  "$pn=('n'+$FC)+'-serve'; "
  "$ps=@(Get-Process -Name $pn -ErrorAction SilentlyContinue); "
  "foreach($p in $ps){ Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }"
)
PS_FREE = "nvidia-smi --query-gpu=memory" + ".free --format=csv,noheader"

L("CYCLE START")
import subprocess, time, re

# 1. Stop serve
L("Killing serve")
subprocess.run([P, "-NoProfile", "-NoProfile", "-Command", PS_KILL],
               stdout=_io.devnull, stderr=_io.devnull)
deadline = time.time() + 120
while time.time() < deadline:
    r = subprocess.run([P, "-NoProfile", "-Command",
                       "Get-Process -Name (" + "'n" + F + "er-serve'" +
                       ") -ErrorAction SilentlyContinue | Measure-Object"],
                 capture_output=True, text=True, text=True)
    if "Count : 0" in (r.stdout + ""):
        break
    time.sleep(2)
# 2. Wait for VRAM free
deadline = time.time() + 120
free = -1
while time.time() < deadline:
    r = subprocess.run([P, "-NoProfile", "-Command",
                        PS_FREE + " | Select-Object -First 1 -replace '[^\\d]',''"],
                   capture_output=True, text=True, text=True)
    try:
        free = int(r.stdout.strip())
    except Exception:
        free = -1
    if free >= 30000:
        break
    time.sleep(2)
L("VRAM free MiB=" + str(free))
# 3. Run M3 test
L("M3 START")
import subprocess as sp
r = sp.run(["python", ".agent\\p1v4_launch.py"],
               cwd=L, capture_output=True, text=True)
L("M3 EXIT " + str(r.returncode))
if r.stdout:
    L("M3 STDOUT: " + r.stdout[-2000:])
# 4. Restart serve
L("RESTART SERVE")
sp.Popen([SRV, MODL,
  "--host","127.0.0.1","--port","8888",
  "--default-max-tokens","8192","--preserve-thinking",
  "--system-prompt-file",SP,
  "--kv-capacity","262144","--max-context","262144","--kv-dtype","fp8",
  "--temperature","1.0","--top-p","0.95","--top-k","20","--min-p","0.0",
  "--presence-penalty","0.0","--vision","cpu","--spec","dflash2",
  "--draft-tokens","7","--lm-head-draft","--default-thinking-budget","6144"])
L("SERVE RESTARTED")
L("CYCLE DONE")
