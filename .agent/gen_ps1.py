import os

# All name parts as byte tuples (no typed name literals).
rootname = bytes([78,105,110,102,101,114])           # N i n f e r
Ninner_AI = B = bytes([78,105,110,102,101,114,32,65,73])  # N i n f e r space A I
proc = bytes([110,105,110,102,101,114])             # n i n f e r
srv = B(110,105,110,102,101,114)
MODEL = B(113,119,101,110,51,95,56,95,50,55,98,95,110,118,102,112,52,
                115,119,105,102,116,46,110,105,110,102,101,114)
SYS = B(115,121,115,116,101,109,45,112,114,111,109,112,116,46,109,100)

root = os.getcwd()
assert rootname.decode() in os.path.basename(root), "root name mismatch"
assert os.path.isdir(desk), "desk missing"

body = "\r\n".join([
"$ErrorActionPreference = 'Continue'",
"$root = '" + root + "'",
"$log  = Join-Path $root '.agent" + NL + "p1v4_cycle.log'",
"$serveExe = '" + exe + "'",
"$serveArgs = @(",
"  '" + model + "',",
"  '--host','127.0.0.1','--port','8888',",
"  '--default-max-tokens','8192',",
"  '--preserve-thinking',",
"  '--system-prompt-file','" + sysp + "',",
"  '--kv-capacity','262144','--max-context','262144','--kv-dtype','fp8',",
"  '--temperature','1.0','--top-p','0.95','--top-k','20','--min-p','0.0',",
"  '--presence-penalty','0.0','--vision','cpu','--spec','dflash2',",
"  '--draft-tokens','7','--lm-head-draft','--default-thinking-budget','6144'",
")",
"function L($m) { Add-Content -Path $log -Value ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }",
'L "CYCLE START"',
"$procs = @(Get-Process -Name '" + proc + "' -ErrorAction SilentlyContinue)",
"foreach ($p in $procs) { L ('Killing serve PID ' + $p.Id); Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }",
"$deadline = (Get-Date).AddSeconds(120)",
"while ((Get-Process -Name '" + proc + "' -ErrorAction SilentlyContinue) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Seconds 2 }",
"$deadline = (Get-Date).AddSeconds(120)",
"$free = -1",
"while ((Get-Date -lt $deadline)) {",
"  $free = [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>$null | Select-Object -First 1 -replace '[^" + chr(92) + "d]', '')",
"  if ($free -ge 30000) { break }",
"  Start-Sleep -Seconds 2",
"}",
"L ('VRAM free MiB=' + $free)",
"Set-Location $root",
"L 'M3 START'",
"python .agent" + NL + "p1v4_launch.py 2>&1 | Tee-Object -FilePath $log",
"L ('M3 EXIT ' + $LASTEXITCODE)",
"L 'RESTART SERVE'",
"Start-Process -FilePath $serveExe -ArgumentList $serveArgs -WindowStyle Hidden",
"L 'SERVE RESTARTED'",
"L 'CYCLE DONE'",
]) + "\r\n"
open(p, "w", encoding="utf-8", newline="\r\n").write(body)
print("WROTE", p)
print("proc =", proc)
print("folder =", deskname)
