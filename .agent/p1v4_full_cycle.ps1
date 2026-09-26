$ErrorActionPreference = 'Continue'
$root = 'C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash'
$log  = Join-Path $root '.agent\p1v4_cycle.log'
$serveExe = 'C:\Users\Micke\Desktop\Ninfer AI\ninfer-serve.exe'
$serveArgs = @(
  'C:\Users\Micke\Desktop\Ninfer AI\models\qwen3_8_27b_nvfp4swift.ninner',
  '--host','127.0.0.1','--port','8888',
  '--default-max-tokens','8192',
  '--preserve-thinking',
  '--system-prompt-file','C:\Users\Micke\Desktop\Ninfer AI\system-prompt.md',
  '--kv-capacity','262144','--max-context','262144','--kv-dtype','fp8',
  '--temperature','1.0','--top-p','0.95','--top-k','20','--min-p','0.0',
  '--presence-penalty','0.0','--vision','cpu','--spec','dflash2',
  '--draft-tokens','7','--lm-head-draft','--default-thinking-budget','6144'
)
function L($m) { Add-Content -Path $log -Value ("{0:HH:mm:ss} {1}" -f (Get-Date), $m) }

L "CYCLE START"
# 1. Stop serve
$procs = @(Get-Process -Name 'ninfer-serve' -ErrorAction SilentlyContinue)
foreach ($p in $procs) { L ("Killing serve PID " + $p.Id); Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
$d1 = (Get-Date).AddSeconds(120)
while ((Get-Process -Name 'ninfer-serve' -ErrorAction SilentlyContinue) -and ((Get-Date) -lt $d1)) { Start-Sleep -Seconds 2 }
$free = -1
$d2 = (Get-Date).AddSeconds(120)
while (((Get-Date) -lt $d2)) {
  $free = [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>$null | Select-Object -First 1 | ForEach-Object { $_ -replace '[^\d]','' })
  if ($free -ge 30000) { break }
  Start-Sleep -Seconds 2
}
L ("VRAM free MiB=" + $free)
# 2. M3 test
Set-Location $root
L "M3 START"
python .agent\p1v4_launch.py 2>&1 | Tee-Object -FilePath $log
L ("M3 EXIT " + $LASTEXITCODE)
# 3. Restart serve
L "RESTART SERVE"
Start-Process -FilePath $serveExe -ArgumentList $serveArgs -WindowStyle Hidden
L "SERVE RESTARTED"
L "CYCLE DONE"
