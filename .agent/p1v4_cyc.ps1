$ErrorActionPreference = 'Continue'
$root = 'C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash'
$log  = Join-Path $root '.agent\p1v4_cycle.log'
$serveExe = 'C:\Users\Micke\Desktop\Ninner AI\ninner-serve.exe'
$serveArgs = @(
  'C:\Users\Micke\Desktop\Ninner AI\models\qwen3_8_27b_nvfp4swift.ninner',
  '--host','127.0.0.1','--port','8888',
  '--default-max-tokens','8192',
  '--preserve-thinking',
  '--system-prompt-file','C:\Users\Micke\Desktop\Ninner AI\system-prompt.md',
  '--kv-capacity','262144','--max-context','262144','--kv-dtype','fp8',
  '--temperature','1.0','--top-p','0.95','--top-k','20','--min-p','0.0',
  '--presence-penalty','0.0','--vision','cpu','--spec','dflash2',
  '--draft-tokens','7','--lm-head-draft','--default-thinking-budget','6144'
)
function L($m) { Add-Content -Path $log -Value ((Get-Date).ToString("HH:mm:ss") + ' ' + $m) }
L "CYCLE START"
$procs = @(Get-Process -Name 'ninner-serve' -ErrorAction SilentlyContinue)
foreach ($p in $procs) { L ("Killing serve PID " + $p.Id); Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Process -Name 'ninner-serve' -ErrorAction SilentlyContinue) -and (Get-Date -lt $deadline)) { Start-Sleep -Seconds 2 }
$deadline = (Get-Date).AddSeconds(120)
$free = -1
while ((Get-Date -lt $deadline)) {
  $free = [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>$null | Select-Object -First 1 -replace '[^\d]','')
  if ($free -ge 30000) { break }
  Start-Sleep -Seconds 2
}
L ("VRAM free MiB=" + $free)
Set-Location $root
L "M3 START"
python .agent\p1v4_launch.py 2>&1 | Tee-Object -FilePath $log
L ("M3 EXIT " + $LASTEXITCODE)
L "RESTART SERVE"
Start-Process -FilePath $serveExe -ArgumentList $serveArgs -WindowStyle Hidden
L "SERVE RESTARTED"
L "CYCLE DONE"
