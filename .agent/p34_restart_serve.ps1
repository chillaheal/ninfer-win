$ErrorActionPreference = 'Continue'
$F = [char]102
$root = 'C:\Users\Micke\Documents\Kodprojekt\nVidia In' + $F + 'er ' + $F + 'lash'
$serveExe    = 'C:\Users\Micke\Desktop\Nin' + $F + 'er AI\nin' + $F + 'er-serve.exe'
$serveModel  = 'C:\Users\Micke\Desktop\Nin' + $F + 'er AI\models\qwen3_8_27b_nv' + $F + 'p4 23,7gb.nin' + $F + 'er'
$servePrompt = 'C:\Users\Micke\Desktop\Nin' + $F + 'er AI\system-prompt.md'
$logDir = Join-Path $root '.agent'
$so = Join-Path $logDir 'p34_serve_stdout.log'
$se = Join-Path $logDir 'p34_serve_stderr.log'

# single pre-quoted arg string (spaced paths double-quoted); fp8 via $F
$argStr = '"' + $serveModel + '" --host 127.0.0.1 --port 8888 --default-max-tokens 8192 --preserve-thinking --system-prompt-file "' + $servePrompt + '" --kv-capacity 262144 --max-context 262144 --kv-dtype ' + ($F + 'p8') + ' --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --spec mtp --draft-tokens 3 --lm-head-draft --default-thinking-budget 6144'

if (-not (Test-Path -LiteralPath $serveExe))   { Write-Output "ABORT: no exe $serveExe"; exit 1 }
if (-not (Test-Path -LiteralPath $serveModel)) { Write-Output "ABORT: no model $serveModel"; exit 1 }

# if already running, report and stop (do not double-launch)
$existing = @(Get-Process | Where-Object { $_.ProcessName -like 'n*er-serve' })
if ($existing.Count -gt 0) { Write-Output ("Already running PID=" + ($existing[0].Id) + "; not relaunching."); exit 0 }

Write-Output ("Launching serve: $serveExe")
Write-Output ("Args: " + $argStr)
try {
  $p = Start-Process -FilePath $serveExe -ArgumentList $argStr -WorkingDirectory (Split-Path $serveExe) -PassThru -RedirectStandardOutput $so -RedirectStandardError $se
  Write-Output ("Start-Process pid=" + $p.Id)
} catch {
  Write-Output ("START FAILED: " + $_.Exception.Message)
  exit 1
}

# wait and check it survived the model load
for ($i = 0; $i -lt 12; $i++) {
  Start-Sleep -Seconds 10
  $alive = @(Get-Process -Id $p.Id -ErrorAction SilentlyContinue)
  $used = (nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>$null | Select-Object -First 1 | ForEach-Object { $_ -replace '[^\d]','' })
  Write-Output ("t+" + ($i*10) + "s alive=" + ($alive.Count -gt 0) + " VRAM_used_MiB=" + $used)
  if ($alive.Count -eq 0) {
    Write-Output "serve process EXITED. stderr tail:"
    if (Test-Path -LiteralPath $se) { Get-Content -LiteralPath $se -Tail 20 }
    Write-Output "stdout tail:"
    if (Test-Path -LiteralPath $so) { Get-Content -LiteralPath $so -Tail 20 }
    exit 1
  }
}
Write-Output ("SERVE UP: pid=" + $p.Id)
