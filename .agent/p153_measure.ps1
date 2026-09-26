$ErrorActionPreference = 'Continue'
$F = [char]102   # lowercase f (0x66); built via char to be mangle-proof

# ---- project paths (all f's via $F) ----
$root      = 'C:\Users\Micke\Documents\Kodprojekt\nVidia In' + $F + 'er ' + $F + 'lash'
$exeName   = $F + 'lash_next_real_load.exe'
$modelName = 'qwen3_8_' + $F + 'lash_next.nin' + $F + 'er'
$ngramName = 'qwen3_8_' + $F + 'lash_next.ngram'
$log       = Join-Path $root '.agent\p153_measure.log'
$testOut   = Join-Path $root '.agent\p153_test_stdout.log'
$testErr   = Join-Path $root '.agent\p153_test_stderr.log'
$envMoe    = 'NIN' + $F + 'ER_MOE_CPU_LAYERS'
$envM3     = 'NIN' + $F + 'ER_P13_M3'
$envTim    = 'NIN' + $F + 'ER_P13_TIMING'
$envMaxTP  = 'NIN' + $F + 'ER_P13_M3_MAXTP'

# ---- serve restart oracle (copied verbatim from p34_cycle.ps1, 2026-09-25; exact argv via $F) ----
$serveExe    = 'C:\Users\Micke\Desktop\Nin' + $F + 'er AI\nin' + $F + 'er-serve.exe'
$serveModel  = 'C:\Users\Micke\Desktop\Nin' + $F + 'er AI\models\qwen3_8_27b_nv' + $F + 'p4 23,7gb.nin' + $F + 'er'
$servePrompt = 'C:\Users\Micke\Desktop\Nin' + $F + 'er AI\system-prompt.md'
# single pre-quoted arg string (array -ArgumentList drops quotes on the spaced model path)
$argStr = '"' + $serveModel + '" --host 127.0.0.1 --port 8888 --default-max-tokens 8192 --preserve-thinking --system-prompt-file "' + $servePrompt + '" --kv-capacity 262144 --max-context 262144 --kv-dtype ' + ($F+'p8') + ' --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --spec mtp --draft-tokens 3 --lm-head-draft --default-thinking-budget 6144'

function W($m) { Write-Host ("{0:HH:mm:ss} {1}" -f (Get-Date), $m) }

Set-Location $root
Start-Transcript -Path $log -Force | Out-Null
W "P1.5 MEASURE START (CPU-OFF, NINFER_P13_M3=1, NINFER_P13_TIMING=1, NINFER_P13_M3_MAXTP=512, 45-min hard kill)"

# ---- 0. Record live serve oracle (wildcard name; cmdline by PID) ----
$live = @(Get-Process | Where-Object { $_.ProcessName -like 'n*er-serve' })
$serveWasRunning = ($live.Count -gt 0)
if ($serveWasRunning) {
  $sPID = $live[0].Id
  $sCIM = Get-CimInstance Win32_Process -Filter "ProcessId=$sPID"
  W ("LIVE SERVE PID=" + $sPID)
  W ("LIVE SERVE CMDLINE: " + $sCIM.CommandLine)
} else {
  W "NOTE: no running serve detected; proceeding (nothing to stop)."
}

# ---- 1. Verify exe + model + ngram BEFORE touching serve ----
$exe   = Join-Path $root ('build\tests\' + $exeName)
$model = Join-Path $root ('models\' + $modelName)
$ngram = Join-Path $root ('models\' + $ngramName)
$ok = $true
if (-not (Test-Path -LiteralPath $exe))   { W "MISSING exe:   $exe";   $ok = $false }
if (-not (Test-Path -LiteralPath $model)) { W "MISSING model: $model"; $ok = $false }
if (-not (Test-Path -LiteralPath $ngram)) { W "MISSING ngram: $ngram"; $ok = $false }
if ($serveWasRunning -and -not (Test-Path -LiteralPath $serveExe)) { W "MISSING serveExe: $serveExe"; $ok = $false }
if (-not $ok) { W "ABORT: not killing serve. Fix paths above and re-run."; Stop-Transcript | Out-Null; exit 1 }
W "exe:   $exe"
W "model: $model"
W "ngram: $ngram"

$killed = $false
# ---- 2. Stop serve ----
if ($serveWasRunning) {
  $procs = @(Get-Process | Where-Object { $_.ProcessName -like 'n*er-serve' })
  foreach ($p in $procs) {
    W ("Killing serve PID " + $p.Id + " (" + $p.ProcessName + ")")
    try { Stop-Process -Id $p.Id -Force -ErrorAction Stop; W ("Stop-Process ok PID " + $p.Id) }
    catch { W ("Stop-Process FAILED: " + $_.Exception.Message)
            & taskkill /F /T /PID $p.Id 2>&1 | ForEach-Object { W ("taskkill: " + $_) } }
  }
  $deadline = (Get-Date).AddSeconds(120)
  while ((Get-Process | Where-Object { $_.ProcessName -like 'n*er-serve' }) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Seconds 2 }
  if (Get-Process | Where-Object { $_.ProcessName -like 'n*er-serve' }) { W "ABORT: serve still alive after kill."; Stop-Transcript | Out-Null; exit 1 }
  $killed = $true
  W "serve stopped."
}

# ---- 3. Wait for VRAM to free (>= 30000 MiB) so the 75 GB model's LRU/pinned fit ----
$free = -1; $deadline2 = (Get-Date).AddSeconds(180)
while ((Get-Date) -lt $deadline2) {
  $free = [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>$null | Select-Object -First 1 | ForEach-Object { $_ -replace '[^\d]','' })
  W ("VRAM free MiB=" + $free)
  if ($free -ge 30000) { break }
  Start-Sleep -Seconds 5
}
if ($killed -and $free -lt 30000) {
  W "VRAM free did not reach 30000 MiB (last=$free); restarting serve and aborting test."
} else { W "VRAM OK." }

if (-not $killed -or $free -ge 30000) {
  # ---- 4. Bounded M3 T=1 decode, CPU OFF (P1.5.3 device top-k), MAXTP=512 ----
  [Environment]::SetEnvironmentVariable($envMoe, $null, 'Process')   # CPU OFF (default)
  [Environment]::SetEnvironmentVariable($envM3, '1', 'Process')
  [Environment]::SetEnvironmentVariable($envTim, '1', 'Process')
  [Environment]::SetEnvironmentVariable($envMaxTP, '512', 'Process')
  W "M3 CPU-OFF START (NINFER_P13_M3_MAXTP=512, 45-min hard kill)"
  # Single pre-quoted arg string: the model/ngram paths live under the spaced
  # project root ("nVidia Infer Flash"), so an unquoted -ArgumentList array would
  # split them on spaces and argv[3] would no longer be the input-token "0".
  $testArgs = '"' + $model + '" "' + $ngram + '" 0'
  $tp = Start-Process -FilePath $exe -ArgumentList $testArgs -PassThru `
        -RedirectStandardOutput $testOut -RedirectStandardError $testErr -WindowStyle Hidden
  W ("TEST PID=" + $tp.Id)
  $testDeadline = (Get-Date).AddMinutes(45)
  $timedOut = $false
  while (-not $tp.HasExited -and (Get-Date) -lt $testDeadline) { Start-Sleep -Seconds 10 }
  if (-not $tp.HasExited) {
    $timedOut = $true
    W "TIMEOUT: test did not finish in 45 min; killing."
    try { Stop-Process -Id $tp.Id -Force -ErrorAction Stop } catch { W ("kill failed: " + $_.Exception.Message) }
    Start-Sleep -Seconds 5
  }
  $code = if ($timedOut) { -2 } else { $tp.ExitCode }
  W ("TEST EXIT code=$code timeout=" + $timedOut)
} else {
  W "M3 SKIPPED (no VRAM)."
  $code = -1
  $timedOut = $false
}

# ---- 5. Wait for the test's VRAM to free (>= 30000 MiB) before restarting the 27B serve ----
if ($killed) {
  $free2 = -1; $deadline3 = (Get-Date).AddSeconds(180)
  while ((Get-Date) -lt $deadline3) {
    $free2 = [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>$null | Select-Object -First 1 | ForEach-Object { $_ -replace '[^\d]','' })
    if ($free2 -ge 30000) { break }
    Start-Sleep -Seconds 5
  }
  W ("post-test VRAM free MiB=" + $free2)
}

# ---- 6. Restart serve (VISIBLE window; exact fresh argv) ----
if ($killed) {
  W "RESTART SERVE (visible): $serveExe"
  try {
    $sp = Start-Process -FilePath $serveExe -ArgumentList $argStr -WorkingDirectory (Split-Path $serveExe) -PassThru
    W ("RESTART START-PROCESS pid=" + $sp.Id)
  } catch { W ("RESTART FAILED: " + $_.Exception.Message) }
} else {
  W "RESTART SKIPPED (serve was not stopped)."
}

# ---- 7. Verify serve is back (process + VRAM) ----
Start-Sleep -Seconds 30
$back = @(Get-Process | Where-Object { $_.ProcessName -like 'n*er-serve' })
$used = (nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>$null | Select-Object -First 1 | ForEach-Object { $_ -replace '[^\d]','' })
W ("VERIFY serve procs=" + $back.Count + " VRAM_used_MiB=" + $used)
W "================================================"
W ("  P1.5 MEASURE DONE (test exit=$code; timeout=$timedOut; serve restarted=$($back.Count -gt 0))")
W ("  Test stdout: $testOut")
W ("  Test stderr: $testErr")
W "  Log: $log"
W "================================================"
Stop-Transcript | Out-Null
exit $code
