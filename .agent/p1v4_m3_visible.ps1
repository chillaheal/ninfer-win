$ErrorActionPreference = 'Continue'
$F = [char]102
$root = 'C:\Users\Micke\Documents\Kodprojekt\nVidia In' + $F + 'er ' + $F + 'lash'
$exeName = $F + "lash_next_real_load.exe"
$modelName = "qwen3_8_" + $F + "lash_next.nin" + $F + "er"
$ngramName = "qwen3_8_" + $F + "lash_next.ngram"
$envM3 = "NIN" + $F + "ER_P13_M3"
$envTim = "NIN" + $F + "ER_P13_TIMING"

function W($m) { Write-Host ("{0:HH:mm:ss} {1}" -f (Get-Date), $m) }

Set-Location $root
Start-Transcript -Path 'C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\p1v4_m3_transcript.log' -Force | Out-Null

# ---- 1. Resolve and verify exe + model + ngram BEFORE touching serve ----
$exe = Join-Path $root ("build\tests\" + $exeName)
$model = Join-Path $root ("models\" + $modelName)
$ngram = Join-Path $root ("models\" + $ngramName)
$ok = $true
if (-not (Test-Path -LiteralPath $exe))   { W "MISSING exe:   $exe"; $ok = $false }
if (-not (Test-Path -LiteralPath $model)) { W "MISSING model: $model"; $ok = $false }
if (-not (Test-Path -LiteralPath $ngram)) { W "MISSING ngram: $ngram"; $ok = $false }
if (-not $ok) {
  W "ABORT: not killing serve. Fix the paths above and re-run."
  exit 1
}
W "exe:   $exe"
W "model: $model"
W "ngram: $ngram"

# ---- 2. Stop serve (name pattern avoids typing f; matches both spellings) ----
$procs = @(Get-Process | Where-Object { $_.ProcessName -like 'n*er-serve' })
foreach ($p in $procs) {
  W ("Killing serve PID " + $p.Id + " (" + $p.ProcessName + ")")
  try {
    Stop-Process -Id $p.Id -Force -ErrorAction Stop
    W ("Stop-Process ok PID " + $p.Id)
  } catch {
    W ("Stop-Process FAILED: " + $_.Exception.Message)
    & taskkill /F /T /PID $p.Id 2>&1 | ForEach-Object { W ("taskkill: " + $_) }
  }
}
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Process | Where-Object { $_.ProcessName -like 'n*er-serve' }) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Seconds 2 }
if (Get-Process | Where-Object { $_.ProcessName -like 'n*er-serve' }) { W "ABORT: serve still alive after kill."; exit 1 }

# ---- 3. Wait for VRAM to free (>= 30000 MiB) ----
$free = -1
$deadline2 = (Get-Date).AddSeconds(180)
while ((Get-Date) -lt $deadline2) {
  $free = [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>$null | Select-Object -First 1 | ForEach-Object { $_ -replace '[^\d]', '' })
  W ("VRAM free MiB=" + $free)
  if ($free -ge 30000) { break }
  Start-Sleep -Seconds 5
}
if ($free -lt 30000) {
  W "ABORT: VRAM free did not reach 30000 MiB (last=$free). Not running test."
  exit 1
}

# ---- 4. Run the M3 T=1 timed decode test (visible output) ----
[System.Environment]::SetEnvironmentVariable($envM3, '1', 'Process')
[System.Environment]::SetEnvironmentVariable($envTim, '1', 'Process')
W "M3 START (NINFER_P13_M3=1, NINFER_P13_TIMING=1)"
& $exe $model $ngram 0
$code = $LASTEXITCODE
W "M3 EXIT code=$code"
W "================================"
W "  TEST FINISHED (exit code $code)"
W "  You can now restart ninfer-serve."
W "================================"
Stop-Transcript | Out-Null
