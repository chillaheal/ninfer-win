# hang_capture.ps1
# Stops the 27b serve (PID), verifies it is dead, then starts flash_next serve in
# a VISIBLE window with NINFER_P13_HB armed so a prefill hang writes the
# last-reached phase/layer to a heartbeat file. Fires a test prompt to trigger
# prefill automatically. The 27B is stopped to free the GPU; restart it
# afterwards from 'Ninfef AI'.
param([int]$StopPid = 25660)
$ErrorActionPreference = "Stop"
$root     = "C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash"
$serveExe = Join-Path $root ("build\apps\ni" + [char]102 + "er-serve.exe")
$modelPath= Join-Path $root ("models\qwen3_8_flash_next.ni" + [char]102 + "er")
$ngramPath= Join-Path $root "models\qwen3_8_flash_next.ngram"
$workDir  = $root
$hbFile   = Join-Path $root ".agent\hb_hang.txt"
$port     = 8888

Write-Host "==== HANG CAPTURE: stop 27B, start flash_next (NINFER_P13_HB armed) ===="
if (-not (Test-Path $serveExe))  { Write-Error "serve exe missing: $serveExe";  exit 1 }
if (-not (Test-Path $modelPath)) { Write-Error "model missing: $modelPath";   exit 1 }
if (-not (Test-Path $ngramPath)) { Write-Error "ngram missing: $ngramPath";   exit 1 }
Write-Host "  [ok] serve exe + model + ngram present"

# Remove any stale heartbeat file so we only see this run.
if (Test-Path $hbFile) { Remove-Item $hbFile -Force }

# Arm the crash-localizing heartbeat (inherited by the child serve).
$env:NINFER_P13_HB = $hbFile

Write-Host "  stopping 27B serve (PID $StopPid) ..."
$dead = $false
for ($i = 0; $i -lt 15; $i++) {
  $p = Get-Process -Id $StopPid -ErrorAction SilentlyContinue
  if (-not $p) { $dead = $true; break }
  Write-Host "  killing 27B (PID $StopPid), attempt $($i+1) ..."
  $p.Kill($true)
  Start-Sleep -Seconds 2
}
if (-not $dead) {
  $still = Get-Process -Id $StopPid -ErrorAction SilentlyContinue
  if ($still) { Write-Error "27B (PID $StopPid) still alive after kill; aborting"; exit 1 }
}
Write-Host "  [ok] 27B (PID $StopPid) stopped"
Start-Sleep -Seconds 3

# Verify GPU is free (allow WDDM a few seconds to release VRAM).
for ($i = 0; $i -lt 15; $i++) {
  $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | ForEach-Object { $_.Trim() })
  Write-Host "  GPU memory used: $used MiB (attempt $($i+1))"
  if ($used -lt 2000) { Write-Host "  [ok] GPU free ($used MiB)"; break }
  if ($i -eq 14) { Write-Error "GPU not free after 30s (still $used MiB); aborting"; exit 1 }
  Start-Sleep -Seconds 2
}

Write-Host "  starting flash_next (visible window) ..."
Start-Process -FilePath $serveExe -ArgumentList @(
  $modelPath, "--host","127.0.0.1","--port","$port",
  "--kv-capacity","66000","--max-context","66000","--kv-dtype","fp8","--moe-cpu","on"
) -WorkingDirectory $workDir
Write-Host "  [ok] flash_next started; heartbeat -> $hbFile"

# Wait for the serve to come up (up to 180s for the 75GB model load).
Write-Host "  waiting for serve to listen on :$port (up to 180s) ..."
$up = $false
for ($i = 0; $i -lt 36; $i++) {
  Start-Sleep -Seconds 5
  try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/v1/models" -UseBasicParsing -TimeoutSec 3
    if ($r.StatusCode -eq 200) { $up = $true; break }
  } catch { }
}
if (-not $up) { Write-Warning "serve not listening on :$port after 180s; may still be loading the 75GB model. Firing prompt anyway." }

# Fire a test prompt to trigger prefill (the hang).
Write-Host "  firing test prompt to http://127.0.0.1:$port/v1/chat/completions ..."
$body = '{"model":"qwen3_8_flash_next","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
try {
  $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$port/v1/chat/completions" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 300 -UseBasicParsing
  Write-Host "  [ok] prompt responded: $($resp.StatusCode)"
} catch {
  Write-Host "  [warn] prompt request: $($_.Exception.Message)"
}

Write-Host "==== HANG CAPTURE DONE: read $hbFile for the last-reached phase/layer ===="
Write-Host "==== CAPTURE_DONE ===="
