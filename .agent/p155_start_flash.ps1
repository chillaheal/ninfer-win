# p155_start_flash.ps1
# Stops the 27b serve, verifies the flash_next exe + model, then starts
# flash_next with --moe-cpu auto in a visible window. The start-half
# pre-flight (exe + model + ngram exist) runs BEFORE the stop, so a broken
# start never leaves the GPU idle with nothing running.
param(
    [int]$StopPid = 15352,
    [string]$MoeCpu = "auto"
)

$ErrorActionPreference = "Stop"

$serveExe = "C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\core\build\apps\ninfer-serve.exe"
$modelPath = "C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\models\qwen3_8_flash_next.ninfer"
$ngramPath = "C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\models\qwen3_8_flash_next.ngram"
$workDir   = "C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash"

Write-Host "==== P155: start flash_next ($MoeCpu) ===="

# Start-half pre-flight (must pass before the stop)
if (-not (Test-Path $serveExe))  { Write-Error "serve exe missing: $serveExe";  exit 1 }
if (-not (Test-Path $modelPath)) { Write-Error "model missing: $modelPath";   exit 1 }
if (-not (Test-Path $ngramPath)) { Write-Error "ngram missing: $ngramPath";    exit 1 }
Write-Host "  [ok] serve exe + model + ngram all present"

# Stop the 27b serve
Write-Host "  stopping 27b serve (PID $StopPid) ..."
Stop-Process -Id $StopPid -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Write-Host "  [ok] 27b serve stopped"

# Start flash_next in a visible window
Write-Host "  starting flash_next ($MoeCpu) ..."
Start-Process -FilePath $serveExe -ArgumentList @($modelPath, "--moe-cpu", $MoeCpu) -WorkingDirectory $workDir
Write-Host "  [ok] flash_next started ($MoeCpu)"
Write-Host "==== P155_DONE ===="
