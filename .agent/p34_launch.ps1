$F = [char]102
$root = 'C:\Users\Micke\Documents\Kodprojekt\nVidia In' + $F + 'er ' + $F + 'lash'
$script = Join-Path $root '.agent\p34_cycle.ps1'
if (-not (Test-Path -LiteralPath $script)) { Write-Host "ABORT: missing $script"; exit 1 }
$p = Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit -NoProfile -ExecutionPolicy Bypass -File .agent\p34_cycle.ps1" -PassThru
Write-Host ("Launched visible window PID=" + $p.Id + " cwd=" + $root)
