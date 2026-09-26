$ErrorActionPreference = 'Continue'
$root = 'C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash'
$log  = Join-Path $root '.agent\deploy_build_run.log'
Set-Location $root
Start-Transcript -Path $log -Force | Out-Null
Write-Host ("START {0:yyyy-MM-dd HH:mm:ss} DEPLOY BUILD (visible window)" -f (Get-Date))
cmd /c ".agent\deploy_build.bat"
$code = $LASTEXITCODE
Write-Host ("END {0:yyyy-MM-dd HH:mm:ss} deploy build exit={1}" -f (Get-Date), $code)
Stop-Transcript | Out-Null
exit $code
