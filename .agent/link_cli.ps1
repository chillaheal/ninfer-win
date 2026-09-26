$ErrorActionPreference = 'Stop'
$src = 'C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\core\build\apps'
$dst = 'C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\build\apps'
New-Item -ItemType Directory -Path $dst -Force | Out-Null
$files = @('ninfer-cli.exe','avcodec-63.dll','avformat-63.dll','avutil-61.dll','libcurl.dll','msvcp140.dll','swresample-7.dll','swscale-10.dll','vcruntime140.dll','vcruntime140_1.dll','z.dll')
foreach ($f in $files) {
  $t = Join-Path $src $f
  $l = Join-Path $dst $f
  if (-not (Test-Path -LiteralPath $t)) { Write-Host "MISSING src: $t"; continue }
  if (Test-Path -LiteralPath $l) { Write-Host "exists: $f"; continue }
  New-Item -ItemType HardLink -Path $l -Target $t | Out-Null
  Write-Host "linked: $f"
}
Write-Host ('cli exists at dst: ' + (Test-Path -LiteralPath (Join-Path $dst 'ninfer-cli.exe')))
