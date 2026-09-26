$ErrorActionPreference = 'Stop'
# Bulletproof: build "Ninfer" (N,i,n,f,e,r) from char codes.
$ni = -join [char[]]@(78,105,110,102,101,114)          # "Ninfer"
$niw = 'C:\Users\Micke\Documents\Kodprojekt\' + $ni + '\ninfer-win'
Write-Output ("core = " + $niw + "  exists=" + (Test-Path -LiteralPath $niw))

# Overlay source = this repo's src\targets\qwen3_8_flash_next.
$ov = 'C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\src\targets\qwen3_8_flash_next'
if (-not (Test-Path -LiteralPath $niw)) { throw "core missing: $niw" }
if (-not (Test-Path -LiteralPath $ov))  { throw "overlay missing: $ov" }

# 1) copy overlay target dir into core tree (reversible, additive)
Copy-Item -LiteralPath $ov -Destination $dst -Recurse -Force
Write-Output "copied overlay -> " + $dst

# 2) add add_subdirectory after the qwen3_6_35b_a3b line (idempotent)
$raw = [System.IO.File]::ReadAllText($cm)
if ($raw -notmatch 'qwen3_8_flash_next') {
  $anchor = 'add_subdirectory(targets/qwen3_6_35b_a3b)'
  if ($raw -notmatch [regex]::Escape($anchor)) { throw 'anchor not found' }
  $nl = "`n"; if ($raw -match "`r`n") { $nl = "`r`n" }
  $new = $raw -replace [regex]::Escape($anchor), ($anchor + $nl + 'add_subdirectory(targets/qwen3_8_flash_next)')
  [System.IO.File]::WriteAllText($cm, $new, (New-Object System.Text.UTF8Encoding($false)))
  Write-Output "EDIT: appended add_subdirectory(targets/qwen3_8_flash_next)"
} else {
  Write-Output "already attached (no edit)"
}
if ([System.IO.File]::ReadAllText($cm) -match 'add_subdirectory\(targets/qwen3_8_flash_next\)') {
  Write-Output "VERIFIED: add_subdirectory present"
} else { throw 'FAILED: add_subdirectory missing' }
Write-Output "DONE"
