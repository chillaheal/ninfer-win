$ErrorActionPreference = 'Stop'
# Locate the core tree robustly by searching for its CMakeCache (avoids typing the homoglyph).
$cache = Get-ChildItem -Path 'C:\Users\Micke\Documents\Kodprojekt' -Recurse -Filter 'CMakeCache.txt' -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -like '*ninner-win\build\CMakeCache.txt' } | Select-Object -First 1
if (-not $cache) { Write-Output 'CMAKECACHE_NOT_FOUND'; exit 0 }
$build = $cache.Directory.FullName
$core = Split-Path -LiteralPath $build -Parent
Write-Output ("BUILD_DIR=" + $build)
Write-Output ("CORE_DIR=" + $core)

$testsCml = Join-Path $core 'tests\CMakeLists.txt'
Write-Output "=== tests\CMakeLists.txt flash_next/add_test/LABELS ==="
Get-Content -LiteralPath $testsCml | Select-String -Pattern 'flash_next|add_test|LABELS' | ForEach-Object { "{0}: {1}" -f $_.LineNumber, $_.Line }
