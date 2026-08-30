#Requires -Version 5.1
<#
build_windows.ps1 -- reproducible Windows build + dist assembly for NInfer (plan D9).

Steps: 1) check tools   2) CMake configure   3) build Release
       4) assemble dist/  5) verify + report

dist/ layout (D8): Ninfer.exe, ninfer-cli.exe, ninfer-serve.exe, VC runtime DLLs,
libcurl+zlib, the FFmpeg shared libs actually imported by the apps (dumpbin-verified).
The CUDA runtime is linked STATICALLY into the exes -- no CUDA DLLs to ship.
The GPU driver API (nvcuda.dll) comes from the installed NVIDIA driver.

Usage:
  powershell -NoProfile -ExecutionPolicy Bypass -File build_windows.ps1          # incremental
  powershell -NoProfile -ExecutionPolicy Bypass -File build_windows.ps1 -Clean   # from scratch
#>
param([switch]$Clean)

$ErrorActionPreference = "Stop"
$root  = $PSScriptRoot
$src   = Join-Path $root "ninfer-win"
$build = Join-Path $src  "build"
$dist  = Join-Path $root "dist"
$deps  = Join-Path $root "deps"

function Step([string]$msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------
Step "1/5 Tool check"
# ---------------------------------------------------------------------------
# VsDevCmd lives in Common7\Tools for BuildTools installs and VC\Auxiliary\Build
# for full VS installs -- try both.
$btRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$vsdevCandidates = @(
    (Join-Path $btRoot "Common7\Tools\VsDevCmd.bat"),
    (Join-Path $btRoot "VC\Auxiliary\Build\VsDevCmd.bat")
)
$vsdev = $vsdevCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vsdev) { throw "VsDevCmd.bat not found under $btRoot -- install VS 2022 Build Tools with the C++ workload" }

function Find-Tool([string]$name, [string[]]$wingetGlobs) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($g in $wingetGlobs) {
        $hit = Get-ChildItem $g -Filter "$name.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    throw "$name not found (not on PATH, no winget package match)"
}
$cmake = Find-Tool "cmake" @("$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Kitware.CMake*")
$ninja = Find-Tool "ninja" @("$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ninja-build.Ninja*")

# CUDA toolkit: auto-detect newest v13.x (winget installs have no registry key).
$cudaRoot = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^v13\.' } | Sort-Object Name -Descending | Select-Object -First 1
if (-not $cudaRoot) { throw "No CUDA v13.x toolkit found under C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" }
$nvcc = Join-Path $cudaRoot.FullName "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) { throw "nvcc not found at $nvcc" }

# VC redist runtime DLLs (dynamic CRT, D8). Iterate all version dirs -- the
# "v143" symlink sorts last but has no CRT payload.
$vcCrt = $null
foreach ($d in (Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC" -Directory -ErrorAction SilentlyContinue)) {
    $cand = Join-Path $d.FullName "x64\Microsoft.VC143.CRT"
    if (Test-Path (Join-Path $cand "vcruntime140.dll")) { $vcCrt = $cand; break }
}
if (-not $vcCrt) { throw "VC143 x64 CRT redist not found under BuildTools\VC\Redist\MSVC" }

Write-Host "  VsDevCmd : OK"
Write-Host ("  cmake    : {0}" -f $cmake)
Write-Host ("  ninja    : {0}" -f $ninja)
Write-Host ("  CUDA     : {0}" -f $cudaRoot.FullName)
Write-Host ("  VC CRT   : {0}" -f $vcCrt)

# ---------------------------------------------------------------------------
Step "2/5 + 3/5 CMake configure + build (Release)"
# ---------------------------------------------------------------------------
if ($Clean) {
    if (Test-Path $build) { Remove-Item $build -Recurse -Force; Write-Host "  removed $build" }
    if (Test-Path $dist)  { Remove-Item $dist  -Recurse -Force; Write-Host "  removed $dist" }
}

# Configure and build must run inside the VsDevCmd environment (nvcc uses cl.exe
# as host compiler). Generate a small bat that does both, so this works from any
# shell. Markers make it obvious which stage failed.
$bat = Join-Path $env:TEMP ("ninfer_build_{0}.bat" -f [guid]::NewGuid().ToString("N"))
# NOTE: each command must be ONE physical line -- PowerShell silently splits
# trailing-"+" continuations inside array literals into separate elements.
$cfgLine   = "`"$cmake`" -S `"$src`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=`"$($cudaRoot.FullName)`" -DCMAKE_CUDA_COMPILER=`"$nvcc`" -DCMAKE_MAKE_PROGRAM=`"$ninja`" -DNINFER_FFMPEG_ROOT=`"$deps\ffmpeg`" -DNINFER_CURL_ROOT=`"$deps\curl`" || exit /b 1"
$buildLine = "`"$cmake`" --build `"$build`" --config Release || exit /b 1"
$batLines = @('@echo off', "call `"$vsdev`" -arch=x64 -no_logo || exit /b 1", 'echo MARKER_CONFIGURE_START', $cfgLine, 'echo MARKER_BUILD_START', $buildLine, 'echo MARKER_BUILD_DONE')
Set-Content -Path $bat -Value $batLines -Encoding ASCII

$sw = [System.Diagnostics.Stopwatch]::StartNew()
cmd /c "`"$bat`""
$rc = $LASTEXITCODE
Remove-Item $bat -Force -ErrorAction SilentlyContinue
if ($rc -ne 0) { throw "configure/build failed (exit $rc) -- see output above MARKER_* lines" }
Write-Host ("  build finished in {0:N1} s (incremental unless -Clean)" -f $sw.Elapsed.TotalSeconds)

# ---------------------------------------------------------------------------
Step "4/5 Assemble dist/"
# ---------------------------------------------------------------------------
if (Test-Path $dist) { Remove-Item $dist -Recurse -Force }
New-Item -ItemType Directory -Force -Path $dist | Out-Null

# FFmpeg shared libs needed by the apps (dumpbin-verified incl. transitive:
# avcodec pulls swresample; version suffixes globbed so a deps/ffmpeg upgrade
# keeps working).
$ffm = @()
foreach ($base in "avcodec", "avformat", "avutil", "swscale", "swresample") {
    $hit = Get-ChildItem (Join-Path $deps "ffmpeg\bin\$base-*.dll") -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $hit) { throw "FFmpeg DLL not found: deps\ffmpeg\bin\$base-*.dll" }
    $ffm += $hit.FullName
}

$copy = @(
    # Product executables (GUI finds ninfer-cli.exe as its sibling, D5/D6).
    @((Join-Path $build "apps\Ninfer.exe"),      "Ninfer.exe"),
    @((Join-Path $build "apps\ninfer-cli.exe"),  "ninfer-cli.exe"),
    @((Join-Path $build "apps\ninfer-serve.exe"), "ninfer-serve.exe"),
    # VC runtime (dynamic CRT).
    @((Join-Path $vcCrt "vcruntime140.dll"),   "vcruntime140.dll"),
    @((Join-Path $vcCrt "msvcp140.dll"),       "msvcp140.dll"),
    @((Join-Path $vcCrt "vcruntime140_1.dll"), "vcruntime140_1.dll"),
    # libcurl (Schannel TLS) + zlib.
    @((Join-Path $deps "curl\bin\libcurl.dll"), "libcurl.dll"),
    @((Join-Path $deps "curl\bin\z.dll"),       "z.dll")
)
# Unary comma keeps each pair a single element (pipeline unrolling would not).
foreach ($f in $ffm) { $copy += ,@($f, (Split-Path $f -Leaf)) }

foreach ($c in $copy) {
    if (-not (Test-Path $c[0])) { throw "missing source file: $($c[0])" }
    Copy-Item $c[0] (Join-Path $dist $c[1])
}
Write-Host ("  copied {0} files to {1}" -f $copy.Count, $dist)

# ---------------------------------------------------------------------------
Step "5/5 Verify + report"
# ---------------------------------------------------------------------------
$fail = 0

# CLI --help from the dist location (all DLLs must resolve next to the exe).
& (Join-Path $dist "ninfer-cli.exe") --help | Out-Null
if ($LASTEXITCODE -eq 0) { Write-Host "  ninfer-cli.exe --help : exit 0" } else { Write-Host "  ninfer-cli.exe --help : FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; $fail++ }

# GUI launches and shows its main window from the dist location.
$gui = Start-Process (Join-Path $dist "Ninfer.exe") -PassThru
$sw2 = [System.Diagnostics.Stopwatch]::StartNew()
while ($gui.MainWindowHandle -eq [IntPtr]::Zero -and $sw2.Elapsed -lt [TimeSpan]::FromSeconds(15)) { Start-Sleep -Milliseconds 250 }
if ($gui.MainWindowHandle -ne [IntPtr]::Zero) {
    Write-Host "  Ninfer.exe window      : OK"
} else {
    Write-Host "  Ninfer.exe window      : FAILED (no main window in 15 s)" -ForegroundColor Red
    $fail++
}
Stop-Process -Id $gui.Id -Force -ErrorAction SilentlyContinue
Get-Process ninfer-cli -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host ""
Write-Host "  dist/ contents:"
Get-ChildItem $dist | Sort-Object Name | ForEach-Object {
    Write-Host ("    {0,-24} {1,10:N0} B" -f $_.Name, $_.Length)
}

if ($fail -gt 0) { throw "verification failed ($fail problem(s)) -- see above" }
Write-Host ""
Write-Host "BUILD_WINDOWS PASS" -ForegroundColor Green
