@echo off
setlocal
setlocal EnableDelayedExpansion
set "CMK=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin"
set "BUILDDIR=C:\Users\Micke\Documents\Kodprojekt\Ninfer\ninfer-win\build"
set "PATH=%CMK%;%PATH%"
echo === LIST (ctest -N -L flash_next) ===
"%CMK%\ctest.exe" --test-dir "%BUILDDIR%" -N -L flash_next
echo === RUN (ctest -L flash_next, per-test 90s) ===
"%CMK%\ctest.exe" --test-dir "%BUILDDIR%" -L flash_next --timeout 90 --output-on-failure
echo === DONE exitcode=%errorlevel% ===
