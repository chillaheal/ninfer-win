@echo off
rem S1 (P12) gate (b) build: real-geometry DEVICE loader + standalone driver.
rem VsDevCmd x64 env + pinned cmake/ninja. Relative to project root.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
cd /d %~dp0..\..
cmake --build "core\build" --target ninfer_engine flash_next_real_load -j8
