@echo off
rem S1 (P12) gate (a) build: real-geometry header-only bind test. VsDevCmd x64
rem env + pinned cmake/ninja. Pattern: p11_build.bat. Builds ninfer_engine (adds
rem impl/load/real_bindings.cpp) + the S1 real-bind test.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
cmake --build "C:\Users\Micke\Documents\Ninfer\ninfer-win\build" --target ninfer_engine flash_next_s1_real_bind_test -j8
