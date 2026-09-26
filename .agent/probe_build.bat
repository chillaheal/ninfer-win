@echo off
rem Rebuild the engine (registry.cpp probe fix) + relink the three apps into <root>\build.
rem Incremental - no reconfigure. Builds ninfer_engine (registry.cpp changed) and relinks the
rem CLI/serve/GUI exes, copying runtime DLLs to build\apps\.
rem Wrapper (visible window + transcript): .agent\probe_build_run.ps1
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;%PATH%"
cd /d "C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash"
echo ==== PROBE FIX BUILD (ninfer_engine + ninfer + ninfer-serve + ninfer-gui, -j8) ====
cmake --build build --target ninfer_engine ninfer ninfer-serve ninfer-gui -j8
echo ==== PROBE_BUILD_DONE exit=%ERRORLEVEL% ====
exit /b %ERRORLEVEL%
