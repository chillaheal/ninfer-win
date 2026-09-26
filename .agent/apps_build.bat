@echo off
rem Build the missing core-tree APP targets into <project root>\build (no reconfigure).
rem The full build was scoped to flash_next targets to dodge the core-tree unistd.h
rem Windows test bug; this adds the serve + GUI apps so build/apps/ is a runnable set.
rem Wrapper (visible window + transcript): .agent\apps_build_run.ps1
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;%PATH%"
cd /d "C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash"
echo ==== BUILD APPS (ninfer-serve + ninfer-gui, -j8) ====
cmake --build build --target ninfer-serve ninfer-gui -j8
echo ==== APPS_BUILD_DONE exit=%ERRORLEVEL% ====
exit /b %ERRORLEVEL%
