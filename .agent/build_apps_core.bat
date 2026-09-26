@echo off
rem Build the serve + GUI app targets into the authoritative core build dir
rem (core\build), so they pick up the freshly rebuilt engine (MoeCpuMode chain).
rem Modeled on .agent\apps_build.bat, but targets core\build (the authoritative
rem build dir that the p1 scripts and ctest use).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;%PATH%"
cd /d "C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash"
echo ==== BUILD APPS IN core\build (ninfer-serve + ninfer-gui, -j8) ====
cmake --build core\build --target ninfer-serve ninfer-gui -j8
echo ==== APPS_CORE_BUILD_DONE exit=%ERRORLEVEL% ====
exit /b %ERRORLEVEL%
