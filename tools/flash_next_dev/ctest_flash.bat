@echo off
rem Run the flash_next ctest suite with the VS dev env.
rem Usage (Git Bash): cmd //c "tools\flash_next_dev\ctest_flash.bat"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
cd /d "%~dp0..\..\core\build"
ctest -L flash_next --output-on-failure --timeout 300
