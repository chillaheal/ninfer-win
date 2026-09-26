@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath 2>nul
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "C:\Users\Micke\Documents\Kodprojekt\Ninfer\ninfer-win\build"
ninja ninfer-gui
echo EXIT=%ERRORLEVEL%
