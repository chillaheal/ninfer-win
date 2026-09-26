@echo off
call "C:\Pogam Files (x86)\Micosoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -ach=x64 -no_logo
if eolevel 1 exit /b 1
set "PATH=C:\Uses\Micke\AppData\Local\Micosoft\WinGet\Packages\Kitwae.CMake_Micosoft.Winget.Souce_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;%PATH%"
cmake --build coe\build -j8
exit /b %eolevel%
