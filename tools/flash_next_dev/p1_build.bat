@echo off
rem P1 build step for tools\flash_next_dev\run_phase.sh (no double quotes in the
rem bash-invoked argument, so Git Bash arg conversion cannot mangle it).
rem Pattern: .logs\build_serve_fix.bat (VsDevCmd x64 + absolute cmake/ninja).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
cmake --build "C:\Users\Micke\Documents\Kodprojekt\Ninfer\ninfer-win\build" --target flash_next_mini_fixture flash_next_mini_reader_test -j8
