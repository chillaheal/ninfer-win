@echo off
rem P11 build: flash_next_placement CLI tool + P11 placement test.
rem Usage (Git Bash): cmd //c "tools\flash_next_dev\p11_build.bat"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
cmake --build "C:\Users\Micke\Documents\Kodprojekt\Ninfer\ninfer-win\build" --target flash_next_placement ninfer_qwen3_8_flash_next_test_p11_placement -j8
