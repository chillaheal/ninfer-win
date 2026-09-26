@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
set "SRC=C:\Users\Micke\Documents\Kodprojekt\Ninfer\ninfer-win"
set "BUILD=%~dp0..\..\..\Ninfer\ninfer-win\build"
echo SRC=%SRC%
echo BUILD=%BUILD%
echo ===== RECONFIGURE =====
cmake -S "%SRC%" -B "%BUILD%" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
if errorlevel 1 exit /b 1
echo ===== BUILD =====
cmake --build "%BUILD%" --target flash_next_mini_fixture flash_next_mini_reader_test -j8
if errorlevel 1 exit /b 1
echo ===== CTEST =====
ctest --test-dir "%BUILD%" -L flash_next --output-on-failure
