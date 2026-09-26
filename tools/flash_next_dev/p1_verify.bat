@echo off
setlocal enabledelayedexpansion
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 (echo VsDevCmd FAILED & exit /b 1)
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
REM Build the core folder name and subfolder from char codes to avoid f/n font trap
powershell -NoProfile -NoLogo -Command "$f=[char]@(78,105,110,102,101,114); $s=[char]@(110,105,110,102,101,114,45,119,105,110); Set-Content -Path 'C:\Users\Micke\fln_paths.txt' -Value ('' + (($f -join '') + "`n" + ($s -join '')) -NoNewline -Encoding ASCII"
set /p SRCFOLDER= < "C:\Users\Micke\fln_paths.txt"
REM First line is folder, second is subfolder
set /p _SUB= < "C:\Users\Micke\fln_paths.txt"
REM We only need first line for folder; read both lines properly
set /a LINES=1
set "L1="
set "L2="
for /f "usebackq delims=" %%a in ("C:\Users\Micke\fln_paths.txt") do (
  if !LINES! EQU 1 (set "L1=%%a") else (set "L2=%%a")
  set /a LINES+=1
)
set "CORE=C:\Users\Micke\Documents\Kodprojekt\%L1%\%L2%"
set "BUILD=%CORE%\build"
echo CORE=%CORE%
set "BUILD=%CORE%\build"
set "BUILD=%CORE%\build"
echo BUILD=%BUILD%
if not exist "%BUILD%" (mkdir "%BUILD%" >nul 2>&1)
echo ===== RECONFIGURE =====
cmake -S "%CORE%" -B "%BUILD%" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
if errorlevel 1 (echo CFG_FAILED & exit /b 1)
echo ===== BUILD =====
cmake --build "%BUILD%" --target flash_next_mini_fixture flash_next_mini_reader_test -j8
if errorlevel 1 (echo BUILD_FAILED & exit /b 1)
echo ===== CTEST =====
ctest --test-dir "%BUILD%" -L flash_next --output-on-failure
echo ALL_DONE
