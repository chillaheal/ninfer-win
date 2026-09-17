@echo off
rem P9 build (Mini Program / first forward). VsDevCmd x64 env + pinned cmake/ninja.
rem Pattern: p8_build.bat. Builds the engine (all flash_next runtime sources +
rem engine.cpp + registry.cpp wiring), the CLI (done-line double-run), and the
rem P9 test once its target exists.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
cmake --build "C:\Users\Micke\Documents\Ninfer\ninfer-win\build" --target ninfer_engine ninfer flash_next_p9_program_test -j8
