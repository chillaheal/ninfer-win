@echo off
rem P11 build (32 GB placement vs real inventory bytes). VsDevCmd x64 env + pinned
rem cmake/ninja. Pattern: p10_build.bat. Builds the engine (planner.cpp +
rem package.cpp, incl. the new render_placement), the placement dry-run tool
rem (flash_next_placement) and the P11 placement test
rem (flash_next_p11_placement_test). The test spawns flash_next_placement.exe,
rem so that target is listed explicitly and built before ctest runs.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
cmake --build "C:\Users\Micke\Documents\Ninfer\ninfer-win\build" --target ninfer_engine flash_next_placement flash_next_p11_placement_test -j8
