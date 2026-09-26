@echo off
rem Rebuild flash_next engine + relink stale flash_next test exes in the ctest tree.
rem Usage (Git Bash): cmd //c "tools\flash_next_dev\p1_rebuild_tests.bat"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
cd /d "C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash"
cmake --build "core\build" --target ninfer_engine ninfer_qwen3_8_flash_next_test_p2_identity ninfer_qwen3_8_flash_next_test_p7_qsa ninfer_qwen3_8_flash_next_test_p9_program ninfer_qwen3_8_flash_next_test_p10_cpu_moe ninfer -j8
