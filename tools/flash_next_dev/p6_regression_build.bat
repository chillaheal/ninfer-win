@echo off
rem Post-phase regression build: existing GDN suites, artifact reader, load plans.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
cmake --build "C:\Users\Micke\Documents\Ninfer\ninfer-win\build" --target ninfer_gated_delta_net_test ninfer_gated_delta_net_replay_record_test ninfer_gdn_replay_fold_test ninfer_gdn_replay_records_test ninfer_artifact_reader_test ninfer_qwen3_6_27b_load_plan_test ninfer_qwen3_6_35b_a3b_dflash_load_plan_test -j8
