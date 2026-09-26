@echo off
rem FULL flash_next build into <project root>\build (fresh: -S core -B build).
rem Toolchain pinned to match core\build exactly (MSVC 14.44.35207, CUDA v13.3, VS-bundled ninja).
rem Scoped to flash_next targets (avoids the pre-existing core-tree unistd.h Windows build bug).
rem Wrapper (visible window + transcript): .agent\full_build_run.ps1
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;%PATH%"
cd /d "C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash"
if not exist "C:\fn\tests\targets\qwen3_8_flash_next" ( echo ABORT: C:/fn junction missing & exit /b 1 )
echo ==== CONFIGURE (-S core -B build, Ninja, Release) ====
cmake -S C:/Users/Micke/Documents/Kodprojekt/Ninfer/ninfer-win -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_MAKE_PROGRAM="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" -DCMAKE_C_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" -DCMAKE_CXX_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe" -DNINFER_FFMPEG_ROOT=C:/Users/Micke/Documents/Kodprojekt/Ninfer/deps/ffmpeg
if errorlevel 1 ( echo CONFIGURE_FAILED & exit /b 1 )
echo ==== BUILD (18 flash_next targets, -j8) ====
cmake --build build --target ninfer_engine ninfer flash_next_placement flash_next_real_load ninfer_qwen3_8_flash_next_test_flash_next_mini_reader ninfer_qwen3_8_flash_next_test_p2_binder ninfer_qwen3_8_flash_next_test_p2_identity ninfer_qwen3_8_flash_next_test_p2_planner ninfer_qwen3_8_flash_next_test_p3_pager ninfer_qwen3_8_flash_next_test_p4_ngram ninfer_qwen3_8_flash_next_test_p5_gated_residual ninfer_qwen3_8_flash_next_test_p6_gdn ninfer_qwen3_8_flash_next_test_p7_qsa ninfer_qwen3_8_flash_next_test_p8_sparse_moe ninfer_qwen3_8_flash_next_test_p9_program ninfer_qwen3_8_flash_next_test_p10_cpu_moe ninfer_qwen3_8_flash_next_test_p11_placement ninfer_qwen3_8_flash_next_test_s1_real_bind -j8
echo ==== FULL_BUILD_DONE exit=%ERRORLEVEL% ====
exit /b %ERRORLEVEL%
