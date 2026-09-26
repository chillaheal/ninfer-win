@echo off
rem P1V.2: build flash_next test targets and run ctest -L flash_next (bit-exactness).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
set "BUILDDIR=C:\Users\Micke\Documents\Kodprojekt\Ninfer\ninfer-win\build"
cmake --build "%BUILDDIR%" -j8 -- ^
  ninfer_qwen3_8_flash_next_test_flash_next_mini_reader ^
  ninfer_qwen3_8_flash_next_test_p2_binder ^
  ninfer_qwen3_8_flash_next_test_p2_identity ^
  ninfer_qwen3_8_flash_next_test_p2_planner ^
  ninfer_qwen3_8_flash_next_test_p3_pager ^
  ninfer_qwen3_8_flash_next_test_p4_ngram ^
  ninfer_qwen3_8_flash_next_test_p5_gated_residual ^
  ninfer_qwen3_8_flash_next_test_p6_gdn ^
  ninfer_qwen3_8_flash_next_test_p7_qsa ^
  ninfer_qwen3_8_flash_next_test_p8_sparse_moe ^
  ninfer_qwen3_8_flash_next_test_p9_program ^
  ninfer_qwen3_8_flash_next_test_p11_placement ^
  ninfer_qwen3_8_flash_next_test_s1_real_bind
if errorlevel 1 (
  echo BUILD_FAILED
  exit /b 1
)
echo BUILD_OK
ctest --test-dir "%BUILDDIR%" -L flash_next --output-on-failure
