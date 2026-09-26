@echo off
setlocal
rem P1V.4 M3 T=1 timed decode - visible terminal window.
rem Flow: stop ninfer-serve -> wait for VRAM -> run the real-load M3 test (visible).
rem When it finishes, the window says DONE; you then restart ninfer-serve yourself.
setlocal EnableExtensions
cd /d "C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash"

rem 1. Stop ninfer-serve (hold ~32 GiB VRAM)
taskkill /F /IM ninfer-serve.exe >nul 2>nul
echo Stopped ninfer-serve. Waiting for VRAM to free...

rem 2. Wait until >= 30000 MiB free
:wait_vram
for /f "delims=" %%f in ('nvidia-smi --query-gpu=memory.free --format=csv,noheader ^| findstr /r "[0-9]"') do set "FREE=%%f"
echo VRAM free: %FREE% MiB
if "%FREE%" LSS 30000 (
  timeout /t 5 /nobreak >nul
  goto wait_vram
)

rem 3. Run the M3 T=1 timed decode test in THIS window (visible output)
set NINFER_P13_M3=1
set NINFER_P13_TIMING=1
build\tests\flash_next_real_load.exe models\qwen3_8_flash_next.ninner models\qwen3_8_flash_next.ngram 0
echo.
echo ================================================
echo   TEST FINISHED (exit code %ERRORLEVEL%)
echo   You can now restart ninfer-serve.
echo ================================================
pause
