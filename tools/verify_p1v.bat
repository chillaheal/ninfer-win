@echo off
cd /d %~dp0..
set "NINFER_EXPERT_GPU_VERIFY=1"
echo === VERIFY RUN START %date% %time% ===
core\build\apps\ninfer-cli.exe models\qwen3_8_flash_next.ninfer --prompt "Please write a short poem about the sea" --max-new 96 --greedy --max-context 1024
echo EXITCODE=%errorlevel%
echo === VERIFY RUN END %date% %time% ===
