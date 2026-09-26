@echo off
rem pn_moe_cpu_auto.bat - start flash_next serve with --moe-cpu auto on the real model.
rem Run from a visible terminal so you can watch the load + first forward.
rem   Auto  -> CPU-offload the coldest MoE layers when free VRAM < 8 GiB
rem   on    -> force CPU offload
rem   off   -> force all-MoE on GPU
cd /d "C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash"
core\build\apps\ninfer-serve.exe models\qwen3_8_flash_next.ninfer --moe-cpu auto %*
