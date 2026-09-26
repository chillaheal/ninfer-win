@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash"
set "PATH=C:\Users\Micke\AppData\Local\Programs\Python\Python313;C:\Windows\System32;%PATH%"
powershell -NoProfile -NoLogo -ExecutionPolicy Bypass -File .agent\p1v4_full_cycle.ps1
