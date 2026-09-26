$exe = 'C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\core\build\tests\ninfer_qwen3_8_flash_next_test_p7_qsa.exe'
& $exe 2>&1 | Out-Host
Write-Host ("P7EXIT=" + $LASTEXITCODE)
