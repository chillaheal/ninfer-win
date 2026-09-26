content = r'''$root = 'C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash'
Set-Location $root
$log = Join-Path $root '.agent\serve_start_test.log'
Write-Host ("START {0:yyyy-MM-dd HH:mm:ss} serve start test context=8192" -f (Get-Date))
& "$root\build\apps\ninfer-serve.exe" "models\qwen3_8_flash_next.ninfer" --host 127.0.0.1 --port 8123 --max-context 8192 2>&1 | Tee-Object -FilePath $log
Write-Host ("END {0:yyyy-MM-dd HH:mm:ss} exit={1}" -f (Get-Date), $LASTEXITCODE)
'''
with open(r'C:\Users\Micke\Documents\Kodprojekt\nVidia Infer Flash\.agent\serve_start_test.ps1', 'w', encoding='utf-8') as f:
    f.write(content)
print('written', len(content))
