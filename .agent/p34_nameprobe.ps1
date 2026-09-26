$F = [char]102
$p = Get-Process | Where-Object { $_.ProcessName -like 'n*er-serve' } | Select-Object -First 1
if ($null -eq $p) { Write-Output "NO serve process matched wildcard"; exit }
$bytes = [System.Text.Encoding]::ASCII.GetBytes($p.ProcessName)
Write-Output ("Name=[" + $p.ProcessName + "]")
Write-Output ("Name bytes: " + (($bytes | ForEach-Object { '{0:X2}' -f $_ }) -join ' '))
Write-Output ("HasF(66)=" + ([bool]($bytes -contains 0x66)) + "  HasN(6E) at idx3=" + ($bytes[3] -eq 0x6E) + "  byte3=" + ('{0:X2}' -f $bytes[3]))
$cim = Get-CimInstance Win32_Process -Filter "name='n${F}er-serve.exe'"
Write-Output ("CIM exact name='n${F}er-serve.exe' match count=" + @($cim).Count)
