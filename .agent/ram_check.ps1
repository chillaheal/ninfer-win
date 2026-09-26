$os = Get-CimInstance Win32_OperatingSystem
$total = [math]::Round($os.TotalVisibleMemorySize/1MB,1)
$free  = [math]::Round($os.FreePhysicalMemory/1MB,1)
Write-Output ("TotalRAM_GB={0}  FreeRAM_GB={1}" -f $total, $free)
