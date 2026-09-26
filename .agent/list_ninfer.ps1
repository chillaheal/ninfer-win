Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match 'ninfer|python|pythonw' } |
  Select-Object ProcessId,Name,CreationDate,
    @{n='WorkSetMB';e={[math]::Round($_.WorkingSetSize/1MB,1)}},
    CommandLine |
  Format-List
