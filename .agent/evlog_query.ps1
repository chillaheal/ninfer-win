$ErrorActionPreference = 'SilentlyContinue'
$since = (Get-Date).AddHours(-36)
Write-Output "=== Application/WER events mentioning flash_next (last 36h) ==="
Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=$since} -ErrorAction SilentlyContinue |
  Where-Object { $_.Message -like '*flash_next*' -or $_.Message -like '*flash_next_real_load*' } |
  Select-Object -First 8 TimeCreated, Id, ProviderName, Message |
  ForEach-Object { $_ | Format-List } | Out-String
