$ErrorActionPreference = 'SilentlyContinue'
$since = (Get-Date).AddHours(-2)
Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000; StartTime=$since} -MaxEvents 12 |
  Select-Object TimeCreated, @{n='Msg';e={($_.Message -split "`n")[0..6] -join ' | '}} |
  ForEach-Object { Write-Output ('--- ' + $_.TimeCreated); Write-Output $_.Msg } | Out-String
