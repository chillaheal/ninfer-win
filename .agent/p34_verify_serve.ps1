$p = Get-Process | Where-Object { $_.ProcessName -like 'n*er-serve' } | Select-Object -First 1
if ($null -eq $p) { Write-Output "NO serve running"; exit }
$cim = Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)"
Write-Output ("PID=" + $p.Id + "  Started=" + $cim.CreationDate + "  Path=" + $p.Path)
Write-Output ("CMDLINE: " + $cim.CommandLine)
$used = (nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>$null | Select-Object -First 1)
Write-Output ("VRAM: " + $used)
try {
  $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8888/health" -TimeoutSec 5
  Write-Output ("HTTP /health: " + $r.StatusCode + " " + $r.Content)
} catch {
  try { $r2 = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8888/" -TimeoutSec 5; Write-Output ("HTTP /: " + $r2.StatusCode) }
  catch { Write-Output ("HTTP probe: " + $_.Exception.Message) }
}
