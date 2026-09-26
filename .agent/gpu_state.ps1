$ErrorActionPreference = "SilentlyContinue"
$F = [char]102   # the f that gets mangled to n in tool I/O
$needle = "ni" + $F + "nfer"          # real name
$needleM = "ni" + "nn" + "er"        # mangled name (n for f)
$d = nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
Write-Output ("VRAM used/total/util: " + $d)
$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -like ("*" + $needle + "*") -or $_.Name -like ("*" + $needleM + "*") }
if ($procs) {
  foreach ($p in $procs) { Write-Output ("PROC pid=" + $p.ProcessId + " name=" + $p.Name) }
} else {
  Write-Output ("no " + $needle + " process")
}
