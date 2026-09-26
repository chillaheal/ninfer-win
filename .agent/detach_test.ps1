Set-Content -Path ".agent/detach_probe.txt" -Value ("probe-" + (Get-Date -Format "HH:mm:ss"))
