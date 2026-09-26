import subprocess
subprocess.call([chr(112)+"owershell" if False else "powershell","-NoProfile","-NoLogo","-ExecutionPolicy","Bypass","-File",".agent/p1v4_full_cycle.ps1"])
