import subprocess, sys
subprocess.call(['powershell','-NoProfile','-NoLogo','-ExecutionPolicy','Bypass','-File', sys.argv[1]])
