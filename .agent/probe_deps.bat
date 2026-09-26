@echo off
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
echo "=== probe complete: env ok ==="
for /d %%D in ("C:\Users\Micke\Desktop\*inner*AI") do (
  echo ### DIR %%~D
  for %%E in ("%%~D\*.exe") do (
    echo --- %%~nxE ---
    dumpbin /dependents "%%~fE" 2>&1 | findstr /i /r "avcodec avformat avutil swscale swresample"
  )
)
