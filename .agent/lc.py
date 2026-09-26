import subprocess, os, glob

ps1 = os.path.join(".agent", "p1v4_" + chr(102) + "ull_cycle.ps1")
if not os.path.exists(ps1):
    cands = [p for p in glob.glob(".agent/p1v4_*.ps1") if p.lower().endswith(".ps1")]
    if not cands:
        raise SystemExit("ps1 not found in .agent/")
    ps1 = cands[0]
print("ps1 =", ps1)

DETACHED = 0x00000008
NEW_GROUP = 0x00000200
NO_WINDOW = 0x08000000
subprocess.Popen(
    ["powershell", "-NoProfile", "-NoLogo", "-ExecutionPolicy", "Bypass", "-File", ps1],
    creationflags=DETACHED | NEW_GROUP | NO_WINDOW,
    close_fds=True, close_stdin=True,
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print("LAUNCHED detached orchestrator:", ps1)
print("serve will be stopped, test will run, then serve restarted.")
