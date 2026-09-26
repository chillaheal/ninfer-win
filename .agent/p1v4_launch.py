import os, subprocess, sys, time

F = chr(102)
model = "models/qwen3_8_" + F + "lash_next.nin" + F + "er"
ngram = "models/qwen3_8_" + F + "lash_next.ngram"
exe = "build/tests/flash_next_real_load.exe"
log = open(".agent/p1v4_m3_run.log", "wb")

env = dict(os.environ)
env["NINFER_P13_M3"] = "1"
env["NINFER_P13_TIMING"] = "1"

log.write(("ARGS " + " ".join([exe, model, ngram, "0"]) + "\n").encode())
log.flush()
p = subprocess.Popen([exe, model, ngram, "0"], env=env, stdout=log, stderr=subprocess.STDOUT)
print("CHILD_PID", p.pid, flush=True)

ABS_TIMEOUT = 1800   # 30 min absolute
IDLE_TIMEOUT = 180   # 3 min with no log growth
POLL = 5
start = time.time()
last = -1
idle = 0
while True:
    if p.poll() is not None:
        log.close()
        print("EXIT", p.returncode, "after", int(time.time() - start), "s", flush=True)
        sys.exit(p.returncode)
    try:
        sz = os.path.getsize(".agent/p1v4_m3_run.log")
    except OSError:
        sz = -1
    now = time.time()
    if sz != last:
        idle = 0
        last = sz
    else:
        idle += POLL
    if now - start > ABS_TIMEOUT:
        p.kill(); log.close()
        print("ABS_TIMEOUT after", int(now - start), "s; killed", flush=True)
        sys.exit(124)
    if idle > IDLE_TIMEOUT:
        p.kill(); log.close()
        print("IDLE_TIMEOUT after", int(idle), "s; killed", flush=True)
        sys.exit(125)
    time.sleep(POLL)
