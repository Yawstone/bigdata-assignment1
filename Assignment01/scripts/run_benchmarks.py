"""Run the benchmark campaign (Linux/macOS/WSL).

For every (language, n): one fresh process loads the shared input, runs `warmup`
untimed multiplications and `reps` timed ones, prints each timing. The harness
records peak resident set size (RSS) of that process via wait4()/ru_maxrss.

Outputs (results/):
  raw.csv   language,n,rep,time_ns          every measured repetition
  runs.csv  language,n,reps_requested,reps_done,warmup,peak_rss_kb,status,checksum,wall_s
Edit the CONFIG block below to change sizes, repetitions and the time budget.
"""
import csv, os, subprocess, sys, time, signal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build"
RES = ROOT / "results"

# ---------------- CONFIG ----------------
SIZES = [10, 50, 100, 200, 400, 600, 800, 1000]
TIMEOUT_S = 600                    # budget per process (all warm-up + timed reps)
MEM_LIMIT_MB = 3000                # informational budget; stated in the report
def reps_for(lang, n):
    if lang == "python" and n >= 600:
        return 3                   # costly runs: fewer repetitions (justified in report)
    return 5
def warmup_for(lang, n):
    if lang == "java":
        return 2                   # let the JIT compile the kernel
    if lang == "python" and n >= 400:
        return 0                   # too costly; no JIT/caching effect to hide in CPython
    return 1
JAVA_OPTS = ["-Xms512m", "-Xmx3g", "-XX:+UseSerialGC"]
# ----------------------------------------

def cmd_for(lang, n, reps, warm):
    inp = str(ROOT / "inputs" / f"input_{n}.bin")
    tail = [inp, str(n), str(reps), str(warm)]
    if lang == "c":
        return [str(BIN / "matmul")] + tail
    if lang == "java":
        return ["java"] + JAVA_OPTS + ["-cp", str(BIN), "MatMul"] + tail
    return [sys.executable, str(ROOT / "src" / "matmul.py")] + tail


def read_vmhwm_kb(pid):
    """Peak resident set size (VmHWM) of a live process from /proc (Linux)."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def run_one(cmd):
    """Run one process; return (status, stdout lines, peak_rss_kb, wall seconds).

    Peak RSS: each program reads its own VmHWM (kernel high-water mark, Linux/WSL) at
    the end and prints it as "rss_kb,<kB>"; main() uses that value. The polling value
    below is only a fallback (e.g. after a timeout). ru_maxrss from wait4() is avoided
    because a forked child inherits the parent's maximum, inflating small runs.
    """
    tmp = RES / "_stdout.tmp"
    t0 = time.time()
    last_hwm = None
    with open(tmp, "w") as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.DEVNULL)
        status, rusage = "ok", None
        while True:
            pid, st, ru = os.wait4(p.pid, os.WNOHANG)
            if pid:
                rusage = ru
                if os.WIFEXITED(st) and os.WEXITSTATUS(st) != 0:
                    status = f"exit_{os.WEXITSTATUS(st)}"
                elif os.WIFSIGNALED(st):
                    status = f"signal_{os.WTERMSIG(st)}"      # e.g. 9 = killed (likely out of memory)
                break
            hwm = read_vmhwm_kb(p.pid)
            if hwm is not None:
                last_hwm = hwm
            if time.time() - t0 > TIMEOUT_S:
                p.kill()
                _, st, rusage = os.wait4(p.pid, 0)
                status = "timeout"
                break
            time.sleep(0.01)
    lines = tmp.read_text().splitlines()
    tmp.unlink()
    if last_hwm is not None:
        rss = last_hwm
    elif rusage is not None:
        rss = rusage.ru_maxrss // 1024 if sys.platform == "darwin" else rusage.ru_maxrss
    else:
        rss = ""
    return status, lines, rss, time.time() - t0


def main():
    RES.mkdir(exist_ok=True)
    langs = sys.argv[1:] or ["c", "java", "python"]
    raw_new = not (RES / "raw.csv").exists()
    runs_new = not (RES / "runs.csv").exists()
    with open(RES / "raw.csv", "a", newline="") as fr, open(RES / "runs.csv", "a", newline="") as fu:
        raw, runs = csv.writer(fr), csv.writer(fu)
        if raw_new:
            raw.writerow(["language", "n", "rep", "time_ns"])
        if runs_new:
            runs.writerow(["language", "n", "reps_requested", "reps_done", "warmup",
                           "peak_rss_kb", "status", "checksum", "wall_s"])
        for lang in langs:
            for n in SIZES:
                reps, warm = reps_for(lang, n), warmup_for(lang, n)
                print(f"[{lang}] n={n} reps={reps} warmup={warm} ...", end=" ", flush=True)
                status, lines, rss, wall = run_one(cmd_for(lang, n, reps, warm))
                done, checksum = 0, ""
                for ln in lines:
                    parts = ln.split(",")
                    if parts[0] == "rep":
                        raw.writerow([lang, n, int(parts[1]), int(parts[2])])
                        done += 1
                    elif parts[0] == "checksum":
                        checksum = parts[1]
                    elif parts[0] == "rss_kb" and int(parts[1]) > 0:
                        rss = int(parts[1])          # self-reported VmHWM is the most reliable value
                runs.writerow([lang, n, reps, done, warm, rss, status, checksum, f"{wall:.2f}"])
                fr.flush(); fu.flush()
                print(f"{status}, {done} reps, peak RSS {rss} KB, wall {wall:.1f}s")
                if status != "ok" and done < reps:
                    print(f"   stopping larger sizes for {lang}: budget reached ({status})")
                    break


if __name__ == "__main__":
    main()
