"""Record the experimental environment to results/environment.json."""
import json, os, platform, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"unavailable ({e})"


def cpu_model():
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return run(["sysctl", "-n", "machdep.cpu.brand_string"])
    except Exception:
        pass
    return platform.processor() or "unknown"


def physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except Exception:
        return "unknown (install psutil)"


def ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2**30, 2)
    except Exception:
        try:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
        except Exception:
            return "unknown"


gcc_out = run(["gcc", "--version"])
info = {
    "cpu_model": cpu_model(),
    "physical_cores": physical_cores(),
    "logical_processors": os.cpu_count(),
    "ram_gb": ram_gb(),
    "os": platform.platform(),
    "python": sys.version.replace("\n", " "),
    "python_implementation": platform.python_implementation(),
    "gcc": gcc_out.splitlines()[0] if gcc_out else "unavailable",
    "java": run(["java", "-version"]).replace("\n", " | "),
    "c_flags": "-O2 (no -march=native, no -ffast-math)",
    "java_flags": "javac default; run with -Xms512m -Xmx3g -XX:+UseSerialGC",
    "fixed_settings": "double precision, row-major inputs from shared binary file, single thread",
}
out = ROOT / "results"
out.mkdir(exist_ok=True)
(out / "environment.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
