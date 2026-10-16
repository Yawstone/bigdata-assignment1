"""Summarise raw measurements: median, IQR, memory, growth exponent, figures.
Reads results/raw.csv and results/runs.csv, writes results/summary.csv and figures/*.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
NAMES = {"c": "C", "java": "Java", "python": "Python"}
COLORS = {"c": "black", "java": "dimgray", "python": "darkgray"}
MARKERS = {"c": "o", "java": "s", "python": "^"}
LSTYLES = {"c": "-", "java": "--", "python": ":"}


def iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def main():
    raw = pd.read_csv(RES / "raw.csv")
    runs = pd.read_csv(RES / "runs.csv")
    raw["time_s"] = raw["time_ns"] / 1e9
    g = raw.groupby(["language", "n"])["time_s"]
    summ = g.agg(reps="count", median_s="median", iqr_s=iqr, min_s="min", max_s="max").reset_index()
    summ = summ.merge(runs[["language", "n", "peak_rss_kb", "status", "warmup"]],
                      on=["language", "n"], how="left")
    summ["peak_rss_mb"] = summ["peak_rss_kb"] / 1024
    summ["array_storage_mb"] = 3 * 8 * summ["n"] ** 2 / 1e6
    summ.to_csv(RES / "summary.csv", index=False)

    # growth exponent: slope of log(time) vs log(n), using n >= 100 so fixed costs matter less
    rows = []
    for lang, d in summ.groupby("language"):
        d = d[d["n"] >= 100]
        if len(d) >= 3:
            slope, _ = np.polyfit(np.log(d["n"]), np.log(d["median_s"]), 1)
            rows.append({"language": lang, "exponent": slope, "points": len(d)})
    pd.DataFrame(rows, columns=["language", "exponent", "points"]).to_csv(RES / "growth_exponents.csv", index=False)

    # Figure 1: runtime vs n (log-log) with n^3 reference
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for lang, d in summ.groupby("language"):
        ax.errorbar(d["n"], d["median_s"], yerr=d["iqr_s"] / 2, marker=MARKERS[lang], linestyle=LSTYLES[lang], capsize=3,
                    label=NAMES[lang], color=COLORS[lang])
    ref = summ[summ["n"] >= 100]
    if not ref.empty:
        anchor = ref.sort_values("n").iloc[0]
        ns = np.array(sorted(summ["n"].unique()), dtype=float)
        ax.plot(ns, anchor["median_s"] * (ns / anchor["n"]) ** 3, "k--", lw=1, label=r"$n^3$ reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (n x n)"); ax.set_ylabel("Kernel time (s), median")
    ax.set_title("Runtime of the triple-loop multiplication")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "runtime_vs_n.png", dpi=200); plt.close(fig)

    # Figure 2: peak RSS vs n
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for lang, d in summ.groupby("language"):
        ax.plot(d["n"], d["peak_rss_mb"], marker=MARKERS[lang], linestyle=LSTYLES[lang], label=NAMES[lang], color=COLORS[lang])
    d0 = summ.drop_duplicates("n").sort_values("n")
    ax.plot(d0["n"], d0["array_storage_mb"], "k--", lw=1, label="3 x 8n^2 bytes (raw arrays)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (n x n)"); ax.set_ylabel("Peak resident memory (MB)")
    ax.set_title("Peak process memory (RSS)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "memory_vs_n.png", dpi=200); plt.close(fig)

    # Figure 3: slowdown relative to C at common sizes
    piv = summ.pivot(index="n", columns="language", values="median_s")
    if "c" in piv:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for lang in ("java", "python"):
            if lang in piv:
                r = (piv[lang] / piv["c"]).dropna()
                ax.plot(r.index, r.values, marker=MARKERS[lang], linestyle=LSTYLES[lang], label=f"{NAMES[lang]} / C", color=COLORS[lang])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Matrix size n"); ax.set_ylabel("Median time ratio vs C")
        ax.set_title("Relative runtime (tested settings only)")
        ax.grid(True, which="both", alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(FIG / "ratio_vs_c.png", dpi=200); plt.close(fig)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
