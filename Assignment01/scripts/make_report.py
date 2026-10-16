"""Build the PDF report from results/ and figures/.
Usage: python scripts/make_report.py   -> report/Assignment01_<student>_Report.pdf
Edit report_config.json, discussion.md and conclusions.md, then re-run.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "report_config.json").read_text())
RES, FIG = ROOT / "results", ROOT / "figures"
NAMES = {"c": "C", "java": "Java", "python": "Python"}

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman", fontSize=10.5,
                      leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Times-Bold", fontSize=14, spaceBefore=12, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=11.5, spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("title", parent=ss["Title"], fontName="Times-Bold", fontSize=18, leading=22, spaceAfter=8)
center = ParagraphStyle("center", parent=body, alignment=TA_CENTER)
caption = ParagraphStyle("caption", parent=body, fontSize=9, leading=11, alignment=TA_CENTER, textColor=colors.black)
small = ParagraphStyle("small", parent=body, fontSize=9, leading=11.5, alignment=0)
todo = ParagraphStyle("todo", parent=body, textColor=colors.black, fontName="Times-Italic")


def P(t, s=body): return Paragraph(t, s)


def table(data, widths=None, header=True, fs=8.5):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONTNAME", (0, 0), (-1, -1), "Times-Roman"), ("FONTSIZE", (0, 0), (-1, -1), fs),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTNAME", (0, 0), (-1, 0), "Times-Bold")]
    t.setStyle(TableStyle(st))
    return t


def fmt_time(x):
    if pd.isna(x): return "-"
    if x < 1e-3: return f"{x*1e6:.1f} us"
    if x < 1: return f"{x*1e3:.2f} ms"
    return f"{x:.3f} s"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8.5)
    canvas.drawString(2 * cm, 1.2 * cm, f"{CFG['first_name']} {CFG['surname']} - {CFG['student_number']} - Big Data, Assignment {CFG['assignment']}")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def load():
    try:
        summ = pd.read_csv(RES / "summary.csv")
        runs = pd.read_csv(RES / "runs.csv")
        exps = pd.read_csv(RES / "growth_exponents.csv") if (RES / "growth_exponents.csv").exists() else pd.DataFrame()
        return summ, runs, exps
    except FileNotFoundError:
        return None, None, None


def env_table():
    f = RES / "environment.json"
    if not f.exists():
        return P("[Run scripts/env_info.py on the benchmark machine, then rebuild the report.]", todo)
    e = json.loads(f.read_text())
    rows = [["Item", "Value"],
            ["CPU", e["cpu_model"]],
            ["Physical cores / logical processors", f"{e['physical_cores']} / {e['logical_processors']}"],
            ["RAM", f"{e['ram_gb']} GB"],
            ["Operating system", e["os"]],
            ["Python", e["python"][:70]],
            ["C compiler", e["gcc"]],
            ["C flags", e["c_flags"]],
            ["Java", e["java"][:110]],
            ["Java run options", e["java_flags"]]]
    rows = [[Paragraph(str(c), small) for c in r] for r in rows]
    return table(rows, [5.2 * cm, 11.3 * cm])


def auto_text(summ, runs, exps):
    out = []
    ok = summ[summ["status"] == "ok"]
    for lang in ("c", "java", "python"):
        d = ok[ok["language"] == lang]
        if d.empty: continue
        out.append(f"{NAMES[lang]}: largest size completed within budget n = {int(d['n'].max())}.")
    bad = runs[runs["status"] != "ok"]
    if bad.empty:
        out.append("No timeouts or memory failures occurred in the tested range.")
    else:
        for _, r in bad.iterrows():
            out.append(f"{NAMES[r['language']]} n={int(r['n'])}: {r['status']} after {int(r['reps_done'])} of "
                       f"{int(r['reps_requested'])} repetitions (limit reached, no time assigned to missing runs).")
    piv = summ.pivot(index="n", columns="language", values="median_s").dropna()
    if not piv.empty:
        n = int(piv.index.max())
        parts = ", ".join(f"{NAMES[l]} {fmt_time(piv.loc[n, l])}" for l in piv.columns)
        out.append(f"At the largest size completed by all three implementations (n = {n}) the median kernel times were: {parts}.")
    if exps is not None and not exps.empty:
        parts = ", ".join(f"{NAMES[r.language]} {r.exponent:.2f}" for r in exps.itertuples())
        out.append(f"Fitted slope of log(time) against log(n) for n >= 100 (cubic model predicts 3.0): {parts}.")
    return out


def build():
    summ, runs, exps = load()
    pdf = ROOT / "report" / f"Assignment{CFG['assignment']:02d}_{CFG['student_number']}_Report.pdf"
    pdf.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title=CFG["title"], author=f"{CFG['first_name']} {CFG['surname']}")
    s = []
    # ---- title block
    s += [P(CFG["title"], title),
          P(f"{CFG['first_name']} {CFG['surname']} &nbsp;|&nbsp; Student number {CFG['student_number']}", center),
          P(f"Big Data - Individual Assignment {CFG['assignment']} - Universidad de Las Palmas de Gran Canaria, 2026-2027", center),
          Spacer(1, 8)]
    # ---- abstract
    s.append(P("Abstract", h2))
    if summ is not None:
        ok = summ[summ["status"] == "ok"]
        mx = {NAMES[l]: int(d["n"].max()) for l, d in ok.groupby("language")}
        s.append(P("This study measures how matrix size affects kernel runtime and peak memory for the basic triple-loop "
                   "multiplication of dense double-precision matrices in C, Java and Python, using shared input files, "
                   "repeated timed runs and a single-threaded configuration on one machine. Largest completed sizes: "
                   + ", ".join(f"{k} n = {v}" for k, v in mx.items()) +
                   ". Conclusions apply only to the tested implementations, settings and hardware."))
    else:
        s.append(P("[Abstract is generated from the results. Run the benchmarks, then rebuild the report.]", todo))

    # ---- 1 Introduction
    s += [P("1. Introduction and objective", h1),
          P("Matrix multiplication is a standard test of how well a language runtime turns simple loops into machine work. "
            "The classic algorithm multiplies two n x n matrices with three nested loops and performs n<super>3</super> "
            "multiply-add operations [1], [3]. This report asks a focused question:"),
          P("<b>Research question.</b> How does matrix size n affect kernel runtime and peak memory of the same triple-loop "
            "algorithm implemented in Python, Java and C, under fixed and documented conditions?"),
          P("<b>Hypotheses (stated before measuring).</b> H1: runtime grows roughly as n<super>3</super> in all three "
            "implementations once n is large enough for fixed costs to be negligible. H2: for the same n, the compiled C "
            "version is faster than Java after JIT warm-up, and both are much faster than the Python interpreter. "
            "H3: peak memory is dominated by a language-runtime baseline at small n and by matrix storage at large n, and "
            "Python uses far more memory per element than C because it stores boxed floats in lists. "
            "H4: for large n, the column-wise access to B hurts cache behaviour [2], which may make the measured exponent "
            "exceed 3 for compiled code. These are predictions, not results.")]

    # ---- 2 Background
    s += [P("2. Background", h1),
          P("The triple-loop algorithm computes C<sub>ij</sub> = sum over k of A<sub>ik</sub> B<sub>kj</sub>. With the i-j-k "
            "loop order used here, A is read along a row, which is contiguous in memory, while B is read down a column, which "
            "has a stride of n elements in a row-major layout. For large n this strided access can cause many cache misses, "
            "which is why loop reordering and blocking are standard optimisations [2]. Those optimisations are deliberately "
            "not used in this assignment, which isolates the basic algorithm. Floating-point addition is not associative, so "
            "any change in evaluation order can change the last bits of the result, which is why correctness is checked with "
            "a tolerance rather than exact equality [4]. NumPy [5] is used only as a trusted reference for checking results "
            "and for analysis; it is never used inside the timed code.")]

    # ---- 3 Methodology
    s += [P("3. Methodology", h1), P("3.1 Implementations", h2),
          P("All three programs implement the same i-j-k triple loop with a scalar accumulator, on double-precision (64-bit) "
            "values, single-threaded. No library multiplication routine is used. Differences that could not be avoided are "
            "listed in Table 1."),
          table([["", "C", "Java", "Python"],
                 ["Storage layout", "one flat row-major array", "double[n][n] (array of row arrays)", "list of lists of float objects"],
                 ["Access to B", "B[k*n+j]", "B[k][j]", "B[k][j]"],
                 ["Execution", "gcc ahead-of-time, -O2", "JVM with JIT compilation", "CPython interpreter"],
                 ["Warm-up (untimed runs)", "1", "2", "1 (n&lt;400), 0 (n&gt;=400)"],
                 ["Timer", "clock_gettime(MONOTONIC)", "System.nanoTime()", "time.perf_counter_ns()"]],
                [3.6 * cm, 4 * cm, 4.5 * cm, 4.4 * cm]),
          P("Table 1. Implementation details and unavoidable differences.", caption),
          P("3.2 Inputs", h2),
          P("A Python script generates, for each n, one binary file containing A then B as little-endian float64 values drawn "
            "uniformly from [0, 1) with NumPy's default_rng, seeded with 2026 + n. All three programs read the same file, so the "
            "logical inputs are identical, which avoids differences between per-language random generators. Sizes tested: "
            "10, 50, 100, 200, 400, 600, 800, 1000."),
          P("3.3 Environment", h2), env_table(), Spacer(1, 3),
          P("3.4 Timing boundary and repetitions", h2),
          P("Only the multiplication kernel is timed. File loading, allocation and zero-initialisation of C happen before the "
            "timer starts. Each language and size runs in a fresh process; the process performs the untimed warm-up runs and "
            "then 5 timed repetitions (3 for Python at n&gt;=600 because of its cost). Every timing is stored in results/raw.csv. "
            "The report gives the median and the interquartile range (IQR) of the repetitions. The result of each run is "
            "consumed through a checksum of C so that the computation cannot be skipped."),
          P("3.5 Memory metric", h2),
          P("Memory is the <b>peak resident set size</b> of the whole process (Linux VmHWM, read by the program at exit). It "
            "includes the language runtime (for example the JVM or the Python interpreter), so it is not the same quantity as "
            "the 3 x 8n<super>2</super> bytes of raw matrix data. It is not a managed-heap measurement. Peak RSS is measured "
            "once per configuration, so it has no variability estimate."),
          P("3.6 Correctness", h2),
          P("Before benchmarking, scripts/test_correctness.py runs every implementation on: a hand-checked 3 x 3 example, "
            "identity x B and A x identity, zero matrices, and random matrices (n = 10, 64, 100). Results are compared with "
            "NumPy using relative tolerance 1e-9 and absolute tolerance 1e-12. The inputs are doubles in [0, 1) and each entry "
            "is a sum of at most a few hundred products, so rounding differences are of order 1e-14, far below the tolerance."),
          P("3.7 Budgets", h2),
          P("Maximum runtime per process (warm-up plus all repetitions): 600 s. Memory budget: 3 GB (Java heap limited with "
            "-Xmx3g). A run that exceeds a budget is recorded with its status and no time is invented for it. The largest "
            "<i>tested</i> size is reported separately from the largest size <i>attempted</i>.")]
    cf = RES / "correctness.txt"
    if cf.exists():
        txt = cf.read_text()
        npass, nfail = txt.count("PASS"), txt.count("FAIL")
        s.append(P(f"Correctness outcome: {npass} checks passed and {nfail} failed (full log in results/correctness.txt)."))
    else:
        s.append(P("[Run scripts/test_correctness.py and save its output to results/correctness.txt, then rebuild.]", todo))

    # ---- 4 Results
    s.append(P("4. Results", h1))
    if summ is None:
        s.append(P("[No measurements yet. Run the benchmark campaign described in README.md on your own machine, then "
                   "run scripts/analyze.py and scripts/make_report.py. Tables and figures will appear here.]", todo))
    else:
        langs = [l for l in ("c", "java", "python") if l in summ["language"].values]
        ns = sorted(summ["n"].unique())
        hdr = ["n"] + [f"{NAMES[l]} median (IQR)" for l in langs]
        rows = [hdr]
        for n in ns:
            r = [str(n)]
            for l in langs:
                d = summ[(summ.language == l) & (summ.n == n)]
                r.append("-" if d.empty else f"{fmt_time(d.median_s.iloc[0])} ({fmt_time(d.iqr_s.iloc[0])})")
            rows.append(r)
        s += [Spacer(1, 1),
              table(rows, [2 * cm] + [4.6 * cm] * len(langs)),
              P("Table 2. Kernel time (median, IQR) per multiplication. A dash means no run completed within the budget.", caption),
              Spacer(1, 6)]
        rows = [["n"] + [f"{NAMES[l]} peak RSS (MB)" for l in langs] + ["3 x 8n^2 bytes (MB)"]]
        for n in ns:
            r = [str(n)]
            for l in langs:
                d = summ[(summ.language == l) & (summ.n == n)]
                r.append("-" if d.empty or pd.isna(d.peak_rss_mb.iloc[0]) else f"{d.peak_rss_mb.iloc[0]:.1f}")
            r.append(f"{3 * 8 * n * n / 1e6:.2f}")
            rows.append(r)
        s += [table(rows, [2 * cm] + [3.4 * cm] * len(langs) + [3.6 * cm]),
              P("Table 3. Peak resident memory of the whole process versus raw matrix storage.", caption), Spacer(1, 6)]
        for fn, cap in (("runtime_vs_n.png", "Figure 1. Median kernel time against n (log-log). Error bars show the IQR. The dashed line is an n^3 reference anchored at the first point with n >= 100."),
                        ("memory_vs_n.png", "Figure 2. Peak process memory against n (log-log), compared with raw array storage."),
                        ("ratio_vs_c.png", "Figure 3. Median time of Java and Python relative to C at each size.")):
            if (FIG / fn).exists():
                s.append(KeepTogether([Image(str(FIG / fn), width=12.5 * cm, height=12.5 * cm * 4.2 / 6.2),
                                       P(cap, caption), Spacer(1, 6)]))
        s.append(P("Observed facts (generated from the data):", h2))
        for line in auto_text(summ, runs, exps):
            s.append(P("- " + line, small))

    # ---- 5 Discussion
    def md(name):
        f = ROOT / name
        t = f.read_text().strip() if f.exists() else ""
        st = todo if t.startswith("[FILL") else body
        return [P(par, st) for par in t.split("\n\n") if par.strip()]
    s += [P("5. Discussion", h1)] + md("discussion.md")
    s += [P("6. Conclusions", h1)] + md("conclusions.md")

    # ---- 7 AI disclosure
    s += [P("7. Use of generative AI", h1), P(CFG["ai_disclosure"])]

    # ---- 8 References & materials
    s += [P("8. References and materials", h1),
          P("[1] G. H. Golub and C. F. Van Loan, <i>Matrix Computations</i>, 4th ed. Baltimore, MD, USA: Johns Hopkins University Press, 2013.", small),
          P("[2] J. L. Hennessy and D. A. Patterson, <i>Computer Architecture: A Quantitative Approach</i>, 6th ed. Cambridge, MA, USA: Morgan Kaufmann, 2019.", small),
          P("[3] T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, <i>Introduction to Algorithms</i>, 4th ed. Cambridge, MA, USA: MIT Press, 2022.", small),
          P("[4] N. J. Higham, <i>Accuracy and Stability of Numerical Algorithms</i>, 2nd ed. Philadelphia, PA, USA: SIAM, 2002.", small),
          P("[5] C. R. Harris <i>et al.</i>, \"Array programming with NumPy,\" <i>Nature</i>, vol. 585, pp. 357-362, 2020.", small),
          Spacer(1, 4),
          P(f"<b>Reproducibility materials:</b> source code, README with exact build and run commands, shared-input generator, "
            f"raw measurements (results/raw.csv, results/runs.csv), analysis and plotting scripts: {CFG['github_url']}", body)]
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", pdf)


if __name__ == "__main__":
    build()
