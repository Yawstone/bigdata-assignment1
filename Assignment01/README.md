# Assignment 1 - Basic matrix multiplication in Python, Java and C

Student number: N021366D  
Course: Big Data, ULPGC 2026-2027

Compares the plain triple-loop (i-j-k) multiplication of dense double-precision matrices in C,
Java and Python: kernel runtime and peak process memory against matrix size n.
All three programs read the **same binary input file**, so the logical inputs are identical.

## Requirements
- Linux, or Windows with WSL2 (peak memory is read from `/proc`; native Windows is not supported)
- gcc, JDK 17+ (`javac`, `java`), Python 3.10+
- `pip install numpy pandas matplotlib reportlab psutil`

## Run order (from this folder)
```bash
./build.sh                                   # gcc -O2 and javac into build/
python scripts/test_correctness.py | tee results/correctness.txt   # must end with "0 failure(s)"
python scripts/gen_inputs.py                 # inputs/input_<n>.bin, seed 2026+n (sizes in the script)
python scripts/env_info.py                   # results/environment.json
python scripts/run_benchmarks.py             # results/raw.csv, results/runs.csv  (can take a long time)
python scripts/analyze.py                    # results/summary.csv, figures/*.png
# edit discussion.md, conclusions.md, report_config.json (name, GitHub URL, AI disclosure)
python scripts/make_report.py                # report/Assignment01_N021366D_Report.pdf
```
Close other programs and plug the laptop in while benchmarking. Do not run anything else
during the campaign. Run it once on a quiet machine and keep every file in `results/`.

## Settings that stay fixed
double precision; single thread; i-j-k loop order; kernel-only timing (load and allocation
outside the timer); C: `gcc -O2` (no `-march=native`, no `-ffast-math`);
Java: `-Xms512m -Xmx3g -XX:+UseSerialGC`; Python: CPython, no NumPy in the timed code.
Sizes, repetitions, warm-up counts, and the 600 s per-process budget are at the top of
`scripts/run_benchmarks.py`. If your machine is slow, lower the Python sizes there and
say so in the report.

## Files
| Path | Purpose |
|---|---|
| `src/matmul.c`, `src/MatMul.java`, `src/matmul.py` | The three implementations |
| `scripts/test_correctness.py` | Known answers, identity, zero, random vs NumPy (rtol 1e-9, atol 1e-12) |
| `scripts/gen_inputs.py` | Shared input generator |
| `scripts/run_benchmarks.py` | Harness: one fresh process per language and size |
| `scripts/analyze.py` | Median, IQR, memory table, growth exponent, figures |
| `scripts/make_report.py` | Builds the PDF from results and the text files |
| `results/raw.csv` | Every measured repetition (`language,n,rep,time_ns`) |
| `results/runs.csv` | Per-process info: status, peak RSS (kB), checksum |

## Reuse in later assignments
These implementations and the input files are the validated baseline for Assignments 2-4;
cite this repository and its commit hash when you reuse them.
