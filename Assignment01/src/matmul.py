"""Basic dense matrix multiplication, triple loop (i-j-k), C = A * B.

Storage: list of lists of Python floats.
Usage: python matmul.py <input.bin> <n> <reps> <warmup> [output.bin]
Timer covers only the triple loop. File loading and allocation are outside.
No NumPy is used inside the timed code (NumPy is only used by the test/analysis scripts).
"""
import sys
import time
from array import array


def multiply(A, B, C, n):
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += Ai[k] * B[k][j]
            Ci[j] = s


def read_matrices(path, n):
    count = n * n
    a = array("d")
    with open(path, "rb") as f:
        a.fromfile(f, 2 * count)          # little-endian float64 on all common platforms
    flat = a.tolist()
    A = [flat[i * n:(i + 1) * n] for i in range(n)]
    B = [flat[count + i * n:count + (i + 1) * n] for i in range(n)]
    return A, B


def peak_rss_kb():
    """Peak resident set size of this process (Linux VmHWM, kB); -1 if unavailable."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def main():
    if len(sys.argv) < 5:
        print("usage: matmul.py input.bin n reps warmup [output.bin]", file=sys.stderr)
        sys.exit(2)
    path, n, reps, warmup = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    A, B = read_matrices(path, n)
    C = [[0.0] * n for _ in range(n)]

    for _ in range(warmup):
        multiply(A, B, C, n)
    for r in range(reps):
        t0 = time.perf_counter_ns()
        multiply(A, B, C, n)
        t1 = time.perf_counter_ns()
        print(f"rep,{r},{t1 - t0}", flush=True)

    checksum = sum(sum(row) for row in C)
    print(f"checksum,{checksum:.12e}")
    print(f"rss_kb,{peak_rss_kb()}")

    if len(sys.argv) > 5:
        out = array("d")
        for row in C:
            out.extend(row)
        with open(sys.argv[5], "wb") as f:
            out.tofile(f)


if __name__ == "__main__":
    main()
