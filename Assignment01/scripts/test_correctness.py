"""Correctness tests for the three implementations (run before benchmarking).

Cases: identity*B, A*identity, zero matrix, a hand-checked 3x3 example, and random
matrices of several sizes compared with NumPy (trusted reference, used for checking only).
Tolerance: np.allclose(rtol=1e-9, atol=1e-12). Inputs are doubles in [0,1) and each
entry is a sum of n products, so rounding differences of order n*2^-53 are expected;
1e-9 relative is many orders above that and still catches real mistakes.
"""
import subprocess, sys, tempfile, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build"
IMPLS = {
    "c": [str(BIN / "matmul")],
    "java": ["java", "-cp", str(BIN), "MatMul"],
    "python": [sys.executable, str(ROOT / "src" / "matmul.py")],
}
RTOL, ATOL = 1e-9, 1e-12


def run_impl(name, A, B, tmp):
    n = A.shape[0]
    inp, out = os.path.join(tmp, "in.bin"), os.path.join(tmp, f"out_{name}.bin")
    with open(inp, "wb") as f:
        A.astype("<f8").tofile(f)
        B.astype("<f8").tofile(f)
    cmd = IMPLS[name] + [inp, str(n), "1", "0", out]
    subprocess.run(cmd, check=True, capture_output=True)
    return np.fromfile(out, dtype="<f8").reshape(n, n)


def main():
    rng = np.random.default_rng(1)
    known_A = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    known_B = np.array([[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]])
    known_C = np.array([[30., 24., 18.], [84., 69., 54.], [138., 114., 90.]])
    cases = [("known 3x3", known_A, known_B, known_C)]
    for n in (1, 2, 7, 33):
        R = rng.random((n, n)) * 10 - 5
        cases.append((f"identity*B n={n}", np.eye(n), R, R))
        cases.append((f"A*identity n={n}", R, np.eye(n), R))
        cases.append((f"zero n={n}", np.zeros((n, n)), R, np.zeros((n, n))))
    for n in (10, 64, 100):
        A, B = rng.random((n, n)), rng.random((n, n))
        cases.append((f"random n={n}", A, B, A @ B))

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for impl in IMPLS:
            for label, A, B, expected in cases:
                got = run_impl(impl, A, B, tmp)
                ok = np.allclose(got, expected, rtol=RTOL, atol=ATOL)
                err = float(np.max(np.abs(got - expected)))
                print(f"{'PASS' if ok else 'FAIL'}  {impl:7s} {label:20s} max abs err={err:.2e}")
                failures += (not ok)
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
