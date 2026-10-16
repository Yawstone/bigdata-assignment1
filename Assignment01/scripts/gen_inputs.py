"""Generate shared input files. Each file holds A then B, n*n little-endian float64 each,
row-major, values uniform in [0, 1). One RNG stream per n, seeded with SEED + n.
Usage: python scripts/gen_inputs.py 10 50 100 ...
"""
import sys
from pathlib import Path
import numpy as np

SEED = 2026
OUT = Path(__file__).resolve().parent.parent / "inputs"


def main():
    OUT.mkdir(exist_ok=True)
    sizes = [int(s) for s in sys.argv[1:]] or [10, 50, 100, 200, 400, 600, 800, 1000]
    for n in sizes:
        rng = np.random.default_rng(SEED + n)
        A = rng.random((n, n))
        B = rng.random((n, n))
        with open(OUT / f"input_{n}.bin", "wb") as f:
            A.astype("<f8").tofile(f)
            B.astype("<f8").tofile(f)
        print(f"wrote input_{n}.bin ({2 * n * n * 8 / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
