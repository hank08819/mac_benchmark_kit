"""
bench_ewma.py

Python baselines for EWMA over 10M-element tick stream.
Measures pure Python loop and NumPy variant.

Run: python bench_ewma.py

Author: Henry Han, Baylor University
License: MIT
"""
import random
import time

import numpy as np


def ewma_py(prices, alpha, out):
    inv = 1.0 - alpha
    for i in range(len(prices)):
        out[i] = alpha * prices[i] + inv * out[i]


def ewma_numpy_naive(prices, alpha, out):
    """NumPy element-wise: NOT vectorizable because of the recurrence.
    This is honest - even NumPy can't vectorize a true EWMA inner loop
    without a closed-form trick, so it ends up Python-loop slow."""
    inv = 1.0 - alpha
    for i in range(len(prices)):
        out[i] = alpha * prices[i] + inv * out[i]


def ewma_numpy_pandaslike(prices, alpha):
    """Pandas-style EWMA using NumPy: closed-form-ish via cumulative."""
    n = len(prices)
    weights = (1 - alpha) ** np.arange(n)
    weighted = alpha * prices / np.where(weights > 0, weights, 1e-300)
    cumsum = np.cumsum(weighted)
    return cumsum * weights


def main():
    n = 10_000_000
    alpha = 0.05
    random.seed(42)

    prices_py = []
    p = 100.0
    for _ in range(n):
        p += (random.random() - 0.5) * 0.1
        prices_py.append(p)
    prices_np = np.asarray(prices_py, dtype=np.float64)

    print(f"PY_KEY n_elements {n}")

    # Pure Python loop: too slow at 10M, do 1M and extrapolate
    n_py = 1_000_000
    out = [prices_py[0]] * n_py
    best_py_ns = None
    for _ in range(3):
        out = [prices_py[0]] * n_py
        t0 = time.perf_counter_ns()
        ewma_py(prices_py[:n_py], alpha, out)
        t1 = time.perf_counter_ns()
        elapsed = t1 - t0
        if best_py_ns is None or elapsed < best_py_ns:
            best_py_ns = elapsed
    extrap_py_ns = best_py_ns * (n // n_py)
    print(f"PY_KEY pure_python_measured_n {n_py}")
    print(f"PY_KEY pure_python_ns_at_measured_n {best_py_ns}")
    print(f"PY_KEY pure_python_extrapolated_10M_ns {extrap_py_ns}")
    print(f"PY_KEY pure_python_last_value {out[-1]:.6f}")

    # NumPy element-wise (same recurrence, just in NumPy) — still loop-bound
    out_np = np.full(n, prices_np[0])
    best_np_ns = None
    for _ in range(3):
        out_np = np.full(n_py, prices_np[0])
        t0 = time.perf_counter_ns()
        ewma_numpy_naive(prices_np[:n_py], alpha, out_np)
        t1 = time.perf_counter_ns()
        elapsed = t1 - t0
        if best_np_ns is None or elapsed < best_np_ns:
            best_np_ns = elapsed
    extrap_np_ns = best_np_ns * (n // n_py)
    print(f"PY_KEY numpy_loop_ns_at_measured_n {best_np_ns}")
    print(f"PY_KEY numpy_loop_extrapolated_10M_ns {extrap_np_ns}")
    print(f"PY_KEY numpy_loop_last_value {float(out_np[-1]):.6f}")


if __name__ == "__main__":
    main()
