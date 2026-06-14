"""
bench_dot_product.py

Python baselines for the dot-product benchmark. Measures:
  (1) pure Python for-loop
  (2) NumPy dot (BLAS-backed)

Prints results in MOJO_KEY format so the wrapper shell script can
grep them into JSON with the Mojo results.

Run: python bench_dot_product.py

Author: Henry Han, Baylor University
License: MIT
"""
import random
import time

import numpy as np


def dot_py(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def main():
    n = 10_000_000
    random.seed(42)
    print("PY_KEY n_elements", n)

    a = [random.random() for _ in range(n)]
    b = [random.random() for _ in range(n)]
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)

    # Warmup
    _ = dot_py(a[:10_000], b[:10_000])
    _ = np.dot(a_np, b_np)

    # Pure Python: this is slow, so we do it on 1M elements and
    # extrapolate to 10M to keep the script under a minute.
    n_py = 1_000_000
    best_py_ns = None
    for _ in range(3):
        t0 = time.perf_counter_ns()
        result_py = dot_py(a[:n_py], b[:n_py])
        t1 = time.perf_counter_ns()
        elapsed = t1 - t0
        if best_py_ns is None or elapsed < best_py_ns:
            best_py_ns = elapsed
    extrap_py_ns = best_py_ns * (n // n_py)
    print("PY_KEY pure_python_measured_n", n_py)
    print("PY_KEY pure_python_best_ns_at_measured_n", best_py_ns)
    print("PY_KEY pure_python_extrapolated_10M_ns", extrap_py_ns)
    print("PY_KEY pure_python_result_at_measured_n", result_py)

    # NumPy at full 10M
    best_np_ns = None
    for _ in range(5):
        t0 = time.perf_counter_ns()
        result_np = float(np.dot(a_np, b_np))
        t1 = time.perf_counter_ns()
        elapsed = t1 - t0
        if best_np_ns is None or elapsed < best_np_ns:
            best_np_ns = elapsed
    print("PY_KEY numpy_dot_best_ns", best_np_ns)
    print("PY_KEY numpy_dot_result", result_np)


if __name__ == "__main__":
    main()
