"""
bench_monte_carlo.py

Python baselines for Monte Carlo option pricing:
  (1) pure Python loop
  (2) NumPy vectorized

Verifies both against analytical Black-Scholes.

Run: python bench_monte_carlo.py

Author: Henry Han, Baylor University
License: MIT
"""
import math
import random
import time
from statistics import NormalDist

import numpy as np


def black_scholes_call(spot, strike, rate, vol, maturity):
    """Closed-form European call price."""
    d1 = (math.log(spot / strike) +
          (rate + 0.5 * vol * vol) * maturity) / (vol * math.sqrt(maturity))
    d2 = d1 - vol * math.sqrt(maturity)
    N = NormalDist().cdf
    return spot * N(d1) - strike * math.exp(-rate * maturity) * N(d2)


def mc_price_py(spot, strike, rate, vol, maturity, n_paths):
    """Pure-Python Monte Carlo. Slow on purpose: this is the baseline."""
    drift = (rate - 0.5 * vol * vol) * maturity
    diffusion = vol * math.sqrt(maturity)
    total = 0.0
    for _ in range(n_paths):
        z = random.gauss(0.0, 1.0)
        ST = spot * math.exp(drift + diffusion * z)
        total += max(0.0, ST - strike)
    return math.exp(-rate * maturity) * total / n_paths


def mc_price_numpy(spot, strike, rate, vol, maturity, n_paths):
    """NumPy vectorized Monte Carlo. Fast because the inner loop is C."""
    drift = (rate - 0.5 * vol * vol) * maturity
    diffusion = vol * math.sqrt(maturity)
    z = np.random.standard_normal(n_paths)
    ST = spot * np.exp(drift + diffusion * z)
    payoffs = np.maximum(0.0, ST - strike)
    return math.exp(-rate * maturity) * float(np.mean(payoffs))


def main():
    spot, strike, rate, vol, maturity = 100.0, 100.0, 0.05, 0.20, 1.0
    n_paths = 1_000_000

    bs = black_scholes_call(spot, strike, rate, vol, maturity)
    print(f"PY_KEY n_paths {n_paths}")
    print(f"PY_KEY black_scholes_analytical {bs:.8f}")

    # Pure Python: 1M is too slow, do 100k and extrapolate
    n_py = 100_000
    random.seed(42)
    t0 = time.perf_counter_ns()
    price_py = mc_price_py(spot, strike, rate, vol, maturity, n_py)
    t1 = time.perf_counter_ns()
    elapsed_py_ns = t1 - t0
    extrap_py_ns = elapsed_py_ns * (n_paths // n_py)
    print(f"PY_KEY pure_python_measured_n {n_py}")
    print(f"PY_KEY pure_python_ns_at_measured_n {elapsed_py_ns}")
    print(f"PY_KEY pure_python_extrapolated_1M_ns {extrap_py_ns}")
    print(f"PY_KEY pure_python_price_at_measured_n {price_py:.8f}")

    # NumPy at 1M paths, best of 3
    np.random.seed(42)
    best_np_ns = None
    price_np = 0.0
    for _ in range(3):
        np.random.seed(42)
        t0 = time.perf_counter_ns()
        price_np = mc_price_numpy(spot, strike, rate, vol, maturity, n_paths)
        t1 = time.perf_counter_ns()
        elapsed = t1 - t0
        if best_np_ns is None or elapsed < best_np_ns:
            best_np_ns = elapsed
    print(f"PY_KEY numpy_best_1M_ns {best_np_ns}")
    print(f"PY_KEY numpy_price {price_np:.8f}")
    print(f"PY_KEY numpy_vs_bs_abs_error {abs(price_np - bs):.6f}")


if __name__ == "__main__":
    main()
