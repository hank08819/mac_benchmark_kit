"""
python_baselines.py

Pure-Python baselines for the Mojo listings in the paper, so that
reviewers without a Mojo install can verify the algorithms and run
the figures themselves.

Usage:
    python python_baselines.py

Companion to:
  "Mojo: A Deterministic, Scalable Kernel Layer for Financial AI"
  Henry Han, Baylor University (CACM Practice submission, 2026)

Author: Henry Han, Baylor University
License: MIT
"""
import math
import random
import time


# -------------------------------------------------------------
# Pure-Python equivalents of the Mojo listings
# -------------------------------------------------------------
def dot_product_py(a, b):
    """Python equivalent of dot_product.mojo. About 100x slower."""
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def ewma_update_py(prices, alpha, out):
    """Python equivalent of ewma.mojo."""
    for i in range(len(prices)):
        out[i] = alpha * prices[i] + (1.0 - alpha) * out[i]


def monte_carlo_price_py(spot, strike, rate, vol, maturity, n_paths):
    """Python equivalent of price_options.mojo monte_carlo_price.
    About 1000x slower than the Mojo version with parallelize+vectorize.
    """
    drift = (rate - 0.5 * vol * vol) * maturity
    diffusion = vol * math.sqrt(maturity)
    sum_payoff = 0.0
    for _ in range(n_paths):
        z = random.gauss(0.0, 1.0)
        ST = spot * math.exp(drift + diffusion * z)
        sum_payoff += max(0.0, ST - strike)
    return math.exp(-rate * maturity) * sum_payoff / n_paths


# -------------------------------------------------------------
# The deterministic-vs-non-deterministic summation experiment
# from Section 4 of the paper, in pure Python.
# -------------------------------------------------------------
def sum_sequential(a):
    s = 0.0
    for x in a:
        s += x
    return s


def sum_naive_parallel(a, chunks):
    n = len(a)
    per = n // chunks
    partials = []
    for c in range(chunks):
        lo = c * per
        hi = lo + per if c < chunks - 1 else n
        s = 0.0
        for i in range(lo, hi):
            s += a[i]
        partials.append(s)
    # accumulate in reverse: simulates one non-deterministic order
    total = 0.0
    for c in range(chunks - 1, -1, -1):
        total += partials[c]
    return total


def sum_tree_deterministic(a):
    """Pairwise binary tree reduction. Order is data-determined."""
    buf = list(a)
    width = len(buf)
    while width > 1:
        half = width // 2
        for i in range(half):
            buf[i] = buf[2 * i] + buf[2 * i + 1]
        if width % 2 == 1:
            buf[half] = buf[width - 1]
            width = half + 1
        else:
            width = half
    return buf[0]


def sum_kahan(a):
    """Kahan compensated summation: O(eps) total error."""
    s = 0.0
    c = 0.0
    for x in a:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


# -------------------------------------------------------------
# Demonstrations
# -------------------------------------------------------------
def demo_dot_product():
    random.seed(42)
    n = 100_000
    a = [random.random() for _ in range(n)]
    b = [random.random() for _ in range(n)]
    t0 = time.perf_counter()
    result = dot_product_py(a, b)
    elapsed = time.perf_counter() - t0
    print(f"[dot_product_py] n={n}: result = {result:.6f}, "
          f"elapsed = {elapsed*1000:.2f} ms")
    print(f"  (Mojo equivalent typically ~100x faster)")


def demo_ewma():
    random.seed(42)
    n = 1024
    prices = []
    p = 100.0
    for _ in range(n):
        p += (random.random() - 0.5) * 0.1
        prices.append(p)
    ewma = list(prices)
    alpha = 0.05
    t0 = time.perf_counter()
    ewma_update_py(prices, alpha, ewma)
    elapsed = time.perf_counter() - t0
    print(f"[ewma_py] n={n}: first ewma = {ewma[0]:.4f}, "
          f"last = {ewma[-1]:.4f}, elapsed = {elapsed*1000:.3f} ms")


def demo_monte_carlo():
    random.seed(42)
    n_paths = 50_000
    t0 = time.perf_counter()
    price = monte_carlo_price_py(
        spot=100.0, strike=100.0,
        rate=0.05, vol=0.20, maturity=1.0,
        n_paths=n_paths,
    )
    elapsed = time.perf_counter() - t0
    # Black-Scholes analytical for comparison
    bs = black_scholes_call(100.0, 100.0, 0.05, 0.20, 1.0)
    print(f"[monte_carlo_py] n_paths={n_paths}: MC price = {price:.4f}, "
          f"BS analytical = {bs:.4f}, elapsed = {elapsed*1000:.0f} ms")


def black_scholes_call(spot, strike, rate, vol, maturity):
    """Closed-form Black-Scholes call price for sanity check."""
    from math import log, sqrt, exp
    from statistics import NormalDist
    d1 = (log(spot / strike) + (rate + 0.5 * vol * vol) * maturity) / \
         (vol * sqrt(maturity))
    d2 = d1 - vol * sqrt(maturity)
    N = NormalDist().cdf
    return spot * N(d1) - strike * exp(-rate * maturity) * N(d2)


def demo_summation_determinism():
    random.seed(0)
    n = 1_000_000
    a = []
    for i in range(n):
        sign = 1.0 if i % 2 == 0 else -1.0
        mag = random.random() * (1 + (i % 1024))
        a.append(sign * mag)

    s_seq    = sum_sequential(a)
    s_naive2 = sum_naive_parallel(a, 2)
    s_naive4 = sum_naive_parallel(a, 4)
    s_naive8 = sum_naive_parallel(a, 8)
    s_tree   = sum_tree_deterministic(a)
    s_kahan  = sum_kahan(a)

    print()
    print("Section 4 summation determinism experiment (in Python):")
    print(f"  Sequential sum:        {s_seq:.10f}")
    print(f"  Naive parallel (2):    {s_naive2:.10f}"
          f"  delta = {s_naive2 - s_seq:+.3e}")
    print(f"  Naive parallel (4):    {s_naive4:.10f}"
          f"  delta = {s_naive4 - s_seq:+.3e}")
    print(f"  Naive parallel (8):    {s_naive8:.10f}"
          f"  delta = {s_naive8 - s_seq:+.3e}")
    print(f"  Tree (deterministic):  {s_tree:.10f}"
          f"  delta = {s_tree - s_seq:+.3e}")
    print(f"  Kahan compensated:     {s_kahan:.10f}"
          f"  delta = {s_kahan - s_seq:+.3e}")
    print()
    print("Note: the naive-parallel deltas vary with chunk count;")
    print("the tree and Kahan variants would be bit-identical across")
    print("any thread/chunk count because their accumulation order")
    print("is data-determined, not schedule-determined.")


if __name__ == "__main__":
    print("=" * 60)
    print("Pure-Python baselines for the Mojo listings")
    print("=" * 60)
    demo_dot_product()
    demo_ewma()
    demo_monte_carlo()
    demo_summation_determinism()
