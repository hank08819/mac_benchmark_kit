# =============================================================
# bench_ewma.mojo
#
# Exponentially weighted moving average over a 10M-element tick
# stream, using Mojo's vectorize primitive.
#
# Run: pixi run mojo run bench_ewma.mojo
#
# Author: Henry Han, Baylor University
# License: MIT
# =============================================================

from collections import List
from algorithm import vectorize
from sys.info import simdwidthof
from random import random_float64, seed
from time import perf_counter_ns


alias DT = DType.float64
alias W = simdwidthof[DT]()


def ewma_scalar(prices: List[Float64], alpha: Float64, out: List[Float64]):
    """Scalar baseline EWMA."""
    var inv_alpha = 1.0 - alpha
    for i in range(len(prices)):
        out[i] = alpha * prices[i] + inv_alpha * out[i]


def ewma_vectorized(
    prices: List[Float64], alpha: Float64, out: List[Float64]
):
    """SIMD-vectorized EWMA via the @parameter + vectorize idiom."""
    var inv_alpha = 1.0 - alpha
    @parameter
    def step[w: Int](i: Int):
        out.store[w](
            i,
            alpha * prices.load[w](i) + inv_alpha * out.load[w](i),
        )
    vectorize[step, W](len(prices))


def main():
    var n = 10_000_000
    var alpha: Float64 = 0.05

    seed(42)
    var prices = List[Float64]()
    var out_a = List[Float64]()
    var out_b = List[Float64]()
    var p: Float64 = 100.0
    for _ in range(n):
        p += (random_float64() - 0.5) * 0.1
        prices.append(p)
        out_a.append(p)
        out_b.append(p)

    print("MOJO_KEY n_elements", n)
    print("MOJO_KEY hardware_simd_width", W)

    # Warmup
    ewma_scalar(prices, alpha, out_a)
    ewma_vectorized(prices, alpha, out_b)

    # Reset state and time scalar
    for i in range(n):
        out_a[i] = prices[0]
    var best_scalar_ns: Int = 0
    for trial in range(3):
        for i in range(n):
            out_a[i] = prices[0]
        var t0 = perf_counter_ns()
        ewma_scalar(prices, alpha, out_a)
        var t1 = perf_counter_ns()
        var elapsed = t1 - t0
        if trial == 0 or elapsed < best_scalar_ns:
            best_scalar_ns = elapsed
    print("MOJO_KEY mojo_scalar_best_ns", best_scalar_ns)
    print("MOJO_KEY mojo_scalar_last_value", out_a[n - 1])

    # Time vectorized
    var best_vec_ns: Int = 0
    for trial in range(3):
        for i in range(n):
            out_b[i] = prices[0]
        var t0 = perf_counter_ns()
        ewma_vectorized(prices, alpha, out_b)
        var t1 = perf_counter_ns()
        var elapsed = t1 - t0
        if trial == 0 or elapsed < best_vec_ns:
            best_vec_ns = elapsed
    print("MOJO_KEY mojo_vectorized_best_ns", best_vec_ns)
    print("MOJO_KEY mojo_vectorized_last_value", out_b[n - 1])


main()
