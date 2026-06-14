# =============================================================
# bench_dot_product.mojo
#
# Benchmarks two Mojo implementations of dot product on 10M
# double-precision elements:
#   (1) scalar loop (looks like Python)
#   (2) SIMD-vectorized using hardware-native vector width
#
# Prints results in a one-line-per-key format that the wrapper
# shell script greps into results_dot_product.json.
#
# Run: pixi run mojo run bench_dot_product.mojo
#
# Author: Henry Han, Baylor University
# License: MIT
# =============================================================

from collections import List
from random import random_float64, seed
from sys.info import simdwidthof
from time import perf_counter_ns


alias DT = DType.float64
alias W = simdwidthof[DT]()


def dot_scalar(a: List[Float64], b: List[Float64]) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def dot_simd[width: Int](a: List[Float64], b: List[Float64]) -> Float64:
    alias VW = width
    var acc = SIMD[DT, VW](0.0)
    var i = 0
    var n = len(a)
    while i + VW <= n:
        acc += a.load[VW](i) * b.load[VW](i)
        i += VW
    var tail: Float64 = 0.0
    while i < n:
        tail += a[i] * b[i]
        i += 1
    return acc.reduce_add() + tail


def bench_one(label: String, n: Int, n_warmup: Int, n_repeats: Int):
    seed(42)
    var a = List[Float64]()
    var b = List[Float64]()
    for _ in range(n):
        a.append(random_float64())
        b.append(random_float64())

    print("MOJO_KEY n_elements", n)
    print("MOJO_KEY hardware_simd_width", W)

    # Warmup
    for _ in range(n_warmup):
        _ = dot_scalar(a, b)

    var best_scalar_ns: Int = 0
    var result_scalar: Float64 = 0.0
    for trial in range(n_repeats):
        var t0 = perf_counter_ns()
        result_scalar = dot_scalar(a, b)
        var t1 = perf_counter_ns()
        var elapsed = t1 - t0
        if trial == 0 or elapsed < best_scalar_ns:
            best_scalar_ns = elapsed
    print("MOJO_KEY mojo_scalar_best_ns", best_scalar_ns)
    print("MOJO_KEY mojo_scalar_result", result_scalar)

    for _ in range(n_warmup):
        _ = dot_simd[W](a, b)

    var best_simd_ns: Int = 0
    var result_simd: Float64 = 0.0
    for trial in range(n_repeats):
        var t0 = perf_counter_ns()
        result_simd = dot_simd[W](a, b)
        var t1 = perf_counter_ns()
        var elapsed = t1 - t0
        if trial == 0 or elapsed < best_simd_ns:
            best_simd_ns = elapsed
    print("MOJO_KEY mojo_simd_best_ns", best_simd_ns)
    print("MOJO_KEY mojo_simd_result", result_simd)


def main():
    # 10 million elements; 3 warmup, 5 timed repeats
    bench_one("dot10M", 10_000_000, 3, 5)
