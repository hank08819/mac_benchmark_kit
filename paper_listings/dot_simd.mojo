# =============================================================
# dot_simd.mojo
#
# Listing from Section 3.1 of:
#   "Mojo: A Promising Tool for Scalable Financial AI Efficiency"
#   Henry Han, Baylor University (CACM Practice submission, 2026)
#
# SIMD-vectorized dot product using compile-time-parameterized
# vector width. The same source compiles to AVX-512 on Intel
# servers, ARM SVE on ARM trading boxes, NEON on Apple silicon.
#
# Author: Henry Han, Baylor University
# License: MIT
# =============================================================

from collections import List
from sys.info import simdwidthof
from random import random_float64, seed
from time import now


alias DT = DType.float64
alias W = simdwidthof[DT]()  # hardware-native SIMD width


def dot_simd[width: Int](a: List[Float64], b: List[Float64]) -> Float64:
    """SIMD dot product. The 'width' parameter is compile-time,
    allowing the compiler to specialize and emit vector instructions
    of the requested width directly."""
    alias VW = width
    var acc = SIMD[DT, VW](0.0)
    var i = 0
    var n = len(a)
    # Process VW elements per iteration
    while i + VW <= n:
        acc += a.load[VW](i) * b.load[VW](i)
        i += VW
    # Handle remainder scalar-wise
    var tail: Float64 = 0.0
    while i < n:
        tail += a[i] * b[i]
        i += 1
    return acc.reduce_add() + tail


def main():
    seed(42)
    var n = 1_000_000
    var a = List[Float64]()
    var b = List[Float64]()
    for _ in range(n):
        a.append(random_float64())
        b.append(random_float64())

    print("Hardware SIMD width for Float64:", W)
    var t0 = now()
    var result = dot_simd[W](a, b)
    var t1 = now()
    var elapsed_ms = Float64(t1 - t0) / 1_000_000.0
    print("dot_simd[", W, "] result =", result)
    print("elapsed:", elapsed_ms, "ms over", n, "elements")
