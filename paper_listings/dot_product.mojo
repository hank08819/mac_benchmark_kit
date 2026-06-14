# =============================================================
# dot_product.mojo
#
# Listing from Section 3.1 of:
#   "Mojo: A Promising Tool for Scalable Financial AI Efficiency"
#   Henry Han, Baylor University (CACM Practice submission, 2026)
#
# A minimal dot product. Pythonic syntax, native compiled performance.
# Run with: pixi run mojo run dot_product.mojo
#
# Author: Henry Han, Baylor University
# License: MIT
# =============================================================

from collections import List
from random import random_float64, seed


def dot_product(a: List[Float64], b: List[Float64]) -> Float64:
    """Compute the dot product of two equal-length vectors.

    Three additions to Python-like code make this fast:
      - explicit type annotations let the compiler specialize
      - var declares a stack-allocated mutable variable
      - Float64 means IEEE-754 doubles, not boxed Python floats

    Note: earlier Mojo versions distinguished `fn` (typed) from
    `def` (Python-like). In the current language line these are
    unified and `fn` is on a deprecation path; we use `def`
    throughout this package.
    """
    var s: Float64 = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def main():
    seed(42)
    var n = 1_000_000
    var a = List[Float64]()
    var b = List[Float64]()
    for _ in range(n):
        a.append(random_float64())
        b.append(random_float64())

    var result = dot_product(a, b)
    print("dot_product over", n, "elements =", result)
    print("expected near n/4 =", Float64(n) / 4.0)
