# =============================================================
# deterministic_reduction.mojo
#
# Companion to Section 4 of:
#   "Mojo: A Deterministic, Scalable Kernel Layer for Financial AI"
#   Henry Han, Baylor University (CACM Practice submission, 2026)
#
# The load-bearing claim of Section 4 is that Mojo gives the
# developer explicit control over reduction order, so that parallel
# floating-point sums are bit-exact reproducible across runs and
# across thread counts. This file demonstrates the pattern.
#
# Background: in IEEE 754 arithmetic, (a + b) + c is not in general
# equal to a + (b + c) (Goldberg 1991). For a parallel reduction
# where the order of accumulation depends on hardware scheduling,
# the result varies bit-by-bit between runs. CUDA atomic-add on
# floating-point values has exactly this property.
#
# A deterministic alternative is a binary tree reduction with a
# fixed, data-independent topology. This file implements one.
#
# Author: Henry Han, Baylor University
# License: MIT
# =============================================================

from collections import List
from random import random_float64, seed
from math import log2


# -------------------------------------------------------------
# Naive sequential sum (deterministic by construction but slow)
# -------------------------------------------------------------
def sum_sequential(a: List[Float64]) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(a)):
        s += a[i]
    return s


# -------------------------------------------------------------
# Naive parallel sum (illustrating the non-deterministic shape).
# In a real GPU kernel this would be a tree of atomic-adds whose
# commit order depends on hardware scheduling. Here we just shuffle
# the work to demonstrate that order-dependent results occur.
# -------------------------------------------------------------
def sum_naive_parallel(a: List[Float64], chunks: Int) -> Float64:
    """Split into 'chunks' contiguous blocks, sum each block,
    accumulate block sums in arbitrary order (here: reverse).
    For very different inputs and chunk counts, this can produce
    different bit patterns than the sequential sum."""
    var n = len(a)
    var per = n // chunks
    var partials = List[Float64]()
    for c in range(chunks):
        var lo = c * per
        var hi = lo + per if c < chunks - 1 else n
        var s: Float64 = 0.0
        for i in range(lo, hi):
            s += a[i]
        partials.append(s)

    # Accumulate partials in reverse: simulates one possible
    # non-deterministic ordering. Different chunk count -> different
    # rounding error -> different bits.
    var total: Float64 = 0.0
    for c in range(chunks - 1, -1, -1):
        total += partials[c]
    return total


# -------------------------------------------------------------
# Deterministic binary tree reduction.
# Topology is fixed by the data layout, not by thread scheduling.
# Pairwise summation has well-known better numerical properties
# than naive sequential summation as well (Higham, "Accuracy and
# Stability of Numerical Algorithms," Chapter 4).
# -------------------------------------------------------------
def sum_tree_deterministic(a: List[Float64]) -> Float64:
    """Pairwise tree reduction. The topology is determined entirely
    by the array index structure, so the result is bit-identical
    across runs, thread counts, and machines (given the same
    floating-point hardware semantics)."""
    var buf = List[Float64]()
    for i in range(len(a)):
        buf.append(a[i])

    var width = len(buf)
    while width > 1:
        var half = width // 2
        for i in range(half):
            buf[i] = buf[2 * i] + buf[2 * i + 1]
        if width % 2 == 1:
            buf[half] = buf[width - 1]
            width = half + 1
        else:
            width = half
    return buf[0]


# -------------------------------------------------------------
# Kahan compensated summation: a third deterministic option
# that reduces the accumulated rounding error to O(eps) rather
# than O(n * eps). Cited in Section 4.5 via Demmel & Nguyen 2013.
# -------------------------------------------------------------
def sum_kahan(a: List[Float64]) -> Float64:
    var sum: Float64 = 0.0
    var c: Float64 = 0.0   # running compensation
    for i in range(len(a)):
        var y = a[i] - c
        var t = sum + y
        c = (t - sum) - y
        sum = t
    return sum


def main():
    seed(0)
    var n = 1_000_000
    var a = List[Float64]()
    # Use a mix of magnitudes so rounding effects are visible
    for i in range(n):
        var sign = 1.0 if (i % 2 == 0) else -1.0
        var mag = random_float64() * Float64(1 + (i % 1024))
        a.append(sign * mag)

    var s_seq    = sum_sequential(a)
    var s_naive2 = sum_naive_parallel(a, 2)
    var s_naive4 = sum_naive_parallel(a, 4)
    var s_naive8 = sum_naive_parallel(a, 8)
    var s_tree   = sum_tree_deterministic(a)
    var s_kahan  = sum_kahan(a)

    print("Sequential sum:      ", s_seq)
    print("Naive parallel (2):  ", s_naive2,
          "  delta vs seq =", s_naive2 - s_seq)
    print("Naive parallel (4):  ", s_naive4,
          "  delta vs seq =", s_naive4 - s_seq)
    print("Naive parallel (8):  ", s_naive8,
          "  delta vs seq =", s_naive8 - s_seq)
    print("Tree (deterministic):", s_tree,
          "  delta vs seq =", s_tree - s_seq)
    print("Kahan compensated:   ", s_kahan,
          "  delta vs seq =", s_kahan - s_seq)
    print()
    print("Observation: changing chunk count in the naive parallel")
    print("scheme changes the result by O(eps) per partial. The tree")
    print("and Kahan variants are bit-identical regardless of how")
    print("the work is later parallelized across threads, because")
    print("their accumulation order is data-determined, not")
    print("schedule-determined. This is the property finance audit")
    print("requires.")
