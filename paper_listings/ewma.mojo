# =============================================================
# ewma.mojo
#
# Listing from Section 5.2 of:
#   "Mojo: A Promising Tool for Scalable Financial AI Efficiency"
#   Henry Han, Baylor University (CACM Practice submission, 2026)
#
# Exponentially-weighted moving average over a tick price stream,
# computed with hardware-native SIMD via Mojo's vectorize primitive.
# A common signal-processing pattern in market-making and
# microstructure analytics.
#
# Author: Henry Han, Baylor University
# License: MIT
# =============================================================

from collections import List
from algorithm import vectorize
from sys.info import simdwidthof
from random import random_float64, seed


alias DT = DType.float64
alias W = simdwidthof[DT]()


def ewma_update(
    prices: List[Float64],
    alpha: Float64,
    out: List[Float64],
):
    """In-place EWMA: out[i] = alpha * prices[i] + (1 - alpha) * out[i].

    Operates with SIMD width W chosen from the hardware. The
    @parameter / vectorize combination emits one SIMD instruction
    per iteration up to the array boundary.

    Note: when used for true streaming EWMA, this assumes 'out'
    is initialized with the previous-window state.
    """
    @parameter
    def step[w: Int](i: Int):
        out.store[w](
            i,
            alpha * prices.load[w](i) +
            (1.0 - alpha) * out.load[w](i),
        )
    vectorize[step, W](len(prices))


def main():
    seed(42)
    var n = 1024
    var prices = List[Float64]()
    var ewma = List[Float64]()
    var p: Float64 = 100.0
    for _ in range(n):
        # Simulate random walk price series
        p += (random_float64() - 0.5) * 0.1
        prices.append(p)
        ewma.append(p)   # initial state = first price

    var alpha: Float64 = 0.05   # half-life ~14 ticks
    ewma_update(prices, alpha, ewma)

    print("First 10 prices and their EWMA:")
    for i in range(10):
        print("  tick", i, "  price =", prices[i],
              "  ewma =", ewma[i])
    print("Last 10:")
    for i in range(n - 10, n):
        print("  tick", i, "  price =", prices[i],
              "  ewma =", ewma[i])
