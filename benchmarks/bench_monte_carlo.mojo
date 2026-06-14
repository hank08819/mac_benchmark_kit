# =============================================================
# bench_monte_carlo.mojo
#
# Monte Carlo European call option pricing benchmark.
# Compares serial Mojo against parallelize'd Mojo at 1M paths.
# Updates Figure 3 panel (a) of the paper.
#
# Run: pixi run mojo run bench_monte_carlo.mojo
#
# Author: Henry Han, Baylor University
# License: MIT
# =============================================================

from collections import List
from algorithm import parallelize
from math import exp, log, sqrt
from random import random_float64, seed
from time import perf_counter_ns


# -------------------------------------------------------------
# Polynomial cosine (portable, avoids math.cos availability issues)
# -------------------------------------------------------------
def _cos_approx(x: Float64) -> Float64:
    var t = x
    while t > 3.14159265358979:
        t -= 2.0 * 3.14159265358979
    while t < -3.14159265358979:
        t += 2.0 * 3.14159265358979
    var t2 = t * t
    var t4 = t2 * t2
    return 1.0 - 0.5 * t2 + (1.0 / 24.0) * t4 - (1.0 / 720.0) * t4 * t2


# -------------------------------------------------------------
# Single-thread Monte Carlo call pricer
# -------------------------------------------------------------
def mc_price_serial(
    spot: Float64, strike: Float64, rate: Float64,
    vol: Float64, maturity: Float64, n_paths: Int,
) -> Float64:
    var drift = (rate - 0.5 * vol * vol) * maturity
    var diffusion = vol * sqrt(maturity)
    var total: Float64 = 0.0
    for _ in range(n_paths):
        var u1 = random_float64()
        var u2 = random_float64()
        if u1 < 1e-12:
            u1 = 1e-12
        var z = sqrt(-2.0 * log(u1)) * _cos_approx(2.0 * 3.14159265358979 * u2)
        var ST = spot * exp(drift + diffusion * z)
        var payoff = ST - strike
        if payoff < 0.0:
            payoff = 0.0
        total += payoff
    return exp(-rate * maturity) * total / Float64(n_paths)


# -------------------------------------------------------------
# Parallel price-many-options pattern from §5.2 of the paper
# -------------------------------------------------------------
def price_many_options_parallel(
    strikes: List[Float64], outputs: List[Float64],
    spot: Float64, rate: Float64, vol: Float64,
    maturity: Float64, n_paths: Int,
):
    @parameter
    def price_one(i: Int):
        outputs[i] = mc_price_serial(
            spot, strikes[i], rate, vol, maturity, n_paths
        )
    parallelize[price_one](len(strikes))


def main():
    var spot: Float64 = 100.0
    var rate: Float64 = 0.05
    var vol: Float64 = 0.20
    var maturity: Float64 = 1.0
    var n_paths = 1_000_000

    seed(42)

    print("MOJO_KEY n_paths", n_paths)

    # Single option serial
    var t0 = perf_counter_ns()
    var price_serial = mc_price_serial(
        spot, 100.0, rate, vol, maturity, n_paths
    )
    var t1 = perf_counter_ns()
    var serial_ns = t1 - t0
    print("MOJO_KEY mojo_serial_one_option_ns", serial_ns)
    print("MOJO_KEY mojo_serial_one_option_price", price_serial)

    # Parallel: 16 strikes priced together
    var strikes = List[Float64]()
    var k: Float64 = 80.0
    for _ in range(16):
        strikes.append(k)
        k += 2.5
    var outputs = List[Float64]()
    for _ in range(16):
        outputs.append(0.0)

    seed(42)
    var t2 = perf_counter_ns()
    price_many_options_parallel(
        strikes, outputs, spot, rate, vol, maturity, n_paths
    )
    var t3 = perf_counter_ns()
    var parallel_ns = t3 - t2
    var per_option_ns = parallel_ns // 16
    print("MOJO_KEY mojo_parallel_16_options_total_ns", parallel_ns)
    print("MOJO_KEY mojo_parallel_per_option_ns", per_option_ns)
    print("MOJO_KEY mojo_parallel_first_price", outputs[0])
    print("MOJO_KEY mojo_parallel_atm_price", outputs[8])


main()
