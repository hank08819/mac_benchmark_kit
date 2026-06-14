"""
risk_reduction_case_study.py

Concrete worked example for Section 4 of the paper. A portfolio
risk-aggregation calculation, of the kind that runs every end-of-day
in every regulated trading firm, produces multiple distinct
floating-point values when the parallel reduction order is allowed
to vary. We measure this directly and report what we see.

The setup: 100,000 risk contributions across the portfolio
(legs, scenarios, factor exposures), aggregated into a single
end-of-day risk number. We run the same arithmetic 200 times
with different reduction schedules and count how many distinct
values come out.

This is the same arithmetic pattern that PyTorch's GPU atomic-add
reductions follow internally. We use pure Python here so the
experiment runs anywhere and is fully reproducible.

Author: Henry Han, Baylor University
Run:    python risk_reduction_case_study.py
"""
import json
import numpy as np


def naive_parallel_sum(arr, chunks, perm):
    """Non-deterministic parallel sum: split into `chunks`, sum each
    block locally, accumulate block sums in the given permutation."""
    n = len(arr)
    per = n // chunks
    partials = []
    for c in range(chunks):
        lo = c * per
        hi = lo + per if c < chunks - 1 else n
        s = 0.0
        for i in range(lo, hi):
            s += arr[i]
        partials.append(s)
    total = 0.0
    for idx in perm:
        total += partials[idx]
    return total


def tree_sum(arr):
    """Deterministic tree reduction. Bit-identical regardless of
    parallelization choices."""
    buf = list(arr)
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


def main():
    print("=" * 70)
    print("CASE STUDY: end-of-day portfolio risk reduction")
    print("=" * 70)
    print()

    # Simulate 100,000 risk contributions across a large portfolio.
    # Mix tiny and huge magnitudes, alternating signs -> ill-conditioned
    # for floating-point summation. This is realistic for a portfolio
    # that combines tens of thousands of small option legs with a few
    # large directional positions.
    rng = np.random.default_rng(11)
    n = 100_000
    base = rng.uniform(-1, 1, n)
    magnitudes = rng.lognormal(mean=0, sigma=6, size=n)
    risk_contribs = (base * magnitudes).tolist()

    print(f"Portfolio risk legs: {n:,}")
    print(f"Magnitude range:     {min(risk_contribs):.2e} to "
          f"{max(risk_contribs):.2e}")
    print()

    # Reference sums (deterministic)
    seq = 0.0
    for x in risk_contribs:
        seq += x
    tree = tree_sum(risk_contribs)

    print(f"Sequential reference:   {seq:.15e}")
    print(f"Tree reduction (det):   {tree:.15e}  "
          f"(delta from seq = {tree - seq:+.3e})")
    print()

    # Now the non-deterministic experiment: 200 random thread schedules
    chunks = 64
    n_schedules = 200
    schedule_results = []
    base_perm = list(range(chunks))
    rng2 = np.random.default_rng(42)
    for trial in range(n_schedules):
        perm = base_perm.copy()
        rng2.shuffle(perm)
        v = naive_parallel_sum(risk_contribs, chunks, perm)
        schedule_results.append(v)

    distinct_vals = sorted(set(schedule_results))
    print(f"Non deterministic parallel sum, {chunks} chunks, "
          f"{n_schedules} random schedules:")
    print(f"  Distinct floating point results: "
          f"{len(distinct_vals)} out of {n_schedules}")
    print(f"  Spread: {max(distinct_vals) - min(distinct_vals):.3e}")
    print(f"  As fraction of the risk number: "
          f"{(max(distinct_vals) - min(distinct_vals)) / abs(seq):.3e}")
    print()
    print(f"  Sample of distinct values (first 5):")
    for v in distinct_vals[:5]:
        print(f"    {v:.15e}  (delta vs seq = {v - seq:+.3e})")
    print()

    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print(f"On a regulated trading desk, this means that the same end of")
    print(f"day risk model, fed the same {n:,} risk contributions, can")
    print(f"produce {len(distinct_vals)} different numbers depending on")
    print(f"which thread in the GPU happens to finish first. In each of")
    print(f"those cases, a different number lands in the audit log, in")
    print(f"the regulatory report, and in the next day's position limit.")
    print()
    print(f"The deterministic tree reduction produces exactly ONE value")
    print(f"regardless of how the work is parallelized. That is the")
    print(f"property finance audit requires, and it is what a Mojo")
    print(f"implementation of this kernel using the tree reduction")
    print(f"pattern delivers.")

    return {
        "n_legs": n,
        "sequential_reference": seq,
        "tree_reduction": tree,
        "n_schedules": n_schedules,
        "distinct_values": len(distinct_vals),
        "spread": max(distinct_vals) - min(distinct_vals),
        "relative_spread": (max(distinct_vals) - min(distinct_vals)) / abs(seq),
    }


if __name__ == "__main__":
    result = main()
    with open("risk_reduction_result.json", "w") as f:
        json.dump(result, f, indent=2)
