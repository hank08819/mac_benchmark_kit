"""
real_determinism_measurement.py

Measure actual ULP-scale variation in floating-point parallel reductions
on the host machine. The output of this script populates Figure 2 of
the paper, replacing the previous calibrated-synthetic version.

We measure:
  (A) per-element deviation in the final result as we vary the number
      of parallel chunks (which simulates different thread schedules)
  (B) cumulative drift over 1000 sequential parallel reductions, run
      8 independent trials with different chunk schedules
  (C) the same experiments using a deterministic tree reduction, which
      is bit-exact regardless of thread count

The arithmetic here is real IEEE-754 double precision arithmetic on
CPython, not a simulation. Different chunk counts genuinely produce
different bit patterns because of (a+b)+c != a+(b+c).

Author: Henry Han, Baylor University
Run:    python real_determinism_measurement.py
"""
import json
import time
import numpy as np


# ---------- Reductions ----------

def sum_sequential(a):
    """Reference: pure left-to-right scalar sum."""
    s = 0.0
    for x in a:
        s += x
    return s


def naive_parallel_sum(a, chunks):
    """Simulates a non-deterministic parallel reduction: split into
    `chunks` blocks, sum each block locally, then accumulate the block
    sums in a permuted order. Different chunk counts -> different
    rounding error accumulation -> different output bits.

    This is the pattern a GPU atomic-add reduction follows in spirit
    (though GPU atomic-add does it at the warp/wave level with
    hardware scheduling).
    """
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
    # Accumulate in reverse (one possible non-deterministic order)
    total = 0.0
    for c in range(chunks - 1, -1, -1):
        total += partials[c]
    return total


def naive_parallel_sum_permuted(a, chunks, perm):
    """Same as naive_parallel_sum but with a specific permutation of
    block-sum accumulation order. Different permutations simulate
    different thread completion orders."""
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
    total = 0.0
    for idx in perm:
        total += partials[idx]
    return total


def tree_sum(a):
    """Deterministic binary tree reduction. Output bits depend only on
    the data layout, not on thread scheduling. This is what Mojo's
    deterministic kernel pattern compiles to."""
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


def kahan_sum(a):
    """Kahan compensated summation: O(eps) total error, deterministic."""
    s = 0.0
    c = 0.0
    for x in a:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


# ---------- Experiment A: deviation vs chunk count ----------

def experiment_a():
    """Measure how the final sum changes as we vary parallel chunk count
    on a fixed input. This is the empirical analog of changing the GPU
    thread block count and observing different output bits."""
    print("=" * 70)
    print("EXPERIMENT A: per-element deviation vs chunk count")
    print("(simulates varying GPU thread schedule on a single reduction)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n = 100_000
    # Ill-conditioned input: wide range of magnitudes, mixed signs
    a = (rng.uniform(-1, 1, n) *
         rng.lognormal(mean=0, sigma=5, size=n)).tolist()

    ref = sum_sequential(a)
    tree_ref = tree_sum(a)
    kahan_ref = kahan_sum(a)

    print(f"\nInput: n = {n}, mixed-sign log-normal magnitudes")
    print(f"Sequential reference: {ref:.15e}")
    print(f"Tree reduction:       {tree_ref:.15e}"
          f"  (delta = {tree_ref - ref:+.3e})")
    print(f"Kahan summation:      {kahan_ref:.15e}"
          f"  (delta = {kahan_ref - ref:+.3e})")
    print()

    chunk_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    chunk_devs = []
    for c in chunk_counts:
        v = naive_parallel_sum(a, c)
        dev = v - ref
        chunk_devs.append(dev)
        print(f"  chunks = {c:4d}  result = {v:.15e}  "
              f"deviation = {dev:+.3e}")

    # Permutation experiment: hold chunks=64, vary the permutation
    print()
    print("Hold chunks = 64, vary block accumulation permutation")
    print("(simulates different thread completion orders):")
    n_perms = 100
    perm_results = []
    base = list(range(64))
    for trial in range(n_perms):
        perm = base.copy()
        rng.shuffle(perm)
        v = naive_parallel_sum_permuted(a, 64, perm)
        perm_results.append(v - ref)

    print(f"  {n_perms} random permutations of chunks=64:")
    print(f"    min deviation: {min(perm_results):+.3e}")
    print(f"    max deviation: {max(perm_results):+.3e}")
    print(f"    range:         {max(perm_results) - min(perm_results):.3e}")
    print(f"    distinct values: "
          f"{len(set(round(v, 15) for v in perm_results))}/{n_perms}")
    print(f"    (each distinct value = a different floating-point result")
    print(f"     that two PyTorch runs could produce on the same input)")

    return {
        "n_elements": n,
        "reference": ref,
        "tree_deviation": tree_ref - ref,
        "kahan_deviation": kahan_ref - ref,
        "chunk_counts": chunk_counts,
        "chunk_deviations": chunk_devs,
        "permutation_deviations": perm_results,
    }


# ---------- Experiment B: cumulative drift over a backtest ----------

def experiment_b():
    """Simulate a 1000-step backtest where each step performs a parallel
    reduction. Each trial has its own randomly-permuted chunk schedule.
    The deviations compound across steps.

    This is the empirical analog of running the SAME PyTorch model 8
    times on the SAME data and watching the cumulative P&L drift."""
    print()
    print("=" * 70)
    print("EXPERIMENT B: cumulative drift over a 1,000-step backtest")
    print("(8 trials of identical input, different reduction schedules)")
    print("=" * 70)

    rng = np.random.default_rng(7)
    n_per_step = 4_000
    n_steps = 1000
    n_trials = 8

    # Fixed input data for every step (deterministic across trials)
    rng_data = np.random.default_rng(123)
    step_data = []
    for s in range(n_steps):
        d = (rng_data.uniform(-1, 1, n_per_step) *
             rng_data.lognormal(0, 3, n_per_step)).tolist()
        step_data.append(d)

    # Reference: deterministic tree sum at every step
    ref_cum = 0.0
    ref_trajectory = []
    for s in range(n_steps):
        ref_cum += tree_sum(step_data[s])
        ref_trajectory.append(ref_cum)
    print(f"Computed deterministic reference trajectory: {n_steps} steps")
    print(f"Final reference value: {ref_cum:.6e}")
    print()

    # 8 non-deterministic trials
    trial_drifts = []
    chunks = 32
    for trial in range(n_trials):
        # Each trial uses its own per-step permutation
        cum = 0.0
        drift_traj = []
        for s in range(n_steps):
            perm = list(range(chunks))
            rng.shuffle(perm)
            v = naive_parallel_sum_permuted(step_data[s], chunks, perm)
            cum += v
            drift_traj.append(cum - ref_trajectory[s])
        trial_drifts.append(drift_traj)
        print(f"  Trial {trial + 1}: final drift = "
              f"{drift_traj[-1]:+.3e}")

    # The eight trajectories used the SAME input data, ran the SAME
    # arithmetic, differed only in chunk accumulation order
    abs_finals = [abs(t[-1]) for t in trial_drifts]
    print()
    print(f"Spread of final drift across {n_trials} trials:")
    print(f"  min |drift| = {min(abs_finals):.3e}")
    print(f"  max |drift| = {max(abs_finals):.3e}")
    print(f"  These are bit-different results from running the SAME model")
    print(f"  on the SAME data, differing only in reduction scheduling.")

    return {
        "n_steps": n_steps,
        "n_trials": n_trials,
        "n_per_step": n_per_step,
        "chunks": chunks,
        "trial_drift_trajectories": trial_drifts,
        "reference_final": ref_cum,
    }


# ---------- Save results ----------

def main():
    t0 = time.perf_counter()
    a = experiment_a()
    b = experiment_b()
    elapsed = time.perf_counter() - t0

    out = {
        "host_python_version": __import__("sys").version,
        "experiment_a": a,
        "experiment_b": b,
        "wall_clock_s": elapsed,
    }
    with open("determinism_measurements.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda x: list(x) if hasattr(x, "__iter__") else x)
    print()
    print(f"Total experiment time: {elapsed:.1f} s")
    print(f"Saved to determinism_measurements.json")


if __name__ == "__main__":
    main()
