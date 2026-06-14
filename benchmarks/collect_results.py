"""
collect_results.py

Two roles:
  (1) merge per-benchmark py/mojo JSON files into a combined JSON for
      one benchmark
  (2) print a summary table across all three benchmarks

Usage:
    python collect_results.py <bench_name> <py.json> <mojo.json>
    python collect_results.py summary <combined1.json> <combined2.json> ...

Author: Henry Han, Baylor University
License: MIT
"""
import json
import platform
import sys
from datetime import datetime, timezone


def merge_one_benchmark(bench_name, py_path, mojo_path):
    with open(py_path) as f:
        py = json.load(f)
    with open(mojo_path) as f:
        mojo = json.load(f)
    return {
        "benchmark": bench_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_results": py,
        "mojo_results": mojo,
    }


def fmt_time(ns):
    """Format nanoseconds as human-readable."""
    if ns < 1_000:
        return f"{ns} ns"
    if ns < 1_000_000:
        return f"{ns/1_000:.1f} us"
    if ns < 1_000_000_000:
        return f"{ns/1_000_000:.1f} ms"
    return f"{ns/1_000_000_000:.2f} s"


def fmt_speedup(slow_ns, fast_ns):
    if fast_ns == 0:
        return "(timing too coarse)"
    return f"{slow_ns/fast_ns:.1f}x"


def print_summary(combined_files):
    print()
    print("=" * 70)
    print("MEASURED SPEEDUP SUMMARY")
    print("=" * 70)

    for f in combined_files:
        with open(f) as fh:
            data = json.load(fh)
        bench = data["benchmark"]
        py = data["python_results"]
        mojo = data["mojo_results"]

        print()
        print(f"--- {bench} ---")

        if bench == "dot_product":
            py_loop_ns = py.get("pure_python_extrapolated_10M_ns")
            np_ns = py.get("numpy_dot_best_ns")
            mojo_scalar_ns = mojo.get("mojo_scalar_best_ns")
            mojo_simd_ns = mojo.get("mojo_simd_best_ns")
            simd_w = mojo.get("hardware_simd_width", "?")

            print(f"  Pure Python loop (extrap to 10M): {fmt_time(py_loop_ns)}")
            print(f"  NumPy dot (BLAS):                 {fmt_time(np_ns)}")
            print(f"  Mojo scalar:                      {fmt_time(mojo_scalar_ns)}")
            print(f"  Mojo SIMD (width={simd_w}):           {fmt_time(mojo_simd_ns)}")
            print()
            print(f"  Mojo scalar vs pure Python:    "
                  f"{fmt_speedup(py_loop_ns, mojo_scalar_ns)}")
            print(f"  Mojo SIMD vs pure Python:      "
                  f"{fmt_speedup(py_loop_ns, mojo_simd_ns)}")
            print(f"  Mojo SIMD vs NumPy dot:        "
                  f"{fmt_speedup(np_ns, mojo_simd_ns)}")

        elif bench == "monte_carlo":
            py_loop_ns = py.get("pure_python_extrapolated_1M_ns")
            np_ns = py.get("numpy_best_1M_ns")
            mojo_serial_ns = mojo.get("mojo_serial_one_option_ns")
            mojo_par_ns = mojo.get("mojo_parallel_per_option_ns")

            print(f"  Pure Python (extrap to 1M paths): {fmt_time(py_loop_ns)}")
            print(f"  NumPy vectorized (1M paths):      {fmt_time(np_ns)}")
            print(f"  Mojo serial (1 option, 1M paths): {fmt_time(mojo_serial_ns)}")
            print(f"  Mojo parallelize'd (per option,")
            print(f"    16 options in parallel):        {fmt_time(mojo_par_ns)}")
            print()
            print(f"  Mojo serial vs pure Python:    "
                  f"{fmt_speedup(py_loop_ns, mojo_serial_ns)}")
            print(f"  Mojo serial vs NumPy:          "
                  f"{fmt_speedup(np_ns, mojo_serial_ns)}")
            print(f"  Mojo parallel vs Mojo serial:  "
                  f"{fmt_speedup(mojo_serial_ns, mojo_par_ns)}")
            print(f"  Mojo parallel vs NumPy:        "
                  f"{fmt_speedup(np_ns, mojo_par_ns)}")

            # Black-Scholes sanity
            bs = py.get("black_scholes_analytical")
            mc_np = py.get("numpy_price")
            mc_mojo = mojo.get("mojo_serial_one_option_price")
            if bs and mc_np and mc_mojo:
                print()
                print(f"  Black-Scholes analytical:      {bs:.6f}")
                print(f"  NumPy Monte Carlo:             {mc_np:.6f}"
                      f"  (error {abs(mc_np-bs):.6f})")
                print(f"  Mojo Monte Carlo:              {mc_mojo:.6f}"
                      f"  (error {abs(mc_mojo-bs):.6f})")

        elif bench == "ewma":
            py_loop_ns = py.get("pure_python_extrapolated_10M_ns")
            np_loop_ns = py.get("numpy_loop_extrapolated_10M_ns")
            mojo_scalar_ns = mojo.get("mojo_scalar_best_ns")
            mojo_vec_ns = mojo.get("mojo_vectorized_best_ns")
            simd_w = mojo.get("hardware_simd_width", "?")

            print(f"  Pure Python loop (extrap to 10M): {fmt_time(py_loop_ns)}")
            print(f"  NumPy loop (same recurrence):     {fmt_time(np_loop_ns)}")
            print(f"  Mojo scalar:                      {fmt_time(mojo_scalar_ns)}")
            print(f"  Mojo vectorize (width={simd_w}):       {fmt_time(mojo_vec_ns)}")
            print()
            print(f"  Mojo scalar vs pure Python:    "
                  f"{fmt_speedup(py_loop_ns, mojo_scalar_ns)}")
            print(f"  Mojo vectorize vs pure Python: "
                  f"{fmt_speedup(py_loop_ns, mojo_vec_ns)}")
            print(f"  Mojo vectorize vs Mojo scalar: "
                  f"{fmt_speedup(mojo_scalar_ns, mojo_vec_ns)}")

    print()
    print("=" * 70)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    if args[0] == "summary":
        print_summary(args[1:])
        return

    # Merge one benchmark
    bench_name = args[0]
    py_path = args[1]
    mojo_path = args[2]
    combined = merge_one_benchmark(bench_name, py_path, mojo_path)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
