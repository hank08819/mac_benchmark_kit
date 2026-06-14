# mac_benchmark_kit

**Code companion to the CACM Practice paper *"Mojo: A Promising Tool for Scalable Financial AI Efficiency"*.**

![License](https://img.shields.io/badge/license-MIT-green)
![Mojo](https://img.shields.io/badge/Mojo-1.0.0%20beta1-orange)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-blue)
![Reproducible](https://img.shields.io/badge/reproducible-yes-brightgreen)

> Every measured number in the paper's Table 1 and Figure 2 came from
> running the scripts in `benchmarks/`. Every Mojo listing shown in
> the paper body is in `paper_listings/`. The real JSON outputs are
> in `measured_results/`. You can re-run, re-measure, and verify on
> any Apple Silicon Mac.

This repository is the code companion to:

**Henry Han.** *"Mojo: A Promising Tool for Scalable Financial AI
Efficiency."* Communications of the ACM, Practice Section, 2026.

The paper's primary code contribution, a small open-source library of
bit-reproducible reduction kernels, lives in its own repository:
**[hank08819/mojo-deterministic](https://github.com/hank08819/mojo-deterministic)**.

---

## What's in here

```
mac_benchmark_kit/
├── README.md                          this file
├── LICENSE                            MIT
├── benchmarks/                        reproduce Table 1 and Figure 2
├── paper_listings/                    the Mojo code shown in the paper body
└── measured_results/                  the real JSON outputs that became
                                       Table 1 and Figure 2
```

### `benchmarks/` — reproduce Table 1 and Figure 2

The driver `RUN_EVERYTHING.sh` runs every benchmark in the paper on
an Apple Silicon Mac. After ~15-20 minutes you have all measured
numbers in JSON.

```bash
cd benchmarks
chmod +x RUN_EVERYTHING.sh
./RUN_EVERYTHING.sh
```

It will (in order): detect Python and PyTorch, install Mojo via pixi
if missing, run the MPS GPU determinism experiment (~3-5 min), run
the three CPU benchmarks (~10 min), and bundle results into a
timestamped zip.

Individual files:

| File | Produces |
|---|---|
| `bench_dot_product.{py,mojo}` | Table 1, row 1 — 10M-element dot product |
| `bench_monte_carlo.{py,mojo}` | Table 1, row 2 — 1M-path European call (with Black-Scholes verification) |
| `bench_ewma.{py,mojo}` | Table 1, row 3 — 10M-tick EWMA |
| `mps_quick_check.py` | 30-second MPS sanity check |
| `mps_determinism_test_v2.py` | Figure 2 — four-probe GPU non-determinism experiment |
| `collect_results.py` | Merges JSON outputs and prints a summary table |
| `RUN_EVERYTHING.sh` | One-script driver |
| `run_all_benchmarks.sh` | Mojo CPU benchmarks only (skips MPS) |

### `paper_listings/` — the Mojo code shown in the paper body

These are the (mostly small) Mojo listings shown in the paper text,
plus their Python baselines, plus the three figure-generator scripts.
They are the **demonstration** versions, suitable for inclusion in
the paper body; the `benchmarks/` versions are the **measurement**
versions, instrumented with timing and best-of-N selection.

| File | Paper section |
|---|---|
| `dot_product.mojo` | §3.1 listing 1 — scalar dot product |
| `dot_simd.mojo` | §3.1 listing 2 — SIMD-vectorized dot product |
| `price_options.mojo` | §5.2 listing — parallel Monte Carlo with `parallelize` |
| `ewma.mojo` | §5.2 listing — EWMA with `vectorize` |
| `python_interop.mojo` | §6.1 listing — Mojo calling pandas/yfinance |
| `deterministic_reduction.mojo` | §4 demonstration of bit-exact reductions |
| `python_baselines.py` | Pure-Python equivalents for the listings above |
| `real_determinism_measurement.py` | The host-CPU determinism experiment |
| `risk_reduction_case_study.py` | §4.3 portfolio risk reduction case study |
| `figures/make_fig1.py` | Generates Figure 1 (calibrated synthetic) |
| `figures/make_fig2.py` | Generates Figure 2 from the measured JSON |
| `figures/make_fig3.py` | Generates Figure 3 (calibrated synthetic) |

### `measured_results/` — the real JSON outputs

These are the JSON files that became Table 1 and Figure 2 of the
paper. They are committed so you can compare your re-runs against
the authors' original measurements.

| File | Used by |
|---|---|
| `results_dot_product.json` | Table 1, row 1 |
| `results_monte_carlo.json` | Table 1, row 2 |
| `results_ewma.json` | Table 1, row 3 |
| `mps_determinism_results_v2.json` | Figure 2 (left and right panels) |

---

## Headline measured numbers

### Table 1 of the paper

| Kernel | Pure Python | Mojo best | Speedup |
|---|---|---|---|
| Dot product, 10M elements | 228.7 ms | 3.7 ms (SIMD) | **61.3x** |
| Monte Carlo call, 1M paths | 291.3 ms | 11.9 ms (parallelize) | **24.6x** |
| EWMA, 10M ticks | 292.5 ms | 1.7 ms (vectorize) | **177.1x** |

Black-Scholes verification: Mojo Monte Carlo price 10.434 vs.
analytical 10.451, error 0.016. The implementation is numerically
correct.

### Figure 2 of the paper

GPU determinism on Apple Silicon MPS (PyTorch 2.5.1, float32). The
script holds the mathematical value fixed and varies only the
reduction order, then counts distinct bit patterns:

| Probe | CPU distinct | MPS distinct |
|---|---|---|
| Permuted reduction order (/200) | 6 | 4 |
| Chunk count (/11) | 4 | 3 |
| Matmul contraction split (/7) | 5 | **6** |
| Run-to-run, same call (/100) | 1 | 1 |

The same 1024x1024 matmul on CPU vs MPS differs by 1.46e-2 — the
cross-device reproducibility gap.

---

## Quick start

```bash
git clone https://github.com/hank08819/mac_benchmark_kit.git
cd mac_benchmark_kit/benchmarks
chmod +x RUN_EVERYTHING.sh
./RUN_EVERYTHING.sh
```

---

## Requirements

- macOS 12.3+ (Monterey or later) — required for MPS
- Apple Silicon (M1/M2/M3/M4) recommended
- Python 3.10+ with pip
- Xcode command line tools (`xcode-select --install`)
- ~5 GB free disk for PyTorch + Mojo installs

The script auto-installs PyTorch and Mojo if they are missing.

---

## Honest interpretation

The paper draws conclusions from these numbers; this repository
provides the data. Three caveats worth knowing:

- **NumPy can beat Mojo on a bare dot product** (2.8 ms vs 3.7 ms)
  because NumPy dispatches to Apple's Accelerate BLAS. This is the
  point of §3.2: Mojo's advantage is in the glue code around the
  math, not in beating BLAS at a single matrix operation.
- **MPS is run-to-run deterministic for fixed calls** in our tests
  (probe D = 1/100). The non-determinism that matters comes from
  reduction order and kernel/device choice (probes A, B, C). CUDA
  atomic-add reductions behave differently.
- **Pure Python timings are extrapolated** from a smaller input
  (since 10M iterations in pure Python is impractical). This is
  conservative: interpreter overhead does not improve with scale.

All of this is discussed honestly in §5.3 and §8 of the paper.

---

## Companion repository

The deterministic-reduction library described in §4.5 of the paper
is published separately at:

**https://github.com/hank08819/mojo-deterministic**

That repository is a self-contained library (`from
deterministic_reduce import deterministic_risk_aggregate`); this one
is the experimental harness. They are independent — you can use
either or both.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{han2026mojo,
  author  = {Han, Henry},
  title   = {Mojo: A Promising Tool for Scalable Financial AI Efficiency},
  year    = {2026}
}
```

---

## Author

**Henry Han**, Data Science and Artificial Intelligence Innovation
Laboratory, School of Engineering and Computer Science, Baylor
University.

This work is partially supported by NASA Grant 80NSSC22K1015, NSF
2229138, and the McCollum endowed chair startup fund.

## License

[MIT](LICENSE) © 2026 Henry Han, Baylor University
