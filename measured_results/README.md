# Measured Results (Apple Silicon)

These are the real benchmark measurements reported in the paper,
collected on:

- **Machine:** Apple Silicon (ARM64)
- **OS:** macOS 15.4.1
- **Python:** 3.12.7 / PyTorch 2.5.1
- **Mojo:** 1.0.0 beta1
- **Date:** 2026-06-14

## Files

- `results_dot_product.json` — 10M-element f64 dot product (Table 1)
- `results_monte_carlo.json` — 1M-path European call Monte Carlo (Table 1)
- `results_ewma.json` — 10M-tick EWMA (Table 1)
- `mps_determinism_results_v2.json` — GPU determinism on MPS (Figure 2)

## Performance (Table 1)

| Kernel | Pure Python | Mojo best | Speedup |
|---|---|---|---|
| Dot product (SIMD) | 228.7 ms | 3.7 ms | 61.3x |
| Monte Carlo (parallelize) | 291.3 ms | 11.9 ms/option | 24.6x |
| EWMA (vectorize) | 292.5 ms | 1.7 ms | 177.1x |

## GPU determinism (Figure 2), measured on Apple Silicon MPS

| Probe | CPU distinct | MPS distinct |
|---|---|---|
| Permuted reduction order (/200) | 6 | 4 |
| Chunk count (/11) | 4 | 3 |
| Matmul contraction split (/7) | 5 | **6** |
| Run-to-run, same call (/100) | 1 | 1 |

Key findings:
1. Varying reduction order produces multiple distinct float32 results
   on both CPU and GPU. One 1024x1024 matmul gave 6 distinct results
   on MPS just by changing the contraction tiling.
2. The same matmul on CPU vs MPS differs by 1.46e-2 (cross-device
   reproducibility gap).
3. Run-to-run is deterministic on MPS for a fixed call (1/100),
   unlike CUDA atomic-add. The non-determinism that matters comes
   from reduction order and kernel/device choice, not random jitter.
   This is why language-level control of reduction order is the
   robust solution.

## Verification

The Monte Carlo Mojo price (10.434) matches analytical Black-Scholes
(10.451) to within 0.016, confirming numerical correctness.

## Reproduce

- Performance: `RUN_EVERYTHING.sh` from mac_test_kit
- GPU determinism: `python mps_determinism_test_v2.py` (in python/)
