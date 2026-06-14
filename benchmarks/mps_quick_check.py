"""
mps_quick_check.py

30-second sanity check that PyTorch + MPS works on your Mac.
Run this BEFORE the full mps_determinism_test.py to make sure
the MPS backend is functional.

Run: python mps_quick_check.py

Author: Henry Han, Baylor University
License: MIT
"""
import platform
import sys

import torch


def main():
    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"macOS:   {platform.mac_ver()[0]}")
    print(f"Machine: {platform.machine()}")
    print()

    if not torch.backends.mps.is_available():
        print("[FAIL] torch.backends.mps.is_available() = False")
        print("       PyTorch cannot see the MPS backend.")
        print("       Possible fixes:")
        print("         pip install --upgrade torch")
        print("         (requires macOS 12.3+)")
        sys.exit(1)

    if not torch.backends.mps.is_built():
        print("[FAIL] torch.backends.mps.is_built() = False")
        print("       Your PyTorch build lacks MPS support.")
        sys.exit(1)

    print("[OK]   MPS backend available and built.")
    print()

    # Quick functional test
    try:
        x = torch.randn(1024, 1024, device="mps")
        y = x @ x
        torch.mps.synchronize()
        result = float(y.sum().cpu())
        print(f"[OK]   1024x1024 matmul on MPS: sum = {result:.4f}")
    except Exception as e:
        print(f"[FAIL] Matmul on MPS raised: {e}")
        sys.exit(1)

    # Quick non-determinism preview
    print()
    print("Quick non-determinism preview (large reduction sum, 50 runs):")
    torch.manual_seed(7)
    torch.mps.manual_seed(7)
    x = torch.randn(500_000, device="mps") * \
        torch.exp(torch.randn(500_000, device="mps") * 3.0)

    results = []
    for _ in range(50):
        s = x.sum()
        torch.mps.synchronize()
        results.append(float(s.cpu()))

    distinct = len(set(results))
    if distinct == 1:
        print(f"  All 50 sums identical. MPS is deterministic for this op.")
        print(f"  The main script will test more aggressive workloads.")
    else:
        spread = max(results) - min(results)
        print(f"  Distinct values across 50 runs: {distinct}/50")
        print(f"  Spread: {spread:.3e}")
        print(f"  -> The main test will quantify this for the paper.")

    print()
    print("All good. Run: python mps_determinism_test.py")


if __name__ == "__main__":
    main()
