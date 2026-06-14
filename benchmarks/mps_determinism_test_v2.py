"""
mps_determinism_test_v2.py  (corrected - actually exposes non-determinism)

Measure PyTorch MPS (Metal Performance Shaders) numerical
non-determinism on Apple Silicon.

NOTE: This replaces the earlier mps_determinism_test.py. The old
version fixed the seed and ran the same call repeatedly, which could
report "all identical" even when floating-point order-sensitivity is
present. Delete the old file and use this one. Output goes to
mps_determinism_results_v2.json.

WHY v1 WAS WRONG:
    v1 fixed the seed and ran the SAME operation 100 times. If the
    GPU's reduction kernel happens to be internally deterministic for
    that exact call, you get 100 identical results and conclude
    (wrongly) that there is no problem. Fixing the seed only fixes the
    INPUT; it does nothing to vary the REDUCTION ORDER, which is the
    actual source of floating-point non-determinism.

WHAT v2 DOES INSTEAD:
    Non-associativity of floating point means (a+b)+c != a+(b+c). The
    question is not "does the same call give the same answer" but
    "does the SAME MATHEMATICAL RESULT change when the reduction is
    carried out in a different ORDER." We probe this four ways, each
    of which holds the true mathematical value fixed while varying the
    execution path:

    TEST A - permuted reduction order:
        Sum the same vector, but with the elements permuted before the
        reduction. Mathematically the sum is identical; in floating
        point the result changes if the hardware accumulates in input
        order. We measure how many distinct sums arise from 200 random
        permutations, on CPU and on MPS.

    TEST B - chunked vs whole reduction:
        Sum the whole vector at once vs. summing in K contiguous
        chunks then adding the chunk totals, for many values of K.
        Different K = different reduction tree = different rounding.

    TEST C - matmul tiling sensitivity:
        Compute the same matmul with different internal shapes
        (splitting the contraction dimension and summing partials).
        This mimics how different GPU kernels tile the same matmul.

    TEST D - run-to-run on MPS (the v1 test, kept as a control):
        Same call, same input, 100 times. If MPS is internally
        non-deterministic this alone shows it; if not, tests A-C still
        demonstrate the order-sensitivity that matters for the paper.

Author: Henry Han, Baylor University
Run:    python mps_determinism_test.py
Requires: pip install torch numpy
"""
import json
import platform
import sys

import torch


def have_mps():
    return (hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built())


def to_dev(t, device):
    return t.to(device)


def distinct_count(values):
    """Count distinct float64 bit patterns in a list of Python floats."""
    return len(set(v.hex() if isinstance(v, float) else float(v)
                   for v in values))


# -------------------------------------------------------------------
# TEST A: permuted reduction order
# -------------------------------------------------------------------
def test_permuted_sum(device, n=1_000_000, n_perms=200, seed=0):
    """Sum the SAME data under many permutations. Same math, different
    accumulation order. Counts distinct floating-point results."""
    g = torch.Generator().manual_seed(seed)
    # Ill-conditioned: wide magnitude range, mixed signs.
    base = torch.randn(n, generator=g, dtype=torch.float32)
    scale = torch.exp(torch.randn(n, generator=g, dtype=torch.float32) * 5.0)
    x_cpu = (base * scale)

    results = []
    for i in range(n_perms):
        pg = torch.Generator().manual_seed(1000 + i)
        perm = torch.randperm(n, generator=pg)
        xp = x_cpu[perm].to(device)
        s = xp.sum()
        if device == "mps":
            torch.mps.synchronize()
        results.append(float(s.cpu().item()))

    d = distinct_count(results)
    spread = max(results) - min(results)
    return {
        "n_elements": n,
        "n_permutations": n_perms,
        "distinct_values": d,
        "spread": spread,
        "min": min(results),
        "max": max(results),
        "sample": sorted(set(results))[:6],
    }


# -------------------------------------------------------------------
# TEST B: chunked vs whole reduction
# -------------------------------------------------------------------
def test_chunked_sum(device, n=1_000_000, seed=1):
    """Sum the same vector using different chunk counts. Each chunk
    count produces a different reduction tree, hence different
    rounding. Same mathematical sum."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, generator=g, dtype=torch.float32)
    scale = torch.exp(torch.randn(n, generator=g, dtype=torch.float32) * 5.0)
    x = (base * scale).to(device)

    chunk_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    results = []
    for k in chunk_counts:
        # Split into k contiguous chunks, sum each, then sum the totals
        chunks = torch.chunk(x, k)
        partials = torch.stack([c.sum() for c in chunks])
        total = partials.sum()
        if device == "mps":
            torch.mps.synchronize()
        results.append(float(total.cpu().item()))

    d = distinct_count(results)
    spread = max(results) - min(results)
    return {
        "n_elements": n,
        "chunk_counts": chunk_counts,
        "results": results,
        "distinct_values": d,
        "spread": spread,
    }


# -------------------------------------------------------------------
# TEST C: matmul contraction-split sensitivity
# -------------------------------------------------------------------
def test_matmul_split(device, dim=1024, seed=2):
    """Compute A @ B directly, vs splitting the contraction dimension
    into s pieces and summing partial products. Same mathematical
    result; different reduction order over the contraction axis. This
    mirrors how different GPU kernels tile the same matmul."""
    g = torch.Generator().manual_seed(seed)
    A_cpu = torch.randn(dim, dim, generator=g, dtype=torch.float32)
    B_cpu = torch.randn(dim, dim, generator=g, dtype=torch.float32)
    A = A_cpu.to(device)
    B = B_cpu.to(device)

    # Reference: single fused matmul
    C_ref = A @ B
    if device == "mps":
        torch.mps.synchronize()
    ref_val = float(C_ref.sum().cpu().item())

    splits = [1, 2, 4, 8, 16, 32]
    results = []
    for s in splits:
        A_parts = torch.chunk(A, s, dim=1)
        B_parts = torch.chunk(B, s, dim=0)
        C = torch.zeros(dim, dim, device=device)
        for ap, bp in zip(A_parts, B_parts):
            C = C + ap @ bp
        if device == "mps":
            torch.mps.synchronize()
        results.append(float(C.sum().cpu().item()))

    all_vals = [ref_val] + results
    d = distinct_count(all_vals)
    spread = max(all_vals) - min(all_vals)
    return {
        "dim": dim,
        "reference_sum": ref_val,
        "splits": splits,
        "split_results": results,
        "distinct_values": d,
        "spread": spread,
    }


# -------------------------------------------------------------------
# TEST D: run-to-run control (v1 style, kept for completeness)
# -------------------------------------------------------------------
def test_run_to_run(device, n=2_000_000, n_runs=100, seed=3):
    """Same call, same input, repeated. Tests whether the device is
    internally non-deterministic for a fixed call. This is the weakest
    probe; tests A-C are the substantive ones."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, generator=g, dtype=torch.float32)
    scale = torch.exp(torch.randn(n, generator=g, dtype=torch.float32) * 5.0)
    x = (base * scale).to(device)

    results = []
    for _ in range(n_runs):
        s = x.sum()
        if device == "mps":
            torch.mps.synchronize()
        results.append(float(s.cpu().item()))

    return {
        "n_elements": n,
        "n_runs": n_runs,
        "distinct_values": distinct_count(results),
        "spread": max(results) - min(results),
    }


# -------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------
def run_all(device):
    print(f"\n{'='*64}")
    print(f"  DEVICE: {device.upper()}")
    print(f"{'='*64}")

    out = {}

    print("\n[A] Permuted reduction order (same data, 200 permutations)")
    a = test_permuted_sum(device)
    out["permuted_sum"] = a
    print(f"    distinct sums: {a['distinct_values']}/{a['n_permutations']}"
          f"   spread: {a['spread']:.3e}")

    print("\n[B] Chunked vs whole reduction (varying chunk count)")
    b = test_chunked_sum(device)
    out["chunked_sum"] = b
    print(f"    distinct sums: {b['distinct_values']}/{len(b['chunk_counts'])}"
          f"   spread: {b['spread']:.3e}")

    print("\n[C] Matmul contraction-split (1024x1024, varying split)")
    c = test_matmul_split(device)
    out["matmul_split"] = c
    print(f"    distinct sums: {c['distinct_values']}/{len(c['splits'])+1}"
          f"   spread: {c['spread']:.3e}")

    print("\n[D] Run-to-run control (same call x100)")
    d = test_run_to_run(device)
    out["run_to_run"] = d
    print(f"    distinct sums: {d['distinct_values']}/{d['n_runs']}"
          f"   spread: {d['spread']:.3e}")

    return out


def main():
    print("PyTorch MPS Determinism Test (v2)")
    print(f"PyTorch {torch.__version__}, "
          f"macOS {platform.mac_ver()[0]}, {platform.machine()}")

    results = {
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
    }

    # CPU is the deterministic reference
    results["cpu"] = run_all("cpu")

    if have_mps():
        results["mps"] = run_all("mps")
    else:
        print("\n[!] MPS not available; ran CPU only.")
        print("    pip install --upgrade torch  (needs macOS 12.3+)")
        results["mps"] = None

    # Summary
    print(f"\n{'='*64}")
    print("  SUMMARY: distinct floating-point results from identical math")
    print(f"{'='*64}")
    print(f"\n  {'Probe':<32}{'CPU':>14}{'MPS':>14}")
    print(f"  {'-'*32}{'-'*14}{'-'*14}")
    labels = {
        "permuted_sum": "A: permuted order (/200)",
        "chunked_sum": "B: chunk count (/11)",
        "matmul_split": "C: matmul split (/7)",
        "run_to_run": "D: run-to-run (/100)",
    }
    for key, label in labels.items():
        cpu_d = results["cpu"][key]["distinct_values"]
        mps_d = (results["mps"][key]["distinct_values"]
                 if results["mps"] else "n/a")
        print(f"  {label:<32}{cpu_d:>14}{str(mps_d):>14}")

    print(f"\n  Interpretation:")
    print(f"    Tests A-C vary the REDUCTION ORDER while holding the")
    print(f"    mathematical value fixed. distinct > 1 means the device")
    print(f"    is order-sensitive: the same risk number computed two")
    print(f"    ways gives different bits. This is the property that")
    print(f"    breaks bit-exact audit reproducibility.")
    print(f"    CPU float32 is also order-sensitive (that's just IEEE")
    print(f"    754); the paper's point is that on GPU you cannot")
    print(f"    CONTROL the order, whereas Mojo lets you fix it.")

    with open("mps_determinism_results_v2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved mps_determinism_results_v2.json - send this back.")


if __name__ == "__main__":
    main()
