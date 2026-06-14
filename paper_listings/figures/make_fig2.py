"""
make_fig2.py  (MEASURED ON APPLE SILICON MPS GPU)

Generates Figure 2 from real measurements taken on an Apple Silicon
GPU (Metal Performance Shaders) and CPU, via mps_determinism_test_v2.py.

Source: mps_determinism_results_v2.json

Companion to:
  "Mojo: A Promising Tool for Scalable Financial AI Efficiency"
  Henry Han, Baylor University (CACM Practice submission, 2026)

Every number here is a real IEEE-754 float32 computation on Apple
Silicon. The experiment varies the REDUCTION ORDER while holding the
mathematical value fixed, then counts how many distinct bit patterns
result. This is the floating-point non-associativity that breaks
bit-exact audit reproducibility.

Author: Henry Han, Baylor University
License: MIT
"""
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    with open("mps_determinism_results_v2.json") as f:
        data = json.load(f)

    cpu = data["cpu"]
    mps = data["mps"]
    pt_ver = data["pytorch_version"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ============================================================
    # LEFT: distinct FP results per probe, CPU vs MPS
    # ============================================================
    probes = ["permuted_sum", "chunked_sum", "matmul_split", "run_to_run"]
    probe_labels = [
        "Permuted\nreduction order\n(200 orders)",
        "Chunk count\n(1..1024)",
        "Matmul\ncontraction split\n(1..32)",
        "Run-to-run\nsame call\n(x100)",
    ]
    cpu_distinct = [cpu[p]["distinct_values"] for p in probes]
    mps_distinct = [mps[p]["distinct_values"] for p in probes]

    x = np.arange(len(probes))
    w = 0.36

    bars1 = ax1.bar(x - w/2, cpu_distinct, w, label="CPU (float32)",
                    color="#34699a", edgecolor="black", linewidth=0.4)
    bars2 = ax1.bar(x + w/2, mps_distinct, w, label="MPS GPU (float32)",
                    color="#c0392b", edgecolor="black", linewidth=0.4)

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width()/2, h + 0.05, str(int(h)),
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.axhline(1, color="green", linewidth=1.2, linestyle="--", alpha=0.7,
                label="Bit-exact target (1 value)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(probe_labels, fontsize=8.5)
    ax1.set_ylabel("Distinct floating-point results\nfrom identical mathematics",
                   fontsize=10.5)
    ax1.set_title("MEASURED on Apple Silicon: how many bit patterns\n"
                  "does the same risk number produce?",
                  fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9, frameon=True)
    ax1.set_ylim(0, max(max(cpu_distinct), max(mps_distinct)) + 1.5)
    ax1.grid(True, axis="y", alpha=0.3)

    ax1.text(0.02, 0.97,
             "When reduction order varies,\nboth CPU and MPS produce\n"
             "multiple distinct results.\nOnly 'run-to-run' (fixed order)\n"
             "is reproducible.",
             transform=ax1.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8f0",
                       edgecolor="#888", alpha=0.9))

    # ============================================================
    # RIGHT: matmul split sensitivity - the actual FP values
    # ============================================================
    cpu_ms = cpu["matmul_split"]
    mps_ms = mps["matmul_split"]
    splits = cpu_ms["splits"]

    cpu_vals = cpu_ms["split_results"]
    mps_vals = mps_ms["split_results"]
    cpu_ref = cpu_ms["reference_sum"]
    mps_ref = mps_ms["reference_sum"]

    # Plot deviation from the CPU reference (in absolute FP units)
    cpu_dev = [v - cpu_ref for v in cpu_vals]
    mps_dev = [v - cpu_ref for v in mps_vals]

    xs = np.arange(len(splits))
    ax2.plot(xs, cpu_dev, "o-", color="#34699a", linewidth=1.8,
             markersize=7, label="CPU (split varies)")
    ax2.plot(xs, mps_dev, "s-", color="#c0392b", linewidth=1.8,
             markersize=7, label="MPS GPU (split varies)")
    ax2.axhline(0, color="#34699a", linewidth=0.8, linestyle=":",
                alpha=0.6)
    ax2.axhline(mps_ref - cpu_ref, color="#c0392b", linewidth=0.8,
                linestyle=":", alpha=0.6)

    # Annotate the cross-device gap
    ax2.annotate(
        f"Same matmul, CPU vs MPS\ndiffer by {abs(mps_ref-cpu_ref):.2e}",
        xy=(0, mps_ref - cpu_ref),
        xytext=(1.6, 0.004),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0",
                  edgecolor="#c0392b", alpha=0.9),
    )

    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{s}" for s in splits])
    ax2.set_xlabel("Number of contraction-dimension splits\n"
                   "(simulates different GPU kernel tilings)",
                   fontsize=10.5)
    ax2.set_ylabel("Deviation from CPU single-pass result\n(float32 units)",
                   fontsize=10.5)
    ax2.set_title("MEASURED: same 1024x1024 matmul, different tiling\n"
                  "= different bits (CPU and MPS both order-sensitive)",
                  fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9, frameon=True)
    ax2.grid(True, alpha=0.3)

    n_mps_distinct = mps_ms["distinct_values"]
    ax2.text(0.03, 0.96,
             f"{n_mps_distinct} distinct results on MPS\n"
             f"from one matmul, just by\nchanging how it is tiled.",
             transform=ax2.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8f0",
                       edgecolor="#888", alpha=0.9))

    fig.suptitle(
        f"Figure 2: GPU Numerical Determinism, Measured on Apple Silicon "
        f"(MPS, PyTorch {pt_ver})",
        fontsize=12.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("fig2_determinism.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_determinism.png (from real MPS measurements)")
    print(f"  CPU permuted-order distinct:  {cpu['permuted_sum']['distinct_values']}/200")
    print(f"  MPS permuted-order distinct:  {mps['permuted_sum']['distinct_values']}/200")
    print(f"  MPS matmul-split distinct:    {n_mps_distinct}/7")
    print(f"  CPU vs MPS matmul gap:        {abs(mps_ref-cpu_ref):.4e}")
    print(f"  Run-to-run: CPU={cpu['run_to_run']['distinct_values']}, "
          f"MPS={mps['run_to_run']['distinct_values']}")


if __name__ == "__main__":
    main()
