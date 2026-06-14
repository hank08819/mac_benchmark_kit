"""
Figure 1: Performance landscape — relative throughput across five
representative Financial-AI workloads, by language/runtime.

Numbers are calibrated against published benchmarks (Modular blog
posts, MLPerf-style microbenchmarks, NVIDIA cuBLAS specs). They are
illustrative orders-of-magnitude, not measurements from one machine.

Author: Henry Han, Baylor University
License: MIT
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Workloads (rows) x languages (cols), normalized so C++ = 1.0
# Sources:
#   - Mandelbrot/Monte Carlo: Modular published 35,000-68,000x vs Python
#   - matmul: Modular matmul blog (~90% OpenBLAS for hand-tuned Mojo)
#   - JSON parsing: typical 50-150x Python->C++ gap for parsing
#   - Sort: 80-200x typical Python->C++ for sort
#   - LLM tokens/sec: MAX engine published numbers vs PyTorch HF Transformers
workloads = [
    "Monte Carlo\n(option pricing)",
    "Matrix Multiply\n(risk covariance)",
    "JSON Parsing\n(tick data ingest)",
    "Numerical Sort\n(order book)",
    "LLM Inference\n(sentiment)",
]
# Relative throughput, normalized to C++ (= 1.0 for each row)
data = {
    "Pure Python":   [0.00003, 0.001,  0.015, 0.012, 0.05],
    "PyTorch+Python":[0.0001,  0.85,   0.02,  0.02,  0.55],  # math fast, glue slow
    "Rust":          [0.95,    0.92,   1.10,  1.05,  0.70],  # safe + fast
    "C++":           [1.00,    1.00,   1.00,  1.00,  1.00],  # baseline
    "Mojo":          [0.92,    0.95,   0.95,  0.97,  1.10],  # MAX wins for LLM
}

colors = {
    "Pure Python":   "#888888",
    "PyTorch+Python":"#c0392b",
    "Rust":          "#d68910",
    "C++":           "#1f4e8c",
    "Mojo":          "#2a9d3f",
}

fig, ax = plt.subplots(figsize=(11, 5.5))
n_lang = len(data)
n_work = len(workloads)
x = np.arange(n_work)
width = 0.16

for i, (lang, vals) in enumerate(data.items()):
    offset = (i - n_lang/2 + 0.5) * width
    bars = ax.bar(x + offset, vals, width, color=colors[lang],
                  label=lang, alpha=0.9, edgecolor="black", linewidth=0.4)

ax.set_yscale("log")
ax.set_ylabel("Relative throughput (C++ = 1.0, log scale)", fontsize=11)
ax.set_title("Figure 1: Relative Performance Across Financial-AI Workloads\n"
             "(calibrated from published benchmarks, illustrative)",
             fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(workloads, fontsize=9)
ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3, axis="y", which="both")
ax.legend(loc="lower right", fontsize=9, frameon=True, ncol=5)
ax.set_ylim(1e-5, 3)

# Annotate the Python column
ax.annotate("Pure Python:\n10,000–100,000× slower\nthan C++ on compute",
            xy=(0, 3e-5), xytext=(0.5, 3e-3),
            fontsize=8.5, color="#444444",
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))

plt.tight_layout()
plt.savefig("fig1_perf_landscape.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("Saved fig1_perf_landscape.png")
