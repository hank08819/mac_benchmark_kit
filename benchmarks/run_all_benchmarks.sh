#!/usr/bin/env bash
# =============================================================
# run_all_benchmarks.sh
#
# Runs all three benchmarks on a Mac, parses the MOJO_KEY/PY_KEY
# lines from each, and writes JSON summaries.
#
# Usage:
#   chmod +x run_all_benchmarks.sh
#   ./run_all_benchmarks.sh
#
# Author: Henry Han, Baylor University
# License: MIT
# =============================================================
set -e

# Auto-detect Mojo invocation
if command -v mojo >/dev/null 2>&1; then
    MOJO="mojo"
elif command -v pixi >/dev/null 2>&1; then
    MOJO="pixi run mojo"
else
    echo "ERROR: neither 'mojo' nor 'pixi' is on PATH."
    echo "Install Mojo first; see README.md step 1."
    exit 1
fi
echo "Using Mojo invocation: $MOJO"

# Detect Python
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    PY="python"
fi
echo "Using Python invocation: $PY"

# Detect system info for the report
SYSTEM_INFO=$(uname -a)
CPU_INFO=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
PY_VERSION=$($PY --version 2>&1)
MOJO_VERSION=$($MOJO --version 2>&1 | head -1)

echo
echo "============================================================"
echo "  System: $SYSTEM_INFO"
echo "  CPU:    $CPU_INFO"
echo "  Python: $PY_VERSION"
echo "  Mojo:   $MOJO_VERSION"
echo "============================================================"
echo

# -------------------------------------------------------------
# Parse "KEY name value" lines into JSON
# -------------------------------------------------------------
parse_keys_to_json() {
    local prefix="$1"   # MOJO_KEY or PY_KEY
    local out_file="$2"
    local in_file="$3"

    # Extract lines starting with prefix, convert to JSON
    awk -v prefix="$prefix" '
        $1 == prefix {
            $1 = "";
            key = $2;
            $2 = "";
            sub(/^[ \t]+/, "");
            # Trim leading spaces
            value = $0;
            sub(/^[ \t]+/, "", value);
            printf "  \"%s\": %s,\n", key, value;
        }
    ' "$in_file" > "$out_file.tmp"

    # Wrap in JSON braces, strip trailing comma
    {
        echo "{"
        sed '$ s/,$//' "$out_file.tmp"
        echo "}"
    } > "$out_file"
    rm -f "$out_file.tmp"
}

# -------------------------------------------------------------
# Benchmark 1: Dot product
# -------------------------------------------------------------
echo ">>> [1/3] Dot product benchmark"
echo
echo "Running Python reference..."
$PY bench_dot_product.py 2>&1 | tee bench_dot_product_py.log

echo
echo "Running Mojo benchmark (may take a minute to compile first time)..."
$MOJO run bench_dot_product.mojo 2>&1 | tee bench_dot_product_mojo.log

cat bench_dot_product_py.log bench_dot_product_mojo.log > bench_dot_product.combined.log
parse_keys_to_json "PY_KEY"   results_dot_product_py.json   bench_dot_product_py.log
parse_keys_to_json "MOJO_KEY" results_dot_product_mojo.json bench_dot_product_mojo.log

# Merge into final
$PY collect_results.py dot_product \
    results_dot_product_py.json \
    results_dot_product_mojo.json \
    > results_dot_product.json

echo
echo "Wrote results_dot_product.json"

# -------------------------------------------------------------
# Benchmark 2: Monte Carlo
# -------------------------------------------------------------
echo
echo ">>> [2/3] Monte Carlo option pricing"
echo
echo "Running Python reference..."
$PY bench_monte_carlo.py 2>&1 | tee bench_monte_carlo_py.log
echo
echo "Running Mojo benchmark..."
$MOJO run bench_monte_carlo.mojo 2>&1 | tee bench_monte_carlo_mojo.log

parse_keys_to_json "PY_KEY"   results_monte_carlo_py.json   bench_monte_carlo_py.log
parse_keys_to_json "MOJO_KEY" results_monte_carlo_mojo.json bench_monte_carlo_mojo.log
$PY collect_results.py monte_carlo \
    results_monte_carlo_py.json \
    results_monte_carlo_mojo.json \
    > results_monte_carlo.json

echo
echo "Wrote results_monte_carlo.json"

# -------------------------------------------------------------
# Benchmark 3: EWMA
# -------------------------------------------------------------
echo
echo ">>> [3/3] EWMA over tick stream"
echo
echo "Running Python reference..."
$PY bench_ewma.py 2>&1 | tee bench_ewma_py.log
echo
echo "Running Mojo benchmark..."
$MOJO run bench_ewma.mojo 2>&1 | tee bench_ewma_mojo.log

parse_keys_to_json "PY_KEY"   results_ewma_py.json   bench_ewma_py.log
parse_keys_to_json "MOJO_KEY" results_ewma_mojo.json bench_ewma_mojo.log
$PY collect_results.py ewma \
    results_ewma_py.json \
    results_ewma_mojo.json \
    > results_ewma.json

echo
echo "Wrote results_ewma.json"

# -------------------------------------------------------------
# Final summary
# -------------------------------------------------------------
echo
echo "============================================================"
echo "  ALL BENCHMARKS COMPLETE"
echo "============================================================"
echo
$PY collect_results.py summary \
    results_dot_product.json \
    results_monte_carlo.json \
    results_ewma.json

echo
echo "Send these three JSON files back:"
echo "  - results_dot_product.json"
echo "  - results_monte_carlo.json"
echo "  - results_ewma.json"
