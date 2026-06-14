#!/usr/bin/env bash
# =============================================================
# RUN_EVERYTHING.sh
#
# One-script driver for all measurements needed by the CACM paper:
#   1. Check Python + PyTorch + MPS install
#   2. Run MPS GPU non-determinism experiment (3-5 min)
#   3. Check / install Mojo
#   4. Run all three Mojo benchmarks (dot product, Monte Carlo, EWMA)
#   5. Bundle all results into ONE zip for you to send back
#
# Usage:
#   chmod +x RUN_EVERYTHING.sh
#   ./RUN_EVERYTHING.sh
#
# Author: Henry Han, Baylor University
# License: MIT
#
# If a step fails, the script prints what to do and exits.
# You can re-run safely; it will skip steps that already finished.
# =============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for legibility
if [ -t 1 ]; then
    BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'
    YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
    BOLD=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

say() { echo "${BLUE}${BOLD}$*${RESET}"; }
ok()  { echo "${GREEN}[OK]${RESET} $*"; }
warn(){ echo "${YELLOW}[WARN]${RESET} $*"; }
fail(){ echo "${RED}[FAIL]${RESET} $*" >&2; }

# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
clear
echo "${BOLD}================================================================${RESET}"
echo "${BOLD}  CACM Mojo Paper - Mac Benchmark Suite${RESET}"
echo "${BOLD}  Henry Han, Baylor University${RESET}"
echo "${BOLD}================================================================${RESET}"
echo
echo "This script will run two experiment sets on your Mac:"
echo "  1. PyTorch MPS GPU non-determinism (3-5 min, no install needed)"
echo "  2. Mojo CPU performance benchmarks (10 min, installs Mojo if missing)"
echo
echo "Total time: ~15-20 minutes"
echo "Output: one zip file you send back."
echo
read -rp "Press ENTER to start, or Ctrl-C to abort... "
echo

# -------------------------------------------------------------
# Detect Python
# -------------------------------------------------------------
say ">>> Step 1/6: Detecting Python"
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    fail "Python not found. Install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi
PY_VER=$($PY --version 2>&1)
ok "Using $PY ($PY_VER)"
echo

# -------------------------------------------------------------
# Check / install PyTorch
# -------------------------------------------------------------
say ">>> Step 2/6: Checking PyTorch with MPS"
if $PY -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    PT_VER=$($PY -c "import torch; print(torch.__version__)")
    ok "PyTorch $PT_VER with MPS backend"
else
    warn "PyTorch with MPS not installed. Installing now..."
    $PY -m pip install --upgrade pip
    $PY -m pip install --upgrade torch numpy
    if $PY -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        PT_VER=$($PY -c "import torch; print(torch.__version__)")
        ok "PyTorch $PT_VER installed with MPS backend"
    else
        fail "PyTorch install failed or MPS still unavailable."
        echo "       Requirements: macOS 12.3+, Apple Silicon or modern Intel Mac."
        echo "       Manual fix:   pip3 install --upgrade torch"
        exit 1
    fi
fi
echo

# -------------------------------------------------------------
# Run MPS quick check
# -------------------------------------------------------------
say ">>> Step 3/6: MPS quick sanity check"
$PY mps_quick_check.py
echo

# -------------------------------------------------------------
# Run full MPS determinism experiment
# -------------------------------------------------------------
say ">>> Step 4/6: MPS GPU non-determinism experiment (3-5 min)"
echo "Running 4 tests, each on CPU (reference) and MPS GPU (test)..."
echo
if [ -f mps_determinism_results.json ]; then
    warn "mps_determinism_results.json already exists, skipping. Delete the file to re-run."
else
    $PY mps_determinism_test.py
fi
ok "MPS experiment complete. Output: mps_determinism_results.json"
echo

# -------------------------------------------------------------
# Check / install Mojo
# -------------------------------------------------------------
say ">>> Step 5/6: Checking Mojo install"
if command -v mojo >/dev/null 2>&1; then
    MOJO="mojo"
    ok "mojo command on PATH"
elif command -v pixi >/dev/null 2>&1 && [ -f pixi.toml ]; then
    MOJO="pixi run mojo"
    if pixi run mojo --version >/dev/null 2>&1; then
        ok "pixi + mojo configured in this directory"
    else
        warn "pixi found but mojo not in env. Installing..."
        pixi add modular
    fi
else
    warn "Mojo not installed. Installing via pixi..."
    if ! command -v pixi >/dev/null 2>&1; then
        echo "Installing pixi (Modular's package manager)..."
        curl -fsSL https://pixi.sh/install.sh | bash
        # pixi installer modifies ~/.bashrc / ~/.zshrc but not current shell
        export PATH="$HOME/.pixi/bin:$PATH"
        if ! command -v pixi >/dev/null 2>&1; then
            fail "pixi install completed but command still not found."
            echo "       Close this terminal, open a new one, cd back here, and re-run."
            exit 1
        fi
        ok "pixi installed"
    fi
    if [ ! -f pixi.toml ]; then
        pixi init . 2>&1 | tail -3
    fi
    say "Installing Mojo via pixi (this can take 3-5 minutes)..."
    pixi add modular 2>&1 | tail -5
    MOJO="pixi run mojo"
fi

if $MOJO --version >/dev/null 2>&1; then
    MOJO_VER=$($MOJO --version 2>&1 | head -1)
    ok "Mojo available: $MOJO_VER"
else
    fail "Mojo install incomplete."
    echo "       Try: curl -ssL https://magic.modular.com/install | bash"
    echo "       Then re-run this script."
    exit 1
fi
echo

# -------------------------------------------------------------
# Run Mojo benchmarks
# -------------------------------------------------------------
say ">>> Step 6/6: Mojo CPU benchmarks (10 min)"
echo

parse_keys_to_json() {
    local prefix="$1"
    local out_file="$2"
    local in_file="$3"
    awk -v prefix="$prefix" '
        $1 == prefix {
            $1 = "";
            key = $2;
            $2 = "";
            sub(/^[ \t]+/, "");
            value = $0;
            sub(/^[ \t]+/, "", value);
            printf "  \"%s\": %s,\n", key, value;
        }
    ' "$in_file" > "$out_file.tmp"
    {
        echo "{"
        sed '$ s/,$//' "$out_file.tmp"
        echo "}"
    } > "$out_file"
    rm -f "$out_file.tmp"
}

# Benchmark 1: dot product
say "  [1/3] Dot product benchmark"
$PY bench_dot_product.py 2>&1 | tee bench_dot_product_py.log
$MOJO run bench_dot_product.mojo 2>&1 | tee bench_dot_product_mojo.log
parse_keys_to_json "PY_KEY"   results_dot_product_py.json   bench_dot_product_py.log
parse_keys_to_json "MOJO_KEY" results_dot_product_mojo.json bench_dot_product_mojo.log
$PY collect_results.py dot_product \
    results_dot_product_py.json results_dot_product_mojo.json \
    > results_dot_product.json
ok "Wrote results_dot_product.json"
echo

# Benchmark 2: Monte Carlo
say "  [2/3] Monte Carlo option pricing"
$PY bench_monte_carlo.py 2>&1 | tee bench_monte_carlo_py.log
$MOJO run bench_monte_carlo.mojo 2>&1 | tee bench_monte_carlo_mojo.log
parse_keys_to_json "PY_KEY"   results_monte_carlo_py.json   bench_monte_carlo_py.log
parse_keys_to_json "MOJO_KEY" results_monte_carlo_mojo.json bench_monte_carlo_mojo.log
$PY collect_results.py monte_carlo \
    results_monte_carlo_py.json results_monte_carlo_mojo.json \
    > results_monte_carlo.json
ok "Wrote results_monte_carlo.json"
echo

# Benchmark 3: EWMA
say "  [3/3] EWMA over tick stream"
$PY bench_ewma.py 2>&1 | tee bench_ewma_py.log
$MOJO run bench_ewma.mojo 2>&1 | tee bench_ewma_mojo.log
parse_keys_to_json "PY_KEY"   results_ewma_py.json   bench_ewma_py.log
parse_keys_to_json "MOJO_KEY" results_ewma_mojo.json bench_ewma_mojo.log
$PY collect_results.py ewma \
    results_ewma_py.json results_ewma_mojo.json \
    > results_ewma.json
ok "Wrote results_ewma.json"
echo

# -------------------------------------------------------------
# Summary
# -------------------------------------------------------------
say "================================================================"
say "  ALL EXPERIMENTS COMPLETE"
say "================================================================"
echo
$PY collect_results.py summary \
    results_dot_product.json results_monte_carlo.json results_ewma.json

# -------------------------------------------------------------
# Bundle results for transmission
# -------------------------------------------------------------
echo
say ">>> Bundling results into one zip for you to send back"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MACHINE=$(uname -m)
CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null | tr ' ' '_' | tr -d ',' || echo "unknown")
RESULTS_ZIP="cacm_paper_results_${MACHINE}_${TIMESTAMP}.zip"

# system_info.txt: what hardware we ran on
cat > system_info.txt <<EOF
Run timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
System:        $(uname -a)
CPU:           $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)
Memory:        $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.1f GB", $1/1024/1024/1024}' || echo unknown)
macOS:         $(sw_vers -productVersion 2>/dev/null || echo unknown)
Python:        $($PY --version 2>&1)
PyTorch:       $($PY -c "import torch; print(torch.__version__)" 2>/dev/null || echo unavailable)
Mojo:          $($MOJO --version 2>&1 | head -1)
EOF

zip -q "$RESULTS_ZIP" \
    system_info.txt \
    mps_determinism_results.json \
    results_dot_product.json \
    results_monte_carlo.json \
    results_ewma.json \
    bench_dot_product_py.log \
    bench_dot_product_mojo.log \
    bench_monte_carlo_py.log \
    bench_monte_carlo_mojo.log \
    bench_ewma_py.log \
    bench_ewma_mojo.log \
    2>/dev/null

ok "Bundled into: ${BOLD}$RESULTS_ZIP${RESET}"
echo
echo "${BOLD}====================================================================${RESET}"
echo "${BOLD} HOW TO SEND THE RESULTS BACK${RESET}"
echo "${BOLD}====================================================================${RESET}"
echo
echo "  Send this one file:"
echo
echo "    ${GREEN}$SCRIPT_DIR/$RESULTS_ZIP${RESET}"
echo
echo "  Drag and drop it into the Claude chat, or upload as an attachment."
echo
echo "  After receiving it, I will:"
echo "    - Update Figure 1 and Figure 3 with your measured Mojo speedups"
echo "    - Update Figure 2 with your measured MPS GPU non-determinism"
echo "    - Update §3.2 and §8 prose to reflect 'measured on Apple Silicon'"
echo "    - Add a new paragraph in §4 about MPS GPU specifically"
echo
echo "${BOLD}====================================================================${RESET}"
