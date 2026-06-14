# =============================================================
# python_interop.mojo
#
# Listing from Section 6.1 of:
#   "Mojo: A Promising Tool for Scalable Financial AI Efficiency"
#   Henry Han, Baylor University (CACM Practice submission, 2026)
#
# Demonstrates Mojo's native Python interop. A quant researcher can
# call yfinance, pandas, scikit-learn, statsmodels, and the rest of
# the scientific Python ecosystem from Mojo code, while keeping hot
# inner loops in native Mojo. This is the "no need to rewrite the
# whole ecosystem before shipping anything" property that makes
# incremental adoption feasible at a real trading firm.
#
# Requires yfinance and pandas in the local Python environment.
#
# Author: Henry Han, Baylor University
# License: MIT
# =============================================================

from python import Python


def fetch_and_analyze(ticker: String) raises:
    """Fetch two years of daily price history for `ticker` and
    compute a simple realized-volatility estimate in Mojo."""
    var yf = Python.import_module("yfinance")
    var ticker_obj = yf.Ticker(ticker)
    var history = ticker_obj.history(period="2y")
    var closes = history["Close"].to_numpy()

    # Convert numpy array back to a Mojo list for native processing
    var n = Int(closes.shape[0])
    print("Fetched", n, "daily closes for", ticker)

    # Compute log returns and realized volatility in native Mojo
    var sum_sq: Float64 = 0.0
    var prev = Float64(closes[0].to_float64())
    var count = 0
    for i in range(1, n):
        var cur = Float64(closes[i].to_float64())
        if prev > 0.0 and cur > 0.0:
            var lr = _log(cur / prev)
            sum_sq += lr * lr
            count += 1
        prev = cur

    if count == 0:
        print("No valid return observations")
        return

    # Annualized realized volatility (252 trading days)
    var rv_daily_sq = sum_sq / Float64(count)
    var rv_annual = _sqrt(rv_daily_sq * 252.0)
    print("Annualized realized vol:", rv_annual * 100.0, "%")


def _log(x: Float64) -> Float64:
    # Natural log via repeated halving; for production use math.log
    var t = x
    var k: Float64 = 0.0
    while t > 2.0:
        t = t / 2.0
        k += 0.6931471805599453   # ln(2)
    while t < 1.0:
        t = t * 2.0
        k -= 0.6931471805599453
    # Taylor expansion of ln(1+y) where y = t-1, y in [-1, 1)
    var y = t - 1.0
    var y2 = y * y
    return k + y - y2 / 2.0 + y2 * y / 3.0 - y2 * y2 / 4.0


def _sqrt(x: Float64) -> Float64:
    # Newton iteration; for production use math.sqrt
    if x <= 0.0:
        return 0.0
    var s = x
    for _ in range(30):
        s = 0.5 * (s + x / s)
    return s


def main() raises:
    fetch_and_analyze("AAPL")
    print()
    fetch_and_analyze("JPM")
