from __future__ import annotations
import numpy as np
import pandas as pd

# Appropriate Sharpe threshold for white-noise canary tests.
# For T trading days the annualised Sharpe of a zero-mean series has a standard
# deviation of ~sqrt(252/T). With T=504 that is ~0.71, so the 99th-percentile
# absolute value is ~1.8. Use WHITE_NOISE_SHARPE_THRESHOLD = 2.0 to avoid
# false positives while still catching any genuine artificial alpha.
WHITE_NOISE_SHARPE_THRESHOLD = 2.0

def white_noise_panel(n_dates: int, n_stocks: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-03", periods=n_dates, freq="B")
    tickers = [f"S{i:04d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    returns = rng.normal(0, 0.01, n_dates * n_stocks)
    return pd.DataFrame({"return": returns}, index=idx)

def leakage_test_signal(returns_panel: pd.DataFrame) -> pd.DataFrame:
    # Signal at t = return at t+1 (perfect look-ahead leakage).
    # If the evaluation pipeline correctly applies the +1 trading-day shift,
    # this signal should produce Sharpe >> 15. If it doesn't, the eval is broken.
    r = returns_panel["return"].unstack("ticker")
    sig = r.shift(-1).stack()  # t+1 return, presented at t
    sig = sig.dropna()  # pandas-version-safe: explicitly drop NaN rows
    sig.name = "signal"
    return sig.to_frame()
