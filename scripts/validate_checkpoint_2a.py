"""
CHECKPOINT 2a: Unscaled momentum positive Sharpe pre-2008.
Run with: python scripts/validate_checkpoint_2a.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.data.loaders import load_ohlcv
from src.data.universe import get_universe
from src.strategy.momentum import momentum_signal
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.eval.metrics import sharpe, max_drawdown

start = pd.Timestamp("1998-01-01")
end   = pd.Timestamp("2007-12-31")

tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print(f"Loading OHLCV for {len(tickers)} tickers...")
ohlcv = load_ohlcv(tickers, start, end)

returns_frames = []
for ticker, grp in ohlcv.groupby(level="ticker"):
    close = grp.droplevel("ticker")["close"]
    r = np.log(close / close.shift(1))
    r.name = "return"
    idx = pd.MultiIndex.from_arrays([r.index, [ticker]*len(r)], names=["date","ticker"])
    returns_frames.append(r.set_axis(idx))
returns_panel = pd.concat(returns_frames).to_frame()

prices_panel = ohlcv[["close"]].copy()

print("Computing momentum signal...")
signal = momentum_signal(prices_panel, lookback=252, skip=21)

print("Building long-short quintile portfolio...")
weights = build_portfolios(signal, mode="long_short_quintile")

print("Applying transaction costs...")
net = apply_costs(weights, returns_panel, cost_bps=10.0)
net = net.dropna()
net_2000_2007 = net.loc["2000":"2007"]

sr = sharpe(net_2000_2007)
equity = (1 + net_2000_2007).cumprod()
mdd = max_drawdown(equity)

print(f"\nCHECKPOINT 2a results (2000-2007):")
print(f"  Annualized Sharpe: {sr:.3f}  (gate: > 0)")
print(f"  Max Drawdown:      {mdd:.3f}")
print(f"  Ann Return:        {net_2000_2007.mean() * 252:.3f}")
print(f"\nResult: {'PASS' if sr > 0 else 'FAIL'}")
