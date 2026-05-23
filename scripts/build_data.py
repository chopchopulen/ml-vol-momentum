"""Build data: universe, OHLCV cache, features, targets."""
import warnings; warnings.filterwarnings("ignore")
import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import pandas as pd
from pathlib import Path
from src.data.universe import get_universe, build_membership_table
from src.data.loaders import load_ohlcv, load_vix
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
import numpy as np

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("data/cache").mkdir(parents=True, exist_ok=True)

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")

print("Building S&P 500 membership table...")
build_membership_table(Path("data/processed/sp500_membership.parquet"))

print("Loading universe for 2002-01-01...")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print(f"  {len(tickers)} tickers")

print("Downloading OHLCV...")
ohlcv = load_ohlcv(tickers, start, end)
print(f"  {len(ohlcv)} rows")

print("Loading VIX...")
vix = load_vix(start, end)

print("Building feature panel...")
features = build_feature_panel(ohlcv, vix)
print(f"  {len(features)} rows, {len(features.columns)} features")

print("Building targets...")
returns_frames = []
for ticker, grp in ohlcv.groupby(level="ticker"):
    close = grp.droplevel("ticker")["close"]
    r = np.log(close / close.shift(1))
    r.name = "return"
    idx = pd.MultiIndex.from_arrays([r.index, [ticker]*len(r)], names=["date","ticker"])
    returns_frames.append(r.set_axis(idx))
returns_panel = pd.concat(returns_frames).to_frame()
targets = forward_rv(returns_panel)
panel = features.join(targets, how="inner").dropna(subset=["target_log_rv"])
panel.to_parquet("data/processed/panel.parquet")
print(f"  Panel saved: {len(panel)} rows")
print("Data build complete.")
