# scripts/validate_checkpoint_3b.py
"""
CHECKPOINT 3b: LSTM seed stability.
Gate: seed-to-seed IC std < 0.10.
Run with: python scripts/validate_checkpoint_3b.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.data.loaders import load_ohlcv, load_vix
from src.data.universe import get_universe
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.tests import cross_sectional_ic
from src.models.lstm_model import LSTMForecaster
from src.config import load_config

cfg = load_config()
start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2010-12-31")   # shorter window for speed
tickers = get_universe(pd.Timestamp("2002-01-01"))[:40]

print(f"Loading data for {len(tickers)} tickers (2000-2010)...")
ohlcv = load_ohlcv(tickers, start, end)
vix   = load_vix(start, end)

returns_frames = []
for ticker, grp in ohlcv.groupby(level="ticker"):
    close = grp.droplevel("ticker")["close"]
    r = np.log(close / close.shift(1))
    r.name = "return"
    idx = pd.MultiIndex.from_arrays([r.index, [ticker]*len(r)], names=["date","ticker"])
    returns_frames.append(r.set_axis(idx))
returns_panel = pd.concat(returns_frames).to_frame()

features = build_feature_panel(ohlcv, vix)
targets  = forward_rv(returns_panel)
panel    = features.join(targets, how="inner").dropna(subset=["target_log_rv"])

windows = generate_windows(start, end, first_test_year=2003)
realized_rv = panel["target_rv"]

seeds = cfg["models"]["lstm"]["seeds"]  # [0, 1, 2, 3, 4]
ic_per_seed = {}

for seed in seeds:
    print(f"Running LSTM seed={seed}...")

    class SingleSeedWrapper:
        name = f"lstm_seed{seed}"
        _seed = seed
        def __init__(self): self._m = LSTMForecaster()
        def fit(self, train): self._m.fit(train, seed=self.__class__._seed)
        def predict(self, history): return self._m.predict(history)

    oos = run_walk_forward(SingleSeedWrapper(), panel, windows)
    if len(oos) == 0:
        print(f"  Seed {seed}: no OOS predictions")
        continue
    ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
    ic_per_seed[seed] = ic.mean()
    print(f"  Seed {seed}: mean IC = {ic.mean():.4f}")

if ic_per_seed:
    ic_vals = list(ic_per_seed.values())
    ic_mean = np.mean(ic_vals)
    ic_std  = np.std(ic_vals)
    print(f"\nCHECKPOINT 3b results:")
    print(f"  Mean IC across seeds: {ic_mean:.4f}")
    print(f"  Std  IC across seeds: {ic_std:.4f}  (gate: < 0.10)")
    gate = ic_std < 0.10
    print(f"\nCHECKPOINT 3b: {'PASS' if gate else 'FAIL'}")
    if not gate:
        print("  WARNING: High seed-to-seed variance. Check training stability.")
        print("  Consider increasing max_epochs or reducing learning_rate in default.yaml.")
else:
    print("CHECKPOINT 3b: FAIL — no OOS predictions produced")
