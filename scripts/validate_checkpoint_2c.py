"""
CHECKPOINT 2c: DM and MZ on HAR-RV vs RollingVol forecasts.
Run with: python scripts/validate_checkpoint_2c.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.data.loaders import load_ohlcv, load_vix
from src.data.universe import get_universe
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.models.baselines import HARRV, RollingVolModel
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.tests import diebold_mariano, mincer_zarnowitz
from src.eval.comparison import build_results_table, build_dm_matrix, build_mz_table

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2005-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:40]
print(f"Loading data for {len(tickers)} tickers...")
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
panel    = returns_panel.join(features.join(targets, how="inner"), how="inner").dropna(subset=["target_log_rv"])
windows  = generate_windows(start, end, first_test_year=2003)

print("Running walk-forward for HAR-RV and RollingVol...")
har_oos  = run_walk_forward(HARRV(), panel, windows)
roll_oos = run_walk_forward(RollingVolModel(), panel, windows)

realized_log = panel["target_log_rv"]

# Align forecasts and realized
common_idx = har_oos.index.intersection(roll_oos.index)
har_f  = har_oos.loc[common_idx, "forecast_log_rv"]
roll_f = roll_oos.loc[common_idx, "forecast_log_rv"]
real   = realized_log.reindex(common_idx).dropna()
har_f  = har_f.reindex(real.index)
roll_f = roll_f.reindex(real.index)

# DM test (MSE on log_rv)
e_har  = real - har_f
e_roll = real - roll_f
dm_stat, dm_p = diebold_mariano(e_har, e_roll, loss="mse")
print(f"\nDM test (HAR-RV vs RollingVol, MSE on log_rv):")
print(f"  DM stat = {dm_stat:.3f},  p-value = {dm_p:.3f}")
print(f"  (Negative stat means HAR-RV has lower MSE)")

# MZ regression for HAR-RV
mz = mincer_zarnowitz(real, har_f)
print(f"\nMincer-Zarnowitz (HAR-RV on log_rv):")
for k, v in mz.items():
    print(f"  {k}: {v:.4f}")
print(f"  (Ideal: alpha=0, beta=1, p_joint > 0.05)")

# DM matrix via comparison module (uses forecast_rv, QLIKE loss)
har_rv_s  = har_oos["forecast_rv"]
roll_rv_s = roll_oos["forecast_rv"]
realized_rv = panel["target_rv"].reindex(har_rv_s.index.union(roll_rv_s.index))

dm_stats, dm_pvals = build_dm_matrix(
    {"HAR-RV": har_rv_s, "RollingVol": roll_rv_s},
    realized_rv,
)
print(f"\nDM matrix (p-values, QLIKE):\n{dm_pvals.to_string()}")

# Sanity checks
assert 0.0 <= float(dm_p) <= 1.0, f"DM p-value {dm_p} out of [0,1]"
print("\nCHECKPOINT 2c: PASS (DM and MZ ran cleanly, p-values in [0,1])")
