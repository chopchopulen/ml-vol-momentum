# scripts/compare_all_models.py
"""
Full 5-model comparison: RollingVol, GARCH, HAR-RV, GBM, LSTM-ensemble.
Produces results/forecasts/{model}.parquet, results/strategies/{model}_scaled.parquet,
and prints the master results table.
Run with: python scripts/compare_all_models.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.loaders import load_ohlcv, load_vix
from src.data.universe import get_universe, get_sector
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.tests import cross_sectional_ic
from src.eval.comparison import build_results_table, build_dm_matrix
from src.models.baselines import RollingVolModel, GARCH11Model, HARRV
from src.models.gbm import GBMForecaster
from src.models.lstm_model import LSTMEnsemble

Path("results/forecasts").mkdir(parents=True, exist_ok=True)
Path("results/strategies").mkdir(parents=True, exist_ok=True)

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print(f"Loading data for {len(tickers)} tickers (2000-2024)...")
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
rv_panel = returns_panel.join(panel, how="inner")  # for RollingVol (needs "return")
sector_map = {t: get_sector(t, pd.Timestamp("2003-01-01")) for t in tickers}
panel["sector"] = panel.index.get_level_values("ticker").map(sector_map)

windows = generate_windows(start, end, first_test_year=2003)
prices_panel = ohlcv[["close"]]
signal = momentum_signal(prices_panel, lookback=252, skip=21)
realized_rv  = panel["target_rv"]
realized_log = panel["target_log_rv"]

models = {
    "rolling_vol": (RollingVolModel(), rv_panel),
    "har_rv":      (HARRV(),          panel),
    "garch":       (GARCH11Model(),   rv_panel),
    "gbm":         (GBMForecaster(),  panel),
    "lstm":        (LSTMEnsemble(),   panel),
}

all_forecasts = {}
all_strategies = {}

for model_name, (model, mpanel) in models.items():
    print(f"\nRunning {model_name} walk-forward...")
    oos = run_walk_forward(model, mpanel, windows)
    if oos.empty:
        print(f"  {model_name}: no OOS predictions — skipping")
        continue
    oos.to_parquet(f"results/forecasts/{model_name}.parquet")
    all_forecasts[model_name] = oos
    ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
    print(f"  OOS rows: {len(oos)}  Mean IC: {ic.mean():.4f}")

    # Build vol-scaled portfolio
    w_scaled_raw = vol_scale(signal, oos, target_vol=0.10)
    w_scaled = build_portfolios(None, weights=w_scaled_raw, mode="vol_targeted_gross")
    net_scaled = apply_costs(w_scaled, returns_panel, cost_bps=10.0).dropna()
    all_strategies[f"{model_name}_scaled"] = net_scaled
    net_scaled.to_parquet(f"results/strategies/{model_name}_scaled.parquet")

# Unscaled baseline
w_unscaled = build_portfolios(signal, mode="long_short_quintile")
net_unscaled = apply_costs(w_unscaled, returns_panel, cost_bps=10.0).dropna()
all_strategies["unscaled_momentum"] = net_unscaled
net_unscaled.to_parquet("results/strategies/unscaled_momentum.parquet")

# Master results table
print("\n\n=== MASTER RESULTS TABLE ===")
results_tbl = build_results_table(all_strategies)
print(results_tbl.to_string())
results_tbl.to_parquet("results/master_results_table.parquet")
print(results_tbl.to_csv())

# Per-model IC summary
print("\n=== IC SUMMARY ===")
for name, oos in all_forecasts.items():
    ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
    print(f"  {name}: mean_IC={ic.mean():.4f}  std_IC={ic.std():.4f}")

# DM matrix
print("\n=== DM MATRIX (QLIKE, p-values) ===")
log_rv_forecasts = {n: f["forecast_log_rv"] for n, f in all_forecasts.items()}
try:
    dm_stats, dm_pvals = build_dm_matrix(log_rv_forecasts, realized_log)
    print(dm_pvals.round(3).to_string())
    dm_pvals.to_parquet("results/dm_pvalues.parquet")
except Exception as e:
    print(f"DM matrix failed: {e}")

print("\nAll results written to results/")
