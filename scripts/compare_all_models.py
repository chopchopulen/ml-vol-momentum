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
    fcast_path = Path(f"results/forecasts/{model_name}.parquet")
    strat_path = Path(f"results/strategies/{model_name}_scaled.parquet")
    if fcast_path.exists() and strat_path.exists():
        print(f"\n{model_name}: loading cached results...")
        oos = pd.read_parquet(fcast_path)
        all_forecasts[model_name] = oos
        all_strategies[f"{model_name}_scaled"] = pd.read_parquet(strat_path)["net_return"]
        ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
        print(f"  OOS rows: {len(oos)}  Mean IC: {ic.mean():.4f}  (cached)")
        continue
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
    net_scaled.to_frame("net_return").to_parquet(f"results/strategies/{model_name}_scaled.parquet")

# Unscaled baseline.
# The control MUST be scored on the same dates as the scaled strategies. Built
# unrestricted it starts 2001-02-01 (6,015 days) while every scaled strategy
# starts at the first OOS date 2003-02-12 (4,897 days); the extra 2001-2002 days
# carried the entire difference and flipped a pre-registered verdict.
w_unscaled = build_portfolios(signal, mode="long_short_quintile")
net_unscaled = apply_costs(w_unscaled, returns_panel, cost_bps=10.0).dropna()
_scaled_dates = None
for _s in all_strategies.values():
    _scaled_dates = _s.index if _scaled_dates is None else _scaled_dates.union(_s.index)
if _scaled_dates is not None:
    net_unscaled = net_unscaled.reindex(_scaled_dates).dropna()
all_strategies["unscaled_momentum"] = net_unscaled
net_unscaled.to_frame("net_return").to_parquet("results/strategies/unscaled_momentum.parquet")

# Master results table
print("\n\n=== MASTER RESULTS TABLE ===")
results_tbl = build_results_table(all_strategies)
print(results_tbl.to_string())
results_tbl.to_parquet("results/master_results_table.parquet")
print(results_tbl.to_csv())

# Per-model IC summary.
# Own-sample IC is NOT comparable across models — row counts differ. Print the
# common-row IC beside it and always print n_rows (CLAUDE.md rule 5).
print("\n=== IC SUMMARY ===")
_ic_idx = None
for f in all_forecasts.values():
    _ic_idx = f.index if _ic_idx is None else _ic_idx.intersection(f.index)
for name, oos in all_forecasts.items():
    ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
    ic_c = cross_sectional_ic(oos.reindex(_ic_idx),
                              realized_rv.rename("target_rv").to_frame())
    print(f"  {name}: own_IC={ic.mean():.4f} (n={len(oos)})  "
          f"common_IC={ic_c.mean():.4f} (n={len(_ic_idx)})")

# DM matrix.
# QLIKE is defined on VARIANCES. Passing forecast_log_rv / target_log_rv here
# fed negative numbers into a clip at 1e-12, collapsing both arguments to the
# same constant and making the loss identically zero on >99% of rows.
print("\n=== DM MATRIX (QLIKE on variance levels, p-values) ===")
_dm_idx = None
for f in all_forecasts.values():
    _dm_idx = f.index if _dm_idx is None else _dm_idx.intersection(f.index)
_dm_idx = _dm_idx.intersection(realized_rv.dropna().index)
print(f"  common rows across all {len(all_forecasts)} models: {len(_dm_idx)}")
rv_forecasts = {n: f["forecast_rv"].reindex(_dm_idx) for n, f in all_forecasts.items()}
try:
    dm_stats, dm_pvals = build_dm_matrix(rv_forecasts, realized_rv.reindex(_dm_idx))
    print(dm_pvals.round(3).to_string())
    dm_pvals.to_parquet("results/dm_pvalues.parquet")
except Exception as e:
    print(f"DM matrix failed: {e}")

print("\nAll results written to results/")
