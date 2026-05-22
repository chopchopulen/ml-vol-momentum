# scripts/run_phase4.py
"""
Phase 4 analysis: SHAP, regime breakdown, sector-neutral robustness, figures.
Run with:
  cd "/Users/harry/RL:ML Project/ml-vol-momentum"
  PYTHONPATH="/tmp/ml-vol-momentum-venv/lib/python3.14/site-packages" \
    /tmp/ml-vol-momentum-venv/bin/python -u scripts/run_phase4.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
sys.stdout.reconfigure(line_buffering=True)

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
from src.eval.walk_forward import generate_windows
from src.eval.tests import cross_sectional_ic
from src.eval.comparison import build_results_table
from src.models.gbm import GBMForecaster
from src.interp.shap_analysis import compute_shap_importance
from src.interp.regime_analysis import assign_regimes, regime_ic_table
from src.interp.sector_neutral import demean_signal
from src.viz.plots import (plot_shap_importance, plot_ic_by_year,
                            plot_regime_ic_heatmap, plot_equity_curves)

Path("docs/figures").mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print(f"Loading {len(tickers)} tickers...", flush=True)
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
sector_map = {t: get_sector(t, pd.Timestamp("2003-01-01")) for t in tickers}
panel["sector"] = panel.index.get_level_values("ticker").map(sector_map)

windows      = generate_windows(start, end, first_test_year=2003)
prices_panel = ohlcv[["close"]]
signal       = momentum_signal(prices_panel, lookback=252, skip=21)
realized_rv  = panel["target_rv"]

# ── Load forecast + strategy parquets ─────────────────────────────────────
model_names = ["rolling_vol", "har_rv", "garch", "gbm", "lstm"]
all_forecasts  = {n: pd.read_parquet(f"results/forecasts/{n}.parquet") for n in model_names}
all_strategies = {n: pd.read_parquet(f"results/strategies/{n}_scaled.parquet")["net_return"]
                  for n in model_names}
unscaled = pd.read_parquet("results/strategies/unscaled_momentum.parquet")["net_return"]
all_strategies["unscaled_momentum"] = unscaled

# ── 1. SHAP Analysis ───────────────────────────────────────────────────────
print("\n=== SHAP ANALYSIS ===", flush=True)
last_w = windows[-1]
train_mask = panel.index.get_level_values("date") <= last_w.train_end
train_last = panel[train_mask]
gbm_shap = GBMForecaster()
print("Fitting GBM on last window for SHAP...", flush=True)
gbm_shap.fit(train_last)
shap_imp = compute_shap_importance(gbm_shap, train_last, sample_size=5000)
print(shap_imp.to_string())
plot_shap_importance(shap_imp)

# ── 2. VIX-Regime IC Breakdown ─────────────────────────────────────────────
print("\n=== REGIME IC TABLE ===", flush=True)
regimes = assign_regimes(vix, windows)
reg_table = regime_ic_table(all_forecasts, realized_rv.rename("target_rv").to_frame(), regimes)
print(reg_table.round(4).to_string())
reg_table.to_parquet("results/regime_ic_table.parquet")
plot_regime_ic_heatmap(reg_table)

# ── 3. Sector-Neutral Robustness ───────────────────────────────────────────
print("\n=== SECTOR-NEUTRAL ROBUSTNESS ===", flush=True)
signal_sn_series = demean_signal(signal["signal"], sector_map)
signal_sn = signal_sn_series.rename("signal").to_frame()
sn_strategies = {}
for model_name in model_names:
    oos = all_forecasts[model_name]
    w_sn_raw = vol_scale(signal_sn, oos, target_vol=0.10)
    w_sn = build_portfolios(None, weights=w_sn_raw, mode="vol_targeted_gross")
    net_sn = apply_costs(w_sn, returns_panel, cost_bps=10.0).dropna()
    sn_strategies[f"{model_name}_sn"] = net_sn

sn_table = build_results_table(sn_strategies)
print("Sector-neutral Sharpes:")
print(sn_table[["sharpe"]].round(3).to_string())
sn_table.to_parquet("results/sector_neutral_results.parquet")

# ── 4. Figures ─────────────────────────────────────────────────────────────
print("\n=== GENERATING FIGURES ===", flush=True)
plot_ic_by_year(all_forecasts, realized_rv)
plot_equity_curves(all_strategies)

# ── 5. Summary markdown ────────────────────────────────────────────────────
vanilla_tbl = pd.read_parquet("results/master_results_table.parquet")
summary_lines = [
    "# Phase 4 Summary\n",
    "## SHAP Feature Importance (GBM, last training window)\n",
    shap_imp.round(4).to_string(), "\n",
    "## Cross-Sectional IC by VIX Regime\n",
    reg_table.round(4).to_string(), "\n",
    "## Sector-Neutral vs Vanilla Sharpe\n",
    "| Model | Vanilla Sharpe | SN Sharpe |",
    "|---|---|---|",
]
for n in model_names:
    # vanilla_tbl index might be plain model names or with _scaled — try both
    v_key = n if n in vanilla_tbl.index else f"{n}_scaled"
    v_sh = vanilla_tbl.loc[v_key, "sharpe"] if v_key in vanilla_tbl.index else float("nan")
    sn_key = f"{n}_sn"
    sn_sh = sn_table.loc[sn_key, "sharpe"] if sn_key in sn_table.index else float("nan")
    summary_lines.append(f"| {n} | {v_sh:.3f} | {sn_sh:.3f} |")

summary_lines += [
    "\n## Figures",
    "- `docs/figures/shap_importance.png`",
    "- `docs/figures/ic_by_year.png`",
    "- `docs/figures/regime_ic_heatmap.png`",
    "- `docs/figures/equity_curves.png`",
]
Path("results/phase4_summary.md").write_text("\n".join(summary_lines))
print("\nPhase 4 complete. Summary written to results/phase4_summary.md", flush=True)
