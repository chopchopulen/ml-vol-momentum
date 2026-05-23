"""Extension 1: Probabilistic LSTM walk-forward + uncertainty-weighted strategy.

Outputs:
  results/forecasts/prob_lstm.parquet          — (mu, sigma) OOS forecasts
  results/strategies/prob_lstm_scaled.parquet  — plain vol-scaled (using mu only)
  results/strategies/prob_lstm_unc_scaled.parquet — uncertainty-weighted sizing
  results/extension1_summary.md
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from pathlib import Path
from src.data.universe import get_universe, get_sector
from src.data.loaders import load_ohlcv, load_vix
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.models.prob_lstm import ProbLSTMEnsemble
from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.uncertainty_scale import uncertainty_vol_scale
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.eval.walk_forward import generate_windows
from src.eval.tests import cross_sectional_ic
from src.eval.metrics import sharpe, max_drawdown

CKPT_DIR = Path("/tmp/prob_lstm_checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
Path("results/forecasts").mkdir(parents=True, exist_ok=True)
Path("results/strategies").mkdir(parents=True, exist_ok=True)

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print(f"Loading {len(tickers)} tickers...")

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

windows = generate_windows(start, end, first_test_year=2003)
signal = momentum_signal(ohlcv[["close"]], lookback=252, skip=21)
realized_rv = panel["target_rv"]

# Check cache
fcast_path = Path("results/forecasts/prob_lstm.parquet")
if fcast_path.exists():
    print("prob_lstm forecasts: loading cached...")
    oos_all = pd.read_parquet(fcast_path)
else:
    model = ProbLSTMEnsemble()
    oos_frames = []
    for i, w in enumerate(windows):
        ckpt = CKPT_DIR / f"window_{i:02d}.parquet"
        if ckpt.exists():
            print(f"  Window {i+1}/{len(windows)}: {w.test_end.year} (cached)")
            oos_frames.append(pd.read_parquet(ckpt))
            continue
        print(f"  Window {i+1}/{len(windows)}: {w.train_end.year} → {w.test_end.year}")
        train_mask = ((panel.index.get_level_values("date") >= w.train_start) &
                      (panel.index.get_level_values("date") <= w.train_end))
        model.fit(panel[train_mask])
        history = panel[panel.index.get_level_values("date") <= w.test_end]
        preds = model.predict(history)
        if preds.empty:
            print("    WARNING: empty preds")
            continue
        pred_dates = preds.index.get_level_values("date")
        oos = preds[(pred_dates >= w.test_start) & (pred_dates <= w.test_end)]
        print(f"    OOS rows: {len(oos)}")
        oos.to_parquet(ckpt)
        oos_frames.append(oos)
    oos_all = pd.concat(oos_frames).sort_index()
    oos_all.to_parquet(fcast_path)

ic = cross_sectional_ic(oos_all, realized_rv.rename("target_rv").to_frame())
print(f"\nProb LSTM OOS rows: {len(oos_all)}  Mean IC: {ic.mean():.4f}")

# Plain vol-scaled strategy (uses mu forecast, ignores sigma)
strat_path = Path("results/strategies/prob_lstm_scaled.parquet")
if not strat_path.exists():
    w_scaled_raw = vol_scale(signal, oos_all, target_vol=0.10)
    w_scaled = build_portfolios(None, weights=w_scaled_raw, mode="vol_targeted_gross")
    net_scaled = apply_costs(w_scaled, returns_panel, cost_bps=10.0).dropna()
    net_scaled.to_frame("net_return").to_parquet(strat_path)
else:
    net_scaled = pd.read_parquet(strat_path)["net_return"]

# Uncertainty-weighted strategy
unc_path = Path("results/strategies/prob_lstm_unc_scaled.parquet")
if not unc_path.exists():
    w_unc_raw = uncertainty_vol_scale(signal, oos_all, target_vol=0.10)
    w_unc = build_portfolios(None, weights=w_unc_raw, mode="vol_targeted_gross")
    net_unc = apply_costs(w_unc, returns_panel, cost_bps=10.0).dropna()
    net_unc.to_frame("net_return").to_parquet(unc_path)
else:
    net_unc = pd.read_parquet(unc_path)["net_return"]

sh_plain = sharpe(net_scaled)
sh_unc   = sharpe(net_unc)
dd_plain = max_drawdown(net_scaled.cumsum())
dd_unc   = max_drawdown(net_unc.cumsum())

# Load existing LSTM result for comparison
lstm_strat_path = Path("results/strategies/lstm_scaled.parquet")
if lstm_strat_path.exists():
    lstm_strat = pd.read_parquet(lstm_strat_path)["net_return"]
    sh_lstm = sharpe(lstm_strat)
    lstm_row = f"| LSTM point forecast (existing) | — | {sh_lstm:.3f} | — |\n"
else:
    lstm_row = "| LSTM point forecast (existing) | — | N/A (run make ml first) | — |\n"

print(f"\n=== Extension 1 Results ===")
print(f"Prob LSTM plain vol-scale:        Sharpe={sh_plain:.3f}  Max DD={dd_plain:.1%}")
print(f"Prob LSTM uncertainty-weighted:   Sharpe={sh_unc:.3f}  Max DD={dd_unc:.1%}")

summary = f"""# Extension 1: Probabilistic LSTM Results

## Forecast Quality
- OOS rows: {len(oos_all)}
- Mean cross-sectional IC: {ic.mean():.4f}
- IC std across windows: {ic.std():.4f}

## Strategy Comparison

| Strategy | Mean IC | Sharpe | Max DD |
|----------|---------|--------|--------|
{lstm_row}| Prob LSTM — plain vol-scale | {ic.mean():.4f} | {sh_plain:.3f} | {dd_plain:.1%} |
| Prob LSTM — uncertainty-weighted | {ic.mean():.4f} | {sh_unc:.3f} | {dd_unc:.1%} |

## Interpretation
{"Uncertainty weighting IMPROVED Sharpe vs plain vol-scale." if sh_unc > sh_plain else "Uncertainty weighting did NOT improve Sharpe vs plain vol-scale."}
"""

Path("results/extension1_summary.md").write_text(summary)
print("\nResults written to results/extension1_summary.md")
