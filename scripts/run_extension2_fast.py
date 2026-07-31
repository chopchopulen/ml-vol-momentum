"""Extension 2 (fast mode): Architecture comparison — LSTM vs Transformer vs MLP vs TCN.

Fast mode: 2 seeds, 10 max epochs, reduced Transformer d_model=16.
Reuses any existing per-window checkpoints from the full run.

Outputs:
  results/forecasts/{arch}_fast.parquet
  results/strategies/{arch}_fast_scaled.parquet
  results/extension2_summary.md
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from src.data.universe import get_universe, get_sector
from src.data.loaders import load_ohlcv, load_vix
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.models.lstm_model import LSTMEnsemble
from src.models.transformer_model import TransformerEnsemble
from src.models.mlp_model import MLPEnsemble
from src.models.tcn_model import TCNEnsemble
from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.eval.walk_forward import generate_windows
from src.eval.tests import cross_sectional_ic
from src.eval.metrics import sharpe, max_drawdown

Path("results/forecasts").mkdir(parents=True, exist_ok=True)
Path("results/strategies").mkdir(parents=True, exist_ok=True)

# Fast-mode overrides — patch config before models read it
from src import config as _config_mod
_orig_load = _config_mod.load_config
_fast_overrides = {"max_epochs": 10, "patience": 3, "seeds": [0, 1]}
def _patched_load_config():
    cfg = _orig_load()
    cfg["models"]["lstm"].update(_fast_overrides)
    cfg["models"]["transformer"]["d_model"] = 16
    return cfg
_config_mod.load_config = _patched_load_config

# Re-import models after patching so they pick up new config
import importlib
import src.models.transformer_model as _tm
import src.models.mlp_model as _mm
import src.models.tcn_model as _tcn
importlib.reload(_tm)
importlib.reload(_mm)
importlib.reload(_tcn)
TransformerEnsemble = _tm.TransformerEnsemble
MLPEnsemble = _mm.MLPEnsemble
TCNEnsemble = _tcn.TCNEnsemble

SEEDS = [0, 1]

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

ARCHITECTURES = [
    ("lstm",        LSTMEnsemble),
    ("transformer", TransformerEnsemble),
    ("mlp",         MLPEnsemble),
    ("tcn",         TCNEnsemble),
]

results = {}

for arch_name, EnsembleClass in ARCHITECTURES:
    # For LSTM reuse existing full results
    if arch_name == "lstm":
        full_fcast = Path("results/forecasts/lstm.parquet")
        full_strat = Path("results/strategies/lstm_scaled.parquet")
        if full_fcast.exists() and full_strat.exists():
            print(f"\nlstm: loading existing full results...")
            oos = pd.read_parquet(full_fcast)
            net = pd.read_parquet(full_strat)["net_return"]
            ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
            sh = sharpe(net)
            dd = max_drawdown((1 + net).cumprod())
            results[arch_name] = {"mean_ic": ic.mean(), "std_ic": ic.std(), "sharpe": sh, "max_dd": dd, "n_oos": len(oos)}
            print(f"  IC={ic.mean():.4f} ± {ic.std():.4f}  Sharpe={sh:.3f}")
            continue

    fcast_path = Path(f"results/forecasts/{arch_name}_fast.parquet")
    strat_path = Path(f"results/strategies/{arch_name}_fast_scaled.parquet")

    if fcast_path.exists() and strat_path.exists():
        print(f"\n{arch_name}: loading cached fast results...")
        oos = pd.read_parquet(fcast_path)
        net = pd.read_parquet(strat_path)["net_return"]
    else:
        print(f"\n{arch_name}: running fast walk-forward (2 seeds, 10 epochs)...")
        # Reuse checkpoints from full run if available, else use fast checkpoints
        full_ckpt_dir  = Path(f"/tmp/{arch_name}_ext2_checkpoints")
        fast_ckpt_dir  = Path(f"/tmp/{arch_name}_fast_checkpoints")
        fast_ckpt_dir.mkdir(exist_ok=True)

        model = EnsembleClass(seeds=SEEDS)
        oos_frames = []
        window_times = []

        for i, w in enumerate(windows):
            # Prefer full checkpoint, then fast checkpoint
            full_ckpt = full_ckpt_dir / f"window_{i:02d}.parquet"
            fast_ckpt = fast_ckpt_dir / f"window_{i:02d}.parquet"

            if full_ckpt.exists():
                print(f"  Window {i+1}/{len(windows)}: {w.test_end.year} (full cache)")
                oos_frames.append(pd.read_parquet(full_ckpt))
                continue
            if fast_ckpt.exists():
                print(f"  Window {i+1}/{len(windows)}: {w.test_end.year} (fast cache)")
                oos_frames.append(pd.read_parquet(fast_ckpt))
                continue

            t0 = time.time()
            print(f"  Window {i+1}/{len(windows)}: {w.train_end.year} → {w.test_end.year}", end="", flush=True)
            train_mask = ((panel.index.get_level_values("date") >= w.train_start) &
                          (panel.index.get_level_values("date") <= w.train_end))
            model.fit(panel[train_mask])
            history = panel[panel.index.get_level_values("date") <= w.test_end]
            preds = model.predict(history)
            if preds.empty:
                print(" WARNING: empty")
                continue
            pred_dates = preds.index.get_level_values("date")
            oos_w = preds[(pred_dates >= w.test_start) & (pred_dates <= w.test_end)]
            elapsed = time.time() - t0
            window_times.append(elapsed)
            print(f"  rows={len(oos_w)}  {elapsed:.0f}s")
            oos_w.to_parquet(fast_ckpt)
            oos_frames.append(oos_w)

        oos = pd.concat(oos_frames).sort_index()
        oos.to_parquet(fcast_path)
        if window_times:
            print(f"  Avg time per window: {np.mean(window_times):.0f}s")

        w_scaled_raw = vol_scale(signal, oos, target_vol=0.10)
        w_scaled = build_portfolios(None, weights=w_scaled_raw, mode="vol_targeted_gross")
        net = apply_costs(w_scaled, returns_panel, cost_bps=10.0).dropna()
        net.to_frame("net_return").to_parquet(strat_path)

    ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
    sh = sharpe(net)
    dd = max_drawdown((1 + net).cumprod())
    results[arch_name] = {"mean_ic": ic.mean(), "std_ic": ic.std(), "sharpe": sh, "max_dd": dd, "n_oos": len(oos)}
    print(f"  IC={ic.mean():.4f} ± {ic.std():.4f}  Sharpe={sh:.3f}  MaxDD={dd:.1%}")

# Write summary
rows = []
for arch, r in results.items():
    rows.append({
        "Architecture": arch.upper(),
        "Mean XS-IC": f"{r['mean_ic']:.4f}",
        "IC Std": f"{r['std_ic']:.4f}",
        "Sharpe": f"{r['sharpe']:.3f}",
        "Max DD": f"{r['max_dd']:.1%}",
    })

df_results = pd.DataFrame(rows)
print("\n=== Extension 2 Results ===")
print(df_results.to_string(index=False))

summary_lines = [
    "# Extension 2: Architecture Comparison Results\n",
    "All architectures use: same 9 features, seq_len=60, Huber loss,",
    "same walk-forward CV (22 windows, 42-day embargo, 2003-2024).",
    "Fast mode: 2 seeds, 10 max epochs, Transformer d_model=16.",
    "LSTM uses the existing 5-seed full-run result for comparison.\n",
    "## Results\n",
    "| Architecture | Mean XS-IC | IC Std | Sharpe | Max DD |",
    "|-------------|-----------|--------|--------|--------|",
]
for r in rows:
    summary_lines.append(
        f"| {r['Architecture']} | {r['Mean XS-IC']} | {r['IC Std']} | {r['Sharpe']} | {r['Max DD']} |"
    )

summary_lines += [
    "\n## Interpretation",
    f"- Best IC: {max(results, key=lambda k: results[k]['mean_ic']).upper()}",
    f"- Best Sharpe: {max(results, key=lambda k: results[k]['sharpe']).upper()}",
    f"- Does temporal structure help vs MLP? {'YES' if results.get('lstm', {}).get('mean_ic', 0) > results.get('mlp', {}).get('mean_ic', 0) else 'NO'} (LSTM IC > MLP IC)",
    f"- Does Transformer overfit? {'YES (lower IC than LSTM)' if results.get('transformer', {}).get('mean_ic', 1) < results.get('lstm', {}).get('mean_ic', 0) else 'NO (Transformer IC >= LSTM)'}",
    f"- Does TCN match LSTM? {'YES' if abs(results.get('tcn', {}).get('mean_ic', 0) - results.get('lstm', {}).get('mean_ic', 0)) < 0.02 else 'NO, gap > 0.02'}",
]

Path("results/extension2_summary.md").write_text("\n".join(summary_lines))
print("\nResults written to results/extension2_summary.md")
