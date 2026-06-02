"""Extensions analysis: cost sensitivity, Prob LSTM calibration, LSTM gradient sensitivity.

Outputs:
  results/figures/cost_sensitivity.png
  results/figures/calibration.png
  results/figures/gradient_sensitivity.png
  results/extensions_analysis_summary.md
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path

Path("results/figures").mkdir(parents=True, exist_ok=True)

# ── shared data ────────────────────────────────────────────────────────────────
from src.data.universe import get_universe, get_sector
from src.data.loaders import load_ohlcv, load_vix
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.strategy.costs import apply_costs
from src.eval.metrics import sharpe

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print("Loading data...")
ohlcv = load_ohlcv(tickers, start, end)
vix   = load_vix(start, end)

returns_frames = []
for ticker, grp in ohlcv.groupby(level="ticker"):
    close = grp.droplevel("ticker")["close"]
    r = np.log(close / close.shift(1))
    idx = pd.MultiIndex.from_arrays([r.index, [ticker]*len(r)], names=["date","ticker"])
    returns_frames.append(r.rename("return").set_axis(idx))
returns_panel = pd.concat(returns_frames).to_frame()

features = build_feature_panel(ohlcv, vix)
targets  = forward_rv(returns_panel)
panel    = features.join(targets, how="inner").dropna(subset=["target_log_rv"])


# ══════════════════════════════════════════════════════════════════════════════
# 1. COST SENSITIVITY SWEEP
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Cost sensitivity sweep ──")

MODELS = {
    "HAR-RV":      "results/strategies/har_rv_scaled.parquet",
    "GBM":         "results/strategies/gbm_scaled.parquet",
    "LSTM":        "results/strategies/lstm_scaled.parquet",
    "Prob LSTM":   "results/strategies/prob_lstm_unc_scaled.parquet",
    "Transformer": "results/strategies/transformer_fast_scaled.parquet",
    "TCN (partial)": "results/strategies/tcn_partial_scaled.parquet",
}

# Load gross returns (before any costs) by reloading weights and computing
# gross PnL. Simpler: we already have net returns at 10bps. Back out gross
# by adding cost back, then re-apply at different bps.
# Gross return ≈ net + turnover * 5bps.  We don't have turnover stored,
# so instead re-run apply_costs at each bps level using the stored weights.
# Simplest correct approach: load the strategy parquets which store net_return,
# reconstruct gross by noting gross = net + cost_paid.
# We don't have turnover stored — so use a direct approach:
# reload the weight parquets and re-apply costs. Weights ARE stored in strategies.

# Actually strategies store net_return series only. Best approach:
# at 10bps we have net. The DIFFERENCE between gross and net is proportional to bps.
# net(c) = gross - turnover * c/2   (c = round-trip bps, /2 = per-side)
# net(10) = gross - turnover * 5bps
# net(c)  = net(10) + turnover*(5 - c/2) bps
# We can estimate turnover from net(10) vs net(0) if we had net(0), but we don't.
# Clean solution: just re-run apply_costs on the raw weight files.
# But weights aren't stored. Use the numerical approach:
# Estimate avg daily turnover from the Sharpe sensitivity slope.
# OR: just re-compute from scratch using stored forecasts + signal.

# Pragmatic: load net returns and compute Sharpe at 0bps (gross) by
# reconstructing from the cost model. Since we know cost=10bps was applied,
# we sweep by rescaling: net_c = net_10 + (10 - c) * daily_cost_per_bp
# We estimate daily_cost_per_bp from the diff between gross (0bps) run.
# Too circular. Just do the clean thing: recompute strategies at each bps.

from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.portfolio import build_portfolios

signal = momentum_signal(ohlcv[["close"]], lookback=252, skip=21)

cost_bps_range = [0, 2, 5, 10, 15, 20, 25, 30]
cost_results = {name: [] for name in MODELS}

fcast_map = {
    "HAR-RV":      "results/forecasts/har_rv.parquet",
    "GBM":         "results/forecasts/gbm.parquet",
    "LSTM":        "results/forecasts/lstm.parquet",
    "Prob LSTM":   "results/forecasts/prob_lstm.parquet",
    "Transformer": "results/forecasts/transformer_fast.parquet",
    "TCN (partial)": "results/forecasts/tcn_partial.parquet",
}

from src.strategy.uncertainty_scale import uncertainty_vol_scale

for name, fcast_path in fcast_map.items():
    oos = pd.read_parquet(fcast_path)
    if "forecast_sigma" in oos.columns and name == "Prob LSTM":
        w_raw = uncertainty_vol_scale(signal, oos, target_vol=0.10)
    else:
        w_raw = vol_scale(signal, oos, target_vol=0.10)
    w = build_portfolios(None, weights=w_raw, mode="vol_targeted_gross")
    for c in cost_bps_range:
        net = apply_costs(w, returns_panel, cost_bps=float(c)).dropna()
        cost_results[name].append(sharpe(net))
    print(f"  {name}: gross Sharpe={cost_results[name][0]:.3f}  @10bps={cost_results[name][cost_bps_range.index(10)]:.3f}")

fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
for (name, sharpes), color in zip(cost_results.items(), colors):
    ax.plot(cost_bps_range, sharpes, marker="o", label=name, color=color, linewidth=2)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvline(10, color="gray", linewidth=0.8, linestyle=":", label="current (10bps)")
ax.set_xlabel("Round-trip transaction cost (bps)")
ax.set_ylabel("Annualised Sharpe ratio")
ax.set_title("Sharpe vs Transaction Cost — All Forecasters")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("results/figures/cost_sensitivity.png", dpi=150)
plt.close()
print("  Saved: results/figures/cost_sensitivity.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROB LSTM CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Prob LSTM calibration ──")

prob_fc = pd.read_parquet("results/forecasts/prob_lstm.parquet")

# Join with realized log-RV
realized = panel[["target_log_rv"]].copy()
cal = prob_fc.join(realized, how="inner").dropna()

mu    = cal["forecast_log_rv"].values
sigma = cal["forecast_sigma"].values
y     = cal["target_log_rv"].values

# Standardised residuals: z = (y - mu) / sigma
z = (y - mu) / np.clip(sigma, 1e-8, None)

print(f"  N={len(z):,}  mean_z={z.mean():.3f}  std_z={z.std():.3f}  (ideal: 0, 1)")

# Expected calibration error: bin by predicted quantile, compare to empirical
n_bins = 20
quantile_levels = np.linspace(0.025, 0.975, n_bins)
from scipy import stats as scipy_stats
empirical_coverage = []
expected_coverage  = []
for q in np.linspace(0.05, 0.95, 10):
    lo = scipy_stats.norm.ppf((1 - q) / 2)
    hi = scipy_stats.norm.ppf((1 + q) / 2)
    covered = np.mean((z >= lo) & (z <= hi))
    empirical_coverage.append(covered)
    expected_coverage.append(q)

ece = float(np.mean(np.abs(np.array(empirical_coverage) - np.array(expected_coverage))))
print(f"  ECE={ece:.4f}  (0=perfect, 0.1=10pp avg miscalibration)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: reliability diagram
ax = axes[0]
ax.plot(expected_coverage, empirical_coverage, "o-", color="steelblue", linewidth=2, label="Prob LSTM")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
ax.set_xlabel("Expected coverage")
ax.set_ylabel("Empirical coverage")
ax.set_title(f"Reliability Diagram\nECE = {ece:.4f}")
ax.legend()
ax.grid(True, alpha=0.3)

# Right: histogram of standardised residuals vs N(0,1)
ax = axes[1]
z_clipped = np.clip(z, -5, 5)
ax.hist(z_clipped, bins=80, density=True, alpha=0.6, color="steelblue", label="Standardised residuals")
xr = np.linspace(-5, 5, 200)
ax.plot(xr, scipy_stats.norm.pdf(xr), "r-", linewidth=2, label="N(0,1)")
ax.set_xlabel("z = (y − μ) / σ")
ax.set_ylabel("Density")
ax.set_title(f"Residual Distribution\nmean={z.mean():.3f}, std={z.std():.3f}")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("results/figures/calibration.png", dpi=150)
plt.close()
print("  Saved: results/figures/calibration.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. LSTM INPUT GRADIENT SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n── LSTM gradient sensitivity ──")

from src.models.lstm_model import LSTMForecaster, FEATURE_COLS, _SEQ_LEN
from src.eval.walk_forward import generate_windows

windows = generate_windows(start, end, first_test_year=2003)

# Use the first walk-forward window's training data to fit one LSTM seed
# then compute |d output / d input| averaged over a sample of sequences
w0 = windows[0]
train_mask = ((panel.index.get_level_values("date") >= w0.train_start) &
              (panel.index.get_level_values("date") <= w0.train_end))
train = panel[train_mask]

print(f"  Fitting LSTM on window 0 (seed=0, {len(train):,} rows)...")
m = LSTMForecaster()
m.fit(train, seed=0)

# Build sequences from the training data (normalised)
feat_cols = [c for c in FEATURE_COLS if c in train.columns]
norm = train.copy()
norm[feat_cols] = (norm[feat_cols] - m._feat_mean) / m._feat_std
norm["target_log_rv"] = (norm["target_log_rv"] - m._tgt_mean) / m._tgt_std

X, y, _ = m._build_sequences(norm, include_target=True)
print(f"  Sequences built: {X.shape}")

# Sample 2000 sequences for speed
rng = np.random.default_rng(42)
idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
X_sample = torch.from_numpy(X[idx]).requires_grad_(True)

m.model_.eval()
out = m.model_(X_sample)          # (N,)
out.sum().backward()

# grad shape: (N, seq_len, n_features)
grads = X_sample.grad.detach().numpy()  # (N, 60, 9)

# Mean absolute gradient per feature (averaged over time steps and samples)
mean_abs_grad = np.abs(grads).mean(axis=(0, 1))  # (9,)

# Also: gradient norm over time (which timesteps matter most?)
grad_over_time = np.abs(grads).mean(axis=(0, 2))  # (60,)

print("  Feature sensitivities:")
for feat, g in sorted(zip(feat_cols, mean_abs_grad), key=lambda x: -x[1]):
    print(f"    {feat:<12} {g:.6f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: per-feature sensitivity bar chart
ax = axes[0]
sorted_idx = np.argsort(mean_abs_grad)[::-1]
sorted_feats = [feat_cols[i] for i in sorted_idx]
sorted_grads = mean_abs_grad[sorted_idx]
bars = ax.barh(sorted_feats[::-1], sorted_grads[::-1], color="steelblue", alpha=0.8)
ax.set_xlabel("Mean |gradient|")
ax.set_title("LSTM Input Gradient Sensitivity\n(mean |∂output/∂input| over 2000 sequences)")
ax.grid(True, alpha=0.3, axis="x")

# Right: sensitivity over time (which lag matters?)
ax = axes[1]
lags = np.arange(_SEQ_LEN, 0, -1)  # lag 60 = oldest, lag 1 = most recent
ax.plot(lags, grad_over_time[::-1], color="steelblue", linewidth=1.5)
ax.axvline(21, color="red", linestyle="--", linewidth=1, label="21-day (monthly RV)")
ax.axvline(5,  color="orange", linestyle="--", linewidth=1, label="5-day (weekly RV)")
ax.axvline(1,  color="green", linestyle="--", linewidth=1, label="1-day (daily RV)")
ax.set_xlabel("Lag (days ago)")
ax.set_ylabel("Mean |gradient|")
ax.set_title("LSTM Temporal Sensitivity\n(which lags drive the output?)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("results/figures/gradient_sensitivity.png", dpi=150)
plt.close()
print("  Saved: results/figures/gradient_sensitivity.png")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
# Breakeven cost for each model
breakeven = {}
for name, sharpes in cost_results.items():
    # Find cost where Sharpe crosses zero
    for i in range(len(cost_bps_range) - 1):
        if sharpes[i] >= 0 and sharpes[i+1] < 0:
            # Linear interpolate
            c0, c1 = cost_bps_range[i], cost_bps_range[i+1]
            s0, s1 = sharpes[i], sharpes[i+1]
            be = c0 + (0 - s0) / (s1 - s0) * (c1 - c0)
            breakeven[name] = f"{be:.1f} bps"
            break
    else:
        if sharpes[0] < 0:
            breakeven[name] = "< 0 bps (never positive)"
        else:
            breakeven[name] = f"> {cost_bps_range[-1]} bps"

summary = f"""# Extensions Analysis Summary

## 1. Cost Sensitivity

Sharpe ratio at different round-trip transaction costs:

| Model | 0 bps (gross) | 5 bps | 10 bps | 20 bps | Breakeven |
|-------|--------------|-------|--------|--------|-----------|
"""
for name in MODELS:
    s = cost_results[name]
    bps_idx = {c: i for i, c in enumerate(cost_bps_range)}
    summary += f"| {name} | {s[bps_idx[0]]:.3f} | {s[bps_idx[5]]:.3f} | {s[bps_idx[10]]:.3f} | {s[bps_idx[20]]:.3f} | {breakeven[name]} |\n"

summary += f"""
**Key finding:** Most forecasters have positive gross Sharpe but go negative after 5–10 bps costs.
The vol-scaling signal itself has value; transaction costs are the binding constraint.

## 2. Prob LSTM Calibration

- N = {len(z):,} OOS predictions
- Standardised residual mean = {z.mean():.3f} (ideal: 0) — {"well-centred" if abs(z.mean()) < 0.1 else "BIASED"}
- Standardised residual std = {z.std():.3f} (ideal: 1.0) — {"well-scaled" if 0.8 < z.std() < 1.2 else "OVERCONFIDENT" if z.std() < 0.8 else "UNDERCONFIDENT"}
- ECE = {ece:.4f} (0 = perfect; 0.05 = 5pp avg miscalibration)

**Interpretation:** {"The model is reasonably calibrated." if ece < 0.05 else "The model is miscalibrated — predicted intervals are too " + ("narrow (overconfident)" if z.std() < 1.0 else "wide (underconfident)")}

## 3. LSTM Gradient Sensitivity

Top features by mean |∂output/∂input|:

| Feature | Sensitivity | Rank |
|---------|------------|------|
"""
for rank, (feat, g) in enumerate(sorted(zip(feat_cols, mean_abs_grad), key=lambda x: -x[1]), 1):
    summary += f"| {feat} | {g:.6f} | {rank} |\n"

summary += f"""
**Temporal sensitivity:** Peak gradient at lag ~{int(np.argmax(grad_over_time[::-1])+1)} days.
The LSTM assigns most weight to {"recent" if np.argmax(grad_over_time[::-1]) < 10 else "mid-range"} lags.
"""

Path("results/extensions_analysis_summary.md").write_text(summary)
print("\nSaved: results/extensions_analysis_summary.md")
print("\nDone.")
