"""
CHECKPOINT 2b: Barroso & Santa-Clara (2015) replication.
2009 momentum crash visible unscaled; vol-scaling provides uplift.
Run with: python scripts/validate_checkpoint_2b.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.data.loaders import load_ohlcv, load_vix
from src.data.universe import get_universe
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.models.baselines import RollingVolModel
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.metrics import sharpe, max_drawdown

start = pd.Timestamp("1998-01-01")
end   = pd.Timestamp("2010-12-31")

tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
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
# panel for HAR/GARCH needs feature cols + target; RollingVol only needs "return"
feature_target_panel = features.join(targets, how="inner").dropna(subset=["target_log_rv"])
# RollingVolModel.predict requires "return" column
rv_panel = returns_panel.join(feature_target_panel, how="inner")

prices_panel = ohlcv[["close"]]
signal = momentum_signal(prices_panel, lookback=252, skip=21)

# Unscaled portfolio
weights_unscaled = build_portfolios(signal, mode="long_short_quintile")
net_unscaled = apply_costs(weights_unscaled, returns_panel, cost_bps=10.0).dropna()

# Rolling-vol-scaled portfolio (walk-forward, 42-day embargo)
print("Running rolling-vol walk-forward...")
windows = generate_windows(start, end, first_test_year=2002)
rv_model = RollingVolModel(window=126)
oos_forecasts = run_walk_forward(rv_model, rv_panel, windows)
weights_scaled_raw = vol_scale(signal, oos_forecasts, target_vol=0.10)
weights_scaled = build_portfolios(None, weights=weights_scaled_raw,
                                  mode="vol_targeted_gross")
net_scaled = apply_costs(weights_scaled, returns_panel, cost_bps=10.0).dropna()

# Trim to evaluation period
net_unscaled = net_unscaled.loc["2000":"2010"]
net_scaled   = net_scaled.loc["2000":"2010"]

sr_unscaled = sharpe(net_unscaled)
sr_scaled   = sharpe(net_scaled)

equity_unscaled = (1 + net_unscaled).cumprod()

# 2009 crash check
eq_2009 = equity_unscaled.loc["2009"]
if len(eq_2009) > 0:
    dd_2009 = (eq_2009 - eq_2009.cummax()) / eq_2009.cummax()
    max_dd_2009 = float(dd_2009.min())
else:
    max_dd_2009 = float("nan")

print(f"\nCHECKPOINT 2b results (2000-2010):")
print(f"  Unscaled Sharpe:     {sr_unscaled:.3f}  (paper: ~0.5)")
print(f"  Scaled Sharpe:       {sr_scaled:.3f}  (paper: ~0.9)")
print(f"  Sharpe uplift:       {sr_scaled - sr_unscaled:.3f}  (gate: > 0.0)")
print(f"  2009 unscaled MaxDD: {max_dd_2009:.3f}  (gate: < -0.20)")

# Print yearly breakdown
print("\nAnnual returns (unscaled vs scaled):")
for yr in range(2000, 2011):
    yr_str = str(yr)
    mask_u = net_unscaled.index.year == yr
    mask_s = net_scaled.index.year == yr
    r_u = net_unscaled.loc[mask_u].mean() * 252 if mask_u.any() else float("nan")
    r_s = net_scaled.loc[mask_s].mean() * 252 if mask_s.any() else float("nan")
    print(f"  {yr}: unscaled={r_u:+.3f}  scaled={r_s:+.3f}")

gate_1 = max_dd_2009 < -0.20
gate_2 = sr_scaled > sr_unscaled
print(f"\nGates: 2009_crash={gate_1}  scaled>unscaled={gate_2}")
print(f"CHECKPOINT 2b: {'PASS' if gate_1 and gate_2 else 'FAIL'}")
