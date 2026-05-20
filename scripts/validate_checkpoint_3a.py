# scripts/validate_checkpoint_3a.py
"""
CHECKPOINT 3a: GBM OOS cross-sectional IC vs HAR-RV baseline.
Gate: GBM IC in [HAR-RV_IC - 0.03, HAR-RV_IC + 0.10]
      GBM OOS R² in [0.25, 0.70]
Run with: python scripts/validate_checkpoint_3a.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.data.loaders import load_ohlcv, load_vix
from src.data.universe import get_universe, get_sector
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.tests import cross_sectional_ic
from src.models.gbm import GBMForecaster
from src.models.baselines import HARRV

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
sector_map = {t: get_sector(t, pd.Timestamp("2003-01-01")) for t in tickers}
panel["sector"] = panel.index.get_level_values("ticker").map(sector_map)

windows = generate_windows(start, end, first_test_year=2003)
realized_rv  = panel["target_rv"]
realized_log = panel["target_log_rv"]

print("Running GBM walk-forward (22 years)...")
gbm_oos = run_walk_forward(GBMForecaster(), panel, windows)

print("Running HAR-RV walk-forward (22 years)...")
har_oos = run_walk_forward(HARRV(), panel, windows)

# Cross-sectional IC
ic_gbm = cross_sectional_ic(gbm_oos, realized_rv.rename("target_rv").to_frame())
ic_har = cross_sectional_ic(har_oos, realized_rv.rename("target_rv").to_frame())

# OOS R² on log_rv
def oos_r2(forecast_log, realized_log):
    common = forecast_log.index.intersection(realized_log.index)
    f = forecast_log.loc[common].dropna()
    r = realized_log.loc[common].reindex(f.index).dropna()
    f = f.reindex(r.index)
    ss_res = ((r - f) ** 2).sum()
    ss_tot = ((r - r.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

r2_gbm = oos_r2(gbm_oos["forecast_log_rv"], realized_log)
r2_har = oos_r2(har_oos["forecast_log_rv"], realized_log)

print(f"\nCHECKPOINT 3a results:")
print(f"  HAR-RV IC (22-yr pooled):  {ic_har.mean():.4f}  (Phase 1: 0.682)")
print(f"  GBM IC (22-yr pooled):     {ic_gbm.mean():.4f}")
print(f"  GBM IC − HAR-RV IC:        {ic_gbm.mean() - ic_har.mean():+.4f}  (gate: [-0.03, +0.10])")
print(f"  GBM OOS R²:                {r2_gbm:.4f}  (gate: [0.25, 0.70])")
print(f"  HAR-RV OOS R²:             {r2_har:.4f}  (Phase 1: 0.328)")

ic_diff = ic_gbm.mean() - ic_har.mean()
gate_ic = -0.03 <= ic_diff <= 0.10
gate_r2 = 0.25 <= r2_gbm <= 0.70
print(f"\nGates: IC_diff_in_range={gate_ic}  R2_in_range={gate_r2}")
print(f"CHECKPOINT 3a: {'PASS' if gate_ic and gate_r2 else 'FAIL'}")

# Per-year IC breakdown
print("\nPer-year IC (GBM vs HAR-RV):")
for yr in range(2003, 2025):
    yr_mask_g = ic_gbm.index.year == yr
    yr_mask_h = ic_har.index.year == yr
    g = ic_gbm[yr_mask_g].mean() if yr_mask_g.any() else float("nan")
    h = ic_har[yr_mask_h].mean() if yr_mask_h.any() else float("nan")
    print(f"  {yr}: GBM={g:.3f}  HAR-RV={h:.3f}  diff={g-h:+.3f}")

# Save OOS forecasts for use in checkpoint 3b and compare_all_models
import os
os.makedirs("results/forecasts", exist_ok=True)
gbm_oos.to_parquet("results/forecasts/gbm.parquet")
har_oos.to_parquet("results/forecasts/har_rv.parquet")
print("\nForecasts saved to results/forecasts/")
