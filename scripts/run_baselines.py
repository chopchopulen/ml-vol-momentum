"""Run walk-forward for RollingVol, GARCH, HAR-RV baselines."""
import warnings; warnings.filterwarnings("ignore")
import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from pathlib import Path
from src.data.universe import get_universe, get_sector
from src.data.loaders import load_ohlcv, load_vix
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.models.baselines import RollingVolModel, GARCH11Model, HARRV
from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.tests import cross_sectional_ic

Path("results/forecasts").mkdir(parents=True, exist_ok=True)
Path("results/strategies").mkdir(parents=True, exist_ok=True)

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]

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
prices_panel = ohlcv[["close"]]
signal = momentum_signal(prices_panel, lookback=252, skip=21)
realized_rv = panel["target_rv"]

for ModelClass, name in [(RollingVolModel, "rolling_vol"), (GARCH11Model, "garch"), (HARRV, "har_rv")]:
    fcast_path = Path(f"results/forecasts/{name}.parquet")
    strat_path = Path(f"results/strategies/{name}_scaled.parquet")
    if fcast_path.exists() and strat_path.exists():
        print(f"{name}: cached, skipping")
        continue
    print(f"Running {name}...")
    model = ModelClass()
    oos = run_walk_forward(model, panel, windows)
    oos.to_parquet(fcast_path)
    ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
    print(f"  OOS rows: {len(oos)}  Mean IC: {ic.mean():.4f}")
    w_scaled_raw = vol_scale(signal, oos, target_vol=0.10)
    w_scaled = build_portfolios(None, weights=w_scaled_raw, mode="vol_targeted_gross")
    net_scaled = apply_costs(w_scaled, returns_panel, cost_bps=10.0).dropna()
    net_scaled.to_frame("net_return").to_parquet(strat_path)
    print(f"  {name} done.")
print("Baselines complete.")
