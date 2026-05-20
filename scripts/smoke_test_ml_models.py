"""
Smoke test: walk-forward GBM and LSTM on a 3-year window (2003-2005).
Verifies models plug into the harness and produce sensible forecasts.
Run with: python scripts/smoke_test_ml_models.py
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
from src.models.lstm_model import LSTMForecaster

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2005-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:40]

print(f"Loading data for {len(tickers)} tickers (2000-2005)...")
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

# Add sector to panel for GBM
sector_map = {t: get_sector(t, pd.Timestamp("2003-01-01")) for t in tickers}
panel["sector"] = panel.index.get_level_values("ticker").map(sector_map)

windows = generate_windows(start, end, first_test_year=2003)
realized_rv = panel["target_rv"]

print(f"Windows: {len(windows)}")
for w in windows:
    print(f"  train {w.train_start.date()} - {w.train_end.date()} | test {w.test_start.date()} - {w.test_end.date()}")

print("\n=== GBM smoke test ===")
gbm_oos = run_walk_forward(GBMForecaster(), panel, windows)
print(f"OOS rows: {len(gbm_oos)}")
print(f"OOS dates: {gbm_oos.index.get_level_values('date').nunique()}")
print(f"NaN forecast_rv: {gbm_oos['forecast_rv'].isna().sum()}")
print(f"Negative forecast_rv: {(gbm_oos['forecast_rv'] < 0).sum()}")
ic_gbm = cross_sectional_ic(gbm_oos, realized_rv.rename("target_rv").to_frame())
print(f"Mean IC: {ic_gbm.mean():.4f}  (HAR-RV Phase 1 baseline: ~0.68)")

print("\n=== LSTM smoke test (seed=0 only) ===")
class SingleSeedLSTM:
    name = "lstm_seed0"
    def __init__(self):
        from src.models.lstm_model import LSTMForecaster
        self._m = LSTMForecaster()
    def fit(self, train):
        self._m.fit(train, seed=0)
    def predict(self, history):
        return self._m.predict(history)

lstm_oos = run_walk_forward(SingleSeedLSTM(), panel, windows)
print(f"OOS rows: {len(lstm_oos)}")
print(f"NaN forecast_rv: {lstm_oos['forecast_rv'].isna().sum()}")
if len(lstm_oos) > 0:
    ic_lstm = cross_sectional_ic(lstm_oos, realized_rv.rename("target_rv").to_frame())
    print(f"Mean IC: {ic_lstm.mean():.4f}")

# Sanity checks
assert len(gbm_oos) > 0, "GBM produced no OOS predictions"
assert gbm_oos["forecast_rv"].isna().sum() == 0, "GBM has NaN forecast_rv"
assert (gbm_oos["forecast_rv"] > 0).all(), "GBM has non-positive forecast_rv"
assert len(lstm_oos) > 0, "LSTM produced no OOS predictions"
assert lstm_oos["forecast_rv"].isna().sum() == 0, "LSTM has NaN forecast_rv"
assert (lstm_oos["forecast_rv"] > 0).all(), "LSTM has non-positive forecast_rv"

print("\nSmoke test: PASS")
