import numpy as np
import pandas as pd
import pytest
from src.eval.walk_forward import generate_windows, run_walk_forward, CVWindow
from src.models.baselines import RollingVolModel

def _make_panel(n_dates=600, n_tickers=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-03", periods=n_dates, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date","ticker"])
    r = rng.normal(0, 0.012, len(idx))
    return pd.DataFrame({"return": r}, index=idx)

class TestRunWalkForward:
    def setup_method(self):
        self.panel = _make_panel()
        self.windows = generate_windows(
            start=pd.Timestamp("2000-01-01"),
            end=pd.Timestamp("2002-05-31"),
            embargo_days=42,
            first_test_year=2001,
        )

    def test_produces_oos_forecasts_no_nans(self):
        forecaster = RollingVolModel(window=63)
        result = run_walk_forward(forecaster, self.panel, self.windows)
        assert isinstance(result, pd.DataFrame)
        assert "forecast_rv" in result.columns
        oos_dates = result.index.get_level_values("date")
        # All OOS forecast dates must fall within test windows
        for w in self.windows:
            mask = (oos_dates >= w.test_start) & (oos_dates <= w.test_end)
            assert mask.sum() > 0
        # Every returned date must be within the overall test range
        min_test_start = min(w.test_start for w in self.windows)
        max_test_end   = max(w.test_end   for w in self.windows)
        assert oos_dates.min() >= min_test_start
        assert oos_dates.max() <= max_test_end

    def test_train_data_never_includes_oos(self):
        seen_dates = []
        class TrackingForecaster:
            name = "tracker"
            def fit(self_, train):
                seen_dates.append(
                    train.index.get_level_values("date").max()
                )
            def predict(self_, history):
                return pd.DataFrame()
        run_walk_forward(TrackingForecaster(), self.panel, self.windows)
        for i, max_train_date in enumerate(seen_dates):
            w = self.windows[i]
            assert max_train_date <= w.train_end, \
                f"Window {i}: training data leaked into OOS"

    def test_40day_embargo_never_violated(self):
        for w in self.windows:
            gap = (w.test_start - w.train_end).days
            assert gap >= 42
