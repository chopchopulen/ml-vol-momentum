import numpy as np
import pandas as pd
import pytest
from src.models.gbm import GBMForecaster


FEATURE_COLS = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]


def _make_synthetic_panel(n_dates=300, n_tickers=10, seed=42):
    """Synthetic panel with all feature columns, target, and sector."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-03", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    sectors = ["Tech", "Finance", "Health", "Energy", "Consumer"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    data = {}
    for col in FEATURE_COLS:
        data[col] = np.abs(rng.normal(1e-4, 5e-5, len(idx)))
    data["target_log_rv"] = rng.normal(-10, 1, len(idx))
    # Sector: same sector per ticker, consistent
    sector_map = {t: sectors[i % len(sectors)] for i, t in enumerate(tickers)}
    data["sector"] = [sector_map[t] for _, t in idx]
    return pd.DataFrame(data, index=idx)


class TestGBMForecaster:
    def test_implements_forecaster_protocol(self):
        from src.models.forecaster import Forecaster
        m = GBMForecaster()
        assert isinstance(m, Forecaster)

    def test_fit_stores_model(self):
        panel = _make_synthetic_panel()
        m = GBMForecaster()
        m.fit(panel)
        assert m.booster_ is not None

    def test_predict_returns_correct_columns(self):
        panel = _make_synthetic_panel()
        train, test = panel.iloc[:panel.shape[0]//2], panel.iloc[panel.shape[0]//2:]
        m = GBMForecaster()
        m.fit(train)
        out = m.predict(test)
        assert "forecast_log_rv" in out.columns
        assert "forecast_rv" in out.columns

    def test_predict_index_is_date_ticker(self):
        panel = _make_synthetic_panel()
        train, test = panel.iloc[:panel.shape[0]//2], panel.iloc[panel.shape[0]//2:]
        m = GBMForecaster()
        m.fit(train)
        out = m.predict(test)
        assert out.index.names == ["date", "ticker"]

    def test_forecast_rv_is_positive(self):
        """forecast_rv = exp(log_rv_hat + sigma2/2) must always be positive."""
        panel = _make_synthetic_panel()
        train, test = panel.iloc[:panel.shape[0]//2], panel.iloc[panel.shape[0]//2:]
        m = GBMForecaster()
        m.fit(train)
        out = m.predict(test)
        assert (out["forecast_rv"].dropna() > 0).all()

    def test_no_leakage_target_in_predict(self):
        """predict() must not use target_log_rv column — it's not present at test time."""
        panel = _make_synthetic_panel()
        train = panel.iloc[:panel.shape[0]//2]
        # Pass test panel WITHOUT target column to simulate real prediction time
        test_no_target = panel.iloc[panel.shape[0]//2:].drop(columns=["target_log_rv"])
        m = GBMForecaster()
        m.fit(train)
        out = m.predict(test_no_target)
        assert len(out) > 0, "predict() should work even without target column"

    def test_shap_values_returns_array(self):
        """shap_values() returns a 2D array matching the number of input rows."""
        panel = _make_synthetic_panel(n_dates=100, n_tickers=5)
        m = GBMForecaster()
        m.fit(panel)
        X = panel.drop(columns=["target_log_rv"])
        shap_vals = m.shap_values(X)
        assert shap_vals is not None
        assert shap_vals.ndim == 2
        assert shap_vals.shape[0] == len(X.dropna(subset=FEATURE_COLS))

    def test_fit_stores_mse_resid(self):
        """mse_resid_ must be a positive finite float after fit — used for Jensen correction."""
        panel = _make_synthetic_panel()
        m = GBMForecaster()
        m.fit(panel)
        assert isinstance(m.mse_resid_, float)
        assert np.isfinite(m.mse_resid_)
        assert m.mse_resid_ > 0
