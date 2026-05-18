import numpy as np
import pandas as pd
import pytest
from src.models.lstm_model import LSTMForecaster

FEATURE_COLS = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]


def _make_synthetic_panel(n_dates=200, n_tickers=5, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-03", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    data = {}
    for col in FEATURE_COLS:
        data[col] = np.abs(rng.normal(1e-4, 5e-5, len(idx)))
    data["target_log_rv"] = rng.normal(-10, 1, len(idx))
    return pd.DataFrame(data, index=idx)


class TestLSTMForecaster:
    def test_implements_forecaster_protocol(self):
        from src.models.forecaster import Forecaster
        m = LSTMForecaster()
        assert isinstance(m, Forecaster)

    def test_fit_stores_model(self):
        panel = _make_synthetic_panel()
        m = LSTMForecaster()
        m.fit(panel, seed=0)
        assert m.model_ is not None

    def test_predict_returns_correct_columns(self):
        panel = _make_synthetic_panel(n_dates=200)
        train = panel.iloc[:panel.shape[0] * 4 // 5]
        test  = panel.iloc[panel.shape[0] * 4 // 5:]
        m = LSTMForecaster()
        m.fit(train, seed=0)
        out = m.predict(test)
        assert "forecast_log_rv" in out.columns
        assert "forecast_rv" in out.columns

    def test_predict_index_is_date_ticker(self):
        panel = _make_synthetic_panel(n_dates=200)
        train = panel.iloc[:panel.shape[0] * 4 // 5]
        test  = panel.iloc[panel.shape[0] * 4 // 5:]
        m = LSTMForecaster()
        m.fit(train, seed=0)
        out = m.predict(test)
        assert out.index.names == ["date", "ticker"]

    def test_forecast_rv_positive(self):
        panel = _make_synthetic_panel(n_dates=200)
        train = panel.iloc[:panel.shape[0] * 4 // 5]
        test  = panel.iloc[panel.shape[0] * 4 // 5:]
        m = LSTMForecaster()
        m.fit(train, seed=0)
        out = m.predict(test)
        assert (out["forecast_rv"].dropna() > 0).all()

    def test_different_seeds_give_different_predictions(self):
        """Two different seeds should produce different forecasts (randomness matters)."""
        panel = _make_synthetic_panel(n_dates=200)
        train = panel.iloc[:160]
        test  = panel.iloc[160:]
        m0 = LSTMForecaster()
        m0.fit(train, seed=0)
        out0 = m0.predict(test)

        m1 = LSTMForecaster()
        m1.fit(train, seed=1)
        out1 = m1.predict(test)

        common = out0.index.intersection(out1.index)
        assert len(common) > 0
        # Not identical (with probability 1)
        assert not np.allclose(
            out0.loc[common, "forecast_log_rv"].values,
            out1.loc[common, "forecast_log_rv"].values,
        ), "Two different seeds produced identical predictions — seeding not working"

    def test_normalisation_uses_train_stats_only(self):
        """Normalisation stats must be fit on train only — verify they don't change when test data changes."""
        panel = _make_synthetic_panel(n_dates=200)
        train = panel.iloc[:160]
        test_a = panel.iloc[160:170]
        test_b = panel.iloc[160:]

        m = LSTMForecaster()
        m.fit(train, seed=0)
        # Stats stored from training window
        mean_after_fit = m._feat_mean.copy()

        # Predict on two different test windows — stats must be unchanged
        m.predict(test_a)
        assert np.allclose(m._feat_mean, mean_after_fit), "Feature mean changed after predict()"
        m.predict(test_b)
        assert np.allclose(m._feat_mean, mean_after_fit), "Feature mean changed after predict()"

    def test_mse_resid_none_before_fit(self):
        """mse_resid_ must be None before fit() and a positive float after."""
        m = LSTMForecaster()
        assert m.mse_resid_ is None
        panel = _make_synthetic_panel(n_dates=200)
        m.fit(panel, seed=0)
        assert m.mse_resid_ is not None
        assert isinstance(m.mse_resid_, float)
        assert m.mse_resid_ > 0

    def test_predict_raises_before_fit(self):
        """predict() must raise RuntimeError if called before fit()."""
        m = LSTMForecaster()
        panel = _make_synthetic_panel(n_dates=200)
        with pytest.raises(RuntimeError):
            m.predict(panel)
