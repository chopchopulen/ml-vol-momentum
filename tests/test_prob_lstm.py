import pytest
import numpy as np
import pandas as pd
import torch
from src.models.prob_lstm import ProbLSTMForecaster, ProbLSTMEnsemble


def _make_panel(n_dates=200, n_tickers=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    feat_cols = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]
    rows = []
    for t in tickers:
        df = pd.DataFrame(rng.standard_normal((n_dates, len(feat_cols))), columns=feat_cols, index=dates)
        df["target_log_rv"] = rng.standard_normal(n_dates)
        df["target_rv"] = np.exp(df["target_log_rv"])
        df.index = pd.MultiIndex.from_arrays([dates, [t]*n_dates], names=["date","ticker"])
        rows.append(df)
    return pd.concat(rows)


@pytest.fixture
def panel():
    return _make_panel()


class TestProbLSTMForecaster:
    def test_fit_stores_model(self, panel):
        m = ProbLSTMForecaster()
        m.fit(panel, seed=0)
        assert m.model_ is not None

    def test_predict_returns_required_columns(self, panel):
        m = ProbLSTMForecaster()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert "forecast_log_rv" in out.columns
        assert "forecast_rv" in out.columns
        assert "forecast_sigma" in out.columns  # NEW — uncertainty column

    def test_forecast_sigma_positive(self, panel):
        m = ProbLSTMForecaster()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert (out["forecast_sigma"] > 0).all()

    def test_forecast_rv_positive(self, panel):
        m = ProbLSTMForecaster()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert (out["forecast_rv"] > 0).all()

    def test_index_is_date_ticker(self, panel):
        m = ProbLSTMForecaster()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert out.index.names == ["date", "ticker"]

    def test_mc_dropout_variant(self, panel):
        m = ProbLSTMForecaster(variant="mc_dropout", n_mc_samples=10)
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert "forecast_sigma" in out.columns
        assert (out["forecast_sigma"] > 0).all()


class TestProbLSTMEnsemble:
    def test_predict_has_uncertainty_column(self, panel):
        ens = ProbLSTMEnsemble(seeds=[0, 1])
        ens.fit(panel)
        out = ens.predict(panel)
        assert "forecast_sigma" in out.columns

    def test_forecaster_protocol(self, panel):
        from src.models.forecaster import Forecaster
        ens = ProbLSTMEnsemble(seeds=[0])
        assert isinstance(ens, Forecaster)
