import pytest
import numpy as np
import pandas as pd
from src.models.transformer_model import TransformerForecaster, TransformerEnsemble
from src.models.mlp_model import MLPForecaster, MLPEnsemble
from src.models.tcn_model import TCNForecaster, TCNEnsemble
from src.models.forecaster import Forecaster


def _make_panel(n_dates=200, n_tickers=4, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    feat_cols = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]
    rows = []
    for t in tickers:
        df = pd.DataFrame(rng.standard_normal((n_dates, len(feat_cols))),
                          columns=feat_cols, index=dates)
        df["target_log_rv"] = rng.standard_normal(n_dates)
        df["target_rv"] = np.exp(df["target_log_rv"])
        df.index = pd.MultiIndex.from_arrays(
            [dates, [t]*n_dates], names=["date", "ticker"])
        rows.append(df)
    return pd.concat(rows)


@pytest.fixture
def panel():
    return _make_panel()


ARCHITECTURES = [
    ("transformer", TransformerForecaster, TransformerEnsemble),
    ("mlp", MLPForecaster, MLPEnsemble),
    ("tcn", TCNForecaster, TCNEnsemble),
]


@pytest.mark.parametrize("name,ForecasterClass,EnsembleClass", ARCHITECTURES)
class TestArchitectures:
    def test_implements_protocol(self, name, ForecasterClass, EnsembleClass, panel):
        m = EnsembleClass(seeds=[0])
        assert isinstance(m, Forecaster)

    def test_fit_and_predict(self, name, ForecasterClass, EnsembleClass, panel):
        m = ForecasterClass()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert "forecast_log_rv" in out.columns
        assert "forecast_rv" in out.columns
        assert len(out) > 0

    def test_forecast_rv_positive(self, name, ForecasterClass, EnsembleClass, panel):
        m = ForecasterClass()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert (out["forecast_rv"] > 0).all()

    def test_index_names(self, name, ForecasterClass, EnsembleClass, panel):
        m = ForecasterClass()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert out.index.names == ["date", "ticker"]

    def test_ensemble_averages_seeds(self, name, ForecasterClass, EnsembleClass, panel):
        ens = EnsembleClass(seeds=[0, 1])
        ens.fit(panel)
        out = ens.predict(panel)
        assert len(out) > 0
