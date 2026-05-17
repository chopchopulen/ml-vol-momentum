import numpy as np
import pandas as pd
import pytest
from src.models.baselines import RollingVolModel, GARCH11Model, HARRV

def _make_panel(n_dates: int = 500, n_tickers: int = 3,
                seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-03", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date","ticker"])
    returns = rng.normal(0, 0.012, len(idx))
    return pd.DataFrame({"return": returns}, index=idx)

class TestRollingVolModel:
    def test_implements_forecaster_protocol(self):
        from src.models.forecaster import Forecaster
        m = RollingVolModel(window=126)
        assert isinstance(m, Forecaster)

    def test_predict_returns_panel(self):
        panel = _make_panel()
        m = RollingVolModel(window=126)
        m.fit(panel)
        out = m.predict(panel)
        assert "forecast_rv" in out.columns
        assert out.index.names == ["date", "ticker"]

    def test_forecast_rv_nonnegative(self):
        panel = _make_panel()
        m = RollingVolModel(window=126)
        m.fit(panel)
        out = m.predict(panel)
        assert (out["forecast_rv"].dropna() >= 0).all()

class TestHARRV:
    def test_fit_produces_coefficients(self):
        from src.data.targets import forward_rv
        # Use synthetic panel
        panel = _make_panel(n_dates=600)
        tickers = panel.index.get_level_values("ticker").unique()
        # Manually compute rv columns for feature panel
        rows = []
        for tkr in tickers:
            r = panel.xs(tkr, level="ticker")["return"]
            rv_d = (r**2).shift(1)
            rv_w = (r**2).rolling(5).sum().shift(1)
            rv_m = (r**2).rolling(21).sum().shift(1)
            sub = pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m})
            sub.index = pd.MultiIndex.from_product([[tkr], sub.index],
                                                    names=["ticker","date"])
            rows.append(sub.swaplevel().sort_index())
        features = pd.concat(rows)
        targets = forward_rv(panel)
        m = HARRV()
        m.fit(pd.concat([features, targets], axis=1).dropna())
        # Should have a coefficient per ticker
        assert len(m.coef_) == len(tickers)

    def test_harrv_matches_statsmodels(self):
        import statsmodels.api as sm
        # Single ticker test
        rng = np.random.default_rng(5)
        n = 500
        rv_d = rng.uniform(1e-5, 5e-4, n)
        rv_w = pd.Series(rv_d).rolling(5).mean().fillna(rv_d.mean()).values
        rv_m = pd.Series(rv_d).rolling(21).mean().fillna(rv_d.mean()).values
        log_rv_target = (np.log(rv_d) * 0.4 + np.log(rv_w) * 0.3 +
                         np.log(rv_m) * 0.2 + rng.normal(0, 0.1, n))
        X = sm.add_constant(
            np.column_stack([np.log(rv_d), np.log(rv_w), np.log(rv_m)]))
        res = sm.OLS(log_rv_target, X).fit()
        # Fit our HAR-RV on the same data
        dates = pd.date_range("2000-01-03", periods=n, freq="B")
        idx = pd.MultiIndex.from_product([dates, ["T0"]], names=["date","ticker"])
        df = pd.DataFrame({
            "rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m,
            "target_log_rv": log_rv_target,
        }, index=idx)
        m = HARRV()
        m.fit(df)
        our_coef = m.coef_["T0"]
        np.testing.assert_allclose(
            our_coef["beta_d"], res.params[1], atol=0.01)
        np.testing.assert_allclose(
            our_coef["beta_w"], res.params[2], atol=0.01)
        np.testing.assert_allclose(
            our_coef["beta_m"], res.params[3], atol=0.01)

class TestGARCH11:
    def test_fit_converges_on_synthetic_panel(self):
        panel = _make_panel(n_dates=400, n_tickers=2)
        m = GARCH11Model()
        m.fit(panel)
        conv = m.convergence_log_
        assert sum(1 for v in conv.values() if v) >= 1  # at least 1 converges

    def test_predict_returns_panel(self):
        panel = _make_panel(n_dates=400, n_tickers=2)
        m = GARCH11Model()
        m.fit(panel)
        out = m.predict(panel)
        assert "forecast_rv" in out.columns
