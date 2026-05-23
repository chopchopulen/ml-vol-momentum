import pytest
import numpy as np
import pandas as pd
from src.strategy.uncertainty_scale import uncertainty_vol_scale


def _make_forecast(n_dates=10, n_tickers=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_dates, freq="ME")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d, "ticker": t,
                "forecast_rv": rng.uniform(0.001, 0.01),
                "forecast_sigma": rng.uniform(0.1, 1.0),
            })
    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    return df


def _make_signal(n_dates=10, n_tickers=5, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_dates, freq="ME")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({"date": d, "ticker": t, "signal": rng.standard_normal()})
    return pd.DataFrame(rows).set_index(["date", "ticker"])


class TestUncertaintyVolScale:
    def test_returns_weight_column(self):
        fc = _make_forecast()
        sig = _make_signal()
        w = uncertainty_vol_scale(sig, fc, target_vol=0.10)
        assert "weight" in w.columns

    def test_index_is_date_ticker(self):
        fc = _make_forecast()
        sig = _make_signal()
        w = uncertainty_vol_scale(sig, fc, target_vol=0.10)
        assert w.index.names == ["date", "ticker"]

    def test_high_sigma_gets_lower_weight(self):
        """Stock with 2× higher forecast_sigma should get lower absolute weight."""
        dates = pd.date_range("2010-01-01", periods=1, freq="ME")
        fc = pd.DataFrame({
            "forecast_rv": [0.005, 0.005],
            "forecast_sigma": [0.2, 0.4],  # T1 has 2× uncertainty
        }, index=pd.MultiIndex.from_tuples(
            [(dates[0], "T0"), (dates[0], "T1")], names=["date", "ticker"]))
        sig = pd.DataFrame({
            "signal": [1.0, 1.0],  # identical momentum signal
        }, index=fc.index)
        w = uncertainty_vol_scale(sig, fc, target_vol=0.10)
        assert abs(w.loc[(dates[0], "T0"), "weight"]) > abs(w.loc[(dates[0], "T1"), "weight"])

    def test_missing_sigma_falls_back_to_vol_scale(self):
        """If forecast_sigma is not present, should behave like plain vol_scale."""
        from src.strategy.scaling import vol_scale
        fc = _make_forecast()
        sig = _make_signal()
        fc_no_sigma = fc[["forecast_rv"]]
        w_unc = uncertainty_vol_scale(sig, fc_no_sigma, target_vol=0.10)
        w_plain = vol_scale(sig, fc, target_vol=0.10)
        pd.testing.assert_frame_equal(
            w_unc.sort_index(), w_plain.sort_index(), check_exact=False, atol=1e-6
        )
