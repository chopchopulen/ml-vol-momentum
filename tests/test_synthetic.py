import numpy as np
import pandas as pd
import pytest
from src.eval.synthetic import white_noise_panel, leakage_test_signal

class TestWhiteNoisePanel:
    def test_shape(self):
        panel = white_noise_panel(n_dates=252, n_stocks=50, seed=0)
        assert len(panel) == 252 * 50

    def test_values_near_zero_mean(self):
        panel = white_noise_panel(n_dates=2520, n_stocks=100, seed=0)
        assert abs(panel["return"].mean()) < 1e-3

class TestLeakageTestSignal:
    def test_leakage_signal_is_future_return(self):
        rng = np.random.default_rng(99)
        n, m = 1000, 20
        dates = pd.date_range("2010-01-01", periods=n, freq="B")
        tickers = [f"S{i:03d}" for i in range(m)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date","ticker"])
        returns = pd.Series(rng.normal(0, 0.01, n*m), index=idx, name="return")
        panel = returns.to_frame()
        sig = leakage_test_signal(panel)
        # sig at date t should equal return at date t+1
        for tkr in tickers[:3]:
            r_t1 = returns.xs(tkr, level="ticker").shift(-1)
            s_t  = sig.xs(tkr, level="ticker")["signal"]
            common = r_t1.index.intersection(s_t.index)
            pd.testing.assert_series_equal(
                r_t1[common].reset_index(drop=True),
                s_t[common].reset_index(drop=True),
                check_names=False,
            )
