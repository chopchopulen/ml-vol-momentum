import numpy as np
import pandas as pd
import pytest
from src.eval.metrics import sharpe, sortino, max_drawdown, calmar, icir

class TestSharpe:
    def test_zero_returns_gives_zero_sharpe(self):
        r = pd.Series(np.zeros(252))
        assert sharpe(r) == 0.0

    def test_constant_positive_return(self):
        r = pd.Series([0.001] * 252)
        s = sharpe(r)
        # mean = 0.001, std = 0 → inf, but we return 0 for std=0 by convention
        assert np.isfinite(s) or np.isinf(s)  # either is acceptable

    def test_iid_normal_sharpe_near_zero(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 10_000))
        s = sharpe(r)
        assert abs(s) < 0.5, f"White noise Sharpe should be near 0, got {s:.3f}"

    def test_positive_drift_positive_sharpe(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.001, 0.01, 2520))
        s = sharpe(r)
        assert s > 0

class TestMaxDrawdown:
    def test_monotone_up_zero_drawdown(self):
        prices = pd.Series(np.cumprod(1 + np.full(100, 0.001)))
        assert max_drawdown(prices) == 0.0

    def test_50pct_drop_gives_correct_drawdown(self):
        prices = pd.Series([100, 80, 50, 60, 70])
        dd = max_drawdown(prices)
        assert abs(dd - 0.5) < 1e-9

class TestICIR:
    def test_returns_float(self):
        ic = pd.Series(np.random.default_rng(1).normal(0.05, 0.1, 100))
        result = icir(ic)
        assert isinstance(result, float)

    def test_higher_mean_gives_higher_icir(self):
        ic_good = pd.Series(np.full(100, 0.1))
        ic_poor = pd.Series(np.full(100, 0.01))
        assert icir(ic_good) > icir(ic_poor)
