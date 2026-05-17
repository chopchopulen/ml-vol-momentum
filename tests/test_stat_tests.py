import numpy as np
import pandas as pd
import pytest
from src.eval.tests import (diebold_mariano, mincer_zarnowitz,
                             cross_sectional_ic, sharpe_diff_bootstrap)

class TestDieboldMariano:
    def test_identical_forecasts_high_pvalue(self):
        rng = np.random.default_rng(0)
        realized = pd.Series(rng.uniform(1e-5, 1e-3, 500))
        forecast = pd.Series(rng.uniform(1e-5, 1e-3, 500))
        e1 = realized - forecast
        e2 = realized - forecast  # identical errors
        stat, pval = diebold_mariano(e1, e2, h=21)
        assert pval > 0.5, f"Identical forecasts should have p >> 0.5, got {pval:.3f}"

    def test_clearly_better_forecast_low_pvalue(self):
        rng = np.random.default_rng(1)
        realized = pd.Series(rng.uniform(1e-5, 1e-3, 2000))
        good_fc  = realized + rng.normal(0, 1e-5, 2000)   # near-perfect
        bad_fc   = pd.Series(rng.uniform(1e-5, 1e-3, 2000))  # random
        e1 = realized.values - good_fc.values
        e2 = realized.values - bad_fc.values
        _, pval = diebold_mariano(pd.Series(e1), pd.Series(e2), h=21,
                                   loss="mse")
        assert pval < 0.05

    def test_pvalue_uniform_under_h0(self):
        rng = np.random.default_rng(42)
        pvals = []
        for _ in range(200):
            realized = pd.Series(rng.uniform(1e-5, 1e-3, 500))
            f1 = realized + rng.normal(0, 5e-5, 500)
            f2 = realized + rng.normal(0, 5e-5, 500)
            e1 = realized - f1
            e2 = realized - f2
            _, p = diebold_mariano(e1, e2, h=1, loss="mse")
            pvals.append(p)
        # Under H0 p-values should be roughly uniform; test fraction < 0.05 ≈ 5%
        reject_rate = np.mean(np.array(pvals) < 0.05)
        assert 0.01 < reject_rate < 0.15, f"Size distortion: reject_rate={reject_rate:.3f}"

class TestMincerZarnowitz:
    def test_perfect_forecast_alpha0_beta1(self):
        rng = np.random.default_rng(3)
        realized = pd.Series(rng.uniform(0.0001, 0.01, 300))
        forecast = realized.copy()  # perfect
        result = mincer_zarnowitz(realized, forecast)
        assert abs(result["alpha"]) < 0.01
        assert abs(result["beta"] - 1.0) < 0.05

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(4)
        r = pd.Series(rng.normal(0.005, 0.001, 100))
        f = pd.Series(rng.normal(0.005, 0.001, 100))
        result = mincer_zarnowitz(r, f)
        for k in ["alpha", "beta", "p_joint", "r2"]:
            assert k in result

class TestCrosssectionalIC:
    def test_perfect_rank_correlation_gives_one(self):
        dates = pd.date_range("2020-01-02", periods=10, freq="B")
        tickers = [f"S{i}" for i in range(20)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date","ticker"])
        vals = np.tile(np.arange(20, dtype=float), 10)
        forecast = pd.DataFrame({"forecast_rv": vals}, index=idx)
        realized = pd.DataFrame({"target_rv": vals}, index=idx)
        ic = cross_sectional_ic(forecast, realized)
        assert (ic.dropna() > 0.99).all()

class TestSharpeDiffBootstrap:
    def test_same_series_ci_crosses_zero(self):
        rng = np.random.default_rng(10)
        r = pd.Series(rng.normal(0.001, 0.01, 1260))
        result = sharpe_diff_bootstrap(r, r, n_boot=500, seed=7)
        assert result["ci_lo"] < 0 < result["ci_hi"]

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(11)
        r1 = pd.Series(rng.normal(0.001, 0.01, 500))
        r2 = pd.Series(rng.normal(0.0005, 0.01, 500))
        result = sharpe_diff_bootstrap(r1, r2, n_boot=200, seed=0)
        for k in ["sharpe_diff_point", "ci_lo", "ci_hi", "p_value"]:
            assert k in result
