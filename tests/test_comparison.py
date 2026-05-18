import numpy as np
import pandas as pd
import pytest
from src.eval.comparison import build_results_table, build_dm_matrix, build_mz_table


def _make_returns(n=252, mean=0.001, std=0.01, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-04", periods=n, freq="B")
    return pd.Series(rng.normal(mean, std, n), index=dates)


def _make_rv_series(n=200, seed=0):
    """Make a realized variance Series indexed by (date, ticker) MultiIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-04", periods=n // 10, freq="B")
    tickers = [f"T{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    vals = np.abs(rng.normal(0.0001, 0.00005, len(idx)))
    return pd.Series(vals, index=idx)


class TestBuildResultsTable:
    def test_returns_dataframe(self):
        strategies = {"a": _make_returns(seed=0), "b": _make_returns(seed=1)}
        tbl = build_results_table(strategies)
        assert isinstance(tbl, pd.DataFrame)

    def test_has_required_columns(self):
        strategies = {"a": _make_returns(seed=0)}
        tbl = build_results_table(strategies)
        for col in ("sharpe", "sortino", "calmar", "ann_ret"):
            assert col in tbl.columns, f"Missing column: {col}"

    def test_row_per_strategy(self):
        strategies = {"a": _make_returns(seed=0), "b": _make_returns(seed=1)}
        tbl = build_results_table(strategies)
        assert set(tbl.index) == {"a", "b"}

    def test_positive_drift_has_positive_sharpe(self):
        strategies = {"good": _make_returns(mean=0.01, std=0.01, seed=0)}
        tbl = build_results_table(strategies)
        assert tbl.loc["good", "sharpe"] > 0


class TestBuildDMMatrix:
    def test_returns_two_dataframes(self):
        """build_dm_matrix returns (stats_df, pvals_df) tuple."""
        rv1 = _make_rv_series(seed=0)
        rv2 = _make_rv_series(seed=1)
        realized = _make_rv_series(seed=99)
        result = build_dm_matrix({"m1": rv1, "m2": rv2}, realized)
        assert isinstance(result, tuple)
        assert len(result) == 2
        stats, pvals = result
        assert isinstance(stats, pd.DataFrame)
        assert isinstance(pvals, pd.DataFrame)

    def test_pvals_in_zero_one(self):
        rv1 = _make_rv_series(seed=0)
        rv2 = _make_rv_series(seed=1)
        realized = _make_rv_series(seed=99)
        _, pvals = build_dm_matrix({"m1": rv1, "m2": rv2}, realized)
        # Off-diagonal p-values should be in [0, 1]
        for i in pvals.index:
            for j in pvals.columns:
                if i != j:
                    p = pvals.loc[i, j]
                    assert 0.0 <= float(p) <= 1.0, f"p-value {p} out of [0,1] for ({i},{j})"

    def test_symmetric_models(self):
        """Models with identical forecasts: DM stat = 0."""
        rv = _make_rv_series(seed=5)
        realized = _make_rv_series(seed=99)
        stats, pvals = build_dm_matrix({"same1": rv, "same2": rv}, realized)
        assert abs(float(stats.loc["same1", "same2"])) < 1e-6


class TestBuildMZTable:
    def test_returns_dataframe_with_expected_columns(self):
        fc = _make_rv_series(seed=0)
        realized = _make_rv_series(seed=99)
        mz = build_mz_table({"model_a": fc}, realized)
        assert isinstance(mz, pd.DataFrame)
        for col in ("alpha", "beta", "r2"):
            assert col in mz.columns

    def test_row_per_model(self):
        fc1 = _make_rv_series(seed=0)
        fc2 = _make_rv_series(seed=1)
        realized = _make_rv_series(seed=99)
        mz = build_mz_table({"m1": fc1, "m2": fc2}, realized)
        assert set(mz.index) == {"m1", "m2"}
