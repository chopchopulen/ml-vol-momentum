import numpy as np
import pandas as pd
import pytest
from src.data.targets import forward_rv

class TestForwardRV:
    def setup_method(self):
        # Constant return r every day: forward RV at any t should == 21 * r^2
        dates = pd.date_range("2010-01-04", periods=100, freq="B")
        tickers = ["AAA", "BBB"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date","ticker"])
        r = 0.005
        self.panel = pd.DataFrame({"return": r}, index=idx)
        self.r = r

    def test_forward_rv_identity_constant_return(self):
        result = forward_rv(self.panel, horizon=21)
        # At any date with a full 21-day forward window, RV = 21 * r^2
        first_valid = result.index.get_level_values("date").unique()[0]
        rv_vals = result.xs(first_valid, level="date")["target_rv"]
        expected = 21 * self.r ** 2
        np.testing.assert_allclose(rv_vals.values, expected, rtol=1e-9)

    def test_log_rv_is_log_of_rv(self):
        result = forward_rv(self.panel, horizon=21)
        valid = result.dropna()
        np.testing.assert_allclose(
            valid["target_log_rv"].values,
            np.log(valid["target_rv"].values),
            rtol=1e-9,
        )

    def test_last_horizon_rows_are_nan(self):
        result = forward_rv(self.panel, horizon=21)
        dates = result.index.get_level_values("date").unique()
        # Last 21 trading days should be NaN (no full forward window)
        last_date = dates[-1]
        assert result.xs(last_date, level="date")["target_rv"].isna().all()

    def test_columns_present(self):
        result = forward_rv(self.panel)
        assert "target_rv" in result.columns
        assert "target_log_rv" in result.columns

    def test_degenerate_target_canary(self):
        """Stock with all-zero returns for 21 days gets RV=0 → log(0)=-inf.
        The pipeline must drop these rows, NOT produce -inf in target_log_rv.
        If this fails, R²=1.0 artifacts will silently contaminate OOS evaluation."""
        # Mix: one normal ticker, one ticker with all-zero returns
        dates = pd.date_range("2010-01-04", periods=100, freq="B")
        tickers = ["NORMAL", "HALTED"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        # HALTED has exactly zero returns for all dates
        returns = pd.Series(
            [0.005 if t == "NORMAL" else 0.0 for _ in dates for t in tickers],
            index=idx,
            name="return",
        )
        panel = returns.to_frame()
        result = forward_rv(panel, horizon=21)
        # No -inf anywhere in the output
        assert not (result["target_log_rv"] == -np.inf).any(), (
            "target_log_rv contains -inf: zero-RV rows must be dropped at "
            "target construction time (targets.py). dropna() does NOT drop -inf."
        )
        # No +inf either
        assert not (result["target_log_rv"] == np.inf).any()
        # HALTED ticker should either be absent or have all-NaN target_log_rv
        halted_rows = result.xs("HALTED", level="ticker") if "HALTED" in result.index.get_level_values("ticker") else None
        if halted_rows is not None:
            assert halted_rows["target_log_rv"].isna().all() or len(halted_rows.dropna()) == 0, (
                "HALTED ticker has non-NaN target_log_rv — zero-RV rows are not being dropped"
            )
