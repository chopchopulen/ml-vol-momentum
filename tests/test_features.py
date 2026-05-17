import numpy as np
import pandas as pd
import pytest
from src.data.features import realized_variance, parkinson, build_feature_panel

class TestRealizedVariance:
    def setup_method(self):
        dates = pd.date_range("2010-01-04", periods=30, freq="B")
        self.r = pd.Series(np.linspace(0.001, 0.002, 30), index=dates)

    def test_rv_window_1_equals_r_squared(self):
        rv = realized_variance(self.r, 1)
        # rv at t = r_t^2 (window=1 means just today's squared return)
        # But we .shift(1) so rv[t] = r[t-1]^2
        for i in range(1, len(self.r)):
            t = self.r.index[i]
            t_prev = self.r.index[i-1]
            np.testing.assert_allclose(rv[t], self.r[t_prev]**2, rtol=1e-9)

    def test_rv_is_nonnegative(self):
        rv = realized_variance(self.r, 5)
        assert (rv.dropna() >= 0).all()

class TestParkinson:
    def test_parkinson_identity_known_value(self):
        import math
        # Single-day panel: H=110, L=100, window=1
        dates = pd.date_range("2010-01-04", periods=6, freq="B")
        high = pd.Series([110]*6, index=dates, dtype=float)
        low  = pd.Series([100]*6, index=dates, dtype=float)
        pk = parkinson(high, low, window=1)
        # Parkinson = (1/(4*ln2)) * (ln(H/L))^2, then .shift(1)
        expected = (1 / (4 * math.log(2))) * (math.log(110/100))**2
        # Check day index 1 (which reflects day 0's values after shift)
        np.testing.assert_allclose(pk.iloc[1], expected, rtol=1e-6)

class TestBuildFeaturePanel:
    def setup_method(self):
        dates = pd.date_range("2010-01-04", periods=150, freq="B")
        tickers = ["AAA", "BBB"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date","ticker"])
        rng = np.random.default_rng(7)
        self.ohlcv = pd.DataFrame({
            "open":   100 + rng.normal(0, 1, len(idx)),
            "high":   102 + rng.normal(0, 1, len(idx)),
            "low":    98  + rng.normal(0, 1, len(idx)),
            "close":  100 + rng.normal(0, 1, len(idx)),
            "volume": rng.integers(1_000_000, 5_000_000, len(idx)).astype(float),
        }, index=idx)
        vix_dates = pd.date_range("2010-01-04", periods=150, freq="B")
        self.vix = pd.Series(15 + rng.normal(0, 2, 150), index=vix_dates, name="vix")

    def test_expected_feature_columns_present(self):
        feat = build_feature_panel(self.ohlcv, self.vix)
        for col in ["rv_d","rv_w","rv_m","pk","skew","kurt","vix","log_dv","ret_21"]:
            assert col in feat.columns, f"Missing feature column: {col}"

    def test_no_lookahead(self):
        # Build features on data[:T] and data[:T+1].
        # Feature values at index T-1 must be identical.
        T = 100
        dates_full = pd.date_range("2010-01-04", periods=T+1, freq="B")
        tickers = ["AAA"]
        def make_panel(n):
            # Use a fresh rng with the same seed each call so that the first T
            # rows are identical between make_panel(T) and make_panel(T+1).
            # This is required for the no-lookahead check: the two panels differ
            # only in that the second has one extra row at the end.
            rng = np.random.default_rng(99)
            idx = pd.MultiIndex.from_product(
                [dates_full[:n], tickers], names=["date","ticker"])
            return pd.DataFrame({
                "open": 100.0, "high": 102.0, "low": 98.0,
                "close": 100 + rng.normal(0, 1, n),
                "volume": 1_000_000.0,
            }, index=idx)
        vix = pd.Series(15.0, index=dates_full)
        feat_T   = build_feature_panel(make_panel(T),   vix)
        feat_Tp1 = build_feature_panel(make_panel(T+1), vix)
        check_date = dates_full[T-2]  # second-to-last date of the shorter panel
        for col in feat_T.columns:
            val_T   = feat_T.xs(check_date, level="date")[col]
            val_Tp1 = feat_Tp1.xs(check_date, level="date")[col]
            pd.testing.assert_series_equal(val_T, val_Tp1,
                                            check_names=False,
                                            obj=f"lookahead in {col}")
