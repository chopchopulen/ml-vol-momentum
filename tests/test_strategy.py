import numpy as np
import pandas as pd
import pytest
from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs


def _make_prices_panel(n_dates=300, n_tickers=5, trend=0.001, seed=42):
    """Trending prices panel for momentum tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-04", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    trends = np.linspace(trend, -trend, n_tickers)
    log_rets = rng.normal(0, 0.01, (n_dates, n_tickers))
    for i, t in enumerate(trends):
        log_rets[:, i] += t
    log_prices = np.cumsum(log_rets, axis=0)
    prices = np.exp(log_prices)
    price_vals = prices.ravel()
    panel = pd.DataFrame({"close": price_vals}, index=idx)
    return panel


class TestMomentumSignal:
    def test_returns_dataframe_with_signal_column(self):
        prices = _make_prices_panel()
        sig = momentum_signal(prices)
        assert isinstance(sig, pd.DataFrame)
        assert "signal" in sig.columns

    def test_index_is_date_ticker_multiindex(self):
        prices = _make_prices_panel()
        sig = momentum_signal(prices)
        assert sig.index.names == ["date", "ticker"]

    def test_no_signal_before_lookback_warmup(self):
        """No valid signal until at least lookback+skip days of data."""
        prices = _make_prices_panel(n_dates=300)
        sig = momentum_signal(prices, lookback=252, skip=21)
        sig_nonan = sig.dropna()
        first_valid_date = sig_nonan.index.get_level_values("date").min()
        dates = prices.index.get_level_values("date").unique()
        min_required_date = dates[252 + 21 - 1]
        assert first_valid_date >= min_required_date

    def test_positive_trend_ticker_has_higher_signal(self):
        """Ticker with positive trend should have higher signal than negative trend ticker."""
        prices = _make_prices_panel(n_dates=300, n_tickers=2, trend=0.002)
        sig = momentum_signal(prices)
        sig_nonan = sig.dropna()
        dates = sig_nonan.index.get_level_values("date").unique()
        last_date = dates[-1]
        vals = sig_nonan.xs(last_date, level="date")["signal"]
        assert vals["T0"] > vals["T1"]

    def test_no_future_leakage(self):
        """Signal at date t must not change when we add t+1 data."""
        prices = _make_prices_panel(n_dates=300)
        dates = prices.index.get_level_values("date").unique()
        cutoff = dates[280]
        sig_short = momentum_signal(prices[prices.index.get_level_values("date") <= cutoff])
        sig_full  = momentum_signal(prices)
        short_vals = sig_short.xs(cutoff, level="date")["signal"].dropna()
        full_vals  = sig_full.xs(cutoff, level="date")["signal"].reindex(short_vals.index)
        pd.testing.assert_series_equal(short_vals, full_vals, check_names=False)


class TestVolScale:
    def _make_signal_panel(self, n_dates=50, n_tickers=10, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        sig = rng.standard_normal(n_dates * n_tickers)
        return pd.DataFrame({"signal": sig}, index=idx)

    def _make_sigma_panel(self, n_dates=50, n_tickers=10, seed=1):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        rv = np.abs(rng.normal(0.0001, 0.00002, n_dates * n_tickers))
        return pd.DataFrame({"forecast_rv": rv}, index=idx)

    def test_returns_dataframe_with_weight_column(self):
        sig = self._make_signal_panel()
        sigma = self._make_sigma_panel()
        w = vol_scale(sig, sigma, target_vol=0.10)
        assert isinstance(w, pd.DataFrame)
        assert "weight" in w.columns

    def test_higher_vol_gets_lower_weight(self):
        """A stock with 2x the forecast vol should get ~0.5x the weight (same |signal|)."""
        dates = pd.date_range("2015-01-02", periods=5, freq="B")
        tickers = ["LOW", "HIGH"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        # Distinct signals so cross-sectional std > 0; LOW positive, HIGH negative
        sig = pd.DataFrame({"signal": [1.0, -1.0] * 5}, index=idx)
        # HIGH has 4x the daily variance of LOW
        rv_vals = [0.0001, 0.0004] * 5
        sigma = pd.DataFrame({"forecast_rv": rv_vals}, index=idx)
        w = vol_scale(sig, sigma, target_vol=0.10)
        last_date = dates[-1]
        w_low  = w.xs(last_date, level="date").loc["LOW", "weight"]
        w_high = w.xs(last_date, level="date").loc["HIGH", "weight"]
        # LOW: z=+1, ann_vol≈0.159; HIGH: z=-1, ann_vol≈0.317
        # |w_low|/|w_high| = 0.317/0.159 ≈ 2.0
        assert w_low > 0   # positive z, lower vol → positive weight
        assert w_high < 0  # negative z → negative weight
        assert abs(abs(w_low) / abs(w_high) - 2.0) < 0.01

    def test_output_index_matches_signal_index(self):
        sig = self._make_signal_panel()
        sigma = self._make_sigma_panel()
        w = vol_scale(sig, sigma, target_vol=0.10)
        pd.testing.assert_index_equal(w.index, sig.index)

    def test_zero_std_signal_gives_zero_weights(self):
        """If all signals are identical (std=0), weights should be 0."""
        dates = pd.date_range("2015-01-02", periods=5, freq="B")
        tickers = ["A", "B", "C"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        sig = pd.DataFrame({"signal": 1.0}, index=idx)
        rv_vals = [0.0001] * 15
        sigma = pd.DataFrame({"forecast_rv": rv_vals}, index=idx)
        w = vol_scale(sig, sigma, target_vol=0.10)
        assert (w["weight"] == 0.0).all()


class TestBuildPortfolios:
    def _make_signal_panel(self, n_dates=60, n_tickers=20, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")
        tickers = [f"T{i:02d}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        sig = rng.standard_normal(n_dates * n_tickers)
        return pd.DataFrame({"signal": sig}, index=idx)

    def _make_weights_panel(self, n_dates=60, n_tickers=20, seed=2):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")
        tickers = [f"T{i:02d}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        w = rng.standard_normal(n_dates * n_tickers)
        return pd.DataFrame({"weight": w}, index=idx)

    def test_long_short_quintile_has_weight_column(self):
        sig = self._make_signal_panel()
        result = build_portfolios(sig, mode="long_short_quintile")
        assert "weight" in result.columns

    def test_long_short_quintile_weights_sum_near_zero(self):
        """Long and short legs are equal in magnitude, so sum ≈ 0 per date."""
        sig = self._make_signal_panel()
        result = build_portfolios(sig, mode="long_short_quintile")
        result_nonan = result.dropna()
        dates = result_nonan.index.get_level_values("date").unique()
        for dt in dates[:5]:
            total = result_nonan.xs(dt, level="date")["weight"].sum()
            assert abs(total) < 1e-9, f"Sum of weights at {dt} = {total:.6f}, expected ~0"

    def test_weights_shifted_by_one_day(self):
        """First weight date must be AFTER first signal date (weights are shifted +1)."""
        sig = self._make_signal_panel()
        result = build_portfolios(sig, mode="long_short_quintile")
        sig_dates = sig.index.get_level_values("date").unique()
        result_dates = result.dropna(how="all").index.get_level_values("date").unique()
        assert result_dates[0] > sig_dates[0]

    def test_long_only_quintile_weights_nonnegative(self):
        sig = self._make_signal_panel()
        result = build_portfolios(sig, mode="long_only_quintile")
        assert (result["weight"].dropna() >= 0).all()

    def test_no_close_to_close_leakage(self):
        """Weight using signal at t should be NaN at date t itself (shifted to t+1)."""
        sig = self._make_signal_panel(n_dates=60, n_tickers=20)
        result = build_portfolios(sig, mode="long_short_quintile")
        first_sig_date = sig.index.get_level_values("date").unique()[0]
        first_date_weights = result.xs(first_sig_date, level="date")["weight"]
        assert first_date_weights.isna().all(), \
            "Weights at first signal date should be NaN (signals not yet shifted)"

    def test_conservation_canary(self):
        """sum(long) ≈ -sum(short); sum(|weights|) = 2 * sum(long) for equal legs."""
        sig = self._make_signal_panel(n_dates=60, n_tickers=20)
        result = build_portfolios(sig, mode="long_short_quintile")
        w = result["weight"].unstack("ticker").fillna(0.0)
        for dt, row in w.iterrows():
            long_sum  = row[row > 0].sum()
            short_sum = row[row < 0].sum()
            gross     = row.abs().sum()
            if long_sum == 0:
                continue
            assert abs(long_sum + short_sum) < 1e-9, \
                f"Long/short imbalance at {dt}: long={long_sum:.4f} short={short_sum:.4f}"
            assert abs(gross - 2 * long_sum) < 1e-9, \
                f"Gross != 2*long at {dt}: gross={gross:.4f} 2*long={2*long_sum:.4f}"


class TestApplyCosts:
    def _make_constant_weights(self, n_dates=30, n_tickers=5):
        dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        w = pd.DataFrame({"weight": 0.2}, index=idx)
        return w

    def _make_returns(self, n_dates=30, n_tickers=5, r=0.001):
        dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        return pd.DataFrame({"return": r}, index=idx)

    def test_constant_weights_zero_cost_after_first_day(self):
        """After the first rebalance, constant weights have zero turnover."""
        w = self._make_constant_weights()
        r = self._make_returns()
        net = apply_costs(w, r, cost_bps=10.0)
        # Gross return for constant 0.2 in 5 tickers = 0.001 * 5 * 0.2
        expected_gross = 0.001 * 5 * 0.2
        assert abs(net.iloc[-1] - expected_gross) < 1e-9

    def test_high_turnover_reduces_net_returns(self):
        """Full portfolio flip every day reduces net return vs no-cost."""
        dates = pd.date_range("2015-01-02", periods=20, freq="B")
        tickers = ["A", "B"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        flip_weights = []
        for i, dt in enumerate(dates):
            if i % 2 == 0:
                flip_weights.extend([(dt, "A", 1.0), (dt, "B", 0.0)])
            else:
                flip_weights.extend([(dt, "A", 0.0), (dt, "B", 1.0)])
        w_df = pd.DataFrame(flip_weights, columns=["date", "ticker", "weight"])
        w_df = w_df.set_index(["date", "ticker"])
        r = self._make_returns(n_dates=20, n_tickers=2, r=0.001)
        net_with_cost = apply_costs(w_df, r, cost_bps=10.0)
        net_no_cost   = apply_costs(w_df, r, cost_bps=0.0)
        assert net_with_cost.mean() < net_no_cost.mean()

    def test_returns_series_indexed_by_date(self):
        w = self._make_constant_weights()
        r = self._make_returns()
        net = apply_costs(w, r, cost_bps=10.0)
        assert isinstance(net, pd.Series)
        assert net.index.name == "date"
