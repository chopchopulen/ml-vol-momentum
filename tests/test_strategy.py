import numpy as np
import pandas as pd
import pytest
from src.strategy.momentum import momentum_signal


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
