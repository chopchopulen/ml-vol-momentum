import pytest
import numpy as np
import pandas as pd


def _make_signal(n_dates=50, n_tickers=20):
    dates = pd.bdate_range("2010-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(3)
    return pd.Series(rng.standard_normal(len(idx)), index=idx, name="signal")


def _make_sector_map(n_tickers=20):
    sectors = ["Tech", "Finance", "Energy", "Health"]
    return {f"T{i}": sectors[i % len(sectors)] for i in range(n_tickers)}


def test_demean_signal_zero_mean_per_sector():
    from src.interp.sector_neutral import demean_signal
    signal = _make_signal()
    sector_map = _make_sector_map()
    result = demean_signal(signal, sector_map)
    panel = result.to_frame("signal")
    panel["sector"] = panel.index.get_level_values("ticker").map(sector_map)
    for (date, sector), grp in panel.groupby(["date", "sector"]):
        assert abs(grp["signal"].mean()) < 1e-10


def test_demean_signal_preserves_index():
    from src.interp.sector_neutral import demean_signal
    signal = _make_signal()
    sector_map = _make_sector_map()
    result = demean_signal(signal, sector_map)
    assert result.index.equals(signal.index)


def test_demean_signal_returns_series():
    from src.interp.sector_neutral import demean_signal
    signal = _make_signal()
    sector_map = _make_sector_map()
    result = demean_signal(signal, sector_map)
    assert isinstance(result, pd.Series)


def test_demean_signal_single_ticker_sector_is_zero():
    from src.interp.sector_neutral import demean_signal
    signal = _make_signal(n_tickers=3)
    sector_map = {"T0": "Tech", "T1": "Finance", "T2": "Energy"}
    result = demean_signal(signal, sector_map)
    t2_vals = result.loc[result.index.get_level_values("ticker") == "T2"]
    assert (t2_vals.abs() < 1e-10).all()
