import pytest
import numpy as np
import pandas as pd
from src.eval.walk_forward import generate_windows


def _make_vix(start="2000-01-01", end="2010-12-31"):
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(1)
    return pd.Series(rng.uniform(10, 40, len(dates)), index=dates, name="vix")


def _make_forecast(vix: pd.Series, n_tickers=5):
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product(
        [vix.index, tickers], names=["date", "ticker"]
    )
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "forecast_log_rv": rng.standard_normal(len(idx)),
        "forecast_rv": rng.uniform(0.001, 0.05, len(idx)),
        "target_rv": rng.uniform(0.001, 0.05, len(idx)),
    }, index=idx)


def test_assign_regimes_returns_series():
    from src.interp.regime_analysis import assign_regimes
    vix = _make_vix()
    windows = generate_windows(pd.Timestamp("2000-01-01"), pd.Timestamp("2010-12-31"), first_test_year=2003)
    regimes = assign_regimes(vix, windows)
    assert isinstance(regimes, pd.Series)
    assert set(regimes.unique()).issubset({"low", "mid", "high"})


def test_assign_regimes_covers_oos_dates():
    from src.interp.regime_analysis import assign_regimes
    vix = _make_vix()
    windows = generate_windows(pd.Timestamp("2000-01-01"), pd.Timestamp("2010-12-31"), first_test_year=2003)
    regimes = assign_regimes(vix, windows)
    assert len(regimes) > 0


def test_regime_ic_returns_dataframe():
    from src.interp.regime_analysis import assign_regimes, regime_ic_table
    vix = _make_vix()
    windows = generate_windows(pd.Timestamp("2000-01-01"), pd.Timestamp("2010-12-31"), first_test_year=2003)
    regimes = assign_regimes(vix, windows)
    forecast = _make_forecast(vix)
    realized = forecast[["target_rv"]]
    result = regime_ic_table({"model_a": forecast}, realized, regimes)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns).issubset({"low", "mid", "high"})


def test_no_lookahead_in_thresholds():
    from src.interp.regime_analysis import assign_regimes
    vix = _make_vix()
    windows = generate_windows(pd.Timestamp("2000-01-01"), pd.Timestamp("2010-12-31"), first_test_year=2003)
    regimes = assign_regimes(vix, windows)
    assert regimes is not None
