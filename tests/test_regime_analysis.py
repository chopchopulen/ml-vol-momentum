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
    windows = generate_windows(
        pd.Timestamp("2000-01-01"), pd.Timestamp("2010-12-31"), first_test_year=2003
    )
    regimes = assign_regimes(vix, windows)
    # Every business day in a test window that has a VIX observation must be in regimes
    for w in windows:
        oos_vix_dates = vix.loc[
            (vix.index >= w.test_start) & (vix.index <= w.test_end)
        ].index
        for d in oos_vix_dates:
            assert d in regimes.index, f"OOS date {d} missing from regimes"


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
    windows = generate_windows(
        pd.Timestamp("2000-01-01"), pd.Timestamp("2005-12-31"), first_test_year=2003
    )
    # Two VIX series: identical training data, identical test-window values for a
    # shared subset, but the extreme series also has additional extreme test values.
    # The shared test dates must receive identical regime labels in both series,
    # because thresholds derive solely from training data.
    dates = pd.bdate_range("2000-01-01", "2005-12-31")
    rng = np.random.default_rng(42)
    train_values = rng.uniform(10, 40, len(dates))
    base = pd.Series(train_values, index=dates, name="vix")

    # Identify test window dates (from first test year onward)
    test_dates = dates[dates >= pd.Timestamp("2003-02-12")]
    # Pick a subset of test dates that will be checked: set them to a fixed mid value
    # in both series so we can assert identical labels.
    shared_test_value = 20.0  # will land somewhere in low/mid/high based on training

    vix_normal = base.copy()
    vix_extreme = base.copy()
    # Both series: shared test dates get the same fixed value
    vix_normal.loc[test_dates] = shared_test_value
    vix_extreme.loc[test_dates] = shared_test_value
    # Extreme series: overwrite the last half of test dates with extreme values
    # This should NOT affect the labels of the first-half dates if there's no look-ahead
    split = len(test_dates) // 2
    first_half = test_dates[:split]
    second_half = test_dates[split:]
    vix_extreme.loc[second_half] = 999.0  # extreme in test window only

    regimes_normal = assign_regimes(vix_normal, windows)
    regimes_extreme = assign_regimes(vix_extreme, windows)

    # The first-half test dates have identical values in both series and must get
    # identical regime labels — proving extreme second-half values didn't shift thresholds
    for d in first_half:
        if d in regimes_normal.index and d in regimes_extreme.index:
            assert regimes_normal[d] == regimes_extreme[d], (
                f"Look-ahead detected: date {d} got different regime labels "
                f"({regimes_normal[d]} vs {regimes_extreme[d]}) despite identical "
                "training data and identical test value at that date"
            )
