from __future__ import annotations
import pandas as pd
import numpy as np
from src.eval.walk_forward import CVWindow
from src.eval.tests import cross_sectional_ic


def assign_regimes(
    vix: pd.Series,
    windows: list[CVWindow],
    thresholds: tuple[float, float] = (0.33, 0.67),
) -> pd.Series:
    """Assign each OOS date to a VIX regime (low/mid/high).

    Percentile thresholds computed from the TRAINING window only (no look-ahead).
    Returns a Series indexed by date with values 'low', 'mid', 'high'.
    """
    regime_map: dict[pd.Timestamp, str] = {}
    for w in windows:
        train_vix = vix.loc[
            (vix.index >= w.train_start) & (vix.index <= w.train_end)
        ]
        if train_vix.empty:
            continue
        p_lo = train_vix.quantile(thresholds[0])
        p_hi = train_vix.quantile(thresholds[1])

        test_vix = vix.loc[
            (vix.index >= w.test_start) & (vix.index <= w.test_end)
        ]
        for date, v in test_vix.items():
            if v < p_lo:
                regime_map[date] = "low"
            elif v > p_hi:
                regime_map[date] = "high"
            else:
                regime_map[date] = "mid"

    return pd.Series(regime_map, name="regime").sort_index()


def regime_ic_table(
    forecasts: dict[str, pd.DataFrame],
    realized: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Return DataFrame: rows=models, cols=regimes (low/mid/high), values=mean IC."""
    rows = {}
    for model_name, oos in forecasts.items():
        ic_series = cross_sectional_ic(oos, realized)
        row = {}
        for regime in ("low", "mid", "high"):
            regime_dates = regimes[regimes == regime].index
            ic_subset = ic_series.reindex(regime_dates).dropna()
            row[regime] = ic_subset.mean() if len(ic_subset) > 0 else float("nan")
        rows[model_name] = row
    return pd.DataFrame(rows).T[["low", "mid", "high"]]
