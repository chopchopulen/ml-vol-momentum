from __future__ import annotations
import numpy as np
import pandas as pd


def apply_costs(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.Series:
    """Compute daily net portfolio return after transaction costs.

    cost_bps is round-trip; per-side cost = cost_bps / 2 / 10_000.
    Daily cost = sum(|Δweight|) * per_side_cost.
    """
    cost_per_side = cost_bps / 2.0 / 10_000.0

    w = weights["weight"].unstack("ticker").fillna(0.0)
    r = returns["return"].unstack("ticker").reindex(index=w.index).fillna(0.0)

    gross = (w * r).sum(axis=1)

    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()  # first day: from 0 to initial weights

    cost = turnover * cost_per_side
    net = gross - cost
    net.index.name = "date"
    return net
