from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd


def _quintile_weights(signal_date: pd.Series, mode: str) -> pd.Series:
    """Compute equal-weight quintile portfolio for a single date's signal."""
    sig = signal_date.dropna()
    if len(sig) < 10:
        return pd.Series(dtype=float)
    q20 = sig.quantile(0.20)
    q80 = sig.quantile(0.80)
    if mode == "long_short_quintile":
        long_mask  = sig >= q80
        short_mask = sig <= q20
        n_long  = long_mask.sum()
        n_short = short_mask.sum()
        if n_long == 0 or n_short == 0:
            return pd.Series(0.0, index=sig.index)
        w = pd.Series(0.0, index=sig.index)
        w[long_mask]  =  1.0 / n_long
        w[short_mask] = -1.0 / n_short
        return w
    elif mode == "long_only_quintile":
        long_mask = sig >= q80
        n_long = long_mask.sum()
        if n_long == 0:
            return pd.Series(0.0, index=sig.index)
        w = pd.Series(0.0, index=sig.index)
        w[long_mask] = 1.0 / n_long
        return w
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_portfolios(
    signal: pd.DataFrame | None,
    weights: pd.DataFrame | None = None,
    mode: Literal["long_short_quintile", "long_only_quintile",
                  "vol_targeted_gross"] = "long_short_quintile",
) -> pd.DataFrame:
    """
    Build portfolio weight panel from signal (and optional pre-scaled weights).

    PIT: weights are SHIFTED +1 trading day before return attribution.
    Signal at date t produces weights held from open of t+1.
    """
    if mode in ("long_short_quintile", "long_only_quintile"):
        dates = signal.index.get_level_values("date").unique()
        rows = []
        for dt in dates:
            sig_dt = signal.xs(dt, level="date")["signal"]
            w = _quintile_weights(sig_dt, mode)
            if w.empty:
                continue
            sub = w.rename("weight").to_frame()
            sub.index = pd.MultiIndex.from_arrays(
                [[dt] * len(sub), sub.index], names=["date", "ticker"])
            rows.append(sub)
        if not rows:
            return pd.DataFrame(columns=["weight"])
        raw = pd.concat(rows).sort_index()
    elif mode == "vol_targeted_gross":
        if weights is None:
            raise ValueError("vol_targeted_gross mode requires pre-scaled weights")
        raw = weights.copy()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # PIT shift: weight computed at t becomes active at t+1
    w_wide = raw["weight"].unstack("ticker")
    w_shifted = w_wide.shift(1)
    result = w_shifted.stack(future_stack=True).rename("weight").to_frame()
    return result
