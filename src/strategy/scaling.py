from __future__ import annotations
import numpy as np
import pandas as pd

# PIT CONVENTION (enforced throughout this module):
#   sigma_hat_{i,t} is the forecast produced from data through CLOSE of t.
#   The resulting weight is held from open t+1 onward.
#   Signal and forecast use the SAME timestamp t; portfolio.py applies the
#   +1 shift before computing returns. Never use t+1 vol to scale weight at t.


def vol_scale(
    signal: pd.DataFrame,
    sigma_hat: pd.DataFrame,
    target_vol: float,
) -> pd.DataFrame:
    """Cross-sectionally scale signal by inverse forecast vol to target_vol.

    PIT: sigma_hat and signal share timestamp t. portfolio.py shifts +1 day.
    """
    rows = []
    dates = signal.index.get_level_values("date").unique()
    for dt in dates:
        try:
            sig_dt = signal.xs(dt, level="date")["signal"].dropna()
            sig_std = sig_dt.std()
            if sig_std == 0 or np.isnan(sig_std):
                z = pd.Series(0.0, index=sig_dt.index)
            else:
                z = (sig_dt - sig_dt.mean()) / sig_std
            rv_dt = sigma_hat.xs(dt, level="date")["forecast_rv"].reindex(z.index).dropna()
            common = z.index.intersection(rv_dt.index)
            z = z[common]
            rv_dt = rv_dt[common]
            ann_vol = np.sqrt(rv_dt * 252)
            ann_vol = ann_vol.replace(0, np.nan)
            w = (target_vol / ann_vol) * z
            w = w.fillna(0.0)
            sub = w.rename("weight").to_frame()
            sub.index = pd.MultiIndex.from_arrays(
                [[dt] * len(sub), sub.index], names=["date", "ticker"])
            rows.append(sub)
        except KeyError:
            continue
    if not rows:
        return pd.DataFrame(columns=["weight"])
    return pd.concat(rows).sort_index()
