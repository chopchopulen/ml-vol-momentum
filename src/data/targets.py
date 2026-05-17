from __future__ import annotations
import numpy as np
import pandas as pd

def forward_rv(panel: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    r = panel["return"].unstack("ticker")
    # rolling(h).sum() gives sum_{s=t-h+1}^{t} r_s^2
    # shift(-h) puts the sum of the NEXT h days at index t
    rv = (r ** 2).rolling(horizon).sum().shift(-horizon)
    rv_stacked = rv.stack(future_stack=True)
    rv_stacked.name = "target_rv"
    # Zero forward-RV means the stock had zero returns for all 21 future days
    # (halted trading or stale-price data). Drop rather than clip: log(0) = -inf
    # propagates silently through R² calculations, producing artifactual R²=1.0.
    # Drop rows where forward-RV is exactly zero: those stocks had zero returns
    # for all 21 future days (halted trading / stale-price data). log(0) = -inf
    # propagates silently through R² calculations producing artifactual R²=1.0.
    # Keep NaN rows (they mark the trailing horizon window with no future data).
    rv_stacked = rv_stacked[(rv_stacked > 0) | rv_stacked.isna()]
    log_rv = np.log(rv_stacked).rename("target_log_rv")
    return pd.concat([rv_stacked, log_rv], axis=1)
