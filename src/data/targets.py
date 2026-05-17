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
    log_rv = np.log(rv_stacked).rename("target_log_rv")
    return pd.concat([rv_stacked, log_rv], axis=1)
