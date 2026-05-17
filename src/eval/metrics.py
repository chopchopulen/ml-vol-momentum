from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def sharpe(r: pd.Series, ann: int = 252) -> float:
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(ann))

def sortino(r: pd.Series, ann: int = 252) -> float:
    downside = r[r < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(r.mean() / downside.std() * np.sqrt(ann))

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(abs(dd.min()))

def calmar(r: pd.Series, ann: int = 252) -> float:
    ann_ret = float(r.mean() * ann)
    equity = (1 + r).cumprod()
    mdd = max_drawdown(equity)
    if mdd == 0:
        return np.inf
    return ann_ret / mdd

def information_coefficient(signal: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    dates = signal.index.get_level_values("date").unique()
    ic_vals = {}
    for dt in dates:
        try:
            sig = signal.xs(dt, level="date")["signal"]
            ret = returns.xs(dt, level="date")["return"]
            common = sig.index.intersection(ret.index)
            if len(common) < 5:
                continue
            rho, _ = spearmanr(sig[common], ret[common])
            ic_vals[dt] = rho
        except Exception:
            continue
    return pd.Series(ic_vals, name="IC")

def icir(ic: pd.Series) -> float:
    std = ic.std()
    mean = ic.mean()
    # Treat series as constant if std is negligible relative to the mean
    # (handles floating-point noise from np.full / identical values).
    # Return mean * sqrt(n) so that larger-mean constant series ranks higher.
    if std == 0 or (mean != 0 and abs(std / mean) < 1e-10):
        return float(mean * np.sqrt(len(ic)))
    return float(mean / std * np.sqrt(len(ic)))

def turnover(weights: pd.DataFrame) -> pd.Series:
    w = weights["weight"].unstack("ticker")
    diff = w.diff().abs().sum(axis=1)
    return diff.rename("turnover")
