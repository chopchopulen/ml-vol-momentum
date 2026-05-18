from __future__ import annotations
import math
import numpy as np
import pandas as pd

def realized_variance(returns: pd.Series, window: int) -> pd.Series:
    rv = (returns ** 2).rolling(window).sum()
    return rv.shift(1)

def parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    # low=0 (bad data) → log(inf) = inf. Replace with NaN.
    hl_ratio = (high / low).where(low > 0)
    log_hl_sq = (np.log(hl_ratio)) ** 2
    pk = log_hl_sq.rolling(window).mean() / (4 * math.log(2))
    return pk.shift(1)

def build_feature_panel(ohlcv: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    results = []
    for ticker, grp in ohlcv.groupby(level="ticker"):
        grp = grp.droplevel("ticker")
        close = grp["close"]
        r = np.log(close / close.shift(1))

        rv_d  = realized_variance(r, 1)
        rv_w  = realized_variance(r, 5)
        rv_m  = realized_variance(r, 21)
        pk    = parkinson(grp["high"], grp["low"], window=5)
        skew  = r.rolling(63).skew().shift(1)
        kurt  = r.rolling(63).kurt().shift(1)
        vix_s = vix.reindex(close.index).shift(1)
        # volume=0 on halted days → log(0)=-inf. Replace with NaN so
        # downstream dropna() cleans these rows rather than propagating -inf.
        dv = grp["volume"] * close
        log_dv = dv.where(dv > 0).apply(np.log).shift(1)
        ret_21 = r.rolling(21).sum().shift(1)

        feat = pd.DataFrame({
            "rv_d": rv_d,
            "rv_w": rv_w,
            "rv_m": rv_m,
            "pk":   pk,
            "skew": skew,
            "kurt": kurt,
            "vix":  vix_s,
            "log_dv": log_dv,
            "ret_21": ret_21,
        })
        feat.index = pd.MultiIndex.from_product(
            [[ticker], feat.index], names=["ticker", "date"])
        feat = feat.swaplevel().sort_index()
        results.append(feat)
    return pd.concat(results).sort_index()
