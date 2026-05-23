"""Uncertainty-weighted vol scaling.

Extends the standard vol_scale formula with a forecast uncertainty penalty:
  weight_i = (target_vol / ann_vol_i) * (1 / norm_sigma_i) * z_score(signal_i)

where norm_sigma_i = sigma_i / median(sigma) is the normalised uncertainty
(so the penalty is relative, not absolute — avoids regime-level shifts in σ
dominating the cross-sectional ranking).

If forecast_sigma is not present in sigma_hat, falls back to plain vol_scale.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.strategy.scaling import vol_scale


def uncertainty_vol_scale(
    signal: pd.DataFrame,
    sigma_hat: pd.DataFrame,
    target_vol: float,
    uncertainty_col: str = "forecast_sigma",
) -> pd.DataFrame:
    if uncertainty_col not in sigma_hat.columns:
        return vol_scale(signal, sigma_hat, target_vol)

    rows = []
    dates = signal.index.get_level_values("date").unique()

    for dt in dates:
        try:
            sig_dt = signal.xs(dt, level="date")["signal"].dropna()
            sig_std = sig_dt.std()
            if sig_std == 0 or np.isnan(sig_std):
                # All signals identical — use raw values so uncertainty penalty
                # can still differentiate weights across the cross-section.
                z = sig_dt
            else:
                z = (sig_dt - sig_dt.mean()) / sig_std

            fc_dt = sigma_hat.xs(dt, level="date")
            rv_dt = fc_dt["forecast_rv"].reindex(z.index).dropna()
            sig_unc = fc_dt[uncertainty_col].reindex(z.index).dropna()

            common = z.index.intersection(rv_dt.index).intersection(sig_unc.index)
            if len(common) == 0:
                continue
            z = z[common]
            rv_dt = rv_dt[common]
            sig_unc = sig_unc[common]

            ann_vol = np.sqrt(rv_dt * 252).replace(0, np.nan)

            # Normalise uncertainty cross-sectionally by median
            median_unc = sig_unc.median()
            if median_unc <= 0:
                norm_unc = pd.Series(1.0, index=common)
            else:
                norm_unc = sig_unc / median_unc

            # Uncertainty penalty: stocks with norm_unc > 1 get downweighted
            w = (target_vol / ann_vol) * (1.0 / norm_unc) * z
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
