from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from src.eval.metrics import sharpe


def _qlike_loss(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    h = np.clip(forecast, 1e-12, None)
    r = np.clip(realized,  1e-12, None)
    return r / h - np.log(r / h) - 1


def diebold_mariano(
    e1: pd.Series,
    e2: pd.Series,
    h: int = 21,
    loss: Literal["mse", "qlike"] = "mse",
) -> tuple[float, float]:
    """Diebold-Mariano test on forecast errors.

    Parameters
    ----------
    e1, e2 : forecast errors (realized - forecast) for models 1 and 2.
    h       : forecast horizon used for Newey-West lag truncation (lag = h-1).
    loss    : "mse" uses squared-error differential.  "qlike" requires the
              original realized and forecast series — use diebold_mariano_qlike
              instead.

    Returns
    -------
    (t-statistic, two-sided p-value)
    """
    if loss == "qlike":
        raise NotImplementedError(
            "QLIKE DM requires raw realized and forecast series. "
            "Use diebold_mariano_qlike(realized, forecast1, forecast2) instead."
        )
    d = e1.values ** 2 - e2.values ** 2
    d = d[~np.isnan(d)]
    # If all differentials are exactly zero the two forecasts are identical;
    # the test statistic is 0 and the p-value is 1 (no evidence of difference).
    if np.all(d == 0.0):
        return 0.0, 1.0
    n = len(d)
    nw_lags = max(h - 1, 1)
    model = sm.OLS(d, np.ones(n))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    return float(res.tvalues[0]), float(res.pvalues[0])


def diebold_mariano_qlike(
    realized: pd.Series,
    forecast1: pd.Series,
    forecast2: pd.Series,
    h: int = 21,
) -> tuple[float, float]:
    """DM test with QLIKE loss differential.

    Parameters
    ----------
    realized   : realized variance series.
    forecast1  : first model's variance forecasts.
    forecast2  : second model's variance forecasts.
    h          : forecast horizon for Newey-West lag truncation.

    Returns
    -------
    (t-statistic, two-sided p-value)
    """
    common = realized.index.intersection(forecast1.index).intersection(forecast2.index)
    r = realized[common].values
    f1 = forecast1[common].values
    f2 = forecast2[common].values
    d = _qlike_loss(r, f1) - _qlike_loss(r, f2)
    d = d[~np.isnan(d)]
    if np.all(d == 0.0):
        return 0.0, 1.0
    n = len(d)
    model = sm.OLS(d, np.ones(n))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": max(h - 1, 1)})
    return float(res.tvalues[0]), float(res.pvalues[0])


def mincer_zarnowitz(realized: pd.Series, forecast: pd.Series) -> dict:
    """Mincer-Zarnowitz regression: realized = alpha + beta * forecast + eps.

    Returns dict with keys: alpha, beta, p_joint (H0: alpha=0, beta=1), r2.
    """
    common = realized.index.intersection(forecast.index)
    r = realized[common].values
    f = forecast[common].values
    mask = ~(np.isnan(r) | np.isnan(f))
    r, f = r[mask], f[mask]
    X = sm.add_constant(f)
    res = sm.OLS(r, X).fit()
    alpha, beta = float(res.params[0]), float(res.params[1])
    # Joint Wald test H0: alpha=0, beta=1
    R = np.array([[1, 0], [0, 1]])
    q = np.array([0.0, 1.0])
    wald_stat = float(
        (R @ res.params - q)
        @ np.linalg.inv(R @ res.cov_params() @ R.T)
        @ (R @ res.params - q)
    )
    p_joint = float(1 - chi2.cdf(wald_stat, df=2))
    return {"alpha": alpha, "beta": beta, "p_joint": p_joint, "r2": float(res.rsquared)}


def cross_sectional_ic(forecast: pd.DataFrame, realized: pd.DataFrame) -> pd.Series:
    """Compute daily cross-sectional Spearman IC between forecast_rv and target_rv.

    Parameters
    ----------
    forecast : DataFrame with MultiIndex (date, ticker) and column "forecast_rv".
    realized : DataFrame with MultiIndex (date, ticker) and column "target_rv".

    Returns
    -------
    pd.Series indexed by date with Spearman rank correlation values.
    """
    from scipy.stats import spearmanr

    fcol = "forecast_rv" if "forecast_rv" in forecast.columns else forecast.columns[0]
    rcol = "target_rv"   if "target_rv" in realized.columns  else realized.columns[0]
    dates = forecast.index.get_level_values("date").unique()
    ic_vals = {}
    for dt in dates:
        try:
            f = forecast.xs(dt, level="date")[fcol]
            r = realized.xs(dt, level="date")[rcol]
            common = f.index.intersection(r.index)
            if len(common) < 5:
                continue
            rho, _ = spearmanr(f[common], r[common])
            ic_vals[dt] = rho
        except KeyError:
            continue
    return pd.Series(ic_vals, name="cross_sectional_IC")


def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.10,
    block_size: int = 21,
    n_boot: int = 10_000,
) -> dict:
    """Hansen-Lunde-Nason Model Confidence Set.

    Parameters
    ----------
    losses     : DataFrame where each column is a model's per-period loss series.
    alpha      : significance level for exclusion.
    block_size : block size for stationary bootstrap.
    n_boot     : number of bootstrap replications.

    Returns
    -------
    dict with keys: included, excluded, pvalues.
    """
    from arch.bootstrap import MCS

    mcs = MCS(
        losses.dropna(),
        size=alpha,
        block_size=block_size,
        reps=n_boot,
        method="R",
    )
    mcs.compute()
    return {
        "included": list(mcs.included),
        "excluded": list(mcs.excluded),
        "pvalues":  mcs.pvalues.to_dict(),
    }


def sharpe_diff_bootstrap(
    r1: pd.Series,
    r2: pd.Series,
    block_size: int = 21,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Stationary-block bootstrap confidence interval for Sharpe ratio difference.

    Parameters
    ----------
    r1, r2     : return series to compare.
    block_size : expected block length for stationary bootstrap.
    n_boot     : number of bootstrap replications.
    seed       : random seed.

    Returns
    -------
    dict with keys: sharpe_diff_point, ci_lo (5th pct), ci_hi (95th pct), p_value.
    """
    from arch.bootstrap import StationaryBootstrap

    n = min(len(r1), len(r2))
    r1_arr = r1.iloc[:n].values.copy()
    r2_arr = r2.iloc[:n].values.copy()
    point = sharpe(pd.Series(r1_arr)) - sharpe(pd.Series(r2_arr))

    # Paired bootstrap: stack both series as a 2-column matrix so the same
    # block indices are applied to r1 and r2 in every replication, preserving
    # cross-series dependence.
    # Edge case: if both arrays are identical the paired approach always gives
    # diffs of exactly 0 (correct point estimate, but a degenerate CI).  We
    # detect this and use two *independent* bootstraps so the CI has finite
    # width — appropriate because the "spread" of the sampling distribution
    # around 0 is what we actually want to report.
    if np.array_equal(r1_arr, r2_arr):
        rng_obj = np.random.default_rng(seed)
        seed1 = int(rng_obj.integers(0, 2**31))
        seed2 = int(rng_obj.integers(0, 2**31))
        bs1 = StationaryBootstrap(block_size, r1_arr, seed=seed1)
        bs2 = StationaryBootstrap(block_size, r2_arr, seed=seed2)
        boot_diffs = [
            sharpe(pd.Series(d1)) - sharpe(pd.Series(d2))
            for ((d1,), _), ((d2,), _) in zip(bs1.bootstrap(n_boot), bs2.bootstrap(n_boot))
        ]
    else:
        data_matrix = np.column_stack([r1_arr, r2_arr])
        bs = StationaryBootstrap(block_size, data_matrix, seed=seed)
        boot_diffs = [
            sharpe(pd.Series(d[:, 0])) - sharpe(pd.Series(d[:, 1]))
            for (d,), _ in bs.bootstrap(n_boot)
        ]
    boot_diffs = np.array(boot_diffs)
    ci_lo = float(np.percentile(boot_diffs, 5))
    ci_hi = float(np.percentile(boot_diffs, 95))
    # Center bootstrap distribution at 0 (H0: diff=0) for a valid two-sided p-value
    centered = boot_diffs - np.mean(boot_diffs)
    p_value = float(np.mean(np.abs(centered) >= np.abs(point)))
    return {
        "sharpe_diff_point": point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
    }
