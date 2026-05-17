from __future__ import annotations
import numpy as np
import pandas as pd
from src.eval.metrics import sharpe, sortino, max_drawdown, calmar
from src.eval.tests import diebold_mariano_qlike, mincer_zarnowitz


def build_results_table(
    strategies: dict[str, pd.Series],
    ann: int = 252,
) -> pd.DataFrame:
    """Build a summary performance table for a set of return series.

    Parameters
    ----------
    strategies : mapping from strategy name to daily return pd.Series.
    ann        : annualisation factor (default 252 trading days).

    Returns
    -------
    DataFrame indexed by strategy name with columns:
    sharpe, sortino, max_dd, calmar, ann_ret, ann_vol.
    """
    rows = []
    for name, r in strategies.items():
        equity = (1 + r).cumprod()
        rows.append({
            "strategy": name,
            "sharpe":   sharpe(r, ann),
            "sortino":  sortino(r, ann),
            "max_dd":   -max_drawdown(equity),
            "calmar":   calmar(r, ann),
            "ann_ret":  r.mean() * ann,
            "ann_vol":  r.std() * np.sqrt(ann),
        })
    return pd.DataFrame(rows).set_index("strategy")


def build_dm_matrix(
    forecasts: dict[str, pd.Series],
    realized: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise DM-QLIKE test statistics and p-values.

    Entry (i, j) tests whether model i significantly outperforms model j
    (negative t-stat ⟹ model i has lower QLIKE loss).

    Returns
    -------
    (stats_df, pvals_df) — both indexed and columned by model name.
    """
    names = list(forecasts.keys())
    stats = pd.DataFrame(index=names, columns=names, dtype=float)
    pvals = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i == j:
                stats.loc[n1, n2] = 0.0
                pvals.loc[n1, n2] = 1.0
                continue
            s, p = diebold_mariano_qlike(realized, forecasts[n1], forecasts[n2])
            stats.loc[n1, n2] = s
            pvals.loc[n1, n2] = p
    return stats, pvals


def build_mz_table(
    forecasts: dict[str, pd.Series],
    realized: pd.Series,
) -> pd.DataFrame:
    """Run Mincer-Zarnowitz regression for each model and return summary table.

    Returns
    -------
    DataFrame indexed by model name with columns: alpha, beta, p_joint, r2.
    """
    rows = []
    for name, fc in forecasts.items():
        result = mincer_zarnowitz(realized, fc)
        result["model"] = name
        rows.append(result)
    return pd.DataFrame(rows).set_index("model")
