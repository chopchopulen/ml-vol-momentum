from __future__ import annotations
import numpy as np
import pandas as pd


def compute_shap_importance(
    model,
    panel: pd.DataFrame,
    sample_size: int = 5000,
) -> pd.Series:
    """Return mean |SHAP| per feature, sorted descending.

    model must be a fitted GBMForecaster with .shap_values(panel) and ._prepare_X(panel).
    Uses up to sample_size rows sampled from panel to keep runtime under control.
    """
    sub = panel
    if len(sub) > sample_size:
        sub = sub.sample(sample_size, random_state=42)

    X, _ = model._prepare_X(sub)
    feat_names = list(X.columns)

    shap_vals = model.shap_values(sub)          # (n_rows, n_features)
    mean_abs = np.abs(shap_vals).mean(axis=0)   # (n_features,)
    result = pd.Series(mean_abs, index=feat_names)
    return result.sort_values(ascending=False)
