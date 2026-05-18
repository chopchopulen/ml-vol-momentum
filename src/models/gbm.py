from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.config import load_config
from src.models.forecaster import Forecaster

_cfg = load_config()
_GCFG = _cfg["models"]["gbm"]

FEATURE_COLS = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]
CATEGORICAL_COLS = ["sector"]


class GBMForecaster(Forecaster):
    name = "gbm"

    def __init__(self):
        self.booster_: lgb.Booster | None = None
        self.mse_resid_: float = 1.0

    def _prepare_X(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
        """Extract feature matrix and valid-row index from panel."""
        cols = [c for c in FEATURE_COLS + CATEGORICAL_COLS if c in panel.columns]
        X = panel[cols].copy()
        # Encode sector as integer category for LightGBM
        if "sector" in X.columns:
            X["sector"] = X["sector"].astype("category")
        valid = X.dropna(subset=[c for c in FEATURE_COLS if c in X.columns])
        return valid, valid.index

    def fit(self, train: pd.DataFrame) -> None:
        X, idx = self._prepare_X(train)
        y = train.loc[idx, "target_log_rv"].values
        # Drop rows where target is also NaN or infinite
        finite_mask = np.isfinite(y)
        X = X.loc[idx[finite_mask]]
        y = y[finite_mask]

        # Chronological 90/10 split for early stopping — no shuffle
        n = len(y)
        val_frac = _GCFG.get("val_fraction", 0.1)
        n_val = max(int(n * val_frac), 1)
        X_tr, y_tr = X.iloc[: n - n_val], y[: n - n_val]
        X_val, y_val = X.iloc[n - n_val :], y[n - n_val :]

        cat_cols = [c for c in CATEGORICAL_COLS if c in X_tr.columns]
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
        dval   = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain)

        params = {
            "objective":        "regression",
            "metric":           "mse",
            "learning_rate":    _GCFG["learning_rate"],
            "num_leaves":       _GCFG["num_leaves"],
            "max_depth":        _GCFG["max_depth"],
            "min_data_in_leaf": _GCFG["min_data_in_leaf"],
            "feature_fraction": _GCFG["feature_fraction"],
            "bagging_fraction": _GCFG["bagging_fraction"],
            "bagging_freq":     1,
            "verbose":          -1,
            "seed":             _cfg["project"]["seed"],
        }
        callbacks = [lgb.early_stopping(_GCFG["early_stopping_rounds"], verbose=False),
                     lgb.log_evaluation(-1)]
        self.booster_ = lgb.train(
            params,
            dtrain,
            num_boost_round=_GCFG["n_estimators"],
            valid_sets=[dval],
            callbacks=callbacks,
        )
        # Store sigma² = MSE of training residuals for Jensen correction
        train_pred = self.booster_.predict(X_tr)
        self.mse_resid_ = float(np.mean((y_tr - train_pred) ** 2))

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if self.booster_ is None:
            raise RuntimeError("Call fit() before predict()")
        X, idx = self._prepare_X(history)
        log_rv_hat = self.booster_.predict(X)
        rv_hat = np.exp(log_rv_hat + self.mse_resid_ / 2)
        out = pd.DataFrame(
            {"forecast_log_rv": log_rv_hat, "forecast_rv": rv_hat},
            index=idx,
        )
        return out.sort_index()

    def shap_values(self, panel: pd.DataFrame) -> np.ndarray:
        """Return SHAP values for the given panel rows (drops NaN rows first)."""
        import shap
        X, _ = self._prepare_X(panel)
        explainer = shap.TreeExplainer(self.booster_)
        vals = explainer.shap_values(X)
        return np.array(vals) if not isinstance(vals, np.ndarray) else vals
