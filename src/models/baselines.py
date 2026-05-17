from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from joblib import Parallel, delayed
from src.config import load_config

_cfg = load_config()


class RollingVolModel:
    name = "rolling_vol"

    def __init__(self, window: int | None = None):
        self.window = window or _cfg["models"]["rolling_vol"]["window"]

    def fit(self, train: pd.DataFrame) -> None:
        pass  # window-based, no fitting

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for tkr, grp in history.groupby(level="ticker"):
            r = grp.droplevel("ticker")["return"]
            rv = (r ** 2).rolling(self.window).sum()
            sub = pd.DataFrame({"forecast_rv": rv, "forecast_log_rv": np.log(rv)})
            sub.index = pd.MultiIndex.from_arrays(
                [sub.index, [tkr] * len(sub)], names=["date", "ticker"])
            rows.append(sub)
        return pd.concat(rows).sort_index()


class HARRV:
    name = "har_rv"

    def __init__(self):
        self.coef_: dict[str, dict] = {}

    def _fit_ticker(self, tkr: str, df: pd.DataFrame) -> tuple[str, dict]:
        sub = df.loc[df.index.get_level_values("ticker") == tkr].copy()
        sub = sub[["rv_d", "rv_w", "rv_m", "target_log_rv"]].dropna()
        if len(sub) < 30:
            return tkr, {}
        log_rv_d = np.log(sub["rv_d"].clip(lower=1e-12))
        log_rv_w = np.log(sub["rv_w"].clip(lower=1e-12))
        log_rv_m = np.log(sub["rv_m"].clip(lower=1e-12))
        X = sm.add_constant(
            np.column_stack([log_rv_d, log_rv_w, log_rv_m]))
        y = sub["target_log_rv"].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(y, X).fit()
        return tkr, {
            "alpha":  res.params[0],
            "beta_d": res.params[1],
            "beta_w": res.params[2],
            "beta_m": res.params[3],
            "sigma2": res.mse_resid,
        }

    def fit(self, train: pd.DataFrame) -> None:
        tickers = train.index.get_level_values("ticker").unique()
        results = [self._fit_ticker(t, train) for t in tickers]
        self.coef_ = {t: c for t, c in results if c}

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for tkr, coef in self.coef_.items():
            try:
                sub = history.loc[
                    history.index.get_level_values("ticker") == tkr
                ]
                r = sub.droplevel("ticker")["return"]
                rv_d = (r**2).shift(1)
                rv_w = (r**2).rolling(5).sum().shift(1)
                rv_m = (r**2).rolling(21).sum().shift(1)
                log_rv_hat = (
                    coef["alpha"]
                    + coef["beta_d"] * np.log(rv_d.clip(lower=1e-12))
                    + coef["beta_w"] * np.log(rv_w.clip(lower=1e-12))
                    + coef["beta_m"] * np.log(rv_m.clip(lower=1e-12))
                )
                # Jensen correction: E[RV] = exp(mu + sigma^2/2)
                rv_hat = np.exp(log_rv_hat + coef["sigma2"] / 2)
                df_out = pd.DataFrame({
                    "forecast_log_rv": log_rv_hat,
                    "forecast_rv": rv_hat,
                })
                df_out.index = pd.MultiIndex.from_product(
                    [df_out.index, [tkr]], names=["date", "ticker"])
                df_out = df_out.swaplevel().sort_index()
                rows.append(df_out)
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows).sort_index()


def _fit_one_garch(tkr: str, r: pd.Series, dist: str,
                   rescale: bool) -> tuple[str, object, bool]:
    try:
        data = r * 100 if rescale else r
        am = arch_model(data.dropna(), vol="Garch", p=1, q=1,
                        dist=dist, rescale=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)
        converged = res.convergence_flag == 0
        return tkr, res, converged
    except Exception:
        return tkr, None, False


class GARCH11Model:
    name = "garch"

    def __init__(self):
        gcfg = _cfg["models"]["garch"]
        self.dist = gcfg["dist"]
        self.rescale = gcfg["rescale"]
        self.n_jobs = gcfg["n_jobs"]
        self.fitted_: dict[str, object] = {}
        self.convergence_log_: dict[str, bool] = {}

    def fit(self, train: pd.DataFrame) -> None:
        tickers = train.index.get_level_values("ticker").unique().tolist()
        series = [
            train.xs(t, level="ticker")["return"] for t in tickers
        ]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_garch)(t, r, self.dist, self.rescale)
            for t, r in zip(tickers, series)
        )
        for tkr, res, conv in results:
            self.convergence_log_[tkr] = conv
            if res is not None:
                self.fitted_[tkr] = res

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for tkr, res in self.fitted_.items():
            try:
                r = history.xs(tkr, level="ticker")["return"]
                data = r * 100 if self.rescale else r
                # Re-forecast from the last fit (fixed params, rolling update)
                fcast = res.forecast(horizon=21, reindex=False)
                # fcast.variance has shape (n_obs, 21); sum across horizon
                cond_var_sum = fcast.variance.sum(axis=1)
                if self.rescale:
                    cond_var_sum = cond_var_sum / (100 ** 2)
                df_out = pd.DataFrame({"forecast_rv": cond_var_sum,
                                       "forecast_log_rv": np.log(cond_var_sum)})
                df_out.index = pd.MultiIndex.from_product(
                    [df_out.index, [tkr]], names=["date", "ticker"])
                df_out = df_out.swaplevel().sort_index()
                rows.append(df_out)
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows).sort_index()
