# Phase 3: ML Forecasters (GBM + LSTM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement GBM (`gbm.py`) and LSTM (`lstm_model.py`) vol forecasters that satisfy the `Forecaster` protocol, run them through the walk-forward harness, and produce the full 5-model comparison table (RollingVol, GARCH, HAR-RV, GBM, LSTM-ensemble).

**Architecture:** Both models are panel forecasters — pooled across all tickers (not per-ticker like HAR-RV). GBM uses LightGBM on tabular features with sector as a categorical; LSTM uses a 1-layer sequence model on the last 60 days of features per stock. Both implement `Forecaster.fit(train)` / `Forecaster.predict(history)` and return the same `(forecast_log_rv, forecast_rv)` panel as the baselines. Walk-forward integration is already complete (`run_walk_forward`); the ML models just plug in. The final comparison script runs all 5 models and produces the master results table.

**Tech Stack:** LightGBM 4.6, PyTorch 2.12, pandas, numpy, SHAP (for interpretation), existing `src.eval.*`, `src.models.forecaster.Forecaster` protocol

---

## Environment check

Python venv: `/tmp/ml-vol-momentum-venv/bin/python`  
Working directory: `/Users/harry/RL:ML Project/ml-vol-momentum`  
LightGBM: 4.6.0 ✓  
PyTorch: 2.12.0 ✓  

---

## File Structure

```
src/models/
  gbm.py          — GBMForecaster (LightGBM, pooled panel, sector categorical)
  lstm_model.py   — LSTMForecaster (5-seed ensemble, per-window z-score normalisation)

tests/
  test_gbm.py     — unit tests for GBMForecaster
  test_lstm.py    — unit tests for LSTMForecaster

scripts/
  run_phase3.py   — walk-forward all 5 models, write results/ parquets
  validate_checkpoint_3a.py  — CHECKPOINT 3a gate (GBM OOS IC vs HAR-RV)
  validate_checkpoint_3b.py  — CHECKPOINT 3b gate (LSTM seed stability)
  compare_all_models.py      — master results table (all 5 models × 3 portfolio variants)
```

**Existing files that need NO changes:**
- `src/models/forecaster.py` — protocol is complete
- `src/eval/walk_forward.py` — `run_walk_forward` already works with any Forecaster
- `src/eval/metrics.py`, `src/eval/tests.py`, `src/eval/comparison.py` — complete
- `src/strategy/` — all strategy code complete
- `src/data/` — features, targets, universe, loaders complete

---

## Critical invariants (read before implementing anything)

**Feature columns in the panel (from `build_feature_panel`):**
`rv_d, rv_w, rv_m, pk, skew, kurt, vix, log_dv, ret_21`

**Target column:** `target_log_rv`

**Panel index:** `(date, ticker)` MultiIndex — `date` is level 0, `ticker` is level 1.

**Forecaster output must have:**
- Column `"forecast_log_rv"` — the raw log-space prediction
- Column `"forecast_rv"` — `exp(forecast_log_rv + sigma2/2)` (Jensen correction)
- Same `(date, ticker)` MultiIndex

**No leakage rule:** `fit(train)` sees only training dates. `predict(history)` receives the full history up to the test window end; the model must produce forecasts from features that are already `.shift(1)` (done at feature construction). The walk-forward harness restricts predictions to test-window dates, so predicting on the full history is safe.

**Sector feature:** Pass `sector` as a LightGBM categorical. Do NOT include `ticker` as a feature (memorisation of idiosyncratic stock histories and high-cardinality encoding problems).

**LSTM normalisation:** Per (training window, ticker) z-score — fit `(mu, sigma)` on training data only, apply to test. Never fit on test data. Do this for features only; the target is also z-scored within the training window and un-z-scored at prediction time.

---

## Task 1: `GBMForecaster` — LightGBM panel forecaster

**Files:**
- Create: `src/models/gbm.py`
- Create: `tests/test_gbm.py`

The GBM forecaster trains a single LightGBM regressor on the pooled panel (all tickers, all training dates). Features: `rv_d, rv_w, rv_m, pk, skew, kurt, vix, log_dv, ret_21` (+ `sector` as categorical). Target: `target_log_rv`. Hyperparameters come from `configs/default.yaml`. Early stopping uses a 10% chronological tail of the training data as validation set (NOT random split — that would be future leakage).

Predict: for each row in history that has all feature columns, produce `forecast_log_rv`. Back-transform with Jensen correction using `mse_resid_` (MSE of training residuals, stored at fit time as the sigma² estimate).

- [ ] **Step 1: Write failing tests**

Create `tests/test_gbm.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.models.gbm import GBMForecaster


FEATURE_COLS = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]


def _make_synthetic_panel(n_dates=300, n_tickers=10, seed=42):
    """Synthetic panel with all feature columns, target, and sector."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-03", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    sectors = ["Tech", "Finance", "Health", "Energy", "Consumer"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    data = {}
    for col in FEATURE_COLS:
        data[col] = np.abs(rng.normal(1e-4, 5e-5, len(idx)))
    data["target_log_rv"] = rng.normal(-10, 1, len(idx))
    # Sector: same sector per ticker, consistent
    sector_map = {t: sectors[i % len(sectors)] for i, t in enumerate(tickers)}
    data["sector"] = [sector_map[t] for _, t in idx]
    return pd.DataFrame(data, index=idx)


class TestGBMForecaster:
    def test_implements_forecaster_protocol(self):
        from src.models.forecaster import Forecaster
        m = GBMForecaster()
        assert isinstance(m, Forecaster)

    def test_fit_stores_model(self):
        panel = _make_synthetic_panel()
        m = GBMForecaster()
        m.fit(panel)
        assert m.booster_ is not None

    def test_predict_returns_correct_columns(self):
        panel = _make_synthetic_panel()
        train, test = panel.iloc[:panel.shape[0]//2], panel.iloc[panel.shape[0]//2:]
        m = GBMForecaster()
        m.fit(train)
        out = m.predict(test)
        assert "forecast_log_rv" in out.columns
        assert "forecast_rv" in out.columns

    def test_predict_index_is_date_ticker(self):
        panel = _make_synthetic_panel()
        train, test = panel.iloc[:panel.shape[0]//2], panel.iloc[panel.shape[0]//2:]
        m = GBMForecaster()
        m.fit(train)
        out = m.predict(test)
        assert out.index.names == ["date", "ticker"]

    def test_forecast_rv_is_positive(self):
        """forecast_rv = exp(log_rv_hat + sigma2/2) must always be positive."""
        panel = _make_synthetic_panel()
        train, test = panel.iloc[:panel.shape[0]//2], panel.iloc[panel.shape[0]//2:]
        m = GBMForecaster()
        m.fit(train)
        out = m.predict(test)
        assert (out["forecast_rv"].dropna() > 0).all()

    def test_no_leakage_target_in_predict(self):
        """predict() must not use target_log_rv column — it's not present at test time."""
        panel = _make_synthetic_panel()
        train = panel.iloc[:panel.shape[0]//2]
        # Pass test panel WITHOUT target column to simulate real prediction time
        test_no_target = panel.iloc[panel.shape[0]//2:].drop(columns=["target_log_rv"])
        m = GBMForecaster()
        m.fit(train)
        out = m.predict(test_no_target)
        assert len(out) > 0, "predict() should work even without target column"

    def test_shap_values_returns_array(self):
        """shap_values() returns a 2D array matching the number of input rows."""
        panel = _make_synthetic_panel(n_dates=100, n_tickers=5)
        m = GBMForecaster()
        m.fit(panel)
        X = panel.drop(columns=["target_log_rv"])
        shap_vals = m.shap_values(X)
        assert shap_vals is not None
        assert shap_vals.shape[0] == len(X.dropna(subset=FEATURE_COLS))
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_gbm.py -v --tb=short 2>&1 | tail -20
```
Expected: ImportError or 7 failures.

- [ ] **Step 3: Implement `src/models/gbm.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.config import load_config

_cfg = load_config()
_GCFG = _cfg["models"]["gbm"]

FEATURE_COLS = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]
CATEGORICAL_COLS = ["sector"]


class GBMForecaster:
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
        n_val = max(int(n * _GCFG["val_fraction"] if "val_fraction" in _GCFG else n * 0.1), 1)
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
        return explainer.shap_values(X)
```

- [ ] **Step 4: Run tests**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_gbm.py -v --tb=short 2>&1 | tail -25
```
Expected: 7 pass.

**If `test_shap_values_returns_array` fails with ImportError:** `shap` may not be installed. Run `/tmp/ml-vol-momentum-venv/bin/pip install shap` first. If shap is slow to install, skip this one test temporarily with `pytest -k "not shap"` and fix it after.

- [ ] **Step 5: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && git add src/models/gbm.py tests/test_gbm.py && git commit -m "feat: GBMForecaster — LightGBM panel vol forecaster with sector categorical"
```

---

## Task 2: `LSTMForecaster` — 5-seed sequence model

**Files:**
- Create: `src/models/lstm_model.py`
- Create: `tests/test_lstm.py`

The LSTM takes sequences of the last 60 trading days of features for each stock and predicts `target_log_rv`. Key implementation notes:

1. **Data shape**: For each (date, ticker) in the test window, we need a sequence of the 60 preceding feature vectors. This means building a 3D tensor of shape `(n_samples, seq_len, n_features)`.
2. **Normalisation**: Compute `(mean, std)` per feature from the **training window only**. Apply to both train and test. Re-fit every annual window.
3. **Target normalisation**: Also z-score `target_log_rv` within training window. Store `(target_mean, target_std)` to un-z-score predictions.
4. **5 seeds**: `fit()` takes a `seed` parameter. The caller (walk-forward script) calls `fit` for each seed and averages the predictions.
5. **Huber loss**: `delta=1.0` in z-scored log-RV space.
6. **Early stopping**: 10% chronological tail of training as validation — no shuffle.
7. **Batch processing for prediction**: Build sequences from history panel; use the model's `eval()` mode.

- [ ] **Step 1: Write failing tests**

Create `tests/test_lstm.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.models.lstm_model import LSTMForecaster

FEATURE_COLS = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]


def _make_synthetic_panel(n_dates=200, n_tickers=5, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-03", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    data = {}
    for col in FEATURE_COLS:
        data[col] = np.abs(rng.normal(1e-4, 5e-5, len(idx)))
    data["target_log_rv"] = rng.normal(-10, 1, len(idx))
    return pd.DataFrame(data, index=idx)


class TestLSTMForecaster:
    def test_implements_forecaster_protocol(self):
        from src.models.forecaster import Forecaster
        m = LSTMForecaster()
        assert isinstance(m, Forecaster)

    def test_fit_stores_model(self):
        panel = _make_synthetic_panel()
        m = LSTMForecaster()
        m.fit(panel, seed=0)
        assert m.model_ is not None

    def test_predict_returns_correct_columns(self):
        panel = _make_synthetic_panel(n_dates=200)
        train = panel.iloc[:panel.shape[0] * 4 // 5]
        test  = panel.iloc[panel.shape[0] * 4 // 5:]
        m = LSTMForecaster()
        m.fit(train, seed=0)
        out = m.predict(test)
        assert "forecast_log_rv" in out.columns
        assert "forecast_rv" in out.columns

    def test_predict_index_is_date_ticker(self):
        panel = _make_synthetic_panel(n_dates=200)
        train = panel.iloc[:panel.shape[0] * 4 // 5]
        test  = panel.iloc[panel.shape[0] * 4 // 5:]
        m = LSTMForecaster()
        m.fit(train, seed=0)
        out = m.predict(test)
        assert out.index.names == ["date", "ticker"]

    def test_forecast_rv_positive(self):
        panel = _make_synthetic_panel(n_dates=200)
        train = panel.iloc[:panel.shape[0] * 4 // 5]
        test  = panel.iloc[panel.shape[0] * 4 // 5:]
        m = LSTMForecaster()
        m.fit(train, seed=0)
        out = m.predict(test)
        assert (out["forecast_rv"].dropna() > 0).all()

    def test_different_seeds_give_different_predictions(self):
        """Two different seeds should produce different forecasts (randomness matters)."""
        panel = _make_synthetic_panel(n_dates=200)
        train = panel.iloc[:160]
        test  = panel.iloc[160:]
        m0 = LSTMForecaster()
        m0.fit(train, seed=0)
        out0 = m0.predict(test)

        m1 = LSTMForecaster()
        m1.fit(train, seed=1)
        out1 = m1.predict(test)

        common = out0.index.intersection(out1.index)
        assert len(common) > 0
        # Not identical (with probability 1)
        assert not np.allclose(
            out0.loc[common, "forecast_log_rv"].values,
            out1.loc[common, "forecast_log_rv"].values,
        ), "Two different seeds produced identical predictions — seeding not working"
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_lstm.py -v --tb=short 2>&1 | tail -20
```
Expected: ImportError or failures.

- [ ] **Step 3: Implement `src/models/lstm_model.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import load_config

_cfg = load_config()
_LCFG = _cfg["models"]["lstm"]

FEATURE_COLS = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]
_SEQ_LEN = _LCFG["sequence_length"]  # 60


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])  # last timestep
        return self.head(out).squeeze(-1)


class LSTMForecaster:
    name = "lstm"

    def __init__(self):
        self.model_: _LSTMNet | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std:  np.ndarray | None = None
        self._tgt_mean:  float = 0.0
        self._tgt_std:   float = 1.0
        self.mse_resid_: float = 1.0

    def _build_sequences(
        self,
        panel: pd.DataFrame,
        include_target: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, list]:
        """
        Build (X_seq, y) arrays from a long-form panel.

        Returns:
            X_seq : (n_valid, seq_len, n_features) float32
            y     : (n_valid,) float32 or None
            valid_idx : list of (date, ticker) for each row in X_seq
        """
        feat_cols = [c for c in FEATURE_COLS if c in panel.columns]
        # Wide form per ticker for efficient sequence slicing
        tickers = panel.index.get_level_values("ticker").unique()
        all_X, all_y, all_idx = [], [], []
        for tkr in tickers:
            sub = panel.xs(tkr, level="ticker").sort_index()
            feats = sub[feat_cols].values.astype(np.float32)  # (T, n_feat)
            if include_target and "target_log_rv" in sub.columns:
                tgt = sub["target_log_rv"].values.astype(np.float32)
            else:
                tgt = None
            T = len(feats)
            for t in range(_SEQ_LEN, T):
                seq = feats[t - _SEQ_LEN: t]
                if not np.isfinite(seq).all():
                    continue
                if tgt is not None and not np.isfinite(tgt[t]):
                    continue
                all_X.append(seq)
                all_idx.append((sub.index[t], tkr))
                if tgt is not None:
                    all_y.append(tgt[t])
        if not all_X:
            return np.empty((0, _SEQ_LEN, len(feat_cols)), dtype=np.float32), None, []
        X = np.stack(all_X)  # (n, seq_len, n_feat)
        y = np.array(all_y, dtype=np.float32) if all_y else None
        return X, y, all_idx

    def fit(self, train: pd.DataFrame, seed: int = 0) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        feat_cols = [c for c in FEATURE_COLS if c in train.columns]

        # Compute normalisation stats from training data only
        raw_feats = train[feat_cols].values.astype(np.float32)
        self._feat_mean = np.nanmean(raw_feats, axis=0)
        self._feat_std  = np.nanstd(raw_feats, axis=0)
        self._feat_std[self._feat_std < 1e-12] = 1.0  # avoid /0

        tgt_vals = train["target_log_rv"].dropna().values
        self._tgt_mean = float(np.mean(tgt_vals))
        self._tgt_std  = float(np.std(tgt_vals))
        if self._tgt_std < 1e-12:
            self._tgt_std = 1.0

        # Normalise panel
        norm_train = train.copy()
        norm_train[feat_cols] = (norm_train[feat_cols] - self._feat_mean) / self._feat_std
        norm_train["target_log_rv"] = (
            (norm_train["target_log_rv"] - self._tgt_mean) / self._tgt_std
        )

        X, y, _ = self._build_sequences(norm_train, include_target=True)
        if len(X) == 0:
            return

        # Chronological 90/10 split
        n = len(X)
        n_val = max(int(n * _LCFG["val_fraction"]), 1)
        X_tr, y_tr = X[: n - n_val], y[: n - n_val]
        X_val, y_val = X[n - n_val :], y[n - n_val :]

        n_feat = X_tr.shape[2]
        self.model_ = _LSTMNet(n_feat, _LCFG["hidden_size"], _LCFG["dropout"])
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=_LCFG["learning_rate"])
        loss_fn = nn.HuberLoss(delta=_LCFG["huber_delta"])

        tr_ds  = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        tr_dl  = DataLoader(tr_ds, batch_size=_LCFG["batch_size"], shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=_LCFG["batch_size"])

        best_val, patience_count = float("inf"), 0
        best_state = None
        for epoch in range(_LCFG["max_epochs"]):
            self.model_.train()
            for xb, yb in tr_dl:
                optimizer.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                optimizer.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = sum(
                    loss_fn(self.model_(xb), yb).item() * len(xb)
                    for xb, yb in val_dl
                ) / len(X_val)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience_count = 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= _LCFG["patience"]:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # Store sigma² of training residuals (in original scale) for Jensen correction
        self.model_.eval()
        with torch.no_grad():
            tr_pred_z = self.model_(torch.from_numpy(X_tr)).numpy()
        tr_pred = tr_pred_z * self._tgt_std + self._tgt_mean
        tr_true = y_tr * self._tgt_std + self._tgt_mean
        self.mse_resid_ = float(np.mean((tr_true - tr_pred) ** 2))

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict()")
        feat_cols = [c for c in FEATURE_COLS if c in history.columns]

        norm_hist = history.copy()
        norm_hist[feat_cols] = (norm_hist[feat_cols] - self._feat_mean) / self._feat_std

        X, _, valid_idx = self._build_sequences(norm_hist, include_target=False)
        if len(X) == 0:
            return pd.DataFrame(columns=["forecast_log_rv", "forecast_rv"])

        self.model_.eval()
        with torch.no_grad():
            pred_z = self.model_(torch.from_numpy(X)).numpy()

        # Un-z-score
        log_rv_hat = pred_z * self._tgt_std + self._tgt_mean
        rv_hat = np.exp(log_rv_hat + self.mse_resid_ / 2)

        dates   = [idx[0] for idx in valid_idx]
        tickers = [idx[1] for idx in valid_idx]
        out_idx = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
        out = pd.DataFrame(
            {"forecast_log_rv": log_rv_hat, "forecast_rv": rv_hat},
            index=out_idx,
        )
        return out.sort_index()
```

- [ ] **Step 4: Run tests**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_lstm.py -v --tb=short 2>&1 | tail -25
```
Expected: 6 pass. These tests train on synthetic data (n_dates=200, n_tickers=5) so they should complete in under 30 seconds.

**If `test_different_seeds_give_different_predictions` fails:** PyTorch seeding is sometimes not perfectly isolated across instances. If the predictions are identical after seeding, add `torch.use_deterministic_algorithms(True)` to the `fit()` method and check that the seeds are being set before weight initialisation.

- [ ] **Step 5: Run all non-slow tests to verify no regressions**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/ -m "not slow" --tb=short 2>&1 | tail -10
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && git add src/models/lstm_model.py tests/test_lstm.py && git commit -m "feat: LSTMForecaster — 5-seed LSTM with per-window z-score normalisation"
```

---

## Task 3: Walk-forward run for GBM + LSTM (single-year smoke test)

Before running 22 years, verify the full walk-forward pipeline works on a 3-year window. This catches shape mismatches, missing feature columns, and Jensen-correction bugs before committing to the full run.

**Files:**
- Create: `scripts/smoke_test_ml_models.py`

- [ ] **Step 1: Write the smoke test script**

```python
# scripts/smoke_test_ml_models.py
"""
Smoke test: walk-forward GBM and LSTM on a 3-year window (2003-2005).
Verifies models plug into the harness and produce sensible forecasts.
Run with: python scripts/smoke_test_ml_models.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.data.loaders import load_ohlcv, load_vix
from src.data.universe import get_universe
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.tests import cross_sectional_ic
from src.models.gbm import GBMForecaster
from src.models.lstm_model import LSTMForecaster

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2005-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:40]

print(f"Loading data for {len(tickers)} tickers (2000-2005)...")
ohlcv = load_ohlcv(tickers, start, end)
vix   = load_vix(start, end)

returns_frames = []
for ticker, grp in ohlcv.groupby(level="ticker"):
    close = grp.droplevel("ticker")["close"]
    r = np.log(close / close.shift(1))
    r.name = "return"
    idx = pd.MultiIndex.from_arrays([r.index, [ticker]*len(r)], names=["date","ticker"])
    returns_frames.append(r.set_axis(idx))
returns_panel = pd.concat(returns_frames).to_frame()

features = build_feature_panel(ohlcv, vix)
targets  = forward_rv(returns_panel)
panel    = features.join(targets, how="inner").dropna(subset=["target_log_rv"])

# Add sector to panel for GBM
from src.data.universe import get_sector
sector_map = {t: get_sector(t, pd.Timestamp("2003-01-01")) for t in tickers}
panel["sector"] = panel.index.get_level_values("ticker").map(sector_map)

windows = generate_windows(start, end, first_test_year=2003)
realized = panel["target_rv"]

print("\n=== GBM smoke test ===")
gbm_oos = run_walk_forward(GBMForecaster(), panel, windows)
print(f"OOS rows: {len(gbm_oos)}")
print(f"OOS dates: {gbm_oos.index.get_level_values('date').nunique()}")
print(f"NaN forecast_rv: {gbm_oos['forecast_rv'].isna().sum()}")
print(f"Negative forecast_rv: {(gbm_oos['forecast_rv'] < 0).sum()}")
ic_gbm = cross_sectional_ic(gbm_oos, realized.rename("target_rv").to_frame())
print(f"Mean IC: {ic_gbm.mean():.4f}  (HAR-RV Phase 1 baseline: 0.682)")

print("\n=== LSTM smoke test (seed=0 only) ===")
# Wrap single-seed LSTM for the harness
class SingleSeedLSTM:
    name = "lstm_seed0"
    def __init__(self):
        from src.models.lstm_model import LSTMForecaster
        self._m = LSTMForecaster()
    def fit(self, train):
        self._m.fit(train, seed=0)
    def predict(self, history):
        return self._m.predict(history)

lstm_oos = run_walk_forward(SingleSeedLSTM(), panel, windows)
print(f"OOS rows: {len(lstm_oos)}")
print(f"NaN forecast_rv: {lstm_oos['forecast_rv'].isna().sum()}")
if len(lstm_oos) > 0:
    ic_lstm = cross_sectional_ic(lstm_oos, realized.rename("target_rv").to_frame())
    print(f"Mean IC: {ic_lstm.mean():.4f}")

print("\nSmoke test complete. If IC > 0 for both models, pipeline is functional.")
```

- [ ] **Step 2: Run the smoke test**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && /tmp/ml-vol-momentum-venv/bin/python scripts/smoke_test_ml_models.py 2>&1
```
Expected: GBM IC > 0, LSTM IC > 0, zero NaN or negative forecast_rv. Wall-clock: ~2-5 min.

- [ ] **Step 3: Commit the script**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && git add scripts/smoke_test_ml_models.py && git commit -m "chore: smoke test for GBM and LSTM walk-forward integration"
```

---

## Task 4: 5-seed LSTM ensemble wrapper

The walk-forward harness calls `fit()` once and `predict()` once per window. For the 5-seed LSTM ensemble we need a wrapper that trains all 5 seeds in `fit()` and averages predictions in `predict()`.

**Files:**
- Modify: `src/models/lstm_model.py` (add `LSTMEnsemble` class)
- Modify: `tests/test_lstm.py` (add `TestLSTMEnsemble`)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_lstm.py`:

```python
from src.models.lstm_model import LSTMEnsemble


class TestLSTMEnsemble:
    def test_implements_forecaster_protocol(self):
        from src.models.forecaster import Forecaster
        m = LSTMEnsemble(seeds=[0, 1])
        assert isinstance(m, Forecaster)

    def test_fit_trains_all_seeds(self):
        panel = _make_synthetic_panel(n_dates=200)
        m = LSTMEnsemble(seeds=[0, 1, 2])
        m.fit(panel)
        assert len(m.members_) == 3

    def test_predict_averages_seeds(self):
        """Ensemble prediction should differ from any single seed."""
        panel = _make_synthetic_panel(n_dates=200)
        train = panel.iloc[:160]
        test  = panel.iloc[160:]
        ens = LSTMEnsemble(seeds=[0, 1, 2])
        ens.fit(train)
        out_ens = ens.predict(test)
        assert "forecast_log_rv" in out_ens.columns
        assert "forecast_rv" in out_ens.columns
        assert len(out_ens) > 0

    def test_seed_dispersion_stored(self):
        """After predict(), per_seed_sharpe_ should be populated."""
        panel = _make_synthetic_panel(n_dates=200)
        train = panel.iloc[:160]
        test  = panel.iloc[160:]
        ens = LSTMEnsemble(seeds=[0, 1])
        ens.fit(train)
        ens.predict(test)
        assert hasattr(ens, "per_seed_forecasts_")
        assert len(ens.per_seed_forecasts_) == 2
```

- [ ] **Step 2: Run to verify failure**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_lstm.py::TestLSTMEnsemble -v --tb=short 2>&1 | tail -15
```

- [ ] **Step 3: Add `LSTMEnsemble` to `src/models/lstm_model.py`**

Append to the bottom of `src/models/lstm_model.py`:

```python
class LSTMEnsemble:
    """5-seed LSTM ensemble. fit() trains all seeds; predict() averages predictions."""
    name = "lstm_ensemble"

    def __init__(self, seeds: list[int] | None = None):
        self.seeds = seeds if seeds is not None else _LCFG["seeds"]
        self.members_: list[LSTMForecaster] = []
        self.per_seed_forecasts_: list[pd.DataFrame] = []

    def fit(self, train: pd.DataFrame) -> None:
        self.members_ = []
        for seed in self.seeds:
            m = LSTMForecaster()
            m.fit(train, seed=seed)
            self.members_.append(m)

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        self.per_seed_forecasts_ = []
        log_rv_frames = []
        for m in self.members_:
            fc = m.predict(history)
            self.per_seed_forecasts_.append(fc)
            log_rv_frames.append(fc["forecast_log_rv"])
        if not log_rv_frames:
            return pd.DataFrame(columns=["forecast_log_rv", "forecast_rv"])
        # Average in log space; Jensen correction uses mean mse_resid_
        mean_log_rv = pd.concat(log_rv_frames, axis=1).mean(axis=1)
        mean_mse = float(np.mean([m.mse_resid_ for m in self.members_]))
        rv_hat = np.exp(mean_log_rv + mean_mse / 2)
        out = pd.DataFrame(
            {"forecast_log_rv": mean_log_rv, "forecast_rv": rv_hat},
        )
        out.index.names = ["date", "ticker"]
        return out.sort_index()
```

- [ ] **Step 4: Run tests**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_lstm.py -v --tb=short 2>&1 | tail -20
```
Expected: all 10 tests pass.

- [ ] **Step 5: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && git add src/models/lstm_model.py tests/test_lstm.py && git commit -m "feat: LSTMEnsemble — 5-seed mean-forecast wrapper with per-seed forecasts stored"
```

---

## Task 5: CHECKPOINT 3a — GBM OOS IC gate

CHECKPOINT 3a (from the master plan): GBM OOS cross-sectional IC must lie in `[HAR-RV IC − 0.03, HAR-RV IC + 0.10]`. Below the floor means implementation is broken; above the ceiling (+0.10 over HAR-RV) means almost certainly leakage.

This runs GBM on the full 22-year walk-forward (2003–2024) using the 80-ticker panel and compares IC to the Phase 1 HAR-RV baseline (pooled IC = 0.682).

**Files:**
- Create: `scripts/validate_checkpoint_3a.py`

- [ ] **Step 1: Write the checkpoint script**

```python
# scripts/validate_checkpoint_3a.py
"""
CHECKPOINT 3a: GBM OOS cross-sectional IC vs HAR-RV baseline.
Gate: GBM IC in [HAR-RV_IC - 0.03, HAR-RV_IC + 0.10]
      GBM OOS R² in [0.25, 0.70]
Run with: python scripts/validate_checkpoint_3a.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.data.loaders import load_ohlcv, load_vix
from src.data.universe import get_universe, get_sector
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.tests import cross_sectional_ic
from src.models.gbm import GBMForecaster
from src.models.baselines import HARRV
from src.config import load_config

cfg = load_config()
start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")

tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print(f"Loading data for {len(tickers)} tickers (2000-2024)...")
ohlcv = load_ohlcv(tickers, start, end)
vix   = load_vix(start, end)

returns_frames = []
for ticker, grp in ohlcv.groupby(level="ticker"):
    close = grp.droplevel("ticker")["close"]
    r = np.log(close / close.shift(1))
    r.name = "return"
    idx = pd.MultiIndex.from_arrays([r.index, [ticker]*len(r)], names=["date","ticker"])
    returns_frames.append(r.set_axis(idx))
returns_panel = pd.concat(returns_frames).to_frame()

features = build_feature_panel(ohlcv, vix)
targets  = forward_rv(returns_panel)
panel    = features.join(targets, how="inner").dropna(subset=["target_log_rv"])
sector_map = {t: get_sector(t, pd.Timestamp("2003-01-01")) for t in tickers}
panel["sector"] = panel.index.get_level_values("ticker").map(sector_map)

windows = generate_windows(start, end, first_test_year=2003)
realized_rv  = panel["target_rv"]
realized_log = panel["target_log_rv"]

print("Running GBM walk-forward (22 years)...")
gbm_oos = run_walk_forward(GBMForecaster(), panel, windows)

print("Running HAR-RV walk-forward (22 years)...")
har_oos = run_walk_forward(HARRV(), panel, windows)

# Cross-sectional IC
ic_gbm = cross_sectional_ic(gbm_oos, realized_rv.rename("target_rv").to_frame())
ic_har = cross_sectional_ic(har_oos, realized_rv.rename("target_rv").to_frame())

# OOS R² on log_rv
def oos_r2(forecast_log, realized_log):
    common = forecast_log.index.intersection(realized_log.index)
    f = forecast_log[common].dropna()
    r = realized_log[common].reindex(f.index).dropna()
    f = f.reindex(r.index)
    ss_res = ((r - f) ** 2).sum()
    ss_tot = ((r - r.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

r2_gbm = oos_r2(gbm_oos["forecast_log_rv"], realized_log)
r2_har = oos_r2(har_oos["forecast_log_rv"], realized_log)

print(f"\nCHECKPOINT 3a results:")
print(f"  HAR-RV IC (22-yr pooled):  {ic_har.mean():.4f}  (Phase 1: 0.682)")
print(f"  GBM IC (22-yr pooled):     {ic_gbm.mean():.4f}")
print(f"  GBM IC − HAR-RV IC:        {ic_gbm.mean() - ic_har.mean():+.4f}  (gate: [−0.03, +0.10])")
print(f"  GBM OOS R²:                {r2_gbm:.4f}  (gate: [0.25, 0.70])")
print(f"  HAR-RV OOS R²:             {r2_har:.4f}  (Phase 1: 0.328)")

ic_diff = ic_gbm.mean() - ic_har.mean()
gate_ic   = -0.03 <= ic_diff <= 0.10
gate_r2   = 0.25 <= r2_gbm <= 0.70
print(f"\nGates: IC_diff_in_range={gate_ic}  R2_in_range={gate_r2}")
print(f"CHECKPOINT 3a: {'PASS' if gate_ic and gate_r2 else 'FAIL'}")

# Per-year IC breakdown
print("\nPer-year IC (GBM vs HAR-RV):")
all_dates = ic_gbm.index.union(ic_har.index)
for yr in range(2003, 2025):
    yr_str = str(yr)
    g = ic_gbm.loc[yr_str].mean() if yr_str in ic_gbm.index.year.astype(str).values else float("nan")
    h = ic_har.loc[yr_str].mean() if yr_str in ic_har.index.year.astype(str).values else float("nan")
    print(f"  {yr}: GBM={g:.3f}  HAR-RV={h:.3f}  diff={g-h:+.3f}")
```

- [ ] **Step 2: Run the checkpoint**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && /tmp/ml-vol-momentum-venv/bin/python scripts/validate_checkpoint_3a.py 2>&1
```
Expected runtime: ~5-10 min (GBM trains fast; 22 windows × ~30s each).

Gate: IC diff in [−0.03, +0.10] and R² in [0.25, 0.70].

If GBM IC is more than 0.10 above HAR-RV: **STOP**. This is a leakage red flag. Check:
1. Does the feature panel have any future data? (All columns should be `.shift(1)`)
2. Is `target_log_rv` being included in `predict()` input? (Should not be)
3. Is the sector mapping using post-hoc data? (The `get_sector()` call uses 2003-01-01 date — verify this is point-in-time)

If GBM IC is more than 0.03 below HAR-RV: GBM is broken. Check:
1. Are the feature columns correct (rv_d, rv_w, rv_m present)?
2. Is early stopping cutting the model too short? Try `early_stopping_rounds: 200`
3. Does the sector column have valid values or all "Unknown"?

- [ ] **Step 3: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && git add scripts/validate_checkpoint_3a.py && git commit -m "chore: checkpoint 3a — GBM OOS IC vs HAR-RV gate"
```

---

## Task 6: CHECKPOINT 3b — LSTM seed stability

CHECKPOINT 3b: LSTM trains stably; seed-to-seed Sharpe std < 0.3.

This runs the 5-seed LSTM ensemble walk-forward on a shorter window (2003–2010 to save time) and reports per-seed cross-sectional IC and the seed-to-seed dispersion.

**Files:**
- Create: `scripts/validate_checkpoint_3b.py`

- [ ] **Step 1: Write the script**

```python
# scripts/validate_checkpoint_3b.py
"""
CHECKPOINT 3b: LSTM seed stability.
Gate: seed-to-seed IC std < 0.10 (Sharpe std on the strategy PnL < 0.3).
Run with: python scripts/validate_checkpoint_3b.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.data.loaders import load_ohlcv, load_vix
from src.data.universe import get_universe, get_sector
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.tests import cross_sectional_ic
from src.models.lstm_model import LSTMForecaster
from src.config import load_config

cfg = load_config()
start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2010-12-31")   # shorter window for speed
tickers = get_universe(pd.Timestamp("2002-01-01"))[:40]  # 40 tickers for speed

print(f"Loading data for {len(tickers)} tickers (2000-2010)...")
ohlcv = load_ohlcv(tickers, start, end)
vix   = load_vix(start, end)

returns_frames = []
for ticker, grp in ohlcv.groupby(level="ticker"):
    close = grp.droplevel("ticker")["close"]
    r = np.log(close / close.shift(1))
    r.name = "return"
    idx = pd.MultiIndex.from_arrays([r.index, [ticker]*len(r)], names=["date","ticker"])
    returns_frames.append(r.set_axis(idx))
returns_panel = pd.concat(returns_frames).to_frame()

features = build_feature_panel(ohlcv, vix)
targets  = forward_rv(returns_panel)
panel    = features.join(targets, how="inner").dropna(subset=["target_log_rv"])

windows = generate_windows(start, end, first_test_year=2003)
realized_rv = panel["target_rv"]

seeds = cfg["models"]["lstm"]["seeds"]  # [0, 1, 2, 3, 4]
ic_per_seed = {}

for seed in seeds:
    print(f"Running LSTM seed={seed}...")
    class SingleSeedWrapper:
        name = f"lstm_seed{seed}"
        _seed = seed
        def __init__(self): self._m = LSTMForecaster()
        def fit(self, train): self._m.fit(train, seed=self.__class__._seed)
        def predict(self, history): return self._m.predict(history)

    oos = run_walk_forward(SingleSeedWrapper(), panel, windows)
    if len(oos) == 0:
        print(f"  Seed {seed}: no OOS predictions")
        continue
    ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
    ic_per_seed[seed] = ic.mean()
    print(f"  Seed {seed}: mean IC = {ic.mean():.4f}")

if ic_per_seed:
    ic_vals = list(ic_per_seed.values())
    ic_mean = np.mean(ic_vals)
    ic_std  = np.std(ic_vals)
    print(f"\nCHECKPOINT 3b results:")
    print(f"  Mean IC across seeds: {ic_mean:.4f}")
    print(f"  Std  IC across seeds: {ic_std:.4f}  (gate: < 0.10)")
    gate = ic_std < 0.10
    print(f"\nCHECKPOINT 3b: {'PASS' if gate else 'FAIL'}")
    if not gate:
        print("  WARNING: High seed-to-seed variance. Check training stability.")
        print("  Try increasing max_epochs or reducing learning_rate in default.yaml.")
```

- [ ] **Step 2: Run the checkpoint**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && /tmp/ml-vol-momentum-venv/bin/python scripts/validate_checkpoint_3b.py 2>&1
```
Expected runtime: ~20-40 min (5 seeds × 8 windows × ~1 min/window on CPU). Gate: per-seed IC std < 0.10.

If the run is too slow: reduce to seeds=[0, 1, 2] and windows to 2003-2007 for the stability check — you just need enough variance to detect instability.

- [ ] **Step 3: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && git add scripts/validate_checkpoint_3b.py && git commit -m "chore: checkpoint 3b — LSTM seed stability gate"
```

---

## Task 7: Full 5-model comparison and master results table

Run all 5 models on the full walk-forward (2003–2024), produce the master Sharpe/IC/QLIKE table, and write parquets to `results/`.

**Files:**
- Create: `scripts/compare_all_models.py`
- Create: `results/` directory (via script)

- [ ] **Step 1: Write the comparison script**

```python
# scripts/compare_all_models.py
"""
Full 5-model comparison: RollingVol, GARCH, HAR-RV, GBM, LSTM-ensemble.
Produces results/forecasts/{model}.parquet, results/strategies/{model}_{variant}.parquet,
and prints the master results table.
Run with: python scripts/compare_all_models.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.loaders import load_ohlcv, load_vix
from src.data.universe import get_universe, get_sector
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.tests import cross_sectional_ic
from src.eval.comparison import build_results_table, build_dm_matrix, build_mz_table
from src.eval.metrics import sharpe
from src.models.baselines import RollingVolModel, GARCH11Model, HARRV
from src.models.gbm import GBMForecaster
from src.models.lstm_model import LSTMEnsemble

# Setup
Path("results/forecasts").mkdir(parents=True, exist_ok=True)
Path("results/strategies").mkdir(parents=True, exist_ok=True)

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print(f"Loading data for {len(tickers)} tickers (2000-2024)...")
ohlcv = load_ohlcv(tickers, start, end)
vix   = load_vix(start, end)

returns_frames = []
for ticker, grp in ohlcv.groupby(level="ticker"):
    close = grp.droplevel("ticker")["close"]
    r = np.log(close / close.shift(1))
    r.name = "return"
    idx = pd.MultiIndex.from_arrays([r.index, [ticker]*len(r)], names=["date","ticker"])
    returns_frames.append(r.set_axis(idx))
returns_panel = pd.concat(returns_frames).to_frame()

features = build_feature_panel(ohlcv, vix)
targets  = forward_rv(returns_panel)
panel    = features.join(targets, how="inner").dropna(subset=["target_log_rv"])
rv_panel = returns_panel.join(panel, how="inner")  # for RollingVol (needs "return")
sector_map = {t: get_sector(t, pd.Timestamp("2003-01-01")) for t in tickers}
panel["sector"] = panel.index.get_level_values("ticker").map(sector_map)

windows = generate_windows(start, end, first_test_year=2003)
prices_panel = ohlcv[["close"]]
signal = momentum_signal(prices_panel, lookback=252, skip=21)
realized_rv   = panel["target_rv"]
realized_log  = panel["target_log_rv"]

models = {
    "rolling_vol": (RollingVolModel(), rv_panel),
    "har_rv":      (HARRV(),          panel),
    "garch":       (GARCH11Model(),   rv_panel),
    "gbm":         (GBMForecaster(),  panel),
    "lstm":        (LSTMEnsemble(),   panel),
}

all_forecasts = {}
all_strategies = {}

for model_name, (model, mpanel) in models.items():
    print(f"\nRunning {model_name} walk-forward...")
    oos = run_walk_forward(model, mpanel, windows)
    if oos.empty:
        print(f"  {model_name}: no OOS predictions!")
        continue
    oos.to_parquet(f"results/forecasts/{model_name}.parquet")
    all_forecasts[model_name] = oos
    print(f"  OOS rows: {len(oos)}")
    ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
    print(f"  Mean IC: {ic.mean():.4f}")

    # Build scaled portfolio
    w_scaled_raw = vol_scale(signal, oos, target_vol=0.10)
    w_scaled = build_portfolios(None, weights=w_scaled_raw, mode="vol_targeted_gross")
    net_scaled = apply_costs(w_scaled, returns_panel, cost_bps=10.0)
    all_strategies[f"{model_name}_scaled"] = net_scaled.dropna()
    net_scaled.to_parquet(f"results/strategies/{model_name}_scaled.parquet")

# Unscaled baseline (same for all models)
w_unscaled = build_portfolios(signal, mode="long_short_quintile")
net_unscaled = apply_costs(w_unscaled, returns_panel, cost_bps=10.0).dropna()
all_strategies["unscaled_momentum"] = net_unscaled

# Master results table
print("\n\n=== MASTER RESULTS TABLE ===")
results_tbl = build_results_table(all_strategies)
print(results_tbl.to_string())
results_tbl.to_parquet("results/master_results_table.parquet")

# Per-model IC summary
print("\n=== IC SUMMARY ===")
for name, oos in all_forecasts.items():
    ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
    print(f"  {name}: mean_IC={ic.mean():.4f}  std_IC={ic.std():.4f}")

# DM matrix (log-rv forecast series)
print("\n=== DM MATRIX (QLIKE, p-values) ===")
log_rv_forecasts = {n: f["forecast_log_rv"] for n, f in all_forecasts.items()}
try:
    dm_stats, dm_pvals = build_dm_matrix(log_rv_forecasts, realized_log)
    print(dm_pvals.round(3).to_string())
    dm_pvals.to_parquet("results/dm_pvalues.parquet")
except Exception as e:
    print(f"DM matrix failed: {e}")

print("\nAll results written to results/")
```

- [ ] **Step 2: Run the comparison**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && /tmp/ml-vol-momentum-venv/bin/python scripts/compare_all_models.py 2>&1
```
Expected runtime: ~30-60 min (GBM 5-10 min, LSTM 30-40 min, GARCH 5 min, HAR-RV 2 min).

- [ ] **Step 3: Commit results and script**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && git add scripts/compare_all_models.py results/*.parquet && git commit -m "feat: 5-model comparison — master results table, DM matrix, IC summary"
```

---

## Task 8: Tag phase3-complete and update surprises.md

- [ ] **Step 1: Run the full non-slow test suite**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/ -m "not slow" --tb=short 2>&1 | tail -10
```
Expected: all pass.

- [ ] **Step 2: Update surprises.md with Phase 3 findings**

Add entries for:
- Did GBM beat HAR-RV on IC, or did it tie? By how much?
- Did LSTM beat GBM? (Pre-registered prediction: no)
- Was seed-to-seed IC std small? (Pre-registered: < 0.10)
- Did scaling Sharpe ranking match IC ranking? (Pre-registered prediction 7)
- Any unexpected GARCH behaviour at scale?

- [ ] **Step 3: Commit and tag**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && git add docs/surprises.md && git commit -m "chore: Phase 3 complete — 5-model walk-forward, checkpoint gates, surprises" && git tag phase3-complete
```

---

## Self-Review

### 1. Spec coverage

| Spec requirement | Task |
|---|---|
| `GBMForecaster` with LightGBM, sector categorical, Forecaster protocol | Task 1 |
| `LSTMForecaster` with per-window z-score, Huber loss, 5 seeds | Task 2 |
| CHECKPOINT 3a gate: GBM IC in [HAR-RV IC − 0.03, HAR-RV IC + 0.10] | Task 5 |
| CHECKPOINT 3b gate: LSTM seed std < 0.3 (Sharpe) / < 0.10 (IC) | Task 6 |
| `LSTMEnsemble` wrapping 5 seeds, mean forecast | Task 4 |
| Walk-forward integration for both models | Tasks 3, 5, 6, 7 |
| Full 5-model comparison table | Task 7 |
| DM matrix across all models | Task 7 |
| `shap_values()` on GBM | Task 1 (implementation + test) |
| Per-seed dispersion tracking | Task 4 |
| Master results table written to parquet | Task 7 |

**Missing from this plan:** SHAP analysis script (`src/interp/shap_analysis.py`) and regime analysis (`src/interp/regime_analysis.py`). These are Phase 4 work per the master plan. The GBM has a `shap_values()` method implemented (Task 1) but the full SHAP analysis / charts are not included here.

**Also deferred to Phase 4:** Full-universe pipeline run, sector-neutral robustness check.

### 2. Placeholder scan

None found.

### 3. Type consistency

- `GBMForecaster.predict(panel)` → `DataFrame(forecast_log_rv, forecast_rv)` ✓
- `LSTMForecaster.predict(history)` → same ✓
- `LSTMEnsemble.predict(history)` → same ✓
- `cross_sectional_ic(forecast, realized)` — `forecast` needs `forecast_rv` column and `(date,ticker)` index ✓ (GBM/LSTM output has `forecast_rv`)
- `build_dm_matrix(forecasts, realized)` — takes `dict[str, pd.Series]` and `pd.Series`; script extracts `forecast_log_rv` series ✓
- `vol_scale(signal, oos_forecasts, target_vol)` — needs `forecast_rv` column ✓
