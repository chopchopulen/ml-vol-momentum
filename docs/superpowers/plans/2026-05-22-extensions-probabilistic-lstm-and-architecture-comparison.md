# ML Extensions: Probabilistic LSTM & Architecture Comparison

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two ML extensions to the existing vol-momentum project: (1) a probabilistic LSTM that outputs a predictive distribution and uses forecast uncertainty to downweight positions, directly addressing the IC ≠ Sharpe finding; (2) an architecture comparison across LSTM, Transformer, MLP, and TCN on identical CV setup.

**Architecture:** Each extension adds new model classes in `src/models/` that implement the existing `Forecaster` protocol. Extension 1 adds a new scaling function for uncertainty-weighted sizing. Extension 2 adds three new model files. Both integrate with the existing `run_walk_forward` / `compare_all_models` pipeline without modifying core eval code. Results land in `results/forecasts/` and `results/strategies/` alongside the existing 5 models.

**Tech Stack:** PyTorch (existing), existing walk-forward CV, existing `Forecaster` protocol, matplotlib for new plots.

---

## Context for the implementer

Project root: `/Users/harry/RL:ML Project/ml-vol-momentum`
Venv: `/tmp/ml-vol-momentum-venv`

### Existing architecture you must NOT break

- `Forecaster` protocol (src/models/forecaster.py): `name: str`, `fit(train) -> None`, `predict(history) -> DataFrame` with columns `["forecast_log_rv", "forecast_rv"]`
- `run_walk_forward(forecaster, panel, windows)` in `src/eval/walk_forward.py` — calls `.fit(train)` then `.predict(history)` per window
- `vol_scale(signal, sigma_hat, target_vol)` in `src/strategy/scaling.py` — expects `sigma_hat` with column `forecast_rv`
- Feature cols: `["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]` — 9 features, defined in `lstm_model.py` as `FEATURE_COLS`
- Sequence length: 60 (from `configs/default.yaml` → `models.lstm.sequence_length`)

### Existing LSTM internals (critical to understand)

- `_LSTMNet`: `nn.LSTM(n_features, hidden) → nn.Dropout → nn.Linear(hidden, 1)` — outputs a single scalar (point forecast)
- `LSTMForecaster.fit(train, seed)`: z-scores features and target on training data only; stores `_feat_mean`, `_feat_std`, `_tgt_mean`, `_tgt_std`; trains with Huber loss; stores `mse_resid_` from validation residuals for Jensen correction
- `LSTMForecaster.predict(history)`: prepends training tail for lookback, builds sequences, un-z-scores, applies Jensen correction: `rv_hat = exp(log_rv_hat + mse_resid_/2)`
- `LSTMEnsemble`: 5-seed wrapper; averages `forecast_log_rv` across seeds, averages `mse_resid_` for Jensen correction

### Walk-forward setup

- 22 windows (2003–2024), annual retrain, 42-day embargo
- LSTM training takes ~5–10 min per window per seed on CPU — budget ~2–3h per model for full 22-window run
- Use checkpoint pattern (save each window's OOS parquet to disk) so laptop sleep doesn't force restart

---

## File structure

### New files to create

```
src/models/prob_lstm.py          # Extension 1: probabilistic LSTM (Gaussian head + MC dropout)
src/models/transformer_model.py  # Extension 2: Transformer encoder
src/models/mlp_model.py          # Extension 2: MLP on flattened sequence
src/models/tcn_model.py          # Extension 2: Temporal Convolutional Network
src/strategy/uncertainty_scale.py  # Extension 1: uncertainty-weighted vol scaling
scripts/run_extension1.py        # Extension 1 pipeline runner
scripts/run_extension2.py        # Extension 2 pipeline runner
tests/test_prob_lstm.py
tests/test_architecture_comparison.py
tests/test_uncertainty_scale.py
```

### Files to modify

```
configs/default.yaml             # Add prob_lstm, transformer, mlp, tcn config blocks
README.md                        # Add extension results sections
```

### Files NOT to modify

`src/eval/walk_forward.py`, `src/strategy/scaling.py`, `src/models/lstm_model.py`, `src/models/forecaster.py` — these are load-bearing; extensions compose with them rather than modifying them.

---

## Task 1: Probabilistic LSTM model (`src/models/prob_lstm.py`)

The Gaussian head predicts `(μ, log_σ²)` for each sequence. At inference, `μ` is the point forecast (same as existing LSTM), `σ` is the epistemic uncertainty. MC dropout variant keeps dropout active at inference and runs N=50 forward passes.

**Files:**
- Create: `src/models/prob_lstm.py`
- Create: `tests/test_prob_lstm.py`

- [ ] **Step 1: Write the failing tests first**

Create `tests/test_prob_lstm.py`:

```python
import pytest
import numpy as np
import pandas as pd
import torch
from src.models.prob_lstm import ProbLSTMForecaster, ProbLSTMEnsemble


def _make_panel(n_dates=200, n_tickers=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    feat_cols = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]
    rows = []
    for t in tickers:
        df = pd.DataFrame(rng.standard_normal((n_dates, len(feat_cols))), columns=feat_cols, index=dates)
        df["target_log_rv"] = rng.standard_normal(n_dates)
        df["target_rv"] = np.exp(df["target_log_rv"])
        df.index = pd.MultiIndex.from_arrays([dates, [t]*n_dates], names=["date","ticker"])
        rows.append(df)
    return pd.concat(rows)


@pytest.fixture
def panel():
    return _make_panel()


class TestProbLSTMForecaster:
    def test_fit_stores_model(self, panel):
        m = ProbLSTMForecaster()
        m.fit(panel, seed=0)
        assert m.model_ is not None

    def test_predict_returns_required_columns(self, panel):
        m = ProbLSTMForecaster()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert "forecast_log_rv" in out.columns
        assert "forecast_rv" in out.columns
        assert "forecast_sigma" in out.columns  # NEW — uncertainty column

    def test_forecast_sigma_positive(self, panel):
        m = ProbLSTMForecaster()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert (out["forecast_sigma"] > 0).all()

    def test_forecast_rv_positive(self, panel):
        m = ProbLSTMForecaster()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert (out["forecast_rv"] > 0).all()

    def test_index_is_date_ticker(self, panel):
        m = ProbLSTMForecaster()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert out.index.names == ["date", "ticker"]

    def test_mc_dropout_variant(self, panel):
        m = ProbLSTMForecaster(variant="mc_dropout", n_mc_samples=10)
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert "forecast_sigma" in out.columns
        assert (out["forecast_sigma"] > 0).all()


class TestProbLSTMEnsemble:
    def test_predict_has_uncertainty_column(self, panel):
        ens = ProbLSTMEnsemble(seeds=[0, 1])
        ens.fit(panel)
        out = ens.predict(panel)
        assert "forecast_sigma" in out.columns

    def test_forecaster_protocol(self, panel):
        from src.models.forecaster import Forecaster
        ens = ProbLSTMEnsemble(seeds=[0])
        assert isinstance(ens, Forecaster)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
/tmp/ml-vol-momentum-venv/bin/pytest tests/test_prob_lstm.py -v --tb=short 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'ProbLSTMForecaster'`

- [ ] **Step 3: Implement `src/models/prob_lstm.py`**

```python
"""Probabilistic LSTM: outputs (μ, σ) for log-RV forecasts.

Two variants:
  gaussian_head — single forward pass, (mu, log_var) output head
  mc_dropout    — point-head LSTM, N stochastic forward passes at inference
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import load_config
from src.models.forecaster import Forecaster
from src.models.lstm_model import FEATURE_COLS, _SEQ_LEN

_cfg = load_config()
_LCFG = _cfg["models"]["lstm"]  # reuse lstm hyperparams


class _GaussianLSTMNet(nn.Module):
    """LSTM backbone with Gaussian output head: predicts (mu, log_var)."""
    def __init__(self, n_features: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.mu_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])
        mu = self.mu_head(h).squeeze(-1)
        log_var = self.logvar_head(h).squeeze(-1)
        return mu, log_var


class _PointLSTMNet(nn.Module):
    """Standard point LSTM — used for MC dropout variant."""
    def __init__(self, n_features: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])
        return self.head(h).squeeze(-1)


def _gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of N(mu, exp(log_var))."""
    precision = torch.exp(-log_var)
    return 0.5 * (log_var + precision * (target - mu) ** 2).mean()


class ProbLSTMForecaster(Forecaster):
    """Probabilistic LSTM forecaster.

    variant='gaussian_head': trains with NLL loss, outputs (mu, sigma) in one pass.
    variant='mc_dropout': trains with Huber loss, uses N stochastic forward passes at
        inference (dropout active) to estimate predictive uncertainty.
    """
    name = "prob_lstm"

    def __init__(self, variant: str = "gaussian_head", n_mc_samples: int = 50):
        assert variant in ("gaussian_head", "mc_dropout")
        self.variant = variant
        self.n_mc_samples = n_mc_samples
        self.model_: nn.Module | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._tgt_mean: float | None = None
        self._tgt_std: float | None = None
        self.mse_resid_: float | None = None
        self._train_tail_: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    #  Sequence builder — identical to LSTMForecaster._build_sequences    #
    # ------------------------------------------------------------------ #
    def _build_sequences(self, panel, include_target):
        feat_cols = [c for c in FEATURE_COLS if c in panel.columns]
        tickers = panel.index.get_level_values("ticker").unique()
        all_X, all_y, all_idx = [], [], []
        for tkr in tickers:
            sub = panel.xs(tkr, level="ticker").sort_index()
            sub = sub[~sub.index.duplicated(keep="first")]
            feats = sub[feat_cols].values.astype(np.float32)
            tgt = (
                sub["target_log_rv"].values.astype(np.float32)
                if include_target and "target_log_rv" in sub.columns
                else None
            )
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
        return np.array(all_X, dtype=np.float32), (np.array(all_y, dtype=np.float32) if all_y else None), all_idx

    # ------------------------------------------------------------------ #
    #  fit                                                                  #
    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, seed: int = 0) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        feat_cols = [c for c in FEATURE_COLS if c in train.columns]
        feat_vals = train[feat_cols].values
        self._feat_mean = np.nanmean(feat_vals, axis=0)
        self._feat_std = np.nanstd(feat_vals, axis=0) + 1e-8

        tgt_vals = train["target_log_rv"].dropna().values
        self._tgt_mean = float(np.mean(tgt_vals))
        self._tgt_std = float(np.std(tgt_vals)) + 1e-8

        # Store tail for predict() context
        self._train_tail_ = train.copy()

        # Normalise
        norm = train.copy()
        norm[feat_cols] = (norm[feat_cols] - self._feat_mean) / self._feat_std
        norm["target_log_rv"] = (norm["target_log_rv"] - self._tgt_mean) / self._tgt_std

        X, y, _ = self._build_sequences(norm, include_target=True)
        if len(X) == 0:
            return

        # Chronological val split
        val_frac = _LCFG["val_fraction"]
        n_val = max(1, int(len(X) * val_frac))
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        n_features = X.shape[2]
        hidden = _LCFG["hidden_size"]
        dropout = _LCFG["dropout"]
        lr = _LCFG["learning_rate"]
        batch = _LCFG["batch_size"]

        if self.variant == "gaussian_head":
            self.model_ = _GaussianLSTMNet(n_features, hidden, _LCFG["num_layers"], dropout)
            loss_fn = _gaussian_nll
        else:
            self.model_ = _PointLSTMNet(n_features, hidden, _LCFG["num_layers"], dropout)
            loss_fn = nn.HuberLoss(delta=_LCFG["huber_delta"])

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=lr)

        X_tr_t = torch.from_numpy(X_tr)
        y_tr_t = torch.from_numpy(y_tr)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        tr_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch, shuffle=True)

        best_val, patience_count, best_state = float("inf"), 0, None

        for _ in range(_LCFG["max_epochs"]):
            self.model_.train()
            for xb, yb in tr_dl:
                optimizer.zero_grad()
                if self.variant == "gaussian_head":
                    mu, lv = self.model_(xb)
                    loss = loss_fn(mu, lv, yb)
                else:
                    loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                if self.variant == "gaussian_head":
                    mu_v, lv_v = self.model_(X_val_t)
                    val_loss = loss_fn(mu_v, lv_v, y_val_t).item()
                else:
                    val_loss = loss_fn(self.model_(X_val_t), y_val_t).item()

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience_count = 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= _LCFG["patience"]:
                    break

        if best_state:
            self.model_.load_state_dict(best_state)

        # mse_resid_ for Jensen correction — computed on val set mu predictions
        self.model_.eval()
        with torch.no_grad():
            if self.variant == "gaussian_head":
                mu_v, _ = self.model_(X_val_t)
                val_pred_z = mu_v.numpy()
            else:
                val_pred_z = self.model_(X_val_t).numpy()
        val_pred = val_pred_z * self._tgt_std + self._tgt_mean
        val_true = y_val * self._tgt_std + self._tgt_mean
        self.mse_resid_ = float(np.mean((val_true - val_pred) ** 2))

    # ------------------------------------------------------------------ #
    #  predict                                                              #
    # ------------------------------------------------------------------ #
    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("Call fit() first")
        feat_cols = [c for c in FEATURE_COLS if c in history.columns]
        predict_dates = set(history.index.get_level_values("date").unique())

        # Prepend training tail for lookback
        if self._train_tail_ is not None:
            hist_tickers = set(history.index.get_level_values("ticker").unique())
            tail_mask = self._train_tail_.index.get_level_values("ticker").isin(hist_tickers)
            tail = self._train_tail_[tail_mask][feat_cols]
            combined = pd.concat([tail, history[feat_cols]]).sort_index()
        else:
            combined = history[feat_cols].copy()

        norm = combined.copy()
        norm[feat_cols] = (norm[feat_cols] - self._feat_mean) / self._feat_std

        X, _, valid_idx = self._build_sequences(norm, include_target=False)
        if len(X) == 0:
            return pd.DataFrame(columns=["forecast_log_rv", "forecast_rv", "forecast_sigma"])

        X_t = torch.from_numpy(X)
        self.model_.eval()

        if self.variant == "gaussian_head":
            with torch.no_grad():
                mu_z, log_var_z = self.model_(X_t)
            mu_z = mu_z.numpy()
            # sigma in z-scored log-RV space → back-transform scale
            sigma_z = np.sqrt(np.exp(log_var_z.numpy()))
            log_rv_hat = mu_z * self._tgt_std + self._tgt_mean
            sigma_hat = sigma_z * self._tgt_std  # uncertainty in log-RV units

        else:  # mc_dropout
            # Keep dropout active during inference
            def _enable_dropout(m):
                if isinstance(m, nn.Dropout):
                    m.train()
            self.model_.apply(_enable_dropout)
            samples = []
            with torch.no_grad():
                for _ in range(self.n_mc_samples):
                    samples.append(self.model_(X_t).numpy())
            samples = np.stack(samples, axis=0)  # (n_mc, n_seq)
            mu_z = samples.mean(axis=0)
            sigma_z = samples.std(axis=0)
            log_rv_hat = mu_z * self._tgt_std + self._tgt_mean
            sigma_hat = sigma_z * self._tgt_std

        rv_hat = np.exp(log_rv_hat + self.mse_resid_ / 2)

        dates = [idx[0] for idx in valid_idx]
        tickers = [idx[1] for idx in valid_idx]
        mask = np.array([d in predict_dates for d in dates])

        out = pd.DataFrame(
            {
                "forecast_log_rv": log_rv_hat[mask],
                "forecast_rv": rv_hat[mask],
                "forecast_sigma": sigma_hat[mask],
            },
            index=pd.MultiIndex.from_arrays(
                [[d for d, m in zip(dates, mask) if m],
                 [t for t, m in zip(tickers, mask) if m]],
                names=["date", "ticker"],
            ),
        )
        return out.sort_index()


class ProbLSTMEnsemble(Forecaster):
    """5-seed ensemble of ProbLSTMForecaster. Averages μ and propagates σ."""
    name = "prob_lstm_ensemble"

    def __init__(self, seeds: list[int] | None = None, variant: str = "gaussian_head"):
        self.seeds = seeds if seeds is not None else _LCFG["seeds"]
        self.variant = variant
        self.members_: list[ProbLSTMForecaster] = []

    def fit(self, train: pd.DataFrame) -> None:
        self.members_ = []
        for seed in self.seeds:
            m = ProbLSTMForecaster(variant=self.variant)
            m.fit(train, seed=seed)
            self.members_.append(m)

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        frames = [m.predict(history) for m in self.members_]
        if not frames:
            return pd.DataFrame(columns=["forecast_log_rv", "forecast_rv", "forecast_sigma"])

        # Average mu in log space; combine sigmas: total_var = epistemic_var + aleatoric_mean
        # Ensemble mean: μ̄ = mean(μ_k)
        # Ensemble variance: σ̄² = mean(σ_k²) + mean(μ_k²) - μ̄²  (law of total variance)
        mu_df = pd.concat([f["forecast_log_rv"] for f in frames], axis=1)
        sig_df = pd.concat([f["forecast_sigma"] for f in frames], axis=1)

        mean_mu = mu_df.mean(axis=1)
        mean_var = (sig_df ** 2).mean(axis=1) + (mu_df ** 2).mean(axis=1) - mean_mu ** 2
        mean_sigma = np.sqrt(mean_var.clip(lower=1e-8))

        mean_mse = float(np.mean([m.mse_resid_ for m in self.members_ if m.mse_resid_ is not None]))
        rv_hat = np.exp(mean_mu + mean_mse / 2)

        out = pd.DataFrame(
            {"forecast_log_rv": mean_mu, "forecast_rv": rv_hat, "forecast_sigma": mean_sigma},
        )
        out.index = mean_mu.index
        out.index.names = ["date", "ticker"]
        return out.sort_index()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
/tmp/ml-vol-momentum-venv/bin/pytest tests/test_prob_lstm.py -v --tb=short 2>&1 | tail -20
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git add src/models/prob_lstm.py tests/test_prob_lstm.py
git commit -m "feat: probabilistic LSTM — Gaussian head + MC dropout variants with forecast_sigma output"
```

---

## Task 2: Uncertainty-weighted scaling (`src/strategy/uncertainty_scale.py`)

The existing `vol_scale` uses `weight_i = target_vol / sigma_hat_i * z_score(signal_i)`. Uncertainty scaling adds a second factor: `weight_i *= 1 / forecast_sigma_i` (normalised). Stocks where the model is uncertain get downweighted. This is the core hypothesis of Extension 1.

**Files:**
- Create: `src/strategy/uncertainty_scale.py`
- Create: `tests/test_uncertainty_scale.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_uncertainty_scale.py`:

```python
import pytest
import numpy as np
import pandas as pd
from src.strategy.uncertainty_scale import uncertainty_vol_scale


def _make_forecast(n_dates=10, n_tickers=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_dates, freq="ME")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d, "ticker": t,
                "forecast_rv": rng.uniform(0.001, 0.01),
                "forecast_sigma": rng.uniform(0.1, 1.0),
            })
    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    return df


def _make_signal(n_dates=10, n_tickers=5, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_dates, freq="ME")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({"date": d, "ticker": t, "signal": rng.standard_normal()})
    return pd.DataFrame(rows).set_index(["date", "ticker"])


class TestUncertaintyVolScale:
    def test_returns_weight_column(self):
        fc = _make_forecast()
        sig = _make_signal()
        w = uncertainty_vol_scale(sig, fc, target_vol=0.10)
        assert "weight" in w.columns

    def test_index_is_date_ticker(self):
        fc = _make_forecast()
        sig = _make_signal()
        w = uncertainty_vol_scale(sig, fc, target_vol=0.10)
        assert w.index.names == ["date", "ticker"]

    def test_high_sigma_gets_lower_weight(self):
        """Stock with 2× higher forecast_sigma should get lower absolute weight."""
        dates = pd.date_range("2010-01-01", periods=1, freq="ME")
        fc = pd.DataFrame({
            "forecast_rv": [0.005, 0.005],
            "forecast_sigma": [0.2, 0.4],  # T1 has 2× uncertainty
        }, index=pd.MultiIndex.from_tuples(
            [(dates[0], "T0"), (dates[0], "T1")], names=["date", "ticker"]))
        sig = pd.DataFrame({
            "signal": [1.0, 1.0],  # identical momentum signal
        }, index=fc.index)
        w = uncertainty_vol_scale(sig, fc, target_vol=0.10)
        assert abs(w.loc[(dates[0], "T0"), "weight"]) > abs(w.loc[(dates[0], "T1"), "weight"])

    def test_missing_sigma_falls_back_to_vol_scale(self):
        """If forecast_sigma is not present, should behave like plain vol_scale."""
        from src.strategy.scaling import vol_scale
        fc = _make_forecast()
        sig = _make_signal()
        fc_no_sigma = fc[["forecast_rv"]]
        w_unc = uncertainty_vol_scale(sig, fc_no_sigma, target_vol=0.10)
        w_plain = vol_scale(sig, fc, target_vol=0.10)
        pd.testing.assert_frame_equal(
            w_unc.sort_index(), w_plain.sort_index(), check_exact=False, atol=1e-6
        )
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
/tmp/ml-vol-momentum-venv/bin/pytest tests/test_uncertainty_scale.py -v --tb=short 2>&1 | tail -5
```

Expected: `ImportError: cannot import name 'uncertainty_vol_scale'`

- [ ] **Step 3: Implement `src/strategy/uncertainty_scale.py`**

```python
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
                z = pd.Series(0.0, index=sig_dt.index)
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
/tmp/ml-vol-momentum-venv/bin/pytest tests/test_uncertainty_scale.py -v --tb=short 2>&1 | tail -10
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git add src/strategy/uncertainty_scale.py tests/test_uncertainty_scale.py
git commit -m "feat: uncertainty-weighted vol scaling — downweights high-sigma forecast stocks"
```

---

## Task 3: Architecture comparison models

Three new model files implementing the `Forecaster` protocol, sharing sequence-building logic with the existing LSTM via a common base class.

**Files:**
- Create: `src/models/transformer_model.py`
- Create: `src/models/mlp_model.py`
- Create: `src/models/tcn_model.py`
- Create: `tests/test_architecture_comparison.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_architecture_comparison.py`:

```python
import pytest
import numpy as np
import pandas as pd
from src.models.transformer_model import TransformerForecaster, TransformerEnsemble
from src.models.mlp_model import MLPForecaster, MLPEnsemble
from src.models.tcn_model import TCNForecaster, TCNEnsemble
from src.models.forecaster import Forecaster


def _make_panel(n_dates=200, n_tickers=4, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    feat_cols = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]
    rows = []
    for t in tickers:
        df = pd.DataFrame(rng.standard_normal((n_dates, len(feat_cols))),
                          columns=feat_cols, index=dates)
        df["target_log_rv"] = rng.standard_normal(n_dates)
        df["target_rv"] = np.exp(df["target_log_rv"])
        df.index = pd.MultiIndex.from_arrays(
            [dates, [t]*n_dates], names=["date", "ticker"])
        rows.append(df)
    return pd.concat(rows)


@pytest.fixture
def panel():
    return _make_panel()


ARCHITECTURES = [
    ("transformer", TransformerForecaster, TransformerEnsemble),
    ("mlp", MLPForecaster, MLPEnsemble),
    ("tcn", TCNForecaster, TCNEnsemble),
]


@pytest.mark.parametrize("name,ForecasterClass,EnsembleClass", ARCHITECTURES)
class TestArchitectures:
    def test_implements_protocol(self, name, ForecasterClass, EnsembleClass, panel):
        m = EnsembleClass(seeds=[0])
        assert isinstance(m, Forecaster)

    def test_fit_and_predict(self, name, ForecasterClass, EnsembleClass, panel):
        m = ForecasterClass()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert "forecast_log_rv" in out.columns
        assert "forecast_rv" in out.columns
        assert len(out) > 0

    def test_forecast_rv_positive(self, name, ForecasterClass, EnsembleClass, panel):
        m = ForecasterClass()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert (out["forecast_rv"] > 0).all()

    def test_index_names(self, name, ForecasterClass, EnsembleClass, panel):
        m = ForecasterClass()
        m.fit(panel, seed=0)
        out = m.predict(panel)
        assert out.index.names == ["date", "ticker"]

    def test_ensemble_averages_seeds(self, name, ForecasterClass, EnsembleClass, panel):
        ens = EnsembleClass(seeds=[0, 1])
        ens.fit(panel)
        out = ens.predict(panel)
        assert len(out) > 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
/tmp/ml-vol-momentum-venv/bin/pytest tests/test_architecture_comparison.py -v --tb=short 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'TransformerForecaster'`

- [ ] **Step 3: Implement `src/models/transformer_model.py`**

Note: Transformer uses `d_model` that must equal `n_features` (9) or be projected. We project to `d_model=32` via a linear layer first.

```python
"""Transformer encoder forecaster for vol prediction.

Architecture: Linear projection → positional encoding → 2-layer TransformerEncoder
→ mean pooling over sequence → Linear head.

Expected behaviour: overfits on this dataset size (~3M obs but only 9 features and
60-step sequences). Report this honestly in the results.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import load_config
from src.models.forecaster import Forecaster
from src.models.lstm_model import FEATURE_COLS, _SEQ_LEN

_cfg = load_config()
_LCFG = _cfg["models"]["lstm"]  # reuse lstm training hyperparams
_TCFG = _cfg["models"].get("transformer", {})

_D_MODEL   = _TCFG.get("d_model", 32)
_N_HEADS   = _TCFG.get("n_heads", 4)
_N_LAYERS  = _TCFG.get("n_layers", 2)
_DROPOUT   = _TCFG.get("dropout", 0.1)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.drop(x)


class _TransformerNet(nn.Module):
    def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)           # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)        # (B, T, d_model)
        x = x.mean(dim=1)          # mean pooling over sequence
        return self.head(x).squeeze(-1)


class TransformerForecaster(Forecaster):
    name = "transformer"

    def __init__(self):
        self.model_: _TransformerNet | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._tgt_mean: float | None = None
        self._tgt_std: float | None = None
        self.mse_resid_: float | None = None
        self._train_tail_: pd.DataFrame | None = None

    def _build_sequences(self, panel, include_target):
        from src.models.lstm_model import LSTMForecaster
        # Reuse sequence builder from LSTMForecaster to avoid code duplication
        _tmp = LSTMForecaster()
        return _tmp._build_sequences(panel, include_target)

    def fit(self, train: pd.DataFrame, seed: int = 0) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        feat_cols = [c for c in FEATURE_COLS if c in train.columns]
        self._feat_mean = np.nanmean(train[feat_cols].values, axis=0)
        self._feat_std = np.nanstd(train[feat_cols].values, axis=0) + 1e-8
        tgt_vals = train["target_log_rv"].dropna().values
        self._tgt_mean = float(np.mean(tgt_vals))
        self._tgt_std = float(np.std(tgt_vals)) + 1e-8
        self._train_tail_ = train.copy()

        norm = train.copy()
        norm[feat_cols] = (norm[feat_cols] - self._feat_mean) / self._feat_std
        norm["target_log_rv"] = (norm["target_log_rv"] - self._tgt_mean) / self._tgt_std

        X, y, _ = self._build_sequences(norm, include_target=True)
        if len(X) == 0:
            return

        n_val = max(1, int(len(X) * _LCFG["val_fraction"]))
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        self.model_ = _TransformerNet(X.shape[2], _D_MODEL, _N_HEADS, _N_LAYERS, _DROPOUT)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=_LCFG["learning_rate"])
        loss_fn = nn.HuberLoss(delta=_LCFG["huber_delta"])

        X_tr_t = torch.from_numpy(X_tr)
        y_tr_t = torch.from_numpy(y_tr)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)
        tr_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=_LCFG["batch_size"], shuffle=True)

        best_val, patience_count, best_state = float("inf"), 0, None
        for _ in range(_LCFG["max_epochs"]):
            self.model_.train()
            for xb, yb in tr_dl:
                optimizer.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(X_val_t), y_val_t).item()
            if val_loss < best_val - 1e-6:
                best_val, patience_count = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= _LCFG["patience"]:
                    break
        if best_state:
            self.model_.load_state_dict(best_state)

        self.model_.eval()
        with torch.no_grad():
            val_pred_z = self.model_(X_val_t).numpy()
        val_pred = val_pred_z * self._tgt_std + self._tgt_mean
        val_true = y_val * self._tgt_std + self._tgt_mean
        self.mse_resid_ = float(np.mean((val_true - val_pred) ** 2))

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("Call fit() first")
        feat_cols = [c for c in FEATURE_COLS if c in history.columns]
        predict_dates = set(history.index.get_level_values("date").unique())

        if self._train_tail_ is not None:
            hist_tickers = set(history.index.get_level_values("ticker").unique())
            tail_mask = self._train_tail_.index.get_level_values("ticker").isin(hist_tickers)
            tail = self._train_tail_[tail_mask][feat_cols]
            combined = pd.concat([tail, history[feat_cols]]).sort_index()
        else:
            combined = history[feat_cols].copy()
        norm = combined.copy()
        norm[feat_cols] = (norm[feat_cols] - self._feat_mean) / self._feat_std

        X, _, valid_idx = self._build_sequences(norm, include_target=False)
        if len(X) == 0:
            return pd.DataFrame(columns=["forecast_log_rv", "forecast_rv"])

        self.model_.eval()
        with torch.no_grad():
            pred_z = self.model_(torch.from_numpy(X)).numpy()

        log_rv_hat = pred_z * self._tgt_std + self._tgt_mean
        rv_hat = np.exp(log_rv_hat + self.mse_resid_ / 2)

        dates = [idx[0] for idx in valid_idx]
        tickers = [idx[1] for idx in valid_idx]
        mask = np.array([d in predict_dates for d in dates])

        out = pd.DataFrame(
            {"forecast_log_rv": log_rv_hat[mask], "forecast_rv": rv_hat[mask]},
            index=pd.MultiIndex.from_arrays(
                [[d for d, m in zip(dates, mask) if m],
                 [t for t, m in zip(tickers, mask) if m]],
                names=["date", "ticker"],
            ),
        )
        return out.sort_index()


class TransformerEnsemble(Forecaster):
    name = "transformer_ensemble"

    def __init__(self, seeds: list[int] | None = None):
        self.seeds = seeds if seeds is not None else _LCFG["seeds"]
        self.members_: list[TransformerForecaster] = []

    def fit(self, train: pd.DataFrame) -> None:
        self.members_ = [TransformerForecaster() for _ in self.seeds]
        for m, seed in zip(self.members_, self.seeds):
            m.fit(train, seed=seed)

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        frames = [m.predict(history) for m in self.members_]
        if not frames:
            return pd.DataFrame(columns=["forecast_log_rv", "forecast_rv"])
        mean_log_rv = pd.concat([f["forecast_log_rv"] for f in frames], axis=1).mean(axis=1)
        mean_mse = float(np.mean([m.mse_resid_ for m in self.members_]))
        rv_hat = np.exp(mean_log_rv + mean_mse / 2)
        out = pd.DataFrame({"forecast_log_rv": mean_log_rv, "forecast_rv": rv_hat})
        out.index.names = ["date", "ticker"]
        return out.sort_index()
```

- [ ] **Step 4: Implement `src/models/mlp_model.py`**

MLP receives the full flattened sequence as input: `seq_len × n_features = 60 × 9 = 540` inputs. This tests whether the temporal structure in LSTM/TCN/Transformer is necessary or if a flat MLP picks up the same patterns.

```python
"""MLP forecaster — flattened sequence input, no temporal structure.

Tests whether sequence modelling (LSTM/Transformer/TCN) adds value over
a simple MLP that treats the 60×9 feature window as a flat 540-dim vector.
Expected result: similar or slightly lower IC than LSTM, faster training.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import load_config
from src.models.forecaster import Forecaster
from src.models.lstm_model import FEATURE_COLS, _SEQ_LEN

_cfg = load_config()
_LCFG = _cfg["models"]["lstm"]
_MCFG = _cfg["models"].get("mlp", {})
_HIDDEN = _MCFG.get("hidden_sizes", [256, 128, 64])
_DROPOUT = _MCFG.get("dropout", 0.25)


class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) — flatten to (batch, seq_len*n_features)
        return self.net(x.reshape(x.size(0), -1)).squeeze(-1)


class MLPForecaster(Forecaster):
    name = "mlp"

    def __init__(self):
        self.model_: _MLPNet | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._tgt_mean: float | None = None
        self._tgt_std: float | None = None
        self.mse_resid_: float | None = None
        self._train_tail_: pd.DataFrame | None = None

    def _build_sequences(self, panel, include_target):
        from src.models.lstm_model import LSTMForecaster
        return LSTMForecaster()._build_sequences(panel, include_target)

    def fit(self, train: pd.DataFrame, seed: int = 0) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        feat_cols = [c for c in FEATURE_COLS if c in train.columns]
        self._feat_mean = np.nanmean(train[feat_cols].values, axis=0)
        self._feat_std = np.nanstd(train[feat_cols].values, axis=0) + 1e-8
        tgt_vals = train["target_log_rv"].dropna().values
        self._tgt_mean = float(np.mean(tgt_vals))
        self._tgt_std = float(np.std(tgt_vals)) + 1e-8
        self._train_tail_ = train.copy()

        norm = train.copy()
        norm[feat_cols] = (norm[feat_cols] - self._feat_mean) / self._feat_std
        norm["target_log_rv"] = (norm["target_log_rv"] - self._tgt_mean) / self._tgt_std

        X, y, _ = self._build_sequences(norm, include_target=True)
        if len(X) == 0:
            return

        n_val = max(1, int(len(X) * _LCFG["val_fraction"]))
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        input_dim = X.shape[1] * X.shape[2]  # seq_len * n_features
        self.model_ = _MLPNet(input_dim, _HIDDEN, _DROPOUT)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=_LCFG["learning_rate"])
        loss_fn = nn.HuberLoss(delta=_LCFG["huber_delta"])

        X_tr_t = torch.from_numpy(X_tr)
        y_tr_t = torch.from_numpy(y_tr)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)
        tr_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=_LCFG["batch_size"], shuffle=True)

        best_val, patience_count, best_state = float("inf"), 0, None
        for _ in range(_LCFG["max_epochs"]):
            self.model_.train()
            for xb, yb in tr_dl:
                optimizer.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(X_val_t), y_val_t).item()
            if val_loss < best_val - 1e-6:
                best_val, patience_count = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= _LCFG["patience"]:
                    break
        if best_state:
            self.model_.load_state_dict(best_state)

        self.model_.eval()
        with torch.no_grad():
            val_pred_z = self.model_(X_val_t).numpy()
        val_pred = val_pred_z * self._tgt_std + self._tgt_mean
        val_true = y_val * self._tgt_std + self._tgt_mean
        self.mse_resid_ = float(np.mean((val_true - val_pred) ** 2))

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("Call fit() first")
        feat_cols = [c for c in FEATURE_COLS if c in history.columns]
        predict_dates = set(history.index.get_level_values("date").unique())

        if self._train_tail_ is not None:
            hist_tickers = set(history.index.get_level_values("ticker").unique())
            tail_mask = self._train_tail_.index.get_level_values("ticker").isin(hist_tickers)
            tail = self._train_tail_[tail_mask][feat_cols]
            combined = pd.concat([tail, history[feat_cols]]).sort_index()
        else:
            combined = history[feat_cols].copy()
        norm = combined.copy()
        norm[feat_cols] = (norm[feat_cols] - self._feat_mean) / self._feat_std

        X, _, valid_idx = self._build_sequences(norm, include_target=False)
        if len(X) == 0:
            return pd.DataFrame(columns=["forecast_log_rv", "forecast_rv"])

        self.model_.eval()
        with torch.no_grad():
            pred_z = self.model_(torch.from_numpy(X)).numpy()

        log_rv_hat = pred_z * self._tgt_std + self._tgt_mean
        rv_hat = np.exp(log_rv_hat + self.mse_resid_ / 2)
        dates = [idx[0] for idx in valid_idx]
        tickers = [idx[1] for idx in valid_idx]
        mask = np.array([d in predict_dates for d in dates])

        out = pd.DataFrame(
            {"forecast_log_rv": log_rv_hat[mask], "forecast_rv": rv_hat[mask]},
            index=pd.MultiIndex.from_arrays(
                [[d for d, m in zip(dates, mask) if m],
                 [t for t, m in zip(tickers, mask) if m]],
                names=["date", "ticker"],
            ),
        )
        return out.sort_index()


class MLPEnsemble(Forecaster):
    name = "mlp_ensemble"

    def __init__(self, seeds: list[int] | None = None):
        self.seeds = seeds if seeds is not None else _LCFG["seeds"]
        self.members_: list[MLPForecaster] = []

    def fit(self, train: pd.DataFrame) -> None:
        self.members_ = [MLPForecaster() for _ in self.seeds]
        for m, seed in zip(self.members_, self.seeds):
            m.fit(train, seed=seed)

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        frames = [m.predict(history) for m in self.members_]
        if not frames:
            return pd.DataFrame(columns=["forecast_log_rv", "forecast_rv"])
        mean_log_rv = pd.concat([f["forecast_log_rv"] for f in frames], axis=1).mean(axis=1)
        mean_mse = float(np.mean([m.mse_resid_ for m in self.members_]))
        rv_hat = np.exp(mean_log_rv + mean_mse / 2)
        out = pd.DataFrame({"forecast_log_rv": mean_log_rv, "forecast_rv": rv_hat})
        out.index.names = ["date", "ticker"]
        return out.sort_index()
```

- [ ] **Step 5: Implement `src/models/tcn_model.py`**

TCN uses dilated causal convolutions. Dilation doubles each layer: `[1, 2, 4, 8]` covers a receptive field of `2×(1+2+4+8)×kernel_size` timesteps. With `kernel_size=3` and 4 layers, receptive field = 48 — nearly the full 60-step sequence.

```python
"""Temporal Convolutional Network (TCN) for vol forecasting.

Architecture: stack of dilated causal convolutions with residual connections.
Dilation doubles each layer: [1, 2, 4, 8] → receptive field = 48 steps with kernel_size=3.
Expected: matches LSTM IC with faster training (no sequential dependency).

Reference: Bai, Kolter & Koltun (2018) "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling."
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import load_config
from src.models.forecaster import Forecaster
from src.models.lstm_model import FEATURE_COLS, _SEQ_LEN

_cfg = load_config()
_LCFG = _cfg["models"]["lstm"]
_TCNCFG = _cfg["models"].get("tcn", {})
_N_CHANNELS = _TCNCFG.get("n_channels", 32)
_KERNEL_SIZE = _TCNCFG.get("kernel_size", 3)
_N_LAYERS = _TCNCFG.get("n_layers", 4)
_DROPOUT = _TCNCFG.get("dropout", 0.2)


class _CausalConv1d(nn.Module):
    """Causal convolution: pads left so output length == input length."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.drop(self.relu(self.conv1(x)))
        out = self.drop(self.relu(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TCNNet(nn.Module):
    def __init__(self, n_features, n_channels, kernel_size, n_layers, dropout):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = n_features if i == 0 else n_channels
            layers.append(_TCNBlock(in_ch, n_channels, kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(n_channels, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features) → (batch, n_features, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x[:, :, -1]  # last timestep output
        return self.head(x).squeeze(-1)


class TCNForecaster(Forecaster):
    name = "tcn"

    def __init__(self):
        self.model_: _TCNNet | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._tgt_mean: float | None = None
        self._tgt_std: float | None = None
        self.mse_resid_: float | None = None
        self._train_tail_: pd.DataFrame | None = None

    def _build_sequences(self, panel, include_target):
        from src.models.lstm_model import LSTMForecaster
        return LSTMForecaster()._build_sequences(panel, include_target)

    def fit(self, train: pd.DataFrame, seed: int = 0) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        feat_cols = [c for c in FEATURE_COLS if c in train.columns]
        self._feat_mean = np.nanmean(train[feat_cols].values, axis=0)
        self._feat_std = np.nanstd(train[feat_cols].values, axis=0) + 1e-8
        tgt_vals = train["target_log_rv"].dropna().values
        self._tgt_mean = float(np.mean(tgt_vals))
        self._tgt_std = float(np.std(tgt_vals)) + 1e-8
        self._train_tail_ = train.copy()

        norm = train.copy()
        norm[feat_cols] = (norm[feat_cols] - self._feat_mean) / self._feat_std
        norm["target_log_rv"] = (norm["target_log_rv"] - self._tgt_mean) / self._tgt_std

        X, y, _ = self._build_sequences(norm, include_target=True)
        if len(X) == 0:
            return

        n_val = max(1, int(len(X) * _LCFG["val_fraction"]))
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        self.model_ = _TCNNet(X.shape[2], _N_CHANNELS, _KERNEL_SIZE, _N_LAYERS, _DROPOUT)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=_LCFG["learning_rate"])
        loss_fn = nn.HuberLoss(delta=_LCFG["huber_delta"])

        X_tr_t = torch.from_numpy(X_tr)
        y_tr_t = torch.from_numpy(y_tr)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)
        tr_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=_LCFG["batch_size"], shuffle=True)

        best_val, patience_count, best_state = float("inf"), 0, None
        for _ in range(_LCFG["max_epochs"]):
            self.model_.train()
            for xb, yb in tr_dl:
                optimizer.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(X_val_t), y_val_t).item()
            if val_loss < best_val - 1e-6:
                best_val, patience_count = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= _LCFG["patience"]:
                    break
        if best_state:
            self.model_.load_state_dict(best_state)

        self.model_.eval()
        with torch.no_grad():
            val_pred_z = self.model_(X_val_t).numpy()
        val_pred = val_pred_z * self._tgt_std + self._tgt_mean
        val_true = y_val * self._tgt_std + self._tgt_mean
        self.mse_resid_ = float(np.mean((val_true - val_pred) ** 2))

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("Call fit() first")
        feat_cols = [c for c in FEATURE_COLS if c in history.columns]
        predict_dates = set(history.index.get_level_values("date").unique())

        if self._train_tail_ is not None:
            hist_tickers = set(history.index.get_level_values("ticker").unique())
            tail_mask = self._train_tail_.index.get_level_values("ticker").isin(hist_tickers)
            tail = self._train_tail_[tail_mask][feat_cols]
            combined = pd.concat([tail, history[feat_cols]]).sort_index()
        else:
            combined = history[feat_cols].copy()
        norm = combined.copy()
        norm[feat_cols] = (norm[feat_cols] - self._feat_mean) / self._feat_std

        X, _, valid_idx = self._build_sequences(norm, include_target=False)
        if len(X) == 0:
            return pd.DataFrame(columns=["forecast_log_rv", "forecast_rv"])

        self.model_.eval()
        with torch.no_grad():
            pred_z = self.model_(torch.from_numpy(X)).numpy()

        log_rv_hat = pred_z * self._tgt_std + self._tgt_mean
        rv_hat = np.exp(log_rv_hat + self.mse_resid_ / 2)
        dates = [idx[0] for idx in valid_idx]
        tickers = [idx[1] for idx in valid_idx]
        mask = np.array([d in predict_dates for d in dates])

        out = pd.DataFrame(
            {"forecast_log_rv": log_rv_hat[mask], "forecast_rv": rv_hat[mask]},
            index=pd.MultiIndex.from_arrays(
                [[d for d, m in zip(dates, mask) if m],
                 [t for t, m in zip(tickers, mask) if m]],
                names=["date", "ticker"],
            ),
        )
        return out.sort_index()


class TCNEnsemble(Forecaster):
    name = "tcn_ensemble"

    def __init__(self, seeds: list[int] | None = None):
        self.seeds = seeds if seeds is not None else _LCFG["seeds"]
        self.members_: list[TCNForecaster] = []

    def fit(self, train: pd.DataFrame) -> None:
        self.members_ = [TCNForecaster() for _ in self.seeds]
        for m, seed in zip(self.members_, self.seeds):
            m.fit(train, seed=seed)

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        frames = [m.predict(history) for m in self.members_]
        if not frames:
            return pd.DataFrame(columns=["forecast_log_rv", "forecast_rv"])
        mean_log_rv = pd.concat([f["forecast_log_rv"] for f in frames], axis=1).mean(axis=1)
        mean_mse = float(np.mean([m.mse_resid_ for m in self.members_]))
        rv_hat = np.exp(mean_log_rv + mean_mse / 2)
        out = pd.DataFrame({"forecast_log_rv": mean_log_rv, "forecast_rv": rv_hat})
        out.index.names = ["date", "ticker"]
        return out.sort_index()
```

- [ ] **Step 6: Run architecture tests**

```bash
/tmp/ml-vol-momentum-venv/bin/pytest tests/test_architecture_comparison.py -v --tb=short 2>&1 | tail -20
```

Expected: all 15 tests pass (3 architectures × 5 test cases).

- [ ] **Step 7: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git add src/models/transformer_model.py src/models/mlp_model.py src/models/tcn_model.py tests/test_architecture_comparison.py
git commit -m "feat: architecture comparison — Transformer, MLP, TCN sequence forecasters"
```

---

## Task 4: Update `configs/default.yaml`

Add config blocks for the new architectures and the probabilistic LSTM.

**Files:**
- Modify: `configs/default.yaml`

- [ ] **Step 1: Read current default.yaml**

```bash
cat "/Users/harry/RL:ML Project/ml-vol-momentum/configs/default.yaml"
```

- [ ] **Step 2: Append new model config blocks after the `lstm:` block**

Find the end of the `lstm:` block and add:

```yaml
  prob_lstm:
    variant: "gaussian_head"       # or "mc_dropout"
    n_mc_samples: 50               # for mc_dropout variant only
    # inherits sequence_length, hidden_size, dropout, optimizer, lr, batch_size,
    # max_epochs, patience, val_fraction, seeds from lstm block

  transformer:
    d_model: 32
    n_heads: 4
    n_layers: 2
    dropout: 0.1

  mlp:
    hidden_sizes: [256, 128, 64]
    dropout: 0.25

  tcn:
    n_channels: 32
    kernel_size: 3
    n_layers: 4
    dropout: 0.2
```

- [ ] **Step 3: Verify yaml loads without error**

```bash
python3 -c "
import yaml
cfg = yaml.safe_load(open('/Users/harry/RL:ML Project/ml-vol-momentum/configs/default.yaml'))
print('Keys under models:', list(cfg['models'].keys()))
"
```

Expected: `Keys under models: ['rolling_vol', 'garch', 'har_rv', 'gbm', 'lstm', 'prob_lstm', 'transformer', 'mlp', 'tcn']`

- [ ] **Step 4: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git add configs/default.yaml
git commit -m "config: add prob_lstm, transformer, mlp, tcn hyperparameter blocks"
```

---

## Task 5: Extension 1 pipeline runner (`scripts/run_extension1.py`)

Runs the probabilistic LSTM walk-forward, computes the uncertainty-weighted strategy alongside the plain vol-scaled strategy, and saves all outputs.

**Files:**
- Create: `scripts/run_extension1.py`

- [ ] **Step 1: Create the script**

```python
"""Extension 1: Probabilistic LSTM walk-forward + uncertainty-weighted strategy.

Outputs:
  results/forecasts/prob_lstm.parquet          — (mu, sigma) OOS forecasts
  results/strategies/prob_lstm_scaled.parquet  — plain vol-scaled (using mu only)
  results/strategies/prob_lstm_unc_scaled.parquet — uncertainty-weighted sizing
  results/extension1_summary.md
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from pathlib import Path
from src.data.universe import get_universe, get_sector
from src.data.loaders import load_ohlcv, load_vix
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.models.prob_lstm import ProbLSTMEnsemble
from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.uncertainty_scale import uncertainty_vol_scale
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.eval.walk_forward import generate_windows
from src.eval.tests import cross_sectional_ic
from src.eval.metrics import sharpe, max_drawdown

CKPT_DIR = Path("/tmp/prob_lstm_checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
Path("results/forecasts").mkdir(parents=True, exist_ok=True)
Path("results/strategies").mkdir(parents=True, exist_ok=True)

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print(f"Loading {len(tickers)} tickers...")

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
signal = momentum_signal(ohlcv[["close"]], lookback=252, skip=21)
realized_rv = panel["target_rv"]

# Check cache
fcast_path = Path("results/forecasts/prob_lstm.parquet")
if fcast_path.exists():
    print("prob_lstm forecasts: loading cached...")
    oos_all = pd.read_parquet(fcast_path)
else:
    model = ProbLSTMEnsemble()
    oos_frames = []
    for i, w in enumerate(windows):
        ckpt = CKPT_DIR / f"window_{i:02d}.parquet"
        if ckpt.exists():
            print(f"  Window {i+1}/{len(windows)}: {w.test_end.year} (cached)")
            oos_frames.append(pd.read_parquet(ckpt))
            continue
        print(f"  Window {i+1}/{len(windows)}: {w.train_end.year} → {w.test_end.year}")
        train_mask = ((panel.index.get_level_values("date") >= w.train_start) &
                      (panel.index.get_level_values("date") <= w.train_end))
        model.fit(panel[train_mask])
        history = panel[panel.index.get_level_values("date") <= w.test_end]
        preds = model.predict(history)
        if preds.empty:
            print("    WARNING: empty preds")
            continue
        pred_dates = preds.index.get_level_values("date")
        oos = preds[(pred_dates >= w.test_start) & (pred_dates <= w.test_end)]
        print(f"    OOS rows: {len(oos)}")
        oos.to_parquet(ckpt)
        oos_frames.append(oos)
    oos_all = pd.concat(oos_frames).sort_index()
    oos_all.to_parquet(fcast_path)

ic = cross_sectional_ic(oos_all, realized_rv.rename("target_rv").to_frame())
print(f"\nProb LSTM OOS rows: {len(oos_all)}  Mean IC: {ic.mean():.4f}")

# Plain vol-scaled strategy (uses mu forecast, ignores sigma)
strat_path = Path("results/strategies/prob_lstm_scaled.parquet")
if not strat_path.exists():
    w_scaled_raw = vol_scale(signal, oos_all, target_vol=0.10)
    w_scaled = build_portfolios(None, weights=w_scaled_raw, mode="vol_targeted_gross")
    net_scaled = apply_costs(w_scaled, returns_panel, cost_bps=10.0).dropna()
    net_scaled.to_frame("net_return").to_parquet(strat_path)
else:
    net_scaled = pd.read_parquet(strat_path)["net_return"]

# Uncertainty-weighted strategy
unc_path = Path("results/strategies/prob_lstm_unc_scaled.parquet")
if not unc_path.exists():
    w_unc_raw = uncertainty_vol_scale(signal, oos_all, target_vol=0.10)
    w_unc = build_portfolios(None, weights=w_unc_raw, mode="vol_targeted_gross")
    net_unc = apply_costs(w_unc, returns_panel, cost_bps=10.0).dropna()
    net_unc.to_frame("net_return").to_parquet(unc_path)
else:
    net_unc = pd.read_parquet(unc_path)["net_return"]

sh_plain = sharpe(net_scaled)
sh_unc   = sharpe(net_unc)
dd_plain = max_drawdown(net_scaled.cumsum())
dd_unc   = max_drawdown(net_unc.cumsum())

# Load existing LSTM result for comparison
lstm_strat = pd.read_parquet("results/strategies/lstm_scaled.parquet")["net_return"]
sh_lstm = sharpe(lstm_strat)

print(f"\n=== Extension 1 Results ===")
print(f"LSTM point forecast (existing):  Sharpe={sh_lstm:.3f}")
print(f"Prob LSTM plain vol-scale:        Sharpe={sh_plain:.3f}  Max DD={dd_plain:.1%}")
print(f"Prob LSTM uncertainty-weighted:   Sharpe={sh_unc:.3f}  Max DD={dd_unc:.1%}")

summary = f"""# Extension 1: Probabilistic LSTM Results

## Forecast Quality
- OOS rows: {len(oos_all)}
- Mean cross-sectional IC: {ic.mean():.4f} (vs point LSTM 0.739)
- IC std across windows: {ic.std():.4f}

## Strategy Comparison

| Strategy | Sharpe | Max DD |
|----------|--------|--------|
| LSTM point forecast (baseline) | {sh_lstm:.3f} | — |
| Prob LSTM — plain vol-scale | {sh_plain:.3f} | {dd_plain:.1%} |
| Prob LSTM — uncertainty-weighted | {sh_unc:.3f} | {dd_unc:.1%} |

## Interpretation
{"Uncertainty weighting IMPROVED Sharpe vs plain vol-scale." if sh_unc > sh_plain else "Uncertainty weighting did NOT improve Sharpe vs plain vol-scale."}
{"Prob LSTM IMPROVED on point LSTM." if sh_plain > sh_lstm else "Prob LSTM did NOT improve on point LSTM."}
"""

Path("results/extension1_summary.md").write_text(summary)
print("\nResults written to results/extension1_summary.md")
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git add scripts/run_extension1.py
git commit -m "feat: Extension 1 pipeline runner — probabilistic LSTM + uncertainty-weighted strategy"
```

---

## Task 6: Extension 2 pipeline runner (`scripts/run_extension2.py`)

Runs all four architectures (LSTM baseline, Transformer, MLP, TCN) and produces the comparison table.

**Files:**
- Create: `scripts/run_extension2.py`

- [ ] **Step 1: Create the script**

```python
"""Extension 2: Architecture comparison — LSTM vs Transformer vs MLP vs TCN.

For each architecture:
  - Run walk-forward with 5-seed ensemble
  - Report mean IC, IC std across 22 windows, Sharpe, training time per window

Outputs:
  results/forecasts/{arch}.parquet
  results/strategies/{arch}_scaled.parquet
  results/extension2_summary.md
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from pathlib import Path
from src.data.universe import get_universe, get_sector
from src.data.loaders import load_ohlcv, load_vix
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.models.lstm_model import LSTMEnsemble
from src.models.transformer_model import TransformerEnsemble
from src.models.mlp_model import MLPEnsemble
from src.models.tcn_model import TCNEnsemble
from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.eval.walk_forward import generate_windows
from src.eval.tests import cross_sectional_ic
from src.eval.metrics import sharpe, max_drawdown

Path("results/forecasts").mkdir(parents=True, exist_ok=True)
Path("results/strategies").mkdir(parents=True, exist_ok=True)

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print(f"Loading {len(tickers)} tickers...")

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
signal = momentum_signal(ohlcv[["close"]], lookback=252, skip=21)
realized_rv = panel["target_rv"]

ARCHITECTURES = [
    ("lstm",        LSTMEnsemble),
    ("transformer", TransformerEnsemble),
    ("mlp",         MLPEnsemble),
    ("tcn",         TCNEnsemble),
]

results = {}

for arch_name, EnsembleClass in ARCHITECTURES:
    fcast_path = Path(f"results/forecasts/{arch_name}.parquet")
    strat_path = Path(f"results/strategies/{arch_name}_scaled.parquet")

    if fcast_path.exists() and strat_path.exists():
        print(f"\n{arch_name}: loading cached results...")
        oos = pd.read_parquet(fcast_path)
        net = pd.read_parquet(strat_path)["net_return"]
    else:
        print(f"\n{arch_name}: running walk-forward...")
        ckpt_dir = Path(f"/tmp/{arch_name}_ext2_checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        model = EnsembleClass()
        oos_frames = []
        window_times = []

        for i, w in enumerate(windows):
            ckpt = ckpt_dir / f"window_{i:02d}.parquet"
            if ckpt.exists():
                print(f"  Window {i+1}/{len(windows)}: {w.test_end.year} (cached)")
                oos_frames.append(pd.read_parquet(ckpt))
                continue
            t0 = time.time()
            print(f"  Window {i+1}/{len(windows)}: {w.train_end.year} → {w.test_end.year}", end="", flush=True)
            train_mask = ((panel.index.get_level_values("date") >= w.train_start) &
                          (panel.index.get_level_values("date") <= w.train_end))
            model.fit(panel[train_mask])
            history = panel[panel.index.get_level_values("date") <= w.test_end]
            preds = model.predict(history)
            if preds.empty:
                print(" WARNING: empty")
                continue
            pred_dates = preds.index.get_level_values("date")
            oos_w = preds[(pred_dates >= w.test_start) & (pred_dates <= w.test_end)]
            elapsed = time.time() - t0
            window_times.append(elapsed)
            print(f"  rows={len(oos_w)}  {elapsed:.0f}s")
            oos_w.to_parquet(ckpt)
            oos_frames.append(oos_w)

        oos = pd.concat(oos_frames).sort_index()
        oos.to_parquet(fcast_path)
        print(f"  Avg time per window: {np.mean(window_times):.0f}s")

        w_scaled_raw = vol_scale(signal, oos, target_vol=0.10)
        w_scaled = build_portfolios(None, weights=w_scaled_raw, mode="vol_targeted_gross")
        net = apply_costs(w_scaled, returns_panel, cost_bps=10.0).dropna()
        net.to_frame("net_return").to_parquet(strat_path)

    ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
    sh = sharpe(net)
    dd = max_drawdown(net.cumsum())
    results[arch_name] = {
        "mean_ic": ic.mean(),
        "std_ic": ic.std(),
        "sharpe": sh,
        "max_dd": dd,
        "n_oos": len(oos),
    }
    print(f"  IC={ic.mean():.4f} ± {ic.std():.4f}  Sharpe={sh:.3f}  MaxDD={dd:.1%}")

# Write summary
rows = []
for arch, r in results.items():
    rows.append({
        "Architecture": arch.upper(),
        "Mean XS-IC": f"{r['mean_ic']:.4f}",
        "IC Std": f"{r['std_ic']:.4f}",
        "Sharpe": f"{r['sharpe']:.3f}",
        "Max DD": f"{r['max_dd']:.1%}",
    })

df_results = pd.DataFrame(rows)
print("\n=== Extension 2 Results ===")
print(df_results.to_string(index=False))

summary_lines = [
    "# Extension 2: Architecture Comparison Results\n",
    "All architectures use: same 9 features, seq_len=60, 5-seed ensemble, Huber loss,",
    "same walk-forward CV (22 windows, 42-day embargo, 2003-2024).\n",
    "## Results\n",
    "| Architecture | Mean XS-IC | IC Std | Sharpe | Max DD |",
    "|-------------|-----------|--------|--------|--------|",
]
for r in rows:
    summary_lines.append(
        f"| {r['Architecture']} | {r['Mean XS-IC']} | {r['IC Std']} | {r['Sharpe']} | {r['Max DD']} |"
    )

summary_lines += [
    "\n## Interpretation",
    f"- Best IC: {max(results, key=lambda k: results[k]['mean_ic']).upper()}",
    f"- Best Sharpe: {max(results, key=lambda k: results[k]['sharpe']).upper()}",
    f"- Does temporal structure help vs MLP? {'YES' if results.get('lstm', {}).get('mean_ic', 0) > results.get('mlp', {}).get('mean_ic', 0) else 'NO'} (LSTM IC > MLP IC)",
    f"- Does Transformer overfit? {'YES (lower IC than LSTM)' if results.get('transformer', {}).get('mean_ic', 1) < results.get('lstm', {}).get('mean_ic', 0) else 'NO (Transformer IC >= LSTM)'}",
    f"- Does TCN match LSTM? {'YES' if abs(results.get('tcn', {}).get('mean_ic', 0) - results.get('lstm', {}).get('mean_ic', 0)) < 0.01 else 'NO, gap > 0.01'}",
]

Path("results/extension2_summary.md").write_text("\n".join(summary_lines))
print("\nResults written to results/extension2_summary.md")
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git add scripts/run_extension2.py
git commit -m "feat: Extension 2 pipeline runner — 4-architecture comparison with checkpoints"
```

---

## Task 7: Run Extension 1 and update README

Run the probabilistic LSTM walk-forward (this takes ~2h on CPU for 22 windows × 5 seeds). Update README with actual results.

**Files:**
- `results/forecasts/prob_lstm.parquet` (will be created)
- `results/strategies/prob_lstm_scaled.parquet` (will be created)
- `results/strategies/prob_lstm_unc_scaled.parquet` (will be created)
- `results/extension1_summary.md` (will be created)
- Modify: `README.md`

- [ ] **Step 1: Run Extension 1**

```bash
PYTHONPATH="/tmp/ml-vol-momentum-venv/lib/python3.14/site-packages" \
  /tmp/ml-vol-momentum-venv/bin/python -u scripts/run_extension1.py 2>&1 | tee /tmp/ext1.log
```

Monitor progress:
```bash
tail -f /tmp/ext1.log
```

Expected: 22 windows printed, then summary table. Takes ~90-120 min.

- [ ] **Step 2: Read extension1_summary.md**

```bash
cat results/extension1_summary.md
```

Note the actual IC and Sharpe numbers.

- [ ] **Step 3: Add Extension 1 section to README.md**

Append a new section after `## Honest Evaluation` using the actual numbers from step 2:

```markdown
---

## Extension 1: Probabilistic LSTM

**Hypothesis:** Point forecasts cause miscalibrated weights because `weight = target_vol / sigma_hat` is level-sensitive. A distributional forecast outputs `(μ, σ)` — expected log-RV and forecast uncertainty. Downweighting high-σ positions should reduce sizing errors and improve Sharpe.

**Implementation:** Gaussian output head predicts `(μ, log_σ²)` jointly, trained with negative log-likelihood. 5-seed ensemble via law of total variance: `σ_ensemble² = mean(σ_k²) + variance(μ_k)`.

**Uncertainty-weighted sizing:**
```
weight_i = (target_vol / ann_vol_i) × (1 / norm_sigma_i) × z_score(signal_i)
```
where `norm_sigma_i = sigma_i / median(sigma)` — cross-sectionally normalised so regime-level shifts in σ don't dominate.

| Strategy | Mean XS-IC | Sharpe | Max DD |
|----------|-----------|--------|--------|
| LSTM point forecast (baseline) | 0.739 | -0.041 | -74% |
| Prob LSTM — plain vol-scale | [INSERT] | [INSERT] | [INSERT] |
| Prob LSTM — uncertainty-weighted | [INSERT] | [INSERT] | [INSERT] |

[Fill in actual numbers from results/extension1_summary.md]
```

- [ ] **Step 4: Commit results and README update**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git add results/extension1_summary.md results/forecasts/prob_lstm.parquet \
    results/strategies/prob_lstm_scaled.parquet results/strategies/prob_lstm_unc_scaled.parquet \
    README.md
git commit -m "results: Extension 1 — probabilistic LSTM walk-forward complete"
git push origin main
```

---

## Task 8: Run Extension 2 and update README

Run all 4 architecture walk-forwards. The LSTM is already cached so only Transformer, MLP, TCN need training. Update README with actual results.

**Files:**
- `results/forecasts/{transformer,mlp,tcn}.parquet` (will be created)
- `results/strategies/{transformer,mlp,tcn}_scaled.parquet` (will be created)
- `results/extension2_summary.md` (will be created)
- Modify: `README.md`

- [ ] **Step 1: Run Extension 2**

```bash
PYTHONPATH="/tmp/ml-vol-momentum-venv/lib/python3.14/site-packages" \
  /tmp/ml-vol-momentum-venv/bin/python -u scripts/run_extension2.py 2>&1 | tee /tmp/ext2.log
```

Monitor:
```bash
tail -f /tmp/ext2.log
```

Expected: LSTM loads from cache, then Transformer, MLP, TCN train sequentially. Total ~4-6h.

- [ ] **Step 2: Read extension2_summary.md**

```bash
cat results/extension2_summary.md
```

- [ ] **Step 3: Add Extension 2 section to README.md**

Add after the Extension 1 section using actual numbers:

```markdown
---

## Extension 2: Architecture Comparison

**Question:** Does temporal structure help? Does the Transformer overfit? Does TCN match LSTM with faster training?

All architectures use identical setup: 9 features, seq_len=60, 5-seed ensemble, Huber loss, same walk-forward CV.

| Architecture | Mean XS-IC | IC Std | Sharpe | Notes |
|-------------|-----------|--------|--------|-------|
| LSTM (baseline) | 0.739 | — | -0.041 | 1-layer, hidden=48 |
| TCN | [INSERT] | [INSERT] | [INSERT] | Dilated causal conv, 4 layers |
| MLP | [INSERT] | [INSERT] | [INSERT] | Flattened 540-dim input |
| Transformer | [INSERT] | [INSERT] | [INSERT] | 2-layer encoder, d_model=32 |

**TCN receptive field:** With kernel_size=3 and dilations [1,2,4,8], covers 48 of the 60 input timesteps — nearly the full context window.

[Fill in actual numbers and interpretation]
```

- [ ] **Step 4: Commit everything and push**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git add results/extension2_summary.md results/forecasts/transformer.parquet \
    results/forecasts/mlp.parquet results/forecasts/tcn.parquet \
    results/strategies/transformer_scaled.parquet results/strategies/mlp_scaled.parquet \
    results/strategies/tcn_scaled.parquet README.md
git commit -m "results: Extension 2 — architecture comparison complete (Transformer, MLP, TCN)"
git push origin main
```

---

## Self-Review

**Spec coverage:**
- ✅ Probabilistic LSTM (Gaussian head + MC dropout) — Task 1
- ✅ Uncertainty-weighted scaling — Task 2
- ✅ Transformer, MLP, TCN architectures — Task 3
- ✅ Config blocks for all new models — Task 4
- ✅ Extension 1 pipeline runner with checkpointing — Task 5
- ✅ Extension 2 pipeline runner with checkpointing — Task 6
- ✅ Walk-forward results + README update — Tasks 7-8

**Placeholder check:** Tasks 7-8 have `[INSERT]` placeholders in the README templates — these are intentional, to be replaced with actual numbers after training completes.

**Protocol compliance:** All new Forecasters implement `name`, `fit(train)`, `predict(history)` returning `forecast_log_rv` and `forecast_rv`. ProbLSTM also returns `forecast_sigma`. Existing `run_walk_forward` works without modification.

**No-modification guarantee:** `src/eval/walk_forward.py`, `src/strategy/scaling.py`, `src/models/lstm_model.py`, `src/models/forecaster.py` are not touched.
