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
            # Ensure sigma is positive (std could be 0 if n_mc_samples=1 or dropout=0)
            sigma_hat = np.clip(sigma_hat, 1e-8, None)

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
