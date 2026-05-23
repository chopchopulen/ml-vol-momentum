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

        feat_cols_only = [c for c in FEATURE_COLS if c in train.columns]
        self._train_tail_ = (
            train[feat_cols_only]
            .groupby(level="ticker", group_keys=False)
            .apply(lambda g: g.sort_index().iloc[-_SEQ_LEN:])
        )

        norm = train.copy()
        norm[feat_cols] = (norm[feat_cols] - self._feat_mean) / self._feat_std
        norm["target_log_rv"] = (norm["target_log_rv"] - self._tgt_mean) / self._tgt_std

        X, y, _ = self._build_sequences(norm, include_target=True)
        if len(X) == 0:
            self.mse_resid_ = 1.0
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
            combined = pd.concat([self._train_tail_, history[feat_cols]]).sort_index()
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
