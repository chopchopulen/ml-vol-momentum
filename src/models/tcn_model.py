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
