from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import load_config
from src.models.forecaster import Forecaster

_cfg = load_config()
_LCFG = _cfg["models"]["lstm"]

FEATURE_COLS = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]
_SEQ_LEN = _LCFG["sequence_length"]  # 60


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])  # last timestep
        return self.head(out).squeeze(-1)


class LSTMForecaster(Forecaster):
    name = "lstm"

    def __init__(self):
        self.model_: _LSTMNet | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std:  np.ndarray | None = None
        self._tgt_mean:  float | None = None
        self._tgt_std:   float | None = None
        self.mse_resid_: float | None = None
        self._train_tail_: pd.DataFrame | None = None

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
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        feat_cols = [c for c in FEATURE_COLS if c in train.columns]
        # Store tail of training data per ticker for lookback context in predict()
        self._train_tail_: pd.DataFrame = (
            train[feat_cols]
            .groupby(level="ticker", group_keys=False)
            .apply(lambda g: g.sort_index().iloc[-_SEQ_LEN:])
        )

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
        n_feat = len(feat_cols)

        # Always initialise the model (may be untrained if insufficient data)
        self.model_ = _LSTMNet(n_feat, _LCFG["hidden_size"], _LCFG["num_layers"], _LCFG["dropout"])

        if len(X) == 0:
            # No sequences to train on; model stays at random initialisation
            self.mse_resid_ = 1.0
            return

        # Chronological 90/10 split — no shuffle
        n = len(X)
        n_val = max(int(n * _LCFG["val_fraction"]), 1)
        X_tr, y_tr = X[: n - n_val], y[: n - n_val]
        X_val, y_val = X[n - n_val :], y[n - n_val :]
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

        # Store sigma² from validation residuals (unbiased, not overfit) for Jensen correction
        self.model_.eval()
        with torch.no_grad():
            val_pred_z = self.model_(torch.from_numpy(X_val)).numpy()
        val_pred = val_pred_z * self._tgt_std + self._tgt_mean
        val_true = y_val * self._tgt_std + self._tgt_mean
        self.mse_resid_ = float(np.mean((val_true - val_pred) ** 2))

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None or self.mse_resid_ is None:
            raise RuntimeError("Call fit() before predict()")
        feat_cols = [c for c in FEATURE_COLS if c in history.columns]

        # Prepend training tail for lookback context (only feature cols; no target needed)
        # This ensures sequences can be built even when history window < seq_len
        if self._train_tail_ is not None and len(self._train_tail_) > 0:
            tail_feats = self._train_tail_.reindex(columns=feat_cols)
            hist_feats = history.reindex(columns=feat_cols)
            combined_feats = pd.concat([tail_feats, hist_feats]).sort_index()
            # Keep only the feature columns for sequence building
            norm_combined = combined_feats.copy()
            norm_combined[feat_cols] = (norm_combined[feat_cols] - self._feat_mean) / self._feat_std
        else:
            norm_combined = history[feat_cols].copy()
            norm_combined[feat_cols] = (norm_combined[feat_cols] - self._feat_mean) / self._feat_std

        # Track which dates belong to the actual prediction window
        predict_dates = set(history.index.get_level_values("date").unique())

        X, _, valid_idx = self._build_sequences(norm_combined, include_target=False)
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

        # Filter to only dates in the original history (not the prepended tail)
        mask = np.array([d in predict_dates for d in dates])
        dates   = [d for d, m in zip(dates, mask) if m]
        tickers = [t for t, m in zip(tickers, mask) if m]
        log_rv_hat = log_rv_hat[mask]
        rv_hat     = rv_hat[mask]

        out_idx = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
        out = pd.DataFrame(
            {"forecast_log_rv": log_rv_hat, "forecast_rv": rv_hat},
            index=out_idx,
        )
        return out.sort_index()
