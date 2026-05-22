# src/viz/plots.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

FIGURES_DIR = Path("docs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_COLORS = {
    "rolling_vol": "#888888",
    "har_rv":      "#2196F3",
    "garch":       "#FF9800",
    "gbm":         "#4CAF50",
    "lstm":        "#9C27B0",
    "unscaled_momentum": "#F44336",
}


def plot_shap_importance(shap_importance: pd.Series, save_path: str | None = None) -> None:
    """Horizontal bar chart of mean |SHAP| per feature."""
    fig, ax = plt.subplots(figsize=(7, 4))
    shap_importance.sort_values().plot.barh(ax=ax, color="#4CAF50", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("GBM Feature Importance (SHAP)")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    path = save_path or str(FIGURES_DIR / "shap_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ic_by_year(
    forecasts: dict[str, pd.DataFrame],
    realized_rv: pd.Series,
    save_path: str | None = None,
) -> None:
    """Line chart: cross-sectional IC per calendar year, one line per model."""
    from src.eval.tests import cross_sectional_ic
    fig, ax = plt.subplots(figsize=(12, 5))
    for model_name, oos in forecasts.items():
        ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
        ic_annual = ic.groupby(ic.index.year).mean()
        ax.plot(ic_annual.index, ic_annual.values,
                label=model_name, color=_MODEL_COLORS.get(model_name),
                marker="o", markersize=4, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Cross-Sectional IC (Spearman)")
    ax.set_title("OOS Cross-Sectional IC by Year")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    path = save_path or str(FIGURES_DIR / "ic_by_year.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_regime_ic_heatmap(
    regime_table: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Color-coded heatmap: rows=models, cols=low/mid/high VIX regime, values=mean IC."""
    fig, ax = plt.subplots(figsize=(6, 4))
    data = regime_table[["low", "mid", "high"]].values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto",
                   vmin=data[~np.isnan(data)].min() - 0.05,
                   vmax=data[~np.isnan(data)].max() + 0.05)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Low VIX", "Mid VIX", "High VIX"])
    ax.set_yticks(range(len(regime_table)))
    ax.set_yticklabels(regime_table.index)
    for i in range(len(regime_table)):
        for j in range(3):
            v = data[i, j]
            ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "n/a",
                    ha="center", va="center", fontsize=9,
                    color="black" if 0.3 < (v - data.min()) / (data.max() - data.min() + 1e-9) < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Mean IC")
    ax.set_title("Cross-Sectional IC by VIX Regime")
    fig.tight_layout()
    path = save_path or str(FIGURES_DIR / "regime_ic_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_equity_curves(
    strategies: dict[str, pd.Series],
    save_path: str | None = None,
) -> None:
    """Cumulative log-return equity curves for each strategy."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, rets in strategies.items():
        eq = rets.cumsum()
        ax.plot(eq.index, eq.values, label=name,
                color=_MODEL_COLORS.get(name, "#333333"),
                linewidth=1.5,
                linestyle="--" if name == "unscaled_momentum" else "-")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log Return")
    ax.set_title("Vol-Scaled Momentum — Equity Curves (Net of 10bps)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = save_path or str(FIGURES_DIR / "equity_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
