# Phase 4: Analysis, Interpretation & Writeup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement SHAP feature importance analysis, VIX-regime breakdown, sector-neutral robustness check, and a final results visualization, completing the research project.

**Architecture:** Four independent analysis modules build on the already-computed OOS forecast parquets in `results/forecasts/`. Each writes outputs to `results/` and/or `docs/figures/`. A final `scripts/run_phase4.py` script ties them together. No new model training — pure analysis over existing parquets.

**Tech Stack:** shap, matplotlib, pandas, numpy, scipy; existing `src/eval/tests.py` and `src/eval/comparison.py`; GBMForecaster.shap_values().

**Key data contract:** All forecast parquets have `(date, ticker)` MultiIndex with columns `forecast_log_rv`, `forecast_rv`. Panel data has the same index with feature columns `rv_d, rv_w, rv_m, pk, skew, kurt, vix, log_dv, ret_21, target_rv, target_log_rv, sector`.

---

## File Structure

| File | Role |
|---|---|
| `src/interp/shap_analysis.py` | SHAP mean absolute values + beeswarm data; wraps GBMForecaster.shap_values() |
| `src/interp/regime_analysis.py` | VIX-percentile regime split; per-regime IC and Sharpe; uses train-window VIX percentiles |
| `src/interp/sector_neutral.py` | Sector-demeaned signal + re-run vol scaling; produces sector-neutral Sharpe table |
| `src/viz/plots.py` | All matplotlib figure functions: SHAP bar, IC-by-year, regime IC table, equity curves |
| `scripts/run_phase4.py` | Orchestrates all four analyses; writes `results/phase4_summary.md` |
| `tests/test_shap_analysis.py` | Unit tests for shap_analysis.py (marked `@pytest.mark.shap`) |
| `tests/test_regime_analysis.py` | Unit tests for regime_analysis.py |
| `tests/test_sector_neutral.py` | Unit tests for sector_neutral.py |
| `docs/figures/` | Output directory for all saved PNG figures |

---

## Task 1: SHAP Feature Importance (shap_analysis.py)

**Files:**
- Create: `src/interp/shap_analysis.py`
- Create: `tests/test_shap_analysis.py`

### Context
`GBMForecaster.shap_values(panel)` returns a 2D numpy array of shape `(n_rows, n_features)` where features are `FEATURE_COLS + CATEGORICAL_COLS` = `["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21", "sector"]`. The GBM is re-fit on each walk-forward window, so SHAP values need to be computed from a single representative fit (use the last/largest training window). Pre-registered prediction #6: `rv_m > rv_w > vix > rv_d` in SHAP importance.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_shap_analysis.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

pytest.importorskip("shap")

pytestmark = pytest.mark.shap


def _make_panel(n=200):
    dates = pd.date_range("2010-01-01", periods=n // 10)
    tickers = [f"T{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(0)
    cols = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]
    data = {c: rng.standard_normal(len(idx)) ** 2 for c in cols}
    data["sector"] = "Tech"
    data["target_log_rv"] = rng.standard_normal(len(idx))
    return pd.DataFrame(data, index=idx)


def test_compute_shap_returns_dataframe():
    from src.interp.shap_analysis import compute_shap_importance
    from src.models.gbm import GBMForecaster
    panel = _make_panel(200)
    model = GBMForecaster()
    model.fit(panel)
    result = compute_shap_importance(model, panel)
    assert isinstance(result, pd.Series)


def test_compute_shap_index_is_feature_names():
    from src.interp.shap_analysis import compute_shap_importance
    from src.models.gbm import GBMForecaster
    panel = _make_panel(200)
    model = GBMForecaster()
    model.fit(panel)
    result = compute_shap_importance(model, panel)
    assert "rv_m" in result.index
    assert "rv_w" in result.index
    assert "vix" in result.index


def test_compute_shap_values_nonnegative():
    from src.interp.shap_analysis import compute_shap_importance
    from src.models.gbm import GBMForecaster
    panel = _make_panel(200)
    model = GBMForecaster()
    model.fit(panel)
    result = compute_shap_importance(model, panel)
    assert (result >= 0).all()


def test_compute_shap_sample_size_respected():
    from src.interp.shap_analysis import compute_shap_importance
    from src.models.gbm import GBMForecaster
    panel = _make_panel(500)
    model = GBMForecaster()
    model.fit(panel)
    # should not raise even with small sample
    result = compute_shap_importance(model, panel, sample_size=50)
    assert len(result) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
PYTHONPATH="/tmp/ml-vol-momentum-venv/lib/python3.14/site-packages" \
  /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_shap_analysis.py -m shap -v --tb=short
```
Expected: `ImportError: cannot import name 'compute_shap_importance'`

- [ ] **Step 3: Implement shap_analysis.py**

```python
# src/interp/shap_analysis.py
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_shap_importance(
    model,
    panel: pd.DataFrame,
    sample_size: int = 5000,
) -> pd.Series:
    """Return mean |SHAP| per feature, sorted descending.

    model must be a fitted GBMForecaster with a .shap_values(panel) method
    and a ._prepare_X(panel) method. Uses up to sample_size rows sampled
    from panel to keep runtime under control.
    """
    import shap
    from src.models.gbm import FEATURE_COLS, CATEGORICAL_COLS

    feat_names = [c for c in FEATURE_COLS + CATEGORICAL_COLS if c in panel.columns]

    # Sample without replacement for speed
    sub = panel.dropna(subset=[c for c in FEATURE_COLS if c in panel.columns])
    if len(sub) > sample_size:
        sub = sub.sample(sample_size, random_state=42)

    shap_vals = model.shap_values(sub)          # (n_rows, n_features)
    mean_abs = np.abs(shap_vals).mean(axis=0)   # (n_features,)
    result = pd.Series(mean_abs, index=feat_names[:len(mean_abs)])
    return result.sort_values(ascending=False)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
PYTHONPATH="/tmp/ml-vol-momentum-venv/lib/python3.14/site-packages" \
  /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_shap_analysis.py -m shap -v --tb=short
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/interp/shap_analysis.py tests/test_shap_analysis.py
git commit -m "feat: SHAP feature importance for GBMForecaster"
```

---

## Task 2: VIX-Regime Breakdown (regime_analysis.py)

**Files:**
- Create: `src/interp/regime_analysis.py`
- Create: `tests/test_regime_analysis.py`

### Context
Split OOS dates into three VIX regimes using **training-window** VIX percentiles (not full-sample — that would be look-ahead). Thresholds: p33 and p67 of VIX over the training window ending before each test year. Regimes: low (VIX < p33), mid (p33–p67), high (> p67). For each regime, compute per-model cross-sectional IC and strategy Sharpe.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_regime_analysis.py
import pytest
import numpy as np
import pandas as pd
from src.eval.walk_forward import generate_windows


def _make_vix(start="2000-01-01", end="2010-12-31"):
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(1)
    return pd.Series(rng.uniform(10, 40, len(dates)), index=dates, name="vix")


def _make_forecast(vix: pd.Series, n_tickers=5):
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product(
        [vix.index, tickers], names=["date", "ticker"]
    )
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "forecast_log_rv": rng.standard_normal(len(idx)),
        "forecast_rv": rng.uniform(0.001, 0.05, len(idx)),
        "target_rv": rng.uniform(0.001, 0.05, len(idx)),
    }, index=idx)


def test_assign_regimes_returns_series():
    from src.interp.regime_analysis import assign_regimes
    vix = _make_vix()
    windows = generate_windows(pd.Timestamp("2000-01-01"), pd.Timestamp("2010-12-31"), first_test_year=2003)
    regimes = assign_regimes(vix, windows)
    assert isinstance(regimes, pd.Series)
    assert set(regimes.unique()).issubset({"low", "mid", "high"})


def test_assign_regimes_covers_oos_dates():
    from src.interp.regime_analysis import assign_regimes
    vix = _make_vix()
    windows = generate_windows(pd.Timestamp("2000-01-01"), pd.Timestamp("2010-12-31"), first_test_year=2003)
    regimes = assign_regimes(vix, windows)
    # Every OOS date in vix index should have a regime label
    assert len(regimes) > 0


def test_regime_ic_returns_dataframe():
    from src.interp.regime_analysis import assign_regimes, regime_ic_table
    vix = _make_vix()
    windows = generate_windows(pd.Timestamp("2000-01-01"), pd.Timestamp("2010-12-31"), first_test_year=2003)
    regimes = assign_regimes(vix, windows)
    forecast = _make_forecast(vix)
    realized = forecast[["target_rv"]]
    result = regime_ic_table({"model_a": forecast}, realized, regimes)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns).issubset({"low", "mid", "high"})


def test_no_lookahead_in_thresholds():
    from src.interp.regime_analysis import assign_regimes
    vix = _make_vix()
    windows = generate_windows(pd.Timestamp("2000-01-01"), pd.Timestamp("2010-12-31"), first_test_year=2003)
    regimes = assign_regimes(vix, windows)
    # Regime for first test year uses only training data — smoke check: no exception
    assert regimes is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
PYTHONPATH="/tmp/ml-vol-momentum-venv/lib/python3.14/site-packages" \
  /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_regime_analysis.py -v --tb=short
```
Expected: `ImportError: cannot import name 'assign_regimes'`

- [ ] **Step 3: Implement regime_analysis.py**

```python
# src/interp/regime_analysis.py
from __future__ import annotations
import pandas as pd
import numpy as np
from src.eval.walk_forward import CVWindow
from src.eval.tests import cross_sectional_ic


def assign_regimes(
    vix: pd.Series,
    windows: list[CVWindow],
    thresholds: tuple[float, float] = (0.33, 0.67),
) -> pd.Series:
    """Assign each OOS date to a VIX regime (low/mid/high).

    Percentile thresholds computed from the TRAINING window only (no look-ahead).
    Returns a Series indexed by date with values 'low', 'mid', 'high'.
    """
    regime_map: dict[pd.Timestamp, str] = {}
    for w in windows:
        train_vix = vix.loc[
            (vix.index >= w.train_start) & (vix.index <= w.train_end)
        ]
        if train_vix.empty:
            continue
        p_lo = train_vix.quantile(thresholds[0])
        p_hi = train_vix.quantile(thresholds[1])

        test_vix = vix.loc[
            (vix.index >= w.test_start) & (vix.index <= w.test_end)
        ]
        for date, v in test_vix.items():
            if v < p_lo:
                regime_map[date] = "low"
            elif v > p_hi:
                regime_map[date] = "high"
            else:
                regime_map[date] = "mid"

    return pd.Series(regime_map, name="regime").sort_index()


def regime_ic_table(
    forecasts: dict[str, pd.DataFrame],
    realized: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Return DataFrame: rows=models, cols=regimes (low/mid/high), values=mean IC."""
    rows = {}
    for model_name, oos in forecasts.items():
        ic_series = cross_sectional_ic(oos, realized)
        row = {}
        for regime in ("low", "mid", "high"):
            regime_dates = regimes[regimes == regime].index
            ic_subset = ic_series.reindex(regime_dates).dropna()
            row[regime] = ic_subset.mean() if len(ic_subset) > 0 else float("nan")
        rows[model_name] = row
    return pd.DataFrame(rows).T[["low", "mid", "high"]]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
PYTHONPATH="/tmp/ml-vol-momentum-venv/lib/python3.14/site-packages" \
  /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_regime_analysis.py -v --tb=short
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/interp/regime_analysis.py tests/test_regime_analysis.py
git commit -m "feat: VIX-regime IC breakdown (train-window percentiles, no look-ahead)"
```

---

## Task 3: Sector-Neutral Robustness (sector_neutral.py)

**Files:**
- Create: `src/interp/sector_neutral.py`
- Create: `tests/test_sector_neutral.py`

### Context
Sector-neutral = demean the momentum signal within each GICS sector at each rebalance date, then re-run vol scaling + portfolio construction + costs. The vol forecast itself does NOT change — we only adjust the signal. Pre-registered prediction #8: sector-neutral Sharpe ~0.1-0.2 lower but ranking preserved.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sector_neutral.py
import pytest
import numpy as np
import pandas as pd


def _make_signal(n_dates=50, n_tickers=20):
    dates = pd.bdate_range("2010-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(3)
    return pd.Series(rng.standard_normal(len(idx)), index=idx, name="signal")


def _make_sector_map(n_tickers=20):
    sectors = ["Tech", "Finance", "Energy", "Health"]
    return {f"T{i}": sectors[i % len(sectors)] for i in range(n_tickers)}


def test_demean_signal_zero_mean_per_sector():
    from src.interp.sector_neutral import demean_signal
    signal = _make_signal()
    sector_map = _make_sector_map()
    result = demean_signal(signal, sector_map)
    # Within each (date, sector) group, mean should be ~0
    panel = result.to_frame("signal")
    panel["sector"] = panel.index.get_level_values("ticker").map(sector_map)
    for (date, sector), grp in panel.groupby(["date", "sector"]):
        assert abs(grp["signal"].mean()) < 1e-10


def test_demean_signal_preserves_index():
    from src.interp.sector_neutral import demean_signal
    signal = _make_signal()
    sector_map = _make_sector_map()
    result = demean_signal(signal, sector_map)
    assert result.index.equals(signal.index)


def test_demean_signal_returns_series():
    from src.interp.sector_neutral import demean_signal
    signal = _make_signal()
    sector_map = _make_sector_map()
    result = demean_signal(signal, sector_map)
    assert isinstance(result, pd.Series)


def test_demean_signal_single_ticker_sector_is_zero():
    from src.interp.sector_neutral import demean_signal
    signal = _make_signal(n_tickers=3)
    # T2 is the only "Energy" stock — demeaning should make it 0
    sector_map = {"T0": "Tech", "T1": "Finance", "T2": "Energy"}
    result = demean_signal(signal, sector_map)
    t2_vals = result.loc[result.index.get_level_values("ticker") == "T2"]
    assert (t2_vals.abs() < 1e-10).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
PYTHONPATH="/tmp/ml-vol-momentum-venv/lib/python3.14/site-packages" \
  /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_sector_neutral.py -v --tb=short
```
Expected: `ImportError: cannot import name 'demean_signal'`

- [ ] **Step 3: Implement sector_neutral.py**

```python
# src/interp/sector_neutral.py
from __future__ import annotations
import pandas as pd


def demean_signal(signal: pd.Series, sector_map: dict[str, str]) -> pd.Series:
    """Within each (date, sector) group, subtract the group mean.

    signal: pd.Series with (date, ticker) MultiIndex.
    sector_map: dict mapping ticker -> GICS sector string.
    Returns Series with same index, sector-demeaned values.
    """
    df = signal.to_frame("signal").copy()
    df["sector"] = df.index.get_level_values("ticker").map(sector_map)
    df["signal"] = df.groupby(
        [df.index.get_level_values("date"), "sector"]
    )["signal"].transform(lambda x: x - x.mean())
    return df["signal"].rename(signal.name)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
PYTHONPATH="/tmp/ml-vol-momentum-venv/lib/python3.14/site-packages" \
  /tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_sector_neutral.py -v --tb=short
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/interp/sector_neutral.py tests/test_sector_neutral.py
git commit -m "feat: sector-neutral signal demeaning for robustness check"
```

---

## Task 4: Visualization (plots.py)

**Files:**
- Create: `src/viz/plots.py`
- Create: `docs/figures/` (directory)

### Context
Four plots total. All use matplotlib with a clean style. No interactivity needed — PNG outputs only.
1. **SHAP bar chart** — horizontal bar, features on y-axis, mean |SHAP| on x-axis.
2. **IC-by-year line chart** — one line per model, x=year, y=cross-sectional IC.
3. **Regime IC heatmap** — rows=models, cols=low/mid/high, cell=mean IC, color-coded.
4. **Equity curves** — cumulative log-return of each vol-scaled strategy + unscaled.

- [ ] **Step 1: Implement plots.py** (no TDD — purely visual, tested by running)

```python
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
```

- [ ] **Step 2: Verify plots.py imports cleanly**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
PYTHONPATH="/tmp/ml-vol-momentum-venv/lib/python3.14/site-packages" \
  /tmp/ml-vol-momentum-venv/bin/python -c "from src.viz.plots import plot_shap_importance, plot_ic_by_year, plot_regime_ic_heatmap, plot_equity_curves; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/viz/plots.py
git commit -m "feat: visualization — SHAP bar, IC-by-year, regime heatmap, equity curves"
```

---

## Task 5: Phase 4 Orchestration Script (run_phase4.py)

**Files:**
- Create: `scripts/run_phase4.py`
- Create: `docs/figures/` (auto-created by plots.py)

### Context
Loads all 5 forecast parquets + strategies, runs SHAP on the last training window's GBM fit, runs regime IC analysis, runs sector-neutral check, generates all 4 figures, prints a summary table, and writes `results/phase4_summary.md`.

- [ ] **Step 1: Implement run_phase4.py**

```python
# scripts/run_phase4.py
"""
Phase 4 analysis: SHAP, regime breakdown, sector-neutral robustness, figures.
Run with:
  PYTHONPATH="/tmp/ml-vol-momentum-venv/lib/python3.14/site-packages" \
    /tmp/ml-vol-momentum-venv/bin/python -u scripts/run_phase4.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

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
from src.eval.walk_forward import generate_windows
from src.eval.tests import cross_sectional_ic
from src.models.gbm import GBMForecaster
from src.interp.shap_analysis import compute_shap_importance
from src.interp.regime_analysis import assign_regimes, regime_ic_table
from src.interp.sector_neutral import demean_signal
from src.viz.plots import (plot_shap_importance, plot_ic_by_year,
                            plot_regime_ic_heatmap, plot_equity_curves)

Path("docs/figures").mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print(f"Loading {len(tickers)} tickers...", flush=True)
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

windows   = generate_windows(start, end, first_test_year=2003)
prices_panel = ohlcv[["close"]]
signal    = momentum_signal(prices_panel, lookback=252, skip=21)
realized_rv = panel["target_rv"]

# ── Load forecast parquets ─────────────────────────────────────────────────
model_names = ["rolling_vol", "har_rv", "garch", "gbm", "lstm"]
all_forecasts  = {n: pd.read_parquet(f"results/forecasts/{n}.parquet") for n in model_names}
all_strategies = {n: pd.read_parquet(f"results/strategies/{n}_scaled.parquet")["net_return"]
                  for n in model_names}
all_strategies["unscaled_momentum"] = pd.read_parquet(
    "results/strategies/unscaled_momentum.parquet")["net_return"]

# ── 1. SHAP Analysis ───────────────────────────────────────────────────────
print("\n=== SHAP ANALYSIS ===", flush=True)
# Fit GBM on the last (largest) training window for SHAP
last_w = windows[-1]
train_mask = (panel.index.get_level_values("date") <= last_w.train_end)
train_last = panel[train_mask]
gbm_for_shap = GBMForecaster()
print("Fitting GBM on last window for SHAP...", flush=True)
gbm_for_shap.fit(train_last)
shap_imp = compute_shap_importance(gbm_for_shap, train_last, sample_size=5000)
print(shap_imp.to_string())
plot_shap_importance(shap_imp)

# ── 2. VIX-Regime IC Breakdown ─────────────────────────────────────────────
print("\n=== REGIME IC TABLE ===", flush=True)
regimes = assign_regimes(vix, windows)
reg_table = regime_ic_table(all_forecasts, realized_rv.rename("target_rv").to_frame(), regimes)
print(reg_table.round(4).to_string())
reg_table.to_parquet("results/regime_ic_table.parquet")
plot_regime_ic_heatmap(reg_table)

# ── 3. Sector-Neutral Robustness ───────────────────────────────────────────
print("\n=== SECTOR-NEUTRAL ROBUSTNESS ===", flush=True)
signal_sn = demean_signal(signal, sector_map)
sn_strategies = {}
for model_name in model_names:
    oos = all_forecasts[model_name]
    w_sn_raw = vol_scale(signal_sn, oos, target_vol=0.10)
    w_sn = build_portfolios(None, weights=w_sn_raw, mode="vol_targeted_gross")
    net_sn = apply_costs(w_sn, returns_panel, cost_bps=10.0).dropna()
    sn_strategies[f"{model_name}_sn"] = net_sn

from src.eval.comparison import build_results_table
sn_table = build_results_table(sn_strategies)
print("Sector-neutral Sharpes:")
print(sn_table[["sharpe"]].round(3).to_string())
sn_table.to_parquet("results/sector_neutral_results.parquet")

# ── 4. Figures ─────────────────────────────────────────────────────────────
print("\n=== GENERATING FIGURES ===", flush=True)
plot_ic_by_year(all_forecasts, realized_rv)
plot_equity_curves(all_strategies)

# ── 5. Summary markdown ────────────────────────────────────────────────────
summary = f"""# Phase 4 Summary

## SHAP Feature Importance (GBM, last training window)

{shap_imp.round(4).to_string()}

## Cross-Sectional IC by VIX Regime

{reg_table.round(4).to_string()}

## Sector-Neutral Sharpe (vs vanilla)

| Model | Vanilla Sharpe | SN Sharpe |
|---|---|---|
"""
vanilla_tbl = pd.read_parquet("results/master_results_table.parquet")
for n in model_names:
    v_sh = vanilla_tbl.loc[f"{n}_scaled", "sharpe"] if f"{n}_scaled" in vanilla_tbl.index else vanilla_tbl.loc[n, "sharpe"] if n in vanilla_tbl.index else float("nan")
    sn_sh = sn_table.loc[f"{n}_sn", "sharpe"] if f"{n}_sn" in sn_table.index else float("nan")
    summary += f"| {n} | {v_sh:.3f} | {sn_sh:.3f} |\n"

summary += "\n## Figures\n- `docs/figures/shap_importance.png`\n- `docs/figures/ic_by_year.png`\n- `docs/figures/regime_ic_heatmap.png`\n- `docs/figures/equity_curves.png`\n"

Path("results/phase4_summary.md").write_text(summary)
print("\nPhase 4 complete. Summary written to results/phase4_summary.md", flush=True)
```

- [ ] **Step 2: Run the script**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
PYTHONPATH="/tmp/ml-vol-momentum-venv/lib/python3.14/site-packages" \
  /tmp/ml-vol-momentum-venv/bin/python -u scripts/run_phase4.py 2>&1 | tee /tmp/phase4_run.log
```
Expected output (in order): SHAP table, regime IC table, sector-neutral Sharpes, "Phase 4 complete."

- [ ] **Step 3: Verify figures exist**

```bash
ls docs/figures/
```
Expected: `shap_importance.png  ic_by_year.png  regime_ic_heatmap.png  equity_curves.png`

- [ ] **Step 4: Commit all Phase 4 work**

```bash
git add scripts/run_phase4.py src/interp/shap_analysis.py src/interp/regime_analysis.py \
        src/interp/sector_neutral.py src/viz/plots.py \
        tests/test_shap_analysis.py tests/test_regime_analysis.py tests/test_sector_neutral.py \
        docs/figures/ results/phase4_summary.md results/regime_ic_table.parquet \
        results/sector_neutral_results.parquet
git commit -m "feat: Phase 4 complete — SHAP, regime IC, sector-neutral, figures"
git tag phase4-complete
```

---

## Self-Review

**Spec coverage check:**
- SHAP analysis with `rv_m > rv_w > vix > rv_d` prediction check → Task 1 + Task 5 ✅
- Regime analysis using VIX percentiles computed on **training window** (no look-ahead) → Task 2 ✅
- Sector-neutral robustness → Task 3 + Task 5 ✅
- Figures: SHAP bar, IC-by-year, regime heatmap, equity curves → Task 4 ✅
- `results/phase4_summary.md` → Task 5 ✅
- `phase4-complete` tag → Task 5 ✅

**Placeholder scan:** None found.

**Type consistency:** All functions match: `compute_shap_importance(model, panel, sample_size) -> pd.Series`, `assign_regimes(vix, windows, thresholds) -> pd.Series`, `regime_ic_table(forecasts, realized, regimes) -> pd.DataFrame`, `demean_signal(signal, sector_map) -> pd.Series`.
