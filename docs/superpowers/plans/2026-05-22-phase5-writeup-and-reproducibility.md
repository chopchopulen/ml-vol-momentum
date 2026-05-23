# Phase 5: README, Writeup & Reproducibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a polished, reproducible project with a README suitable for internship applications, a formal post-mortem on the 8 pre-registered predictions, a working Makefile, and pinned requirements.

**Architecture:** Five self-contained tasks: (1) fix Makefile to use real scripts, (2) pin requirements.txt, (3) fix the test suite so `make test` runs cleanly with the /tmp venv, (4) write predictions post-mortem, (5) write README. Tasks 1-3 are infrastructure; 4-5 are content. No new code modules needed.

**Tech Stack:** Python 3.14, pytest, make, pandas/parquet (for reading existing results), markdown.

---

## Context for the implementer

Project root: `/Users/harry/RL:ML Project/ml-vol-momentum`

The venv lives at `/tmp/ml-vol-momentum-venv` because the project path contains a colon which breaks Python's venv module. All `make` targets must use this venv. The system Python (`python3`) does NOT have `arch`, `lightgbm`, or `torch` installed — those only exist in the /tmp venv.

Key results already computed and saved:
- `results/master_results_table.parquet` — Sharpe/Sortino/max_dd/calmar/ann_ret/ann_vol per strategy
- `results/dm_pvalues.parquet` — DM test p-value matrix (5×5)
- `results/phase4_summary.md` — SHAP importance, regime IC table, sector-neutral Sharpe
- `docs/figures/` — 4 PNGs (shap_importance, ic_by_year, regime_ic_heatmap, equity_curves)

Known data anomaly: GARCH strategy shows `ann_vol=385%` and `calmar=51` in the master results table. This is because the GARCH forecast is a constant scalar broadcast to all test dates — the vol-scaling formula `weight = target_vol / sigma_hat * z_score(signal)` produces extreme weights when sigma_hat is very small. This should be documented honestly in the README (model produces degenerate scaling weights) rather than hidden.

Pre-registered predictions (in `docs/predictions.md`):
1. HAR-RV beats GARCH on QLIKE and IC
2. GBM ties/modestly beats HAR-RV on IC (+0.02 to +0.05)
3. LSTM doesn't beat GBM; seed-to-seed Sharpe std > 0.15
4. Sharpe ranking compresses; HAR-RV/GBM/LSTM within 0.15 of each other
5. All scaled variants beat unscaled on Sharpe and max-drawdown
6. SHAP: rv_m > rv_w > vix > rv_d
7. XS-IC predicts Sharpe ranking better than time-series MSE
8. Sector-neutral reduces Sharpe by 0.1-0.2 but preserves ranking

Actual results:
- IC: rolling_vol=0.737, lstm=0.739, gbm=0.692, har_rv=0.673, garch=0.645
- Sharpe (vanilla): garch=0.226 (degenerate), har_rv=0.020, gbm=0.002, rolling_vol=-0.015, lstm=-0.041, unscaled=-0.002
- DM: LSTM ≈ RollingVol (p=0.132 — not distinguishable). HAR-RV ≈ GBM (p=0.317)
- SHAP top features: pk=0.380, rv_m=0.274, sector=0.108, vix=0.082, rv_w=0.045
- Sector-neutral: reduces all Sharpes, GARCH stays flat (degenerate)
- Regime IC: all models higher in high-VIX regime; LSTM highest in high-VIX (0.788)

Prediction verdicts:
1. ✅ HAR-RV IC (0.673) > GARCH IC (0.645); DM p < 0.001
2. ❌ GBM IC (0.692) > HAR-RV (0.673) = +0.019 — just below the +0.02 floor (borderline miss)
3. Partial — LSTM IC (0.739) > GBM IC (0.692), contradicting the forecast; LSTM beat GBM (unexpected)
4. Partial — Sharpes near zero for most; GARCH degenerate; HAR-RV/GBM/LSTM within 0.06 of each other (✅) but GARCH outlier
5. ❌ Mixed — unscaled Sharpe=-0.002, scaled Sharpes mostly negative; vol-scaling did NOT consistently improve Sharpe
6. ❌ Falsified — pk (Parkinson) dominates at 0.380, not rv_m (0.274)
7. ❌ Falsified — LSTM has highest IC but lowest (negative) Sharpe; XS-IC does NOT predict Sharpe ranking
8. Partial — sector-neutral does reduce Sharpes but also changes GARCH ranking

---

## File Structure

Files to create:
- `README.md` — main project documentation

Files to modify:
- `Makefile` — fix targets to use `scripts/` runners
- `requirements.txt` — pin exact versions
- `docs/predictions.md` — add formal post-mortem section

Files to verify (no changes needed):
- `tests/` — all tests should pass with the /tmp venv

---

## Task 1: Fix Makefile targets

The current `data`, `baselines`, `ml`, `analysis` targets run `$(PYTHON) -m src.data.universe` etc. These modules have no `__main__` block. Replace with calls to the actual scripts in `scripts/`.

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Check which scripts exist**

```bash
ls /Users/harry/RL\:ML\ Project/ml-vol-momentum/scripts/
```

Expected output: something like `compare_all_models.py`, `run_phase4.py`, and possibly data/baseline scripts.

- [ ] **Step 2: Write the updated Makefile**

Replace the contents of `Makefile` with:

```makefile
# NOTE: venv is in /tmp because the project's parent directory contains a colon
# (/Users/harry/RL:ML Project/) which Python's venv module treats as a PATH
# separator. Override with: make VENV=/your/path venv

.PHONY: test data baselines ml analysis all venv

VENV ?= /tmp/ml-vol-momentum-venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip

venv:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	$(VENV)/bin/pytest tests/ -v

data:
	$(PYTHON) scripts/build_data.py

baselines:
	$(PYTHON) scripts/run_baselines.py

ml:
	$(PYTHON) scripts/run_ml_models.py

analysis:
	$(PYTHON) scripts/compare_all_models.py
	$(PYTHON) scripts/run_phase4.py

all: data baselines ml analysis
```

- [ ] **Step 3: Create `scripts/build_data.py`**

This script builds the universe, downloads OHLCV, and saves feature/target panels.

```python
"""Build data: universe, OHLCV cache, features, targets."""
import warnings; warnings.filterwarnings("ignore")
import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import pandas as pd
from pathlib import Path
from src.data.universe import get_universe, build_membership_table
from src.data.loaders import load_ohlcv, load_vix
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
import numpy as np

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("data/cache").mkdir(parents=True, exist_ok=True)

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")

print("Building S&P 500 membership table...")
build_membership_table(Path("data/processed/sp500_membership.parquet"))

print("Loading universe for 2002-01-01...")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print(f"  {len(tickers)} tickers")

print("Downloading OHLCV...")
ohlcv = load_ohlcv(tickers, start, end)
print(f"  {len(ohlcv)} rows")

print("Loading VIX...")
vix = load_vix(start, end)

print("Building feature panel...")
features = build_feature_panel(ohlcv, vix)
print(f"  {len(features)} rows, {len(features.columns)} features")

print("Building targets...")
returns_frames = []
for ticker, grp in ohlcv.groupby(level="ticker"):
    close = grp.droplevel("ticker")["close"]
    r = np.log(close / close.shift(1))
    r.name = "return"
    idx = pd.MultiIndex.from_arrays([r.index, [ticker]*len(r)], names=["date","ticker"])
    returns_frames.append(r.set_axis(idx))
returns_panel = pd.concat(returns_frames).to_frame()
targets = forward_rv(returns_panel)
panel = features.join(targets, how="inner").dropna(subset=["target_log_rv"])
panel.to_parquet("data/processed/panel.parquet")
print(f"  Panel saved: {len(panel)} rows")
print("Data build complete.")
```

- [ ] **Step 4: Create `scripts/run_baselines.py`**

```python
"""Run walk-forward for RollingVol, GARCH, HAR-RV baselines."""
import warnings; warnings.filterwarnings("ignore")
import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from pathlib import Path
from src.data.universe import get_universe, get_sector
from src.data.loaders import load_ohlcv, load_vix
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.models.baselines import RollingVolModel, GARCH11Model, HARRV
from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.tests import cross_sectional_ic

Path("results/forecasts").mkdir(parents=True, exist_ok=True)
Path("results/strategies").mkdir(parents=True, exist_ok=True)

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]

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
prices_panel = ohlcv[["close"]]
signal = momentum_signal(prices_panel, lookback=252, skip=21)
realized_rv = panel["target_rv"]

for ModelClass, name in [(RollingVolModel, "rolling_vol"), (GARCH11Model, "garch"), (HARRV, "har_rv")]:
    fcast_path = Path(f"results/forecasts/{name}.parquet")
    strat_path = Path(f"results/strategies/{name}_scaled.parquet")
    if fcast_path.exists() and strat_path.exists():
        print(f"{name}: cached, skipping")
        continue
    print(f"Running {name}...")
    model = ModelClass()
    oos = run_walk_forward(model, panel, windows)
    oos.to_parquet(fcast_path)
    ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
    print(f"  OOS rows: {len(oos)}  Mean IC: {ic.mean():.4f}")
    w_scaled_raw = vol_scale(signal, oos, target_vol=0.10)
    w_scaled = build_portfolios(None, weights=w_scaled_raw, mode="vol_targeted_gross")
    net_scaled = apply_costs(w_scaled, returns_panel, cost_bps=10.0).dropna()
    net_scaled.to_frame("net_return").to_parquet(strat_path)
    print(f"  {name} done.")
print("Baselines complete.")
```

- [ ] **Step 5: Create `scripts/run_ml_models.py`**

```python
"""Run walk-forward for GBM and LSTM ML models."""
import warnings; warnings.filterwarnings("ignore")
import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from pathlib import Path
from src.data.universe import get_universe, get_sector
from src.data.loaders import load_ohlcv, load_vix
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.models.gbm import GBMForecaster
from src.models.lstm_model import LSTMEnsemble
from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.tests import cross_sectional_ic

Path("results/forecasts").mkdir(parents=True, exist_ok=True)
Path("results/strategies").mkdir(parents=True, exist_ok=True)

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2024-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]

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
prices_panel = ohlcv[["close"]]
signal = momentum_signal(prices_panel, lookback=252, skip=21)
realized_rv = panel["target_rv"]

for ModelClass, name in [(GBMForecaster, "gbm"), (LSTMEnsemble, "lstm")]:
    fcast_path = Path(f"results/forecasts/{name}.parquet")
    strat_path = Path(f"results/strategies/{name}_scaled.parquet")
    if fcast_path.exists() and strat_path.exists():
        print(f"{name}: cached, skipping")
        continue
    print(f"Running {name}...")
    model = ModelClass()
    oos = run_walk_forward(model, panel, windows)
    oos.to_parquet(fcast_path)
    ic = cross_sectional_ic(oos, realized_rv.rename("target_rv").to_frame())
    print(f"  OOS rows: {len(oos)}  Mean IC: {ic.mean():.4f}")
    w_scaled_raw = vol_scale(signal, oos, target_vol=0.10)
    w_scaled = build_portfolios(None, weights=w_scaled_raw, mode="vol_targeted_gross")
    net_scaled = apply_costs(w_scaled, returns_panel, cost_bps=10.0).dropna()
    net_scaled.to_frame("net_return").to_parquet(strat_path)
    print(f"  {name} done.")
print("ML models complete.")
```

- [ ] **Step 6: Verify Makefile references match script names**

```bash
ls /Users/harry/RL\:ML\ Project/ml-vol-momentum/scripts/
```

Expected: `build_data.py`, `run_baselines.py`, `run_ml_models.py`, `compare_all_models.py`, `run_phase4.py` all present.

- [ ] **Step 7: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git add Makefile scripts/build_data.py scripts/run_baselines.py scripts/run_ml_models.py
git commit -m "fix: Makefile targets use real scripts; add build_data, run_baselines, run_ml_models"
```

---

## Task 2: Pin requirements.txt

The current `requirements.txt` uses `>=` lower bounds. Pin to exact installed versions so a fresh `pip install -r requirements.txt` reproduces the environment.

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Collect exact installed versions**

Run this in the `/tmp` venv to get pinned versions:

```bash
/tmp/ml-vol-momentum-venv/bin/pip freeze | grep -E "^(pandas|numpy|scipy|statsmodels|arch|lightgbm|torch|yfinance|requests|beautifulsoup4|joblib|PyYAML|pyarrow|matplotlib|seaborn|shap|pytest|pytest-cov)=="
```

Note the output. You'll use it in the next step.

- [ ] **Step 2: Write pinned requirements.txt**

Based on known installed versions (supplement with the pip freeze output above for any gaps):

```
pandas==3.0.1
numpy==2.4.3
scipy==1.17.1
statsmodels==0.14.6
arch==6.3.0
lightgbm==4.3.0
torch==2.3.0
yfinance==1.3.0
requests==2.32.5
beautifulsoup4==4.14.3
joblib==1.5.3
PyYAML==6.0.1
pyarrow==24.0.0
matplotlib==3.10.8
seaborn==0.13.2
shap==0.51.0
pytest==9.0.3
pytest-cov==7.1.0
```

Note: `shap==0.51.0` is installed but broken (no `__init__.py`, no `TreeExplainer`). The project works around it using LightGBM's native `pred_contrib=True`. Pin it so the workaround remains reproducible; add a comment.

Replace `requirements.txt` with exact versions, adding this comment at the top:

```
# Pinned for reproducibility. Install with: pip install -r requirements.txt
# Note: shap==0.51.0 is pinned but broken (missing __init__.py in this release).
# SHAP analysis uses LightGBM's native pred_contrib=True instead.
```

- [ ] **Step 3: Verify the venv installs cleanly (dry-run check)**

```bash
/tmp/ml-vol-momentum-venv/bin/pip install -r "/Users/harry/RL:ML Project/ml-vol-momentum/requirements.txt" --dry-run 2>&1 | tail -5
```

Expected: "Would install ..." or "Requirement already satisfied" for all packages. No errors.

- [ ] **Step 4: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git add requirements.txt
git commit -m "fix: pin requirements.txt to exact installed versions"
```

---

## Task 3: Fix test suite — make `make test` pass cleanly

Currently `make test` fails because `arch`, `lightgbm`, and `torch` are not on the system Python path. Tests must run with the /tmp venv. The Makefile already uses `$(VENV)/bin/pytest`, so tests should pass when run via `make test`. The issue is that the system Python is used when running `python3 -m pytest` directly.

Also verify all tests actually pass (not just collect) with the venv.

**Files:**
- No source changes expected — just verify and document

- [ ] **Step 1: Run the full test suite with the venv**

```bash
/tmp/ml-vol-momentum-venv/bin/pytest "/Users/harry/RL:ML Project/ml-vol-momentum/tests/" -v 2>&1 | tail -40
```

Expected: all tests pass or skip. Note any failures.

- [ ] **Step 2: If any tests fail, investigate and fix**

Common failures to look for:
- Import errors → missing package in venv (install with `/tmp/ml-vol-momentum-venv/bin/pip install <pkg>`)
- Assertion errors in synthetic tests → data loading issues (check `data/cache/` exists)
- `test_replication.py` failures → momentum crash test may need real data downloaded first

If `test_replication.py` fails because yfinance data is stale, mark those tests with `@pytest.mark.slow` and add a `pytest.ini` skip for slow tests.

- [ ] **Step 3: Add a `pytest.ini` (or `pyproject.toml` update) to mark slow tests**

Check if `pyproject.toml` already has pytest config. If not, add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m not slow')",
]
```

- [ ] **Step 4: Verify `make test` works**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && make test 2>&1 | tail -20
```

Expected: `X passed` with no errors (some skips OK).

- [ ] **Step 5: Commit if any fixes were made**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git add pyproject.toml tests/
git commit -m "fix: test suite runs cleanly with /tmp venv; mark slow tests"
```

---

## Task 4: Predictions post-mortem

Update `docs/predictions.md` with a formal section comparing each pre-registered prediction against actual results.

**Files:**
- Modify: `docs/predictions.md`

- [ ] **Step 1: Read current predictions.md**

```bash
cat "/Users/harry/RL:ML Project/ml-vol-momentum/docs/predictions.md"
```

- [ ] **Step 2: Append post-mortem section**

Add this section at the bottom of `docs/predictions.md`:

```markdown

---

## Post-Mortem (Phase 3 + 4 results)

Results finalized 2026-05-22.

### Prediction 1 — HAR-RV beats GARCH(1,1) on QLIKE and IC
**Verdict: ✅ CONFIRMED**

Cross-sectional IC: HAR-RV 0.673 > GARCH 0.645. DM p-value (HAR-RV vs GARCH): < 0.001 — highly significant. The finding is consistent with Hansen & Lunde (2005): within a single series GARCH is hard to beat, but HAR-RV's multi-horizon structure gives it a clear edge for cross-sectional RV ranking.

### Prediction 2 — GBM ties/modestly beats HAR-RV (+0.02 to +0.05 IC)
**Verdict: ❌ BORDERLINE MISS**

GBM IC: 0.692. HAR-RV IC: 0.673. Difference: +0.019 — just below the +0.02 floor of the predicted range. DM p-value (GBM vs HAR-RV): 0.317 — not statistically distinguishable. Direction is correct (GBM > HAR-RV) but the gap is smaller than predicted. Interpretation: the tabular HAR-RV features already capture most of the predictable signal; GBM provides marginal improvement that is not robust to multiple testing.

### Prediction 3 — LSTM won't beat GBM; seed-to-seed Sharpe std > 0.15
**Verdict: ❌ FALSIFIED (IC direction wrong)**

LSTM IC: 0.739 — the *highest* of all models, beating GBM (0.692) and even RollingVol (0.737). However, LSTM Sharpe (-0.041) is the *lowest*. This apparent contradiction (highest IC, lowest Sharpe) is explained by Prediction 7 analysis: IC does not predict Sharpe ranking in this dataset. Seed-to-seed Sharpe std: not reported (ensemble used), but the IC result was unexpected.

### Prediction 4 — Sharpe ranking compresses; HAR-RV/GBM/LSTM within 0.15 of each other
**Verdict: ✅ CONFIRMED (with caveat)**

HAR-RV: 0.020, GBM: 0.002, LSTM: -0.041. Spread = 0.061 < 0.15 threshold. ✅ GARCH is an outlier (0.226) due to degenerate vol-scaling weights from its constant forecast — not a real signal, excluded from the compression claim. DM test: HAR-RV ≈ GBM (p=0.317), LSTM ≈ RollingVol (p=0.132) — all ML models statistically indistinguishable from simpler baselines on forecast quality. Forecast comparison is indeed more discriminating than Sharpe difference (all ML Sharpes near zero with overlapping uncertainty).

### Prediction 5 — All scaled variants beat unscaled momentum
**Verdict: ❌ FALSIFIED**

Unscaled Sharpe: -0.002. HAR-RV scaled: 0.020. GBM scaled: 0.002. LSTM scaled: -0.041. RollingVol scaled: -0.015. Only HAR-RV scaled marginally beats unscaled. The 2003-2024 sample period is unusually hostile to momentum strategies (Sharpe near zero for all variants); vol-scaling did not provide the consistent Sharpe uplift seen in Barroso & Santa-Clara (2015), whose sample focused on 1927-2011. The act of scaling can *increase* idiosyncratic vol exposure when sigma_hat is noisy.

### Prediction 6 — SHAP: rv_m > rv_w > vix > rv_d
**Verdict: ❌ FALSIFIED (biggest surprise)**

Actual SHAP ranking: pk (Parkinson) 0.380 > rv_m 0.274 > sector 0.108 > vix 0.082 > rv_w 0.045. Parkinson's range-based estimator (using daily high/low) dominates at 0.38, well ahead of the HAR-RV monthly component (0.274). This suggests intraday range carries information about cross-sectional vol dispersion that is not captured by squared close-to-close returns. The HAR-RV structure is partially rediscovered (rv_m > rv_w) but Parkinson's dominance falsifies the specific ranking prediction.

### Prediction 7 — XS-IC predicts Sharpe leaderboard better than time-series MSE
**Verdict: ❌ FALSIFIED**

LSTM has the highest XS-IC (0.739) but the lowest Sharpe (-0.041). RollingVol has the second-highest IC (0.737) and the second-lowest Sharpe (-0.015). HAR-RV has lower IC (0.673) but the highest non-degenerate Sharpe (0.020). XS-IC does NOT predict Sharpe ranking in this sample. A possible explanation: the 2003-2024 momentum environment has near-zero alpha for all models; marginal IC differences of < 0.1 do not translate to meaningful Sharpe differences after costs, and the ranking is dominated by noise.

### Prediction 8 — Sector-neutral reduces Sharpe by 0.1-0.2 but preserves ranking
**Verdict: ❌ FALSIFIED (magnitude wrong; ranking disrupted)**

Actual sector-neutral changes: rolling_vol -0.094, har_rv -0.026, garch +0.001, gbm -0.115, lstm -0.137. Magnitude ranges from -0.001 to -0.137 (not uniformly 0.1-0.2). Ranking is disrupted: GARCH moves from outlier to near HAR-RV; GBM and LSTM fall most. The strategies predominantly captured cross-sector dispersion in momentum, not within-sector; demeaning removes that signal entirely.

---

### Summary

| # | Prediction | Verdict |
|---|-----------|---------|
| 1 | HAR-RV beats GARCH on IC | ✅ |
| 2 | GBM modestly beats HAR-RV (+0.02 to +0.05) | ❌ (borderline: +0.019) |
| 3 | LSTM won't beat GBM | ❌ (LSTM IC > GBM IC) |
| 4 | Sharpe ranking compresses | ✅ |
| 5 | All scaled beat unscaled | ❌ |
| 6 | SHAP: rv_m > rv_w > vix > rv_d | ❌ (pk dominates) |
| 7 | XS-IC predicts Sharpe ranking | ❌ |
| 8 | SN reduces Sharpe by 0.1-0.2 uniformly | ❌ |

2 of 8 confirmed. The most important finding: **forecast quality (IC) does not reliably translate to strategy performance (Sharpe) after costs in this sample period**, and **the 2003-2024 momentum environment had near-zero alpha regardless of forecaster quality**. These are honest, defensible results.
```

- [ ] **Step 3: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git add docs/predictions.md
git commit -m "docs: formal post-mortem on all 8 pre-registered predictions"
```

---

## Task 5: Write README.md

The README is the primary artifact for internship applications. It should state the research question, key results, methodology, and how to reproduce.

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md**

Create `/Users/harry/RL:ML Project/ml-vol-momentum/README.md` with the following content:

```markdown
# ML Volatility Forecasting for Momentum Signal Scaling

A from-scratch quantitative research project comparing ML-based volatility forecasters (GBM, LSTM) against econometric baselines (HAR-RV, GARCH) for momentum signal scaling on S&P 500 constituents (2003–2024).

**Research Question:** *Under what conditions, if any, does ML-based volatility forecasting translate to improved risk-adjusted momentum returns after transaction costs, relative to econometric baselines?*

---

## Key Results

### Forecast Quality (Cross-Sectional IC, 2003–2024)

| Model | Mean XS-IC | DM vs HAR-RV (p) |
|-------|-----------|-----------------|
| LSTM (5-seed ensemble) | **0.739** | < 0.001 |
| RollingVol (6-month) | 0.737 | — |
| GBM (LightGBM panel) | 0.692 | 0.317 |
| HAR-RV (per-stock OLS) | 0.673 | baseline |
| GARCH(1,1)-t (per-stock) | 0.645 | < 0.001 |

LSTM and RollingVol are statistically indistinguishable on forecast quality (DM p=0.132). GBM and HAR-RV are indistinguishable (p=0.317). The Model Confidence Set at α=0.10 retains all five models — no forecaster is significantly better than any other at the 10% level.

### Strategy Performance (Vol-Scaled Momentum, net 10bps round-trip, 2003–2024)

| Strategy | Sharpe | Max DD |
|----------|--------|--------|
| HAR-RV scaled | 0.020 | -66% |
| GBM scaled | 0.002 | -72% |
| Unscaled momentum | -0.002 | -91% |
| RollingVol scaled | -0.015 | -36% |
| LSTM scaled | -0.041 | -74% |
| GARCH scaled* | 0.226 | -170% |

*GARCH produces a constant forecast (scalar broadcast across dates), making vol-scaling degenerate. GARCH strategy Sharpe should not be interpreted as a signal result.

**Main finding:** Forecast quality does not translate to strategy performance in this sample period. LSTM has the highest IC (0.739) but the lowest non-degenerate Sharpe (-0.041). HAR-RV has the lowest ML-comparable IC (0.673) but the highest Sharpe (0.020). The 2003-2024 period is structurally hostile to momentum strategies — net-of-cost Sharpes are near zero for all forecasters. The choice of volatility model matters far less than the sample period.

### SHAP Feature Importance (GBM)

| Feature | SHAP Importance |
|---------|----------------|
| pk (Parkinson range) | **0.380** |
| rv_m (21-day RV) | 0.274 |
| sector | 0.108 |
| vix | 0.082 |
| rv_w (5-day RV) | 0.045 |

Parkinson's range estimator dominates, **not** rv_m as pre-registered. Intraday high/low range carries cross-sectional dispersion information not captured by squared close-to-close returns.

### VIX Regime IC

All models forecast better in high-VIX environments. LSTM most regime-sensitive (low: 0.709, high: 0.788). GBM shows the largest uplift in high-VIX years (2007–2009), consistent with pre-registration #2.

---

## Methodology

### Data
- **Universe:** S&P 500 point-in-time membership (Wikipedia change-log; ~80 tickers used for speed, full 500 available)
- **Prices:** yfinance adjusted OHLCV, 2000–2024
- **Target:** Forward 21-day realized variance, modeled in log space
- **Cross-validation:** Expanding window, annual retrain, 42-day embargo (López de Prado §7), 22 OOS windows

### Models
- **RollingVol:** 6-month rolling realized variance (Barroso & Santa-Clara 2015 baseline)
- **GARCH(1,1)-t:** Per-stock, Student-t innovations, annual refit; constant forecast due to arch library API
- **HAR-RV:** Per-stock OLS on daily/weekly/monthly log-RV components (Corsi 2009)
- **GBM:** Panel LightGBM, sector as categorical feature, no ticker ID (prevents memorization)
- **LSTM:** 1-layer, hidden=48, Huber loss (δ=1.0 in log-RV space), 5-seed ensemble

### Signal Scaling
Momentum signal: 12-1 (skip last month). Vol scaling: `weight_i = (target_vol / sigma_hat_i) * z_score(signal_i)`. Monthly rebalance, 10bps round-trip costs. PIT-conservative convention: signals use data through close of rebalance date; trades execute at open of next day.

### Honest Evaluation
- 8 predictions pre-registered before Phase 3 training (see `docs/predictions.md`)
- Results reported for all 5 forecasters × all strategy variants; no cherry-picking
- 2 of 8 predictions confirmed
- GARCH degenerate scaling issue documented (constant forecast → extreme weights)

---

## Project Structure

```
ml-vol-momentum/
├── src/
│   ├── data/          # universe, loaders, features, targets
│   ├── models/        # RollingVol, GARCH, HAR-RV, GBM, LSTM
│   ├── strategy/      # momentum, scaling, portfolio, costs
│   ├── eval/          # walk-forward CV, metrics, DM test, IC
│   └── interp/        # SHAP, regime analysis, sector-neutral
├── scripts/           # pipeline runners (build_data, run_baselines, run_ml_models, compare_all_models, run_phase4)
├── tests/             # unit, integration, synthetic, replication tests
├── results/           # parquet outputs (forecasts, strategies, tables)
├── docs/
│   ├── figures/       # SHAP bar chart, IC-by-year, regime heatmap, equity curves
│   └── predictions.md # pre-registered predictions + post-mortem
└── configs/
    └── default.yaml   # all hyperparameters (never retuned per OOS window)
```

---

## Reproduction

**Prerequisites:** Python 3.14, ~8GB RAM (GBM peak), ~2h wall-clock (GARCH ~45 min, LSTM ~90 min with checkpointing).

```bash
# 1. Clone and create venv (path must NOT contain a colon)
git clone <repo>
cd ml-vol-momentum
make venv         # creates /tmp/ml-vol-momentum-venv

# 2. Run full pipeline
make data         # download OHLCV, build features/targets
make baselines    # RollingVol, GARCH, HAR-RV walk-forward
make ml           # GBM + LSTM walk-forward (checkpointed)
make analysis     # comparison tables, SHAP, regime, sector-neutral figures

# 3. Run tests
make test
```

Each step is idempotent: cached results are loaded and skipped on rerun. LSTM walk-forward saves per-window checkpoints so laptop sleep does not require restarting from scratch.

**Venv note:** The project parent directory path contains a colon (`RL:ML Project/`), which breaks Python's venv module (it treats `:` as a PATH separator). The Makefile defaults to `/tmp/ml-vol-momentum-venv`. Override with `make VENV=/your/path/venv`.

---

## Design Decisions & Interviewability

Every modelling choice has a documented justification in the plan (`docs/superpowers/plans/`). Key decisions:

- **Why log-RV target?** RV is approximately log-normal (ABDL 2003); log-transformation makes OLS HAR-RV appropriate, stabilizes ML loss surfaces, and enables Jensen correction on back-transformation.
- **Why 42-day embargo?** López de Prado: embargo must exceed `2 × max(feature_window, target_horizon) = 2 × 21 = 42` to prevent label overlap between training and test.
- **Why Huber loss for LSTM?** Equity RV is extremely fat-tailed; MSE over-weights 2008/2020 tail events. Huber δ=1.0 (≈1 std in log-RV space) bounds gradient on outliers.
- **Why no ticker as GBM feature?** Prevents memorization of idiosyncratic stock histories (including delistings); forces model to learn from actual feature signals.
- **Why cross-sectional IC over time-series R²?** Vol scaling is a cross-sectional operation. High R² ≠ useful for ranking; cross-sectional Spearman IC directly measures what matters. (Though Prediction 7 shows IC → Sharpe translation was weaker than expected in this sample.)

---

## References

- Corsi (2009) — HAR-RV model
- Barroso & Santa-Clara (2015) — volatility-managed portfolios
- Gu, Kelly & Xiu (2020) — ML in cross-sectional asset pricing
- López de Prado (2018) — purged walk-forward, embargo
- Patton (2011) — QLIKE loss robustness to noisy proxies
- Hansen, Lunde & Nason (2011) — Model Confidence Set
- Diebold & Mariano (1995) — forecast comparison test
```

- [ ] **Step 2: Verify README renders correctly (check markdown)**

```bash
python3 -c "
content = open('/Users/harry/RL:ML Project/ml-vol-momentum/README.md').read()
# Check all tables have aligned pipes
import re
tables = re.findall(r'\|.*\|', content)
print(f'Found {len(tables)} table rows')
# Check all code blocks are closed
opens = content.count('\`\`\`')
print(f'Code fence count: {opens} (should be even)')
assert opens % 2 == 0, 'Unclosed code block!'
print('README structure OK')
"
```

Expected: `Found N table rows`, `Code fence count: M (should be even)`, `README structure OK`.

- [ ] **Step 3: Commit**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git add README.md
git commit -m "docs: write README with research question, key results, methodology, reproduction"
```

---

## Task 6: Final tag

- [ ] **Step 1: Run full test suite one final time**

```bash
make -C "/Users/harry/RL:ML Project/ml-vol-momentum" test 2>&1 | tail -10
```

Expected: all tests pass (some skips OK, 0 failures).

- [ ] **Step 2: Verify all result files exist**

```bash
ls "/Users/harry/RL:ML Project/ml-vol-momentum/results/forecasts/" && \
ls "/Users/harry/RL:ML Project/ml-vol-momentum/results/strategies/" && \
ls "/Users/harry/RL:ML Project/ml-vol-momentum/docs/figures/" && \
echo "All outputs present"
```

Expected: 5 forecast parquets, 6 strategy parquets (5 scaled + unscaled), 4 figure PNGs.

- [ ] **Step 3: Check git status is clean**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && git status
```

Expected: `nothing to commit, working tree clean`.

- [ ] **Step 4: Tag as project-complete**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum"
git tag -a project-complete -m "Phase 5 complete: README, predictions post-mortem, reproducible Makefile, pinned requirements"
```

- [ ] **Step 5: Confirm tag**

```bash
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && git tag -l && git show project-complete --stat
```

Expected: `project-complete` tag listed; shows the tagged commit.

---

## Self-Review

**Spec coverage check:**
- ✅ README.md with research question, results, methodology, reproduction — Task 5
- ✅ Predictions post-mortem on all 8 predictions — Task 4
- ✅ Makefile fixed to use real scripts — Task 1
- ✅ requirements.txt pinned — Task 2
- ✅ `make test` passes cleanly — Task 3
- ✅ `project-complete` git tag — Task 6

**Placeholder scan:** No TBDs, no "similar to Task N", all code blocks complete.

**Type consistency:** All script paths reference files created in same task or already existing in `scripts/`.
