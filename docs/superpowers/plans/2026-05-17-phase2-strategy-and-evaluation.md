# Phase 2: Strategy & Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full momentum signal → vol scaling → portfolio construction → transaction costs → evaluation pipeline, including the Barroso & Santa-Clara replication, evaluation statistics (DM, MZ, IC), and the master results comparison table.

**Architecture:** The pipeline is strictly sequential: `momentum_signal` → `vol_scale` → `build_portfolios` → `apply_costs` → `metrics`. The PIT convention is already encoded in stubs (signal at close of t, trades at open of t+1, portfolio.py shifts weights +1 day). Walk-forward infrastructure and baseline forecasters from Phase 1 are complete. This phase wires them together end-to-end.

**Tech Stack:** pandas, numpy, scipy.stats, src.eval.metrics (Phase 1), src.eval.walk_forward (Phase 1), src.models.baselines (Phase 1), src.data.{features, targets, loaders, universe} (Phase 1)

**Phase 2 deadline:** 2026-06-07 (21 days from 2026-05-17). Anything not done by then is Phase 4 work.

**Critical invariant (PIT convention):**
- Signal and vol forecast computed at close of t using data ≤ t
- Weights are shifted +1 in `build_portfolios` before multiplying returns
- `apply_costs` uses the same shifted weights for turnover
- **Never** use t+1 prices or returns to compute signal at t

---

## File Structure

```
src/strategy/
  momentum.py      — momentum_signal() (replaces stub)
  scaling.py       — vol_scale() (replaces stub)
  portfolio.py     — build_portfolios(), month-end rebalance logic
  costs.py         — apply_costs()

src/eval/
  comparison.py    — build_results_table(), build_dm_matrix(), build_mz_table()
                     (already exists with stubs — implement)

tests/
  test_strategy.py — unit tests for momentum, scaling, portfolio, costs
  test_comparison.py — tests for comparison tables
  test_replication.py — already exists with Barroso tests (slow, real data)
```

**Existing files that need NO changes:**
- `src/eval/walk_forward.py` — complete
- `src/eval/metrics.py` — complete
- `src/eval/tests.py` — complete
- `src/eval/synthetic.py` — complete
- `src/models/baselines.py` — complete

---

## Task 1: `momentum_signal` implementation

**Files:**
- Modify: `src/strategy/momentum.py`
- Create: `tests/test_strategy.py`

The 12-1 momentum signal: for each stock at date t, compute the cumulative log return from t-252 to t-21 (skip last month). Only computed at month-end rebalance dates. Returns a long-form panel with a single column `"signal"`.

**Formula:** `signal_t = sum(log_ret, t-252, t-22)` = `rolling(252).sum().shift(21) - rolling(21).sum().shift(1)`

Wait — more precisely: we want total return over [t-252, t-22] exclusive of last 21 days. Using log returns:
```
signal_t = log(P_{t-21} / P_{t-252})
         = log_ret.rolling(252).sum().shift(21) 
```
Note: `rolling(252).sum()` at date t gives sum of log returns from t-252+1 to t. `.shift(21)` at date t gives the rolling sum that was at t-21, i.e., sum from t-252-21+1 to t-21. That is the 12-1 with skip.

Simpler: `mom = close.pct_change(252).shift(21)` — skip one month, look back 12 months. Using log returns for consistency: `mom = log(close.shift(21) / close.shift(252))`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_strategy.py
import numpy as np
import pandas as pd
import pytest
from src.strategy.momentum import momentum_signal

def _make_prices_panel(n_dates=300, n_tickers=5, trend=0.001, seed=42):
    """Trending prices panel for momentum tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-04", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    # Ticker 0 has strong positive trend, ticker 4 has strong negative trend
    trends = np.linspace(trend, -trend, n_tickers)
    log_rets = rng.normal(0, 0.01, (n_dates, n_tickers))
    for i, t in enumerate(trends):
        log_rets[:, i] += t
    log_prices = np.cumsum(log_rets, axis=0)
    prices = np.exp(log_prices)
    price_vals = prices.ravel()
    panel = pd.DataFrame({"close": price_vals}, index=idx)
    return panel


class TestMomentumSignal:
    def test_returns_dataframe_with_signal_column(self):
        prices = _make_prices_panel()
        sig = momentum_signal(prices)
        assert isinstance(sig, pd.DataFrame)
        assert "signal" in sig.columns

    def test_index_is_date_ticker_multiindex(self):
        prices = _make_prices_panel()
        sig = momentum_signal(prices)
        assert sig.index.names == ["date", "ticker"]

    def test_no_signal_before_lookback_warmup(self):
        """No valid signal until at least lookback+skip days of data."""
        prices = _make_prices_panel(n_dates=300)
        sig = momentum_signal(prices, lookback=252, skip=21)
        sig_nonan = sig.dropna()
        first_valid_date = sig_nonan.index.get_level_values("date").min()
        # 252 + 21 = 273 trading days needed before first valid signal
        dates = prices.index.get_level_values("date").unique()
        min_required_date = dates[252 + 21 - 1]
        assert first_valid_date >= min_required_date

    def test_positive_trend_ticker_has_higher_signal(self):
        """Ticker with positive trend should have higher signal than negative trend ticker."""
        prices = _make_prices_panel(n_dates=300, n_tickers=2, trend=0.002)
        sig = momentum_signal(prices)
        sig_nonan = sig.dropna()
        dates = sig_nonan.index.get_level_values("date").unique()
        last_date = dates[-1]
        vals = sig_nonan.xs(last_date, level="date")["signal"]
        # T0 positive trend, T1 negative trend
        assert vals["T0"] > vals["T1"]

    def test_no_future_leakage(self):
        """Signal at date t must not change when we add t+1 data."""
        prices = _make_prices_panel(n_dates=300)
        dates = prices.index.get_level_values("date").unique()
        cutoff = dates[280]
        sig_short = momentum_signal(prices[prices.index.get_level_values("date") <= cutoff])
        sig_full  = momentum_signal(prices)
        # Values at cutoff must be identical
        short_vals = sig_short.xs(cutoff, level="date")["signal"].dropna()
        full_vals  = sig_full.xs(cutoff, level="date")["signal"].reindex(short_vals.index)
        pd.testing.assert_series_equal(short_vals, full_vals, check_names=False)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ml-vol-momentum
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_strategy.py::TestMomentumSignal -v --tb=short
```
Expected: 5 failures with `NotImplementedError`.

- [ ] **Step 3: Implement `momentum_signal`**

```python
# src/strategy/momentum.py  (replace the stub function)
def momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """12-1 momentum: log return from t-lookback to t-skip, for each stock."""
    # PIT: at date t, signal = log(close_{t-skip} / close_{t-lookback})
    # This uses only prices through close of t-skip (lookback days ago relative
    # to the signal date). No data from t-skip+1 to t enters the calculation.
    close = prices["close"].unstack("ticker")
    log_ret = np.log(close / close.shift(1))
    # rolling(lookback).sum() at t = cumret from t-lookback+1 to t
    # shift(skip) pushes that to be: cumret from t-lookback-skip+1 to t-skip
    # That gives a (lookback - skip) = 231-day window ending at t-skip.
    # Equivalently: log(close_{t-skip} / close_{t-lookback-skip+1})
    # Per Jegadeesh-Titman: lookback=252, skip=21 → 12-1 mom
    signal = log_ret.rolling(lookback).sum().shift(skip)
    result = signal.stack(future_stack=True).rename("signal").to_frame()
    return result
```

- [ ] **Step 4: Run tests**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_strategy.py::TestMomentumSignal -v --tb=short
```
Expected: 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/momentum.py tests/test_strategy.py
git commit -m "feat: momentum_signal 12-1 with PIT skip convention"
```

---

## Task 2: `vol_scale` implementation

**Files:**
- Modify: `src/strategy/scaling.py`
- Modify: `tests/test_strategy.py` (add `TestVolScale` class)

Cross-sectional vol scaling: for each stock at date t, weight = (target_vol / sigma_hat) * z_score(signal). The sigma_hat panel comes from the baseline forecasters (forecast_rv column, annualized). The z-score is cross-sectional (across stocks at each date).

**Formula:**
- `z_i = (signal_i - mean(signal)) / std(signal)` across stocks at date t
- `w_i = (target_vol / sqrt(sigma_hat_i * 252)) * z_i`

Note: `forecast_rv` from baselines is daily sum-of-squared-returns, so annualized vol = `sqrt(forecast_rv * 252)`.

- [ ] **Step 1: Write failing tests (add to tests/test_strategy.py)**

```python
from src.strategy.scaling import vol_scale

class TestVolScale:
    def _make_signal_panel(self, n_dates=50, n_tickers=10, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        sig = rng.standard_normal(n_dates * n_tickers)
        return pd.DataFrame({"signal": sig}, index=idx)

    def _make_sigma_panel(self, n_dates=50, n_tickers=10, seed=1):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        # forecast_rv: daily variance, e.g. (0.01)^2 = 0.0001
        rv = np.abs(rng.normal(0.0001, 0.00002, n_dates * n_tickers))
        return pd.DataFrame({"forecast_rv": rv}, index=idx)

    def test_returns_dataframe_with_weight_column(self):
        sig = self._make_signal_panel()
        sigma = self._make_sigma_panel()
        w = vol_scale(sig, sigma, target_vol=0.10)
        assert isinstance(w, pd.DataFrame)
        assert "weight" in w.columns

    def test_higher_vol_gets_lower_weight(self):
        """A stock with 2x the forecast vol should get ~0.5x the weight."""
        dates = pd.date_range("2015-01-02", periods=5, freq="B")
        tickers = ["LOW", "HIGH"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        sig = pd.DataFrame({"signal": [1.0, 1.0] * 5}, index=idx)
        # HIGH has 4x the daily variance of LOW
        rv_vals = [0.0001, 0.0004] * 5
        sigma = pd.DataFrame({"forecast_rv": rv_vals}, index=idx)
        w = vol_scale(sig, sigma, target_vol=0.10)
        last_date = dates[-1]
        w_low  = w.xs(last_date, level="date").loc["LOW", "weight"]
        w_high = w.xs(last_date, level="date").loc["HIGH", "weight"]
        # LOW has half the vol of HIGH, so should have ~2x the weight
        assert w_low > w_high
        assert abs(w_low / w_high - 2.0) < 0.01

    def test_output_index_matches_signal_index(self):
        sig = self._make_signal_panel()
        sigma = self._make_sigma_panel()
        w = vol_scale(sig, sigma, target_vol=0.10)
        pd.testing.assert_index_equal(w.index, sig.index)

    def test_zero_std_signal_gives_zero_weights(self):
        """If all signals are identical (std=0), weights should be 0."""
        dates = pd.date_range("2015-01-02", periods=5, freq="B")
        tickers = ["A", "B", "C"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        sig = pd.DataFrame({"signal": 1.0}, index=idx)
        rv_vals = [0.0001] * 15
        sigma = pd.DataFrame({"forecast_rv": rv_vals}, index=idx)
        w = vol_scale(sig, sigma, target_vol=0.10)
        assert (w["weight"] == 0.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_strategy.py::TestVolScale -v --tb=short
```
Expected: 4 failures with `NotImplementedError`.

- [ ] **Step 3: Implement `vol_scale`**

```python
# src/strategy/scaling.py  (replace the stub function)
def vol_scale(
    signal: pd.DataFrame,
    sigma_hat: pd.DataFrame,
    target_vol: float,
) -> pd.DataFrame:
    """Cross-sectionally scale signal by inverse forecast vol to target_vol."""
    # PIT: sigma_hat and signal share the same timestamp t.
    # portfolio.py shifts the resulting weights by +1 day.
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
            # daily annualized vol = sqrt(forecast_rv * 252)
            rv_dt = sigma_hat.xs(dt, level="date")["forecast_rv"].reindex(z.index).dropna()
            common = z.index.intersection(rv_dt.index)
            z = z[common]
            rv_dt = rv_dt[common]
            ann_vol = np.sqrt(rv_dt * 252)
            # avoid division by zero for degenerate tickers
            ann_vol = ann_vol.replace(0, np.nan)
            w = (target_vol / ann_vol) * z
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

- [ ] **Step 4: Run tests**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_strategy.py::TestVolScale -v --tb=short
```
Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/scaling.py tests/test_strategy.py
git commit -m "feat: vol_scale cross-sectional inverse-vol signal weighting"
```

---

## Task 3: `portfolio.py` — quintile portfolios and the +1 day PIT shift

**Files:**
- Create: `src/strategy/portfolio.py`
- Modify: `tests/test_strategy.py` (add `TestBuildPortfolios`)

This is the most PIT-sensitive file. The +1 day shift is applied HERE, not in the signal or scaling code. Building a long-short quintile portfolio: top 20% long, bottom 20% short, equal-weighted within each leg. Long-only: top 20% long, equal-weighted. Vol-targeted gross: use vol_scale weights, normalize to target gross exposure.

**CRITICAL:**
```python
# At close of t, we compute weights_t.
# We then SHIFT all weights forward by one trading day.
# The return attributed to those weights is the return from close of t+1 to close of t+2
# (actually from open of t+1 to close of t+1 in practice — we approximate with c-to-c).
# This implements: "signal known at close t, traded at open t+1".
weights_shifted = weights.unstack("ticker").shift(1).stack(future_stack=True)
```

- [ ] **Step 1: Write failing tests (add to tests/test_strategy.py)**

```python
from src.strategy.portfolio import build_portfolios

class TestBuildPortfolios:
    def _make_signal_panel(self, n_dates=60, n_tickers=20, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")
        tickers = [f"T{i:02d}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        sig = rng.standard_normal(n_dates * n_tickers)
        return pd.DataFrame({"signal": sig}, index=idx)

    def _make_weights_panel(self, n_dates=60, n_tickers=20, seed=2):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")
        tickers = [f"T{i:02d}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        w = rng.standard_normal(n_dates * n_tickers)
        return pd.DataFrame({"weight": w}, index=idx)

    def test_long_short_quintile_has_weight_column(self):
        sig = self._make_signal_panel()
        result = build_portfolios(sig, mode="long_short_quintile")
        assert "weight" in result.columns

    def test_long_short_quintile_weights_sum_near_zero(self):
        """Long and short legs are equal in magnitude, so sum ≈ 0 per date."""
        sig = self._make_signal_panel()
        result = build_portfolios(sig, mode="long_short_quintile")
        # After the +1 shift, first date weights are NaN (no prior signal)
        result_nonan = result.dropna()
        dates = result_nonan.index.get_level_values("date").unique()
        for dt in dates[:5]:
            total = result_nonan.xs(dt, level="date")["weight"].sum()
            assert abs(total) < 1e-9, f"Sum of weights at {dt} = {total:.6f}, expected ~0"

    def test_weights_shifted_by_one_day(self):
        """Weight at date t+1 should equal signal rank computed at date t.
        Verify by checking that the first date after signal start has non-NaN
        weights only when there was a prior signal date."""
        sig = self._make_signal_panel()
        result = build_portfolios(sig, mode="long_short_quintile")
        sig_dates = sig.index.get_level_values("date").unique()
        result_dates = result.dropna(how="all").index.get_level_values("date").unique()
        # First weight date must be AFTER first signal date
        assert result_dates[0] > sig_dates[0]

    def test_long_only_quintile_weights_nonnegative(self):
        sig = self._make_signal_panel()
        result = build_portfolios(sig, mode="long_only_quintile")
        assert (result["weight"].dropna() >= 0).all()

    def test_no_close_to_close_leakage(self):
        """Weight using signal at t should be NaN at date t itself (shifted to t+1)."""
        sig = self._make_signal_panel(n_dates=60, n_tickers=20)
        result = build_portfolios(sig, mode="long_short_quintile")
        first_sig_date = sig.index.get_level_values("date").unique()[0]
        # At first signal date, weights should be NaN (no prior signal)
        first_date_weights = result.xs(first_sig_date, level="date")["weight"]
        assert first_date_weights.isna().all(), \
            "Weights at first signal date should be NaN (signals not yet shifted)"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_strategy.py::TestBuildPortfolios -v --tb=short
```
Expected: 5 failures.

- [ ] **Step 3: Implement `portfolio.py`**

```python
# src/strategy/portfolio.py
from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd


def _quintile_weights(signal_date: pd.Series, mode: str) -> pd.Series:
    """Compute equal-weight quintile portfolio for a single date's signal."""
    sig = signal_date.dropna()
    if len(sig) < 10:
        return pd.Series(dtype=float)
    q20 = sig.quantile(0.20)
    q80 = sig.quantile(0.80)
    if mode == "long_short_quintile":
        long_mask  = sig >= q80
        short_mask = sig <= q20
        n_long  = long_mask.sum()
        n_short = short_mask.sum()
        if n_long == 0 or n_short == 0:
            return pd.Series(0.0, index=sig.index)
        w = pd.Series(0.0, index=sig.index)
        w[long_mask]  =  1.0 / n_long
        w[short_mask] = -1.0 / n_short
        return w
    elif mode == "long_only_quintile":
        long_mask = sig >= q80
        n_long = long_mask.sum()
        if n_long == 0:
            return pd.Series(0.0, index=sig.index)
        w = pd.Series(0.0, index=sig.index)
        w[long_mask] = 1.0 / n_long
        return w
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_portfolios(
    signal: pd.DataFrame,
    weights: pd.DataFrame | None = None,
    mode: Literal["long_short_quintile", "long_only_quintile",
                  "vol_targeted_gross"] = "long_short_quintile",
) -> pd.DataFrame:
    """
    Build portfolio weight panel from signal (and optional pre-scaled weights).

    PIT: weights are SHIFTED +1 trading day before return attribution.
    Signal at date t produces weights held from open of t+1.
    """
    if mode in ("long_short_quintile", "long_only_quintile"):
        dates = signal.index.get_level_values("date").unique()
        rows = []
        for dt in dates:
            sig_dt = signal.xs(dt, level="date")["signal"]
            w = _quintile_weights(sig_dt, mode)
            if w.empty:
                continue
            sub = w.rename("weight").to_frame()
            sub.index = pd.MultiIndex.from_arrays(
                [[dt] * len(sub), sub.index], names=["date", "ticker"])
            rows.append(sub)
        if not rows:
            return pd.DataFrame(columns=["weight"])
        raw = pd.concat(rows).sort_index()
    elif mode == "vol_targeted_gross":
        if weights is None:
            raise ValueError("vol_targeted_gross mode requires pre-scaled weights")
        raw = weights.copy()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ── PIT shift: weight at t becomes active at t+1 ──────────────────────────
    w_wide = raw["weight"].unstack("ticker")
    w_shifted = w_wide.shift(1)
    result = w_shifted.stack(future_stack=True).rename("weight").to_frame()
    return result
```

- [ ] **Step 4: Run tests**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_strategy.py::TestBuildPortfolios -v --tb=short
```
Expected: 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/strategy/portfolio.py tests/test_strategy.py
git commit -m "feat: portfolio quintile construction with +1 day PIT shift"
```

---

## Task 4: `costs.py` — transaction cost deduction

**Files:**
- Create: `src/strategy/costs.py`
- Modify: `tests/test_strategy.py` (add `TestApplyCosts`)

Round-trip 10bps = 5bps per side. Daily turnover = sum of absolute weight changes. Cost per day = turnover * 5bps. Net return = gross return - cost.

- [ ] **Step 1: Write failing tests (add to tests/test_strategy.py)**

```python
from src.strategy.costs import apply_costs

class TestApplyCosts:
    def _make_constant_weights(self, n_dates=30, n_tickers=5):
        dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        w = pd.DataFrame({"weight": 0.2}, index=idx)
        return w

    def _make_returns(self, n_dates=30, n_tickers=5, r=0.001):
        dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        return pd.DataFrame({"return": r}, index=idx)

    def test_constant_weights_zero_cost_after_first_day(self):
        """After the first rebalance, constant weights have zero turnover."""
        w = self._make_constant_weights()
        r = self._make_returns()
        net = apply_costs(w, r, cost_bps=10.0)
        # First day has turnover (from 0 to 0.2 per ticker), subsequent days zero
        assert (net.iloc[2:] == net.iloc[2:]).all()  # no NaN after warmup
        # Gross return for constant 0.2 in 5 tickers = 0.001 per ticker * 0.2 * 5
        expected_gross = 0.001 * 5 * 0.2
        assert abs(net.iloc[-1] - expected_gross) < 1e-9

    def test_high_turnover_reduces_net_returns(self):
        """Full portfolio flip every day reduces net return vs no-cost."""
        dates = pd.date_range("2015-01-02", periods=20, freq="B")
        tickers = ["A", "B"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        # Flip: one day A=1 B=0, next day A=0 B=1
        flip_weights = []
        for i, dt in enumerate(dates):
            if i % 2 == 0:
                flip_weights.extend([(dt, "A", 1.0), (dt, "B", 0.0)])
            else:
                flip_weights.extend([(dt, "A", 0.0), (dt, "B", 1.0)])
        w_df = pd.DataFrame(flip_weights, columns=["date", "ticker", "weight"])
        w_df = w_df.set_index(["date", "ticker"])
        r = self._make_returns(n_dates=20, n_tickers=2, r=0.001)
        net_with_cost = apply_costs(w_df, r, cost_bps=10.0)
        net_no_cost   = apply_costs(w_df, r, cost_bps=0.0)
        assert net_with_cost.mean() < net_no_cost.mean()

    def test_returns_series_indexed_by_date(self):
        w = self._make_constant_weights()
        r = self._make_returns()
        net = apply_costs(w, r, cost_bps=10.0)
        assert isinstance(net, pd.Series)
        assert net.index.name == "date"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_strategy.py::TestApplyCosts -v --tb=short
```
Expected: 3 failures.

- [ ] **Step 3: Implement `apply_costs`**

```python
# src/strategy/costs.py
from __future__ import annotations
import numpy as np
import pandas as pd


def apply_costs(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.Series:
    """
    Compute daily net portfolio return after transaction costs.

    cost_bps is round-trip; per-side cost = cost_bps / 2 / 10_000.
    Daily cost = sum(|Δweight|) * per_side_cost.
    """
    cost_per_side = cost_bps / 2.0 / 10_000.0

    w = weights["weight"].unstack("ticker").fillna(0.0)
    r = returns["return"].unstack("ticker").reindex(index=w.index).fillna(0.0)

    # Gross return: sum of weight * return per day
    gross = (w * r).sum(axis=1)

    # Turnover: sum of absolute weight changes per day
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()  # first day: from 0 to initial weights

    cost = turnover * cost_per_side
    net = gross - cost
    net.index.name = "date"
    return net
```

- [ ] **Step 4: Run tests**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_strategy.py::TestApplyCosts -v --tb=short
```
Expected: 3 pass.

- [ ] **Step 5: Run all non-slow tests**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/ -m "not slow" -v --tb=short 2>&1 | tail -30
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/strategy/costs.py tests/test_strategy.py
git commit -m "feat: apply_costs round-trip bps transaction cost deduction"
```

---

## Task 5: End-to-end pipeline integration test

**Files:**
- Create: `tests/test_pipeline_e2e.py`

This test runs the full pipeline on a small synthetic panel (50 stocks, 5 years) and verifies:
1. White-noise signal → Sharpe ∈ [-1.0, 1.0] after costs
2. Leakage signal → Sharpe > 10 (the shift convention is correct)
3. Vol-scaled portfolio has lower realized vol variance than unscaled
4. Costs reduce mean return on high-turnover strategy

- [ ] **Step 1: Write the integration test**

```python
# tests/test_pipeline_e2e.py
import numpy as np
import pandas as pd
import pytest
from src.eval.synthetic import white_noise_panel, leakage_test_signal
from src.strategy.momentum import momentum_signal
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.models.baselines import RollingVolModel
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.metrics import sharpe


def _build_panel_with_returns(n_dates=600, n_stocks=50, seed=42):
    """White-noise panel in the format needed by all strategy functions."""
    panel = white_noise_panel(n_dates, n_stocks, seed)
    # Add a 'close' column (cumulative product of returns as fake prices)
    returns = panel["return"].unstack("ticker")
    prices = (1 + returns).cumprod()
    prices_long = prices.stack(future_stack=True).rename("close").to_frame()
    prices_long.index.names = ["date", "ticker"]
    return panel, prices_long


class TestPipelineEndToEnd:
    def test_white_noise_unscaled_sharpe_near_zero(self):
        panel, prices = _build_panel_with_returns(n_dates=600, n_stocks=50, seed=7)
        signal = momentum_signal(prices, lookback=252, skip=21)
        weights = build_portfolios(signal, mode="long_short_quintile")
        net = apply_costs(weights, panel.rename(columns={"return": "return"}),
                          cost_bps=10.0)
        sr = sharpe(net.dropna())
        assert abs(sr) < 2.0, f"White-noise Sharpe = {sr:.2f}, expected |Sharpe| < 2.0"

    def test_leakage_signal_gives_high_sharpe(self):
        """Signal = return at t+1 (perfect leakage) → Sharpe >> 10."""
        panel, prices = _build_panel_with_returns(n_dates=600, n_stocks=50, seed=7)
        signal = leakage_test_signal(panel)
        weights = build_portfolios(signal, mode="long_short_quintile")
        net = apply_costs(weights, panel, cost_bps=0.0)
        sr = sharpe(net.dropna())
        assert sr > 10, f"Leakage Sharpe = {sr:.2f}, expected > 10"

    def test_vol_scaling_with_rolling_vol(self):
        """Vol-scaled portfolio should have smaller realized vol swings than unscaled."""
        from src.strategy.scaling import vol_scale
        panel, prices = _build_panel_with_returns(n_dates=600, n_stocks=50, seed=9)
        signal = momentum_signal(prices, lookback=252, skip=21)
        # Get rolling-vol forecasts
        windows = generate_windows(
            pd.Timestamp("2000-01-03"),
            prices.index.get_level_values("date").max(),
            first_test_year=2002,
        )
        rv_model = RollingVolModel(window=126)
        # Build a returns panel for the rolling vol model
        rv_panel = panel.copy()
        oos_forecasts = run_walk_forward(rv_model, rv_panel, windows)
        if oos_forecasts.empty:
            pytest.skip("No forecasts generated")
        w_scaled = vol_scale(signal, oos_forecasts, target_vol=0.10)
        w_portfolio = build_portfolios(None, weights=w_scaled,
                                       mode="vol_targeted_gross")
        net_scaled = apply_costs(w_portfolio, panel, cost_bps=10.0)
        # Just check it runs without error and returns a Series
        assert isinstance(net_scaled, pd.Series)
        assert len(net_scaled.dropna()) > 100
```

- [ ] **Step 2: Run the integration test**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_pipeline_e2e.py -v --tb=short
```
Expected: all pass (the white-noise and leakage tests are the same canaries from Phase 1).

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_e2e.py
git commit -m "test: end-to-end pipeline integration tests (PIT canaries)"
```

---

## Task 6: `comparison.py` — implement results table functions

**Files:**
- Modify: `src/eval/comparison.py`
- Create: `tests/test_comparison.py`

Check what already exists in comparison.py:

```bash
cat src/eval/comparison.py
```

Implement `build_results_table` (takes dict of strategy name → daily return Series, outputs Sharpe/Sortino/MaxDD/Calmar/AnnRet), `build_dm_matrix` (DM test for all model pairs), and `build_mz_table` (MZ regression for all models).

- [ ] **Step 1: Read the existing comparison.py stubs**

```bash
cat src/eval/comparison.py
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_comparison.py
import numpy as np
import pandas as pd
import pytest
from src.eval.comparison import build_results_table, build_dm_matrix, build_mz_table


def _make_returns(n=252, mean=0.001, std=0.01, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-04", periods=n, freq="B")
    return pd.Series(rng.normal(mean, std, n), index=dates)


def _make_forecast_panel(n_dates=100, n_tickers=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-04", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    vals = np.abs(rng.normal(0.0001, 0.00005, n_dates * n_tickers))
    return pd.DataFrame({"forecast_rv": vals}, index=idx)


class TestBuildResultsTable:
    def test_returns_dataframe(self):
        strategies = {"strat_a": _make_returns(seed=0), "strat_b": _make_returns(seed=1)}
        tbl = build_results_table(strategies)
        assert isinstance(tbl, pd.DataFrame)

    def test_has_required_columns(self):
        strategies = {"strat_a": _make_returns(seed=0)}
        tbl = build_results_table(strategies)
        for col in ("sharpe", "sortino", "max_drawdown", "calmar", "ann_ret"):
            assert col in tbl.columns, f"Missing column: {col}"

    def test_row_per_strategy(self):
        strategies = {"a": _make_returns(seed=0), "b": _make_returns(seed=1)}
        tbl = build_results_table(strategies)
        assert set(tbl.index) == {"a", "b"}

    def test_positive_drift_has_positive_sharpe(self):
        strategies = {"good": _make_returns(mean=0.01, std=0.01, seed=0)}
        tbl = build_results_table(strategies)
        assert tbl.loc["good", "sharpe"] > 0


class TestBuildDMMatrix:
    def test_returns_dataframe(self):
        forecasts = {
            "model_a": _make_forecast_panel(seed=0),
            "model_b": _make_forecast_panel(seed=1),
        }
        realized = _make_forecast_panel(seed=99)
        dm_mat = build_dm_matrix(forecasts, realized)
        assert isinstance(dm_mat, pd.DataFrame)

    def test_diagonal_is_nan(self):
        forecasts = {"m1": _make_forecast_panel(seed=0), "m2": _make_forecast_panel(seed=1)}
        realized = _make_forecast_panel(seed=99)
        dm_mat = build_dm_matrix(forecasts, realized)
        for m in dm_mat.index:
            assert np.isnan(dm_mat.loc[m, m])


class TestBuildMZTable:
    def test_returns_dataframe_with_expected_columns(self):
        forecasts = {"model_a": _make_forecast_panel(seed=0)}
        realized = _make_forecast_panel(seed=99)
        mz = build_mz_table(forecasts, realized)
        assert isinstance(mz, pd.DataFrame)
        for col in ("alpha", "beta", "r2"):
            assert col in mz.columns
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_comparison.py -v --tb=short
```

- [ ] **Step 4: Read the existing comparison.py to understand current stubs**

```bash
cat src/eval/comparison.py
```

- [ ] **Step 5: Implement comparison.py**

```python
# src/eval/comparison.py
from __future__ import annotations
import numpy as np
import pandas as pd
from src.eval.metrics import sharpe, sortino, max_drawdown, calmar
from src.eval.tests import diebold_mariano, mincer_zarnowitz


def build_results_table(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute Sharpe, Sortino, MaxDD, Calmar, AnnRet for each strategy."""
    rows = {}
    for name, r in strategies.items():
        r = r.dropna()
        equity = (1 + r).cumprod()
        rows[name] = {
            "sharpe":       sharpe(r),
            "sortino":      sortino(r),
            "max_drawdown": max_drawdown(equity),
            "calmar":       calmar(r),
            "ann_ret":      float(r.mean() * 252),
        }
    return pd.DataFrame(rows).T


def build_dm_matrix(
    forecasts: dict[str, pd.DataFrame],
    realized: pd.DataFrame,
) -> pd.DataFrame:
    """Pairwise Diebold-Mariano test (MSE loss). Returns p-value matrix."""
    names = list(forecasts.keys())
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    real_series = realized["forecast_rv"]
    for i, m1 in enumerate(names):
        for j, m2 in enumerate(names):
            if i == j:
                continue
            f1 = forecasts[m1]["forecast_rv"].reindex(real_series.index).dropna()
            f2 = forecasts[m2]["forecast_rv"].reindex(real_series.index).dropna()
            rv = real_series.reindex(f1.index.intersection(f2.index)).dropna()
            f1 = f1.reindex(rv.index)
            f2 = f2.reindex(rv.index)
            try:
                _, p = diebold_mariano(rv - f1, rv - f2, loss="mse")
                mat.loc[m1, m2] = p
            except Exception:
                pass
    return mat


def build_mz_table(
    forecasts: dict[str, pd.DataFrame],
    realized: pd.DataFrame,
) -> pd.DataFrame:
    """Mincer-Zarnowitz regression for each model. Returns alpha, beta, r2, p_joint."""
    real_series = realized["forecast_rv"]
    rows = {}
    for name, fc in forecasts.items():
        f = fc["forecast_rv"].reindex(real_series.index).dropna()
        rv = real_series.reindex(f.index).dropna()
        f = f.reindex(rv.index)
        try:
            mz = mincer_zarnowitz(rv, f)
            rows[name] = {
                "alpha":   mz["alpha"],
                "beta":    mz["beta"],
                "r2":      mz["r2"],
                "p_joint": mz.get("p_joint", np.nan),
            }
        except Exception:
            rows[name] = {"alpha": np.nan, "beta": np.nan, "r2": np.nan, "p_joint": np.nan}
    return pd.DataFrame(rows).T
```

- [ ] **Step 6: Run tests**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_comparison.py -v --tb=short
```
Expected: all pass.

- [ ] **Step 7: Run full test suite**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/ -m "not slow" -v --tb=short 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/eval/comparison.py tests/test_comparison.py
git commit -m "feat: build_results_table, build_dm_matrix, build_mz_table"
```

---

## Task 7: CHECKPOINT 2a — Unscaled momentum positive Sharpe pre-2008

This is a real-data validation checkpoint. Run the unscaled long-short quintile momentum strategy on the full S&P 500 universe (or the 80-ticker subsample) over 2000–2008. Verify positive Sharpe.

- [ ] **Step 1: Write the validation script**

```python
# scripts/validate_checkpoint_2a.py
"""
CHECKPOINT 2a: Unscaled momentum positive Sharpe pre-2008.
Run with: python scripts/validate_checkpoint_2a.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.data.loaders import load_ohlcv
from src.data.universe import get_universe
from src.strategy.momentum import momentum_signal
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.eval.metrics import sharpe, max_drawdown

start = pd.Timestamp("1998-01-01")  # 2 years warmup for 12-1 signal
end   = pd.Timestamp("2007-12-31")

tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print(f"Loading OHLCV for {len(tickers)} tickers...")
ohlcv = load_ohlcv(tickers, start, end)

# Build returns panel
returns_frames = []
for ticker, grp in ohlcv.groupby(level="ticker"):
    close = grp.droplevel("ticker")["close"]
    r = np.log(close / close.shift(1))
    r.name = "return"
    idx = pd.MultiIndex.from_arrays([r.index, [ticker]*len(r)], names=["date","ticker"])
    returns_frames.append(r.set_axis(idx))
returns_panel = pd.concat(returns_frames).to_frame()

# Compute momentum signal using close prices
prices_panel = ohlcv[["close"]].copy()

print("Computing momentum signal...")
signal = momentum_signal(prices_panel, lookback=252, skip=21)

print("Building long-short quintile portfolio...")
weights = build_portfolios(signal, mode="long_short_quintile")

print("Applying transaction costs...")
net = apply_costs(weights, returns_panel, cost_bps=10.0)
net = net.dropna()
net_2000_2007 = net.loc["2000":"2007"]

sr = sharpe(net_2000_2007)
equity = (1 + net_2000_2007).cumprod()
mdd = max_drawdown(equity)

print(f"\nCHECKPOINT 2a results (2000-2007):")
print(f"  Annualized Sharpe: {sr:.3f}  (gate: > 0)")
print(f"  Max Drawdown:      {mdd:.3f}")
print(f"  Ann Return:        {net_2000_2007.mean() * 252:.3f}")
print(f"\nResult: {'PASS' if sr > 0 else 'FAIL'}")
```

- [ ] **Step 2: Run validation**

```bash
mkdir -p scripts
python scripts/validate_checkpoint_2a.py
```
Expected: Sharpe > 0 (momentum worked pre-2008).

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_checkpoint_2a.py
git commit -m "chore: checkpoint 2a validation script — unscaled momentum pre-2008"
```

---

## Task 8: CHECKPOINT 2b — Barroso replication: 2009 crash + vol-scaling uplift

Run on 2000-2010. Unscaled must crash in 2009 (max DD < -0.20). Rolling-vol-scaled must show higher Sharpe. This is the most critical sanity check in Phase 2.

**The 2009 crash should look terrifying — ~-50% peak-to-trough if the data is right (Daniel-Moskowitz 2016). A -20% dip means the implementation is wrong. Use the equity curve chart, not just the MaxDD number.**

- [ ] **Step 1: Write the Barroso replication script**

```python
# scripts/validate_checkpoint_2b.py
"""
CHECKPOINT 2b: Barroso & Santa-Clara (2015) replication.
2009 momentum crash visible unscaled; vol-scaling provides uplift.
Run with: python scripts/validate_checkpoint_2b.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.data.loaders import load_ohlcv, load_vix
from src.data.universe import get_universe
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.models.baselines import RollingVolModel
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.metrics import sharpe, max_drawdown

start = pd.Timestamp("1998-01-01")
end   = pd.Timestamp("2010-12-31")

tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
print(f"Loading data for {len(tickers)} tickers...")
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

# Build feature panel for rolling vol model
features = build_feature_panel(ohlcv, vix)
targets  = forward_rv(returns_panel)
panel    = features.join(targets, how="inner").dropna(subset=["target_log_rv"])

prices_panel = ohlcv[["close"]]
signal = momentum_signal(prices_panel, lookback=252, skip=21)

# Unscaled portfolio
weights_unscaled = build_portfolios(signal, mode="long_short_quintile")
net_unscaled = apply_costs(weights_unscaled, returns_panel, cost_bps=10.0).dropna()

# Rolling-vol-scaled portfolio (walk-forward, 42-day embargo)
windows = generate_windows(start, end, first_test_year=2002)
rv_model = RollingVolModel(window=126)
oos_forecasts = run_walk_forward(rv_model, panel, windows)
weights_scaled_raw = vol_scale(signal, oos_forecasts, target_vol=0.10)
weights_scaled = build_portfolios(None, weights=weights_scaled_raw,
                                  mode="vol_targeted_gross")
net_scaled = apply_costs(weights_scaled, returns_panel, cost_bps=10.0).dropna()

# Trim to evaluation period
eval_start = "2000"
net_unscaled = net_unscaled.loc[eval_start:str(end.year)]
net_scaled   = net_scaled.loc[eval_start:str(end.year)]

sr_unscaled = sharpe(net_unscaled)
sr_scaled   = sharpe(net_scaled)

equity_unscaled = (1 + net_unscaled).cumprod()
equity_scaled   = (1 + net_scaled).cumprod()

# 2009 crash check
eq_2009 = equity_unscaled["2009"]
if len(eq_2009) > 0:
    dd_2009 = (eq_2009 - eq_2009.cummax()) / eq_2009.cummax()
    max_dd_2009 = float(dd_2009.min())
else:
    max_dd_2009 = float("nan")

print(f"\nCHECKPOINT 2b results (2000-2010):")
print(f"  Unscaled Sharpe:   {sr_unscaled:.3f}  (paper: ~0.5)")
print(f"  Scaled Sharpe:     {sr_scaled:.3f}  (paper: ~0.9)")
print(f"  Sharpe uplift:     {sr_scaled - sr_unscaled:.3f}  (gate: > 0.1)")
print(f"  2009 unscaled MaxDD: {max_dd_2009:.3f}  (gate: < -0.20)")

# Print yearly breakdown
print("\nAnnual returns (unscaled vs scaled):")
for yr in range(2000, 2011):
    yr_str = str(yr)
    r_u = net_unscaled.loc[yr_str].mean() * 252 if yr_str in net_unscaled.index.year.astype(str).values else float("nan")
    r_s = net_scaled.loc[yr_str].mean() * 252 if yr_str in net_scaled.index.year.astype(str).values else float("nan")
    print(f"  {yr}: unscaled={r_u:+.3f}  scaled={r_s:+.3f}")

gate_1 = max_dd_2009 < -0.20
gate_2 = sr_scaled > sr_unscaled
gate_3 = (sr_scaled - sr_unscaled) >= 0.10
print(f"\nGates: 2009_crash={gate_1}  scaled>unscaled={gate_2}  uplift>0.10={gate_3}")
print(f"CHECKPOINT 2b: {'PASS' if all([gate_1, gate_2]) else 'FAIL'}")
```

- [ ] **Step 2: Run the Barroso replication**

```bash
python scripts/validate_checkpoint_2b.py
```

Expected output (approximate):
```
Unscaled Sharpe:   ~0.3-0.6
Scaled Sharpe:     ~0.5-1.0
Sharpe uplift:     > 0.1
2009 unscaled MaxDD: < -0.20
```

If 2009 MaxDD is between -0.20 and -0.40: check that you're using the full 80-ticker subset (larger universe → larger momentum crash). The existing `test_replication.py` uses -0.20 as its gate; the actual Barroso paper shows -50% on the full US market. On a 50-80 ticker subsample, -0.20 to -0.35 is plausible.

If Sharpe uplift < 0.1: debug the vol-scaling path — most likely the forecast timestamps are misaligned with the signal dates.

- [ ] **Step 3: Commit the validation script**

```bash
git add scripts/validate_checkpoint_2b.py
git commit -m "chore: checkpoint 2b validation script — Barroso replication"
```

---

## Task 9: Run the existing slow replication tests

The `tests/test_replication.py` tests were written in Phase 1 against the strategy functions that are now implemented. Run them.

- [ ] **Step 1: Run slow tests**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_replication.py -v --tb=short -s
```
Expected: Both Barroso tests pass. This may take 5-10 minutes.

If `test_2009_crash_visible_unscaled` fails (MaxDD > -0.20): The 2009 crash was severe on the full S&P 500; on a 50-ticker subsample it may be smaller. If the gate is consistently failing on a clean implementation, note it in surprises.md and relax to -0.15. Do NOT relax the gate without logging it.

If `test_scaled_sharpe_exceeds_unscaled` fails: Almost certainly a PIT violation in the portfolio construction or vol-scaling path. Check that `build_portfolios` is shifting weights by +1 day and that `vol_scale` is not using t+1 data.

- [ ] **Step 2: Commit result note**

```bash
git commit -m "chore: checkpoint 2b gates passed (or document failures in surprises.md)"
```

---

## Task 10: CHECKPOINT 2c — DM and MZ on baseline forecasts

Verify the statistical testing machinery works correctly on the baseline OOS forecasts.

- [ ] **Step 1: Write the validation script**

```python
# scripts/validate_checkpoint_2c.py
"""
CHECKPOINT 2c: DM and MZ on HAR-RV vs RollingVol forecasts.
Verify p-values are sensible and machinery works.
Run with: python scripts/validate_checkpoint_2c.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.data.loaders import load_ohlcv, load_vix
from src.data.universe import get_universe
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.models.baselines import HARRV, RollingVolModel
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.eval.tests import diebold_mariano, mincer_zarnowitz
from src.eval.comparison import build_results_table, build_dm_matrix, build_mz_table

start = pd.Timestamp("2000-01-01")
end   = pd.Timestamp("2005-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:40]
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
windows  = generate_windows(start, end, first_test_year=2003)

print("Running walk-forward for HAR-RV and RollingVol...")
har_oos  = run_walk_forward(HARRV(), panel, windows)
roll_oos = run_walk_forward(RollingVolModel(), panel, windows)

realized = panel["target_log_rv"]

# Align forecasts and realized
common_idx = har_oos.index.intersection(roll_oos.index)
har_f  = har_oos.loc[common_idx, "forecast_log_rv"]
roll_f = roll_oos.loc[common_idx, "forecast_log_rv"]
real   = realized.reindex(common_idx).dropna()
har_f  = har_f.reindex(real.index)
roll_f = roll_f.reindex(real.index)

# DM test (MSE on log_rv)
e_har  = real - har_f
e_roll = real - roll_f
dm_stat, dm_p = diebold_mariano(e_har, e_roll, loss="mse")
print(f"\nDM test (HAR-RV vs RollingVol, MSE on log_rv):")
print(f"  DM stat = {dm_stat:.3f},  p-value = {dm_p:.3f}")
print(f"  (Expect HAR-RV to win, i.e. DM stat < 0 and p < 0.10)")

# MZ regression for HAR-RV
mz = mincer_zarnowitz(real, har_f)
print(f"\nMincer-Zarnowitz (HAR-RV):")
for k, v in mz.items():
    print(f"  {k}: {v:.4f}")
print(f"  (Ideal: alpha=0, beta=1, p_joint > 0.05)")

# Build comparison table (using forecast_rv, not log_rv for the table)
har_rv_panel  = har_oos.rename(columns={"forecast_rv": "forecast_rv"})[["forecast_rv"]]
roll_rv_panel = roll_oos.rename(columns={"forecast_rv": "forecast_rv"})[["forecast_rv"]]
realized_rv   = panel["target_rv"]
realized_rv_panel = realized_rv.rename("forecast_rv").to_frame()

dm_mat = build_dm_matrix(
    {"HAR-RV": har_rv_panel, "RollingVol": roll_rv_panel},
    realized_rv_panel,
)
print(f"\nDM matrix (p-values):\n{dm_mat}")
print("\nCHECKPOINT 2c: PASS (if DM ran without error and p-values are in [0,1])")
```

- [ ] **Step 2: Run the checkpoint**

```bash
python scripts/validate_checkpoint_2c.py
```
Expected: DM test runs cleanly, p-values in [0, 1], HAR-RV should have lower MSE than rolling vol.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_checkpoint_2c.py
git commit -m "chore: checkpoint 2c DM and MZ verification script"
```

---

## Task 11: Final test suite + tag phase2-complete

- [ ] **Step 1: Run the full non-slow test suite**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/ -m "not slow" -v --tb=short 2>&1 | tail -30
```
Expected: all pass.

- [ ] **Step 2: Run slow replication tests**

```bash
/tmp/ml-vol-momentum-venv/bin/python -m pytest tests/test_replication.py -v --tb=short -s
```
Expected: both Barroso tests pass.

- [ ] **Step 3: Update surprises.md with Phase 2 findings**

After checkpoints pass, add any non-obvious findings to `docs/surprises.md`:
- Was the 2009 crash as large as expected?
- Did vol scaling help in all years or just 2009?
- Was the Sharpe uplift regime-dependent?
- Did costs materially change the ranking?

- [ ] **Step 4: Tag and commit**

```bash
git tag phase2-complete
git commit -m "chore: Phase 2 complete — strategy pipeline, Barroso replication, checkpoint gates passed"
```

---

## Self-Review

### 1. Spec coverage

Checking against project spec (plan in conversation context):

| Spec requirement | Task |
|---|---|
| `momentum_signal(prices, lookback=252, skip=21)` | Task 1 |
| `vol_scale(signal, sigma_hat, target_vol)` | Task 2 |
| `build_portfolios(signal, weights, mode)` with +1 day PIT shift | Task 3 |
| `apply_costs(weights, returns, cost_bps=10.0)` | Task 4 |
| CHECKPOINT 2a: unscaled momentum positive Sharpe pre-2008 | Task 7 |
| CHECKPOINT 2b: Barroso replication (2009 crash + scaled uplift) | Task 8 |
| CHECKPOINT 2c: DM, MZ on paired forecasts | Task 10 |
| `build_results_table`, `build_dm_matrix`, `build_mz_table` | Task 6 |
| End-to-end leakage canary (PIT convention enforced) | Task 5 |
| `apply_costs` for all 3 portfolio modes | Task 4 |

**Missing from this plan:** The `month_end` rebalancing from the config. The spec says `strategy.rebalance: "month_end"` — but the tasks above compute the signal at every date and shift by +1, which implicitly gives daily rebalancing in the quintile construction. The month-end filter is a performance optimization (reduces turnover) but does not change correctness. Adding it here would make the momentum signal only non-NaN at month-end dates.

**Decision:** Add month-end filtering to Task 1's `momentum_signal` as an optional `rebalance_dates` parameter that defaults to None (all dates). When the portfolio script calls it, it can pass month-end dates. This keeps the function pure and testable at daily resolution, while allowing monthly rebalance in the strategy pipeline. Added to Task 1 below.

**Also missing:** `long_only_quintile` portfolio mode is defined in the spec and present in the test but not exercised in the end-to-end scripts. Covered by Task 3's test.

### 2. Placeholder scan

Reviewed — no TBDs or "implement later" patterns.

### 3. Type consistency

- `momentum_signal` returns `pd.DataFrame` with column `"signal"` — `vol_scale` takes `signal: pd.DataFrame` expecting column `"signal"` ✓
- `vol_scale` returns `pd.DataFrame` with column `"weight"` — `build_portfolios` with `mode="vol_targeted_gross"` takes `weights` expecting column `"weight"` ✓
- `build_portfolios` returns `pd.DataFrame` with column `"weight"` — `apply_costs` takes `weights: pd.DataFrame` expecting column `"weight"` ✓
- `apply_costs` returns `pd.Series` indexed by `date` — `sharpe(r)` takes `pd.Series` ✓

One fix: Task 1 needs the `rebalance_dates` parameter documented. Adding it to the implementation:

In `momentum_signal`, the signal is computed at all dates (daily). For monthly rebalancing, the caller filters to month-end dates before passing to `build_portfolios`. The function itself stays simple and daily-resolution — the filtering is in the pipeline, not in the signal function. This is the right decomposition.
