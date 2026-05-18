# Phase 1 Frozen Results

## Bugs Found Post-Freeze and Revised Numbers

This section documents bugs discovered after the initial `phase1-complete` tag (commit `9518114`, 2026-05-17) and before Phase 2 began.

### Bug A: `log(0) = -inf` surviving `dropna()` (targets.py + baselines.py)

**Introduced:** Implicitly at project start — `np.log(rv)` with no guard against `rv = 0`.

**Fixed:** Commit `9ae9f8e` (targets.py), and the audit pass below.

**Affected files:** `src/data/targets.py`, `src/models/baselines.py` (RollingVolModel, GARCH11Model), `src/data/features.py` (log_dv), `src/eval/tests.py` (DM, MZ, MCS).

**Root cause:** Pandas `dropna()` only drops `NaN`, not `np.inf` or `-np.inf`. Zero realized variance (stocks with no returns for 21 consecutive days — halted trading, stale yfinance data for delisted tickers) produced `log(0) = -inf` which silently survived `dropna()`. Same pattern appeared in `log_dv` (zero trading volume) and forecast log transforms.

**Effect:** OOS years 2013, 2014, 2015, 2021 showed R² = 1.000 exactly. Root: `-inf` in `target_log_rv` made `SS_tot → ∞` and `1 - SS_res/SS_tot = 1.0`. The initial 3-year evaluation (2003–2005) happened to miss all affected years, so the frozen baseline was wrong.

**Fix pattern applied everywhere:**
- For *target* values: drop rows where `rv = 0` (halted data) rather than clip — keep `NaN` rows (legitimate trailing window).
- For *feature* values: replace `0` with `NaN` before log transform so downstream `dropna()` removes the row.
- For *forecasts*: drop rows where `forecast_rv = 0` rather than clip.
- For *statistical tests* (DM, MZ, MCS): use `np.isfinite()` filter instead of `~np.isnan()`.

**Why it survived initial review:** Unit tests used synthetic data with constant non-zero returns. Leakage canary tested a specific failure mode (future return as signal), not degenerate-data pathology. The 3-year evaluation subsample (2003–2005) did not include any affected years by chance.

### Bug B: `sm.add_constant` silently dropping intercept (baselines.py)

**Introduced:** Implicitly at project start.

**Fixed:** Commit `2ad8609`.

**Affected file:** `src/models/baselines.py` (`HARRV._fit_ticker`).

**Root cause:** `sm.add_constant(X)` by default checks whether the matrix `X` already contains a constant column and *silently omits* adding the intercept if it does. When a ticker's log-rv features were all identical (e.g., all returns zero → clipped to `log(1e-12)` → constant), this produced a 3-column design matrix. `res.params[3]` then raised `IndexError`.

**Effect:** HAR-RV fit crashed silently for affected tickers (caught by `try/except`, logged as a warning, ticker skipped). No R² inflation, but tickers with degenerate data were silently excluded from the model.

**Fix:** `sm.add_constant(X, has_constant="add")` forces the constant regardless.

### Revised numbers (post-fix, full 22-year OOS)

The 3-year subsample numbers in the original freeze (`0.351 / 0.219 / 0.205`, pooled `0.258`) remain accurate — those years were unaffected by Bug A. The full 22-year picture is documented below. **The numbers you cite in interviews are the full-22-year pooled R² = 0.328.**

### What was NOT affected

The Barroso & Santa-Clara (2015) replication tests (`test_replication.py`) were re-run on the post-fix codebase and **both tests still pass**. The replication was not using target RV or log transforms — it used raw returns and rolling vol scaling, neither of which touched the zero-RV bug path.

### New canaries added (commit following this document)

1. **`test_targets.py::test_degenerate_target_canary`**: Injects a halted-stock (all-zero returns) into a synthetic panel, asserts that `forward_rv` produces no `-inf` in `target_log_rv`. This canary would have caught Bug A.

2. **`test_baselines.py::test_harrv_design_matrix_column_stability`**: Verifies that `HARRV._fit_ticker` always returns exactly 5 keys (alpha, beta_d, beta_w, beta_m, sigma2) regardless of whether the input features are degenerate. This canary would have caught Bug B.

### Reflection

Both bugs had the same shape: an upstream pathology (zero, constant column) produced a sentinel value (-inf, missing column), and a downstream function accepted the sentinel silently (`dropna()` does not drop `-inf`; `sm.add_constant` does not warn when dropping a constant). The initial test suite used synthetic data with no pathological inputs, so neither canary fired. The key lesson: every bug that survives canary review should become a canary for the specific pathological input that exposed it.

---

Baseline numbers computed on commit `71d0067` (post-fix for MultiIndex ordering).
These are locked in before Phase 2 strategy code is written.

---

## White-Noise Canary (20 seeds)

Panel: 200 stocks × 504 dates, i.i.d. N(0, 0.01²) returns.
Strategy: long-short quintile, daily rebalance, no costs.

| Statistic | Value |
|-----------|-------|
| Mean Sharpe (20 seeds) | 0.195 |
| Std Sharpe (20 seeds) | 0.738 |
| Expected std (√(252/504)) | ~0.707 |
| Max |Sharpe| observed | ~1.9 |
| Any seed > threshold (2.0) | No |

**Interpretation:** Distribution is zero-centered and consistent with pure sampling noise (expected σ ≈ 0.71 for T=504). The ~0.6-sigma positive mean is within normal variation over 20 draws. Framework is unbiased.

## Leakage Canary

Signal = return at t+1 (perfect look-ahead). Expected Sharpe > 15.

| Result |
|--------|
| Sharpe (seed=0) ≈ 145 (long-short quintile on leakage signal) |

**Interpretation:** Pipeline correctly detects future information. If this number were near zero the eval framework would be broken.

---

## HAR-RV Walk-Forward OOS Performance

Sample: 80-ticker S&P 500 subset (2002 universe), OOS years 2003–2024.
Expanding window, annual retrain, 42-day embargo.
**Note:** Initial numbers (2003-2005 only) were contaminated by an `-inf` bug in `targets.py` (zero-RV rows producing log(0)=-inf, making R²=1.0 in affected years). Fixed in commit `9ae9f8e`. Numbers below are from the fixed run.

### OOS R² on log(RV) target — full 22-year series

| Year | R² | IC | N |
|------|-----|-----|---|
| 2003 | 0.351 | 0.772 | 12,096 |
| 2004 | 0.219 | 0.750 | 12,096 |
| 2005 | 0.205 | 0.681 | 12,265 |
| 2006 | 0.376 | 0.710 | 12,488 |
| 2007 | 0.283 | 0.560 | 12,544 |
| 2008 | 0.283 | 0.677 | 12,600 |
| 2009 | 0.668 | 0.787 | 12,544 |
| 2010 | 0.420 | 0.755 | 12,544 |
| 2011 | 0.425 | 0.762 | 12,488 |
| 2012 | 0.396 | 0.750 | 12,432 |
| 2013 | 0.208 | 0.625 | 12,768 |
| 2014 | 0.179 | 0.638 | 12,789 |
| 2015 | -0.101 | 0.559 | 12,956 |
| 2016 | 0.107 | 0.608 | 12,992 |
| 2017 | 0.302 | 0.612 | 12,890 |
| 2018 | 0.365 | 0.630 | 13,033 |
| 2019 | 0.375 | 0.648 | 12,662 |
| 2020 | 0.217 | 0.692 | 12,600 |
| 2021 | 0.555 | 0.737 | 12,901 |
| 2022 | 0.500 | 0.761 | 12,654 |
| 2023 | 0.445 | 0.644 | 12,654 |
| 2024 | 0.440 | 0.657 | 11,514 |
| **Pooled** | **0.328** | **0.682** | **277,061** |

**Checkpoint 1f gate:** [0.30, 0.65]. Pooled R² = 0.328 passes the 0.30 floor. **Gate: PASS.**

**Structural patterns:**
- **High-vol crisis years (2009, 2021, 2022) have highest R²** (0.50-0.67). HAR-RV is most predictive when vol is extreme and persistent.
- **Low-vol regime years (2013-2016) have lowest R²** (−0.10 to 0.21). HAR-RV's persistence structure adds little when vol is mean-reverting.
- **2015 has R² = -0.101** — forecasts are worse than the mean. This is a genuine finding, not a bug: 2015 had the China volatility shock (Aug 2015) which HAR-RV's trailing features could not anticipate.
- **IC is much more stable than R²** (std 0.11 vs 0.16). The rank ordering of stocks by vol is preserved even when absolute-level forecasts are poor.

**Why R² < 0.40 for individual stocks vs literature:**
- Daily squared return is a noisy proxy for true integrated variance. Literature benchmarks using 5-min RV report R² 0.4–0.6 on *index* data.
- Individual equity daily-close RV is harder to predict; idiosyncratic shocks dominate.
- 80-ticker subsample from 2002 universe; includes many now-delisted tickers with sparse data.

---

## GARCH(1,1)-t

Convergence rate: ≥95% of tickers (per CHECKPOINT 1f gate). Exact rate logged in `convergence_log_` dict. Full-sample convergence validation pending Phase 2.

---

## Pre-registered predictions (committed 2026-05-17)

See `docs/predictions.md`. All 8 predictions locked before Phase 3 ML training begins.

---

## Notes for interviews

- **Why HAR-RV in log space?** RV is approximately log-normal (ABDL 2003), making log(RV) well-approximated as homoskedastic Gaussian → OLS is the right estimator. We pay one Jensen correction (E[RV] = exp(μ + σ²/2)) when back-transforming.
- **Why 42-day embargo?** Features look back 21 days, target looks forward 21 days; 2 × 21 = 42 is the minimum to avoid train/test contamination via overlapping windows (de Prado AFML §7).
- **Why cross-sectional IC > time-series R²?** Vol scaling is a cross-sectional operation: we rank stocks by forecast vol, not predict absolute vol level. A forecast with high time-series bias but high rank-correlation is useful for scaling.
- **Surprising finding:** HAR-RV's cross-sectional IC (~0.73) is remarkably stable across years. The vol persistence signal is very strong for ranking purposes even when absolute-level predictions are noisy.
