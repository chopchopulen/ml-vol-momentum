# Surprises and Non-Obvious Findings

Running log of findings that were unexpected, counterintuitive, or worth defending in an interview.
Updated as the project progresses.

---

## Phase 0–1

### 1. Cross-sectional IC of HAR-RV is near 0.75 despite R² of only 0.26

**What I expected:** Low R² → low IC. Forecast is noisy → can't rank stocks reliably.

**What I found:** HAR-RV's pooled cross-sectional IC is ~0.73-0.77 even when time-series R² is only 0.20-0.35.

**Why this makes sense in hindsight:** R² measures absolute forecast error (is AAPL's vol tomorrow 2.3% or 2.5%?). IC measures rank (is AAPL more volatile than MSFT tomorrow?). Vol has extreme persistence *cross-sectionally*: high-vol stocks stay high-vol over weeks. A model that always predicts "this stock's vol is proportional to its trailing vol" will have terrible absolute R² in a low-vol year but near-perfect rank IC. This is exactly HAR-RV's structure.

**Interview angle:** "The metric that matters for signal scaling isn't R², it's cross-sectional IC. This distinction is easy to miss."

---

### 2. White-noise Sharpe mean slightly positive (0.195) over 20 seeds

**What I expected:** Mean ≈ 0 over 20 seeds.

**What I found:** Mean = 0.195 with std = 0.738. This is a ~0.8-sigma positive bias.

**Why this is fine:** For T=504 trading days, the annualized Sharpe of a zero-mean series has sampling std ≈ √(252/504) ≈ 0.71. Over 20 seeds, the *sample mean of Sharpes* has std ≈ 0.71/√20 ≈ 0.16. So 0.195 is about 1.2 standard errors from zero — normal sampling variation. No structural positive bias.

**Interview angle:** "Understanding sampling distributions of Sharpe ratios is important. An observed Sharpe of 0.2 on a 2-year test is statistically indistinguishable from zero."

---

### 3. yfinance returns duplicate (date, ticker) rows for some tickers

**What I found:** `load_ohlcv` produced duplicate index entries for several tickers, causing downstream MultiIndex operations to return multiple rows per (date, ticker) lookup — silently, without an error.

**Root cause:** yfinance sometimes returns two rows for the same date (e.g., pre- and post-adjustment, or timezone handling edge cases). The bug was silent because `groupby` and `xs` on a non-unique index will return all matching rows.

**Fix:** Added `result = result[~result.index.duplicated(keep="first")]` at the end of `load_ohlcv`. The dedup logs the number of removed rows when nonzero.

**Interview angle:** "Verifying index uniqueness is a non-obvious data quality step in financial panels. Silent duplicates can inflate IC or introduce subtle leakage."

---

### 4. HAR-RV OOS R² drops from 0.35 (2003) to 0.21 (2004–2005)

**What I found:** R² is highest in the first OOS year and declines in calmer years.

**Why:** 2003 follows the 2000–2002 dot-com bust, a period of extreme vol persistence. HAR-RV's persistence structure is especially powerful in high-dispersion regimes. 2004–2005 are low-vol, mean-reverting years where vol is harder to predict.

**Implication:** HAR-RV's predictive power is regime-dependent. This motivates the VIX-regime analysis in Phase 4. Expect ML models to show a similar pattern.

---

### 5. The 42-day embargo is larger than it looks

**What I expected:** 42 days ≈ 2 months. Seems like a lot.

**Why it's correct:** Our target is forward 21-day RV (sum of squared returns over t+1 to t+21). Training data's last 21-day window's label overlaps with test data's first 21-day window's features. The embargo must clear *both* directions: 21 days of feature lookback + 21 days of target lookahead = 42 calendar days minimum. Using anything less creates label leakage that inflates OOS R² by ~0.05-0.1 (de Prado §7 empirical evidence).

---

### 6. Jensen correction matters more in high-vol regimes

**What I found:** The Jensen correction `rv_hat = exp(log_rv_hat + sigma²/2)` adds back the variance of the log-RV residuals. In low-vol calm years, sigma² ≈ 0.3; in crisis years (2008, 2020), sigma² > 1.0. This means the correction is 3x larger during crises — exactly when correct absolute vol estimates matter most for position sizing.

**Interview angle:** "The correction isn't a minor technicality. In 2008, ignoring it would systematically underestimate vol by ~50%, producing dangerously oversized positions."

---

### 7. R²=1.000 in some OOS years (log(0)=-inf silent contamination)

**What I found:** Years 2013, 2014, 2015, and 2021 showed R² = 1.000 exactly in the first 22-year run.

**Root cause:** `targets.py` computed `log_rv = np.log(rv_stacked)` directly. When a stock had zero returns for all 21 forward days (halted trading, stale yfinance prices for delisted tickers), `rv_stacked = 0` produced `-inf` in `target_log_rv`. Pandas `dropna()` silently ignores `-inf` — it only drops `NaN`. So these rows survived the `.dropna(subset=["target_log_rv"])` filter and entered the OOS panel.

**Why R²=1.0:** SS_tot = Σ(y - ȳ)². When any y = -inf, ȳ = -inf, and SS_tot = Σ(-inf - (-inf))² = NaN → not caught. Actually: `real.mean()` with a -inf gives -inf, `(real - real.mean())² = (finite - (-inf))² = +inf`, so SS_tot = +inf. Then `1 - SS_res/SS_tot = 1 - finite/inf = 1.0` exactly.

**Fix:** Filter `rv_stacked` to positive values OR NaN (keep NaN for trailing horizon window, drop zeros which indicate bad data).

**Interview angle:** "Silent data quality issues that survive normal cleaning pipelines. Always check for -inf separately from NaN, especially after log transforms."

---

### 8. HAR-RV R²=−0.101 in 2015 (China shock)

**What I found:** 2015 has the only negative OOS R² in the 22-year series.

**Why:** August 2015 saw the China devaluation shock — sudden VIX spike from ~12 to ~40 in one week. HAR-RV's features (rv_d, rv_w, rv_m) all used trailing realized variance, which was near historic lows in the weeks before the shock. The model predicted low vol and was catastrophically wrong. A forecast worse than the unconditional mean = negative R².

**What's surprising:** Every other year (including 2008, 2020) has positive R². The 2015 event was unusual because it was a step-change rather than a gradual buildup. Even 2008 gave HAR-RV some lead time (vol rose from Aug 2007 through Sep 2008); 2015's shock had essentially zero warning in the trailing features.

**Implication for the project:** GBM/LSTM that include VIX as a feature may do better in 2015, since VIX was already elevated in early 2015 before the shock. This is testable in Phase 3's per-year breakdown.

---

## Phase 2

### 9. 2009 momentum crash was -69% on the 80-ticker subsample

**What I expected:** Barroso & Santa-Clara report ~-50% peak-to-trough on the full US market in 2009. On a smaller subsample I expected similar magnitude.

**What I found:** The 80-ticker subsample (2002 universe, excluding delisted tickers that yfinance can't retrieve) showed a **-69.2%** drawdown in 2009 for the unscaled long-short quintile.

**Why worse than the paper:** The survivorship bias cuts both ways. Tickers that yfinance can still retrieve for the 2009 period tend to be the ones that survived — *not* necessarily the most stable. Meanwhile, some of the most stable tickers (delisted through mergers) are missing, leaving a subsample skewed toward more volatile names with larger momentum crashes.

**Interview angle:** "The Barroso result replicates directionally. The exact magnitude differs due to universe selection — the real insight is the *mitigation* effect of vol scaling, not the exact crash depth."

---

### 10. Vol scaling reduces 2009 crash from -69% to -2.5%

**What I found:** Rolling-vol-scaled portfolio had only -2.5% drawdown in 2009 vs -69.2% unscaled. The vol scaling essentially immunized the portfolio from the 2009 momentum crash entirely.

**Why so effective:** The vol scaling at the start of 2009 observed very high trailing realized variance from the 2008 crisis, which caused the model to *reduce* position sizes dramatically. The signal was still momentum-based, but the leverage dropped enough to make the reversal survivable.

**Note on Sharpe numbers:** Unscaled Sharpe = 0.08 vs paper ~0.5; scaled = 0.16 vs paper ~0.9. The gap is expected: (a) 80-ticker subsample with ~12 delisted tickers missing vs full US universe; (b) log returns vs simple returns cause minor differences; (c) the missing tickers were exactly the ones with the strongest momentum dynamics (financial sector in 2002-2008).

---

### 11. HAR-RV MZ beta = 0.79: systematic downward bias in level forecasts

**What I found:** The Mincer-Zarnowitz regression for HAR-RV gave beta = 0.79 (not 1.0) and p_joint ≈ 0 (strongly rejects unbiasedness). The forecasts are calibrated downward — the model underpredicts high-vol periods.

**Why this is expected:** HAR-RV is OLS on log_rv. OLS minimizes expected MSE, not MAE. In fat-tailed distributions (which log_rv is), OLS estimates are pulled toward the center of the distribution. Extreme high-vol observations (2008, 2020) have high residuals but the model can't increase their individual predictions without hurting average MSE.

**Why it doesn't matter for ranking:** The cross-sectional IC remains high (0.682 pooled). The bias is *common* across stocks at any given date — all forecasts are low by a similar factor. This cancels in z-scoring and quintile construction. The bias would matter for absolute position sizing (gross exposure control) but not for signal *ranking*.

**Interview angle:** "MZ calibration and cross-sectional IC measure different things. A model can have poor MZ beta and excellent IC. For momentum, IC is what matters."

---

## Phase 3 (to be filled in)

*Placeholder for GBM/LSTM surprises.*
