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

## Phase 2 (to be filled in)

*Placeholder for momentum, scaling, and portfolio surprises.*

---

## Phase 3 (to be filled in)

*Placeholder for GBM/LSTM surprises.*
