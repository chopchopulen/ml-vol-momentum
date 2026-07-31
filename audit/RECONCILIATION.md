# audit/RECONCILIATION.md

Independent evaluation of six claims from an outside review, against evidence produced in
Phase 2 before those claims were considered. Nothing here is accepted on authority. One
claim is refuted outright; one is refuted in its mechanism while pointing at a real defect
by accident; four are confirmed, two of them more strongly than stated.

All measurements are reproducible from `bench/BASELINE.md`.

---

## Claim 1 — "IC 0.769 may be reproducing volatility PERSISTENCE rather than adding information"

### **CONFIRMED — and understated.**

The claim predicted naive forecasts reach IC 0.6–0.8 on their own. They do. It did not
predict that a naive forecast would **beat** the TCN. One does.

All four requested baselines plus two more, on the **same test set** as the TCN's 0.769
(124,216 rows / 2,236 dates / 2003-02-12 → 2012-12-31):

| forecaster | IC on the TCN's test set |
|---|---|
| **rolling_vol — 126-day trailing Σr², zero parameters** | **0.7755** |
| transformer_fast | 0.7715 |
| **TCN ← the headline** | **0.7691** |
| **(b) EWMA, λ = 0.94** | **0.7393** |
| gbm | 0.7273 |
| **(d) HAR-RV (Corsi 2009)** | **0.7095** |
| **(a) trailing 21-day realized variance** | **0.7002** |
| **(c) GARCH(1,1)-t** | **0.6700** |
| per-ticker constant, no time-varying information | **≈ 0.57–0.61** |

**This is the single most important number in the audit, as the claim said it would be, and
the answer is that the TCN is third.** A 126-day rolling sum of squared returns — no fitting,
no parameters, three lines of pandas — scores **0.7755 against the TCN's 0.7691**.

Formally, paired daily-IC differentials vs `rolling_vol`, Newey-West(20), same 2,236 dates:

| model | ΔIC vs rolling_vol | t | p |
|---|---|---|---|
| tcn_partial | −0.0064 | −1.76 | 0.079 |
| transformer_fast | −0.0040 | −1.06 | 0.291 |
| lstm | −0.0182 | −4.15 | <0.001 |
| gbm | −0.0482 | −12.79 | <0.001 |
| har_rv | −0.0660 | −10.77 | <0.001 |
| garch | −0.1055 | −13.21 | <0.001 |

**No model in the project beats the zero-parameter baseline.** Four are significantly worse.

Two corrections to the claim's framing:

- **HAR-RV is not "hard to beat" here — it is beaten by nearly everything, including EWMA
  and a plain rolling window.** That is itself a finding: HAR is fit *per ticker*
  (`baselines.py:67-70`) while GBM and the neural models are fit on the pooled panel, so it
  is handicapped on estimation sample size. Its Jensen correction also uses an in-sample,
  *per-ticker* σ² (`baselines.py:64`), which reorders the cross-section, where every ML model
  uses a rank-preserving global scalar. HAR-RV as implemented is not the literature's HAR-RV.
- **The GARCH number is not GARCH.** `baselines.py:179-194` calls `forecast()` once from the
  last in-sample date and broadcasts a **scalar** across the whole test year. Measured unique
  forecast values per (ticker, year): garch **1.0**, every other model **224**. GARCH's 0.670
  is the IC of a per-ticker constant refreshed annually. The comparison the claim asked for
  cannot be made until this is fixed.

The floor matters as much as the ceiling. A per-ticker constant, estimated from data up to
the prior December 31 and refreshed once per fold — zero time-varying information — scores
**≈0.57**. So the entire modelling effort buys about **+0.16 IC over knowing which stocks are
chronically volatile**, and the TCN buys **−0.006 against a rolling window**.

---

## Claim 2 — "IC may be computed on vol LEVELS rather than changes"

### **CONFIRMED.**

`src/eval/tests.py:112-141` correlates `forecast_rv` against `target_rv` — variance
**levels**, cross-sectionally, Spearman, per date, aggregated as an unweighted mean over
dates (`compare_all_models.py:115`, `run_extension2_fast.py:180`). **The headline uses
levels.**

Both requested numbers, same rows:

| model | IC on **levels** (reported) | IC on **log-vol changes** |
|---|---|---|
| rolling_vol | **0.7755** | **0.5368** |
| transformer_fast | 0.7715 | 0.5353 |
| tcn_partial | **0.7691** | **0.5300** |
| lstm | 0.7574 | 0.5221 |
| garch | 0.6700 | 0.4618 |
| gbm | 0.7273 | 0.4448 |
| har_rv | 0.7095 | 0.4035 |

(Changes measured as predicted vs realized log change relative to the trailing 21-day
realized variance already known at forecast time.)

Controlling for that same trailing RV directly — partial rank IC — gives rolling_vol 0.4748,
transformer 0.4822, tcn 0.4680, lstm 0.4479, har_rv 0.2843.

So the inflation the claim describes is real and large: **0.769 → 0.530 on changes**. But
note what does *not* change: **the ordering is identical on both metrics, and the TCN is
third on both.** The levels/changes issue inflates every model's number; it is not what
manufactures the TCN's apparent win. That was the sample (Claim 4 / S0-1).

A second, independent confirmation of the same point: removing the ticker fixed effect from
both sides drops LSTM 0.739 → 0.496 and GARCH 0.642 → 0.334. Roughly 60–77% of the headline
is static cross-sectional ordering.

---

## Claim 3 — "Targets may overlap"

### **CONFIRMED.**

`h = 21` (`configs/default.yaml:31`), and `targets.py:9` computes the target at **every**
step, so consecutive targets share **20/21** of their window exactly as the claim describes.

The claim asked for h, the effective N, and significance computed against it. There are two
effective-N figures because two different statistics are computed in this repo, and they
have different units:

**For statistics built from the daily IC series** (mean IC, IC_A vs IC_B — the cross-section
is already collapsed inside each daily IC):

```
daily IC observations (TCN sample)     2,236
IC autocorrelation 1 / 5 / 21 / 42 / 63   0.957 / 0.815 / 0.474 / 0.418 / 0.421
naive SE = std/sqrt(N) = 0.00156       ->  t = 493
Newey-West(20) SE      = 0.00629       ->  t = 122      variance inflation 16.3
implied effective N                       137 dates
Newey-West(60)                         ->  t = 82       -> N_eff = 62
non-overlapping 21-day blocks             106
```

**For row-level pooled statistics** — the Diebold-Mariano tests, which pool ~275,000 rows:
233 time blocks × 2.61 effective names (measured mean pairwise cross-sectional residual
correlation **0.371**, mean cross-section 55.6 names) ≈ **623 independent rows** full-sample,
≈ **277** on the TCN sample.

**So: raw N = 124,216 rows. Effective N ≈ 106–137 dates, or ≈277 independent panel
observations.** Pooled t-statistics in this repo are overstated by roughly **√441 ≈ 21×**.

Two concrete consequences the claim did not anticipate:

- `README.md:245` reports "N = 275,489 OOS predictions" for the prob-LSTM calibration. The
  honest figure is ~600. ECE = 0.012 sits **below** the sampling floor of ~1/√623 ≈ 0.04, so
  "essentially no miscalibration" (`README.md:243`) is unsupported — the estimate cannot
  resolve miscalibration at that scale.
- `tests.py:48` sets `nw_lags = h − 1 = 20`, which is right for a time series but is applied
  to a **row-major pooled panel** with ~56 rows per date. Lag 20 does not span even one
  cross-section and captures none of the 21-day overlap.

To be clear about what this does and does not overturn: the null "IC > 0" survives easily
(t = 122 even at NW(20)). What does not survive is any claim about **differences between
models**, which is what the project actually asserts.

---

## Claim 4 — "Six architectures is a multiple-comparisons exposure"

### **CONFIRMED, and the reported result is worse than the null.**

**Was the test set touched during selection?** Yes, and it is visible in git.
`4714e26` (probabilistic LSTM) and `78a0a51` (uncertainty weighting) are dated 2026-05-22,
*after* Phase 3 completion (2026-05-19) and after `5de2465` "post-mortem on all 8
pre-registered predictions" (2026-05-22). `README.md:273` states the motivating hypothesis
explicitly as an observed test-set result: *"The IC ≠ Sharpe disconnect arises because…"*.
Extension 1 and Extension 2 exist *because of* what the test set showed.

`run_extension2_fast.py:43,47` then overrides `d_model` 32→16, `max_epochs` 50→10, `seeds`
[0,1,2,3,4]→[0,1] for the challenger architectures only — so LSTM and its challengers were
not even run under one protocol.

**Is the TCN result a best-of-k selection?** Yes. `run_extension2_fast.py:215-216` literally
computes `max(results, key=...)` over the architectures — and does so across mismatched row
sets (S0-1). Counting the whole project: ~10 architectures/variants, ~40 reported Sharpe
cells, ~15 IC cells, all on one test sample.

**The null simulation the claim asked for.** On the Sharpe side, where I can compute the
standard error directly, the answer does not need the seed distribution:

```
SE(annualised Sharpe) on TCN's 2,236 days = sqrt(252/2236) = 0.336
TCN reported Sharpe = +0.238   ->   t = 0.71

E[max of  4] under a pure-noise null = +0.346
E[max of  6]                         = +0.426
E[max of 10]                         = +0.516
```

**The project's best Sharpe is smaller than the expected maximum of four pure coin flips.**
It is not merely unadjusted for selection — it does not clear the null even before
adjustment.

On the IC side, Phase 0 measured σ_seed = **0.0075** across 12 seeds (`bench/BASELINE.md`
§9b). The claim's requested simulation:

```
E[max of  4] under N(0, 0.0075) = +0.0077
E[max of  6]                    = +0.0095
E[max of 10]                    = +0.0115
```

So seed-selection alone manufactures ~0.01 of the 0.046 architecture spread — real, but not
the main driver. The main drivers are the sample mismatch (S0-1) and shared persistence
(Claim 1).

**The more damaging number is the seed range: 0.0321 across 12 seeds** (min 0.7100 at seed 5,
max 0.7421 at seed 43). The TCN's margin over the zero-parameter rolling window is
**−0.0064** — smaller than one seed σ, and one-fifth of the seed range. Swapping seed 5 for
seed 43 moves a single architecture's IC by two-thirds of the entire best-to-worst spread
that `extension2_summary.md` interprets at length.

To be fair to the project: the claim's own threshold was that σ ≥ 0.05 would mean "one draw,
not a result", and σ = 0.0075 passes that comfortably. The headline is not a lucky seed. It
is simply measuring something that a rolling window measures marginally better, on a sample
nothing else was measured on.

---

## Claim 5 — "SHAP at 38% on the Parkinson estimator is a SYMPTOM… check the window arithmetic directly"

### **REFUTED on the stated mechanism.** The window arithmetic is clean. The claim was right to demand the check, and right that a leak would produce exactly this — but there is no leak.

This is the finding I attacked hardest, both directly and through the adversarial pass,
because the claim is correct that SHAP would faithfully report a leaking feature as
important.

`src/data/features.py:10-15`:
```python
pk = log_hl_sq.rolling(window).mean() / (4 * math.log(2))
return pk.shift(1)
```
With `window = 5`, `pk` at index `t` consumes high/low bars **t−5 … t−1**.

`src/data/targets.py:9`:
```python
rv = (r ** 2).rolling(horizon).sum().shift(-horizon)
```
consumes returns **t+1 … t+21**.

Verified numerically rather than by reading: `target_rv[t]` = 0.017158451179953606 versus
Σr² over t+1…t+21 = 0.017158451179953564 (agreement to 1e-16), while the t…t+20 window does
**not** match. On a synthetic monotone series, `pk[t=10]` equals exactly the mean over bars
5…9.

**Newest feature bar = t−1. Oldest target bar = t. There is a full one-bar dead zone — bar
`t` is consumed by neither.** No overlap, not even partial. Every other feature verified the
same way; the sequence models are more conservative still (their newest underlying bar is
`t−2`).

**The economic explanation is sufficient and requires no leak.** Measured standalone
cross-sectional IC against the target: `pk` **0.6454** on a **5-day** window versus `rv_m`
**0.6552** on a **21-day** window. A Parkinson range estimator is roughly 5× more efficient
per observation than squared close-to-close returns, so 5 days of range legitimately carries
about as much information as 21 days of squared returns. Cross-sectional rank correlation
between `pk` and `rv_m` is 0.777 — they are measuring the same thing, and `pk` measures it
with less noise. A gradient-boosted model will prefer the cleaner estimator. That is a real
result, and it should be reported with confidence.

**However — the claim's instinct that the 38% is "a symptom to investigate, not a result"
lands anyway, for an entirely different reason it did not name.** `scripts/run_phase4.py:84-85`:

```python
gbm_shap.fit(train_last)
shap_imp = compute_shap_importance(gbm_shap, train_last, sample_size=5000)
```

**The attribution is computed on the same rows the model was fit on.** It describes what the
model fit, not what generalises — while `README.md:88` calls it "a genuine discovery" and
`docs/predictions.md:59` grades a pre-registered prediction against it. It is also a
5,000-row sample of ~1.4M training rows, and "38%" is a share of total mean|SHAP|, not
variance explained.

And two things nobody had said: the attribution belongs to the **GBM** — the fourth-ranked
model, not the 0.769 TCN, which has no SHAP analysis in the repo at all — and `vix` receives
0.082 SHAP weight despite having **zero cross-sectional variation** (identical for every
ticker on a given date, so its standalone cross-sectional IC is undefined; it cannot rank
stocks at all).

---

## Claim 6 — "The IC-vs-Sharpe disconnect is itself evidence"

### **PARTIALLY CORRECT.** The answer is **(b) and (c)**, not (a) — and there is a fourth cause the three options do not cover.

First, a correction to the premise: **Sharpe 0.576 does not exist in this repository.** The
TCN's recorded Sharpe is **+0.238** (`results/extension2_summary.md:15`). No file in
`README.md`, `docs/`, or `results/` contains 0.576. The disconnect is real but the number is
not one of ours.

Taking the three options in turn:

**(a) "IC measuring persistence that is already priced" — the first half is right, the
causal claim is wrong.** IC *is* mostly persistence (Claim 1, Claim 2). But that is not why
the Sharpe is low, and "already priced" is not the mechanism. The decisive test:

| forecast fed to `vol_scale` (identical rows, 4,897 days, 2003–2024) | Sharpe |
|---|---|
| **ORACLE — the realized forward RV itself, perfect foresight** | **+0.1142** |
| constant RV (zero information) | +0.0288 |
| har_rv | +0.0200 |
| gbm | +0.0019 |
| rolling_vol | −0.0146 |
| lstm | −0.0406 |

Perfect foresight of the exact quantity being forecast earns **+0.085 over a
zero-information constant** and **+0.155 over the LSTM**. If the information were "already
priced", the oracle would earn nothing. It earns the most of anything tested. There is real
headroom; the models capture approximately none of it, and the LSTM is worse than a constant.

*(I initially concluded the opposite — that the oracle bought nothing — because I first
measured it on the TCN's 2003–2012 subsample, where all ten strategies cluster at 0.22–0.32
and nothing discriminates. That was wrong and is corrected here.)*

**(b) "The trading rule failing to monetize a real forecast" — CORRECT, and it is the main
cause.** `src/strategy/scaling.py:37` is `w = (target_vol / ann_vol) * z`, where `z` is the
cross-sectional z-score of a **12-1 momentum** signal. **The volatility forecast never picks
a position — it only rescales a bet whose sign and rank come entirely from momentum.** A
Sharpe reported here is momentum's Sharpe modulated by sizing.

On top of that, costs eat the alpha: measured annualised turnover is **65×** for the LSTM
against a **21-day** forecast horizon, costing 3.27% against 2.43% of gross return.
`configs/default.yaml` specifies `rebalance: "month_end"`; the code re-sizes the entire book
daily and never implements it.

**(c) "An error in one of the two metrics" — CORRECT for the Sharpe.** The reported number is
not a return on capital. Nothing normalises gross exposure (`portfolio.py:64-67`); measured
mean gross is 1.15 for rolling_vol, 2.79 for LSTM, **10.0 for GARCH (max 1810)**, and it
floats 0.68→4.66 *within* the LSTM alone. `results/master_results_table.parquet` shows the
GARCH row at ann_ret 8695%, ann_vol 38542%, **max_dd −170%** — arithmetically impossible for
a return on capital. Max drawdown is computed two incompatible ways in two scripts that write
the same summary file, producing the `inf%` in `results/extension1_summary.md`. And
`scaling.py:35` annualises a 21-day variance with a factor of 252 instead of 252/21, so the
stated "10% annualised vol target" is off by √21 (Sharpe-invariant, but the design property
is false).

**(d) The cause the three options miss: the two numbers are not computed on the same data.**
The IC of 0.769 and the Sharpe of 0.238 both come from the 2003–2012 subsample, but they are
tabled against 2003–2024 numbers for every other model. On matched rows the disconnect
largely dissolves — every strategy is positive on 2003–2012, and the TCN is eighth of ten.
Before asking why IC and Sharpe disagree, the two need to be measured on the same sample.

---

# What Phase 2 found that this list did not ask about

The reconciliation list is sharp on *interpretation* — it correctly anticipated persistence,
levels, overlap, and multiple comparisons. It did not anticipate that the numbers themselves
are not what they appear to be. Ranked by consequence:

1. **The headline is scored on a different sample from its comparators** (S0-1). 124,216 rows
   / 2003-2012 versus 275,489 rows / 2003-2024. This single defect manufactures the TCN's
   apparent win, and no amount of correct statistical interpretation of 0.769 would have
   caught it. It is also the reason the IC-vs-Sharpe disconnect looks as sharp as it does.
2. **Every Diebold-Mariano p-value in the project is computed from a loss that is identically
   zero** (S0-2). Log-variances are fed to a QLIKE function that clips at 1e-12; both
   arguments collapse to the same constant. Between 0.00% and 0.54% of rows carry any signal.
   Correcting it **inverts four published conclusions**, including the README's most-quoted
   sentence. The claim list treats the DM results as sound throughout.
3. **The control group is scored on a different window, and fixing it flips a pre-registered
   verdict in the flattering direction** (S0-3). Unscaled momentum: 6,015 days vs 4,897.
   Aligned, its Sharpe is −0.134 not −0.002, and Prediction 5 goes from ❌ Falsified to ✅
   Confirmed.
4. **GARCH is a constant** (S1-1). One forecast value per ticker per year, broadcast. The
   claim list asked me to compute GARCH's IC as a benchmark; that benchmark does not exist in
   this repo yet.
5. **SHAP is computed in-sample** (S1-5) — the right suspicion about the 38%, wrong mechanism.
6. **The validation split is a ticker holdout, not a chronological one** (S2-1). All four
   sequence architectures select their stopping epoch against contemporaneous data, under a
   comment claiming the opposite. Not a leak — but the four architectures being compared
   differ mainly in overfitting propensity, and the mechanism meant to control overfitting is
   inoperative for all four of them.
7. **`rolling_vol`'s forecast level is 5.94× the target** (S1-2c) — a 126-day sum reported
   against a 21-day target. Invisible to Spearman, fatal to QLIKE, and it makes the simplest
   baseline look like the lowest-risk strategy purely through 2.4× less leverage.
8. **The pinned environment does not install** on any Python on this machine
   (`bench/BASELINE.md` §0), and **136 tests pass** while all of the above is true — no test
   asserts split chronology, none exercises the QLIKE path, none checks that rows in a table
   match.
