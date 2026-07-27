# audit/FINDINGS.md

Blind audit of `ml-vol-momentum` at commit `3e748e8`. Four independent read-only
auditors (leakage, statistics, baselines, backtest) reported without hints; every finding
was then put through an adversarial pass whose default verdict was REFUTED. Only findings
that survived that pass, or that I measured myself, appear here. Refuted claims are
recorded at the bottom — including one of my own.

No model or feature code was modified. Every number below is reproducible from
`bench/BASELINE.md`.

Tags: `statistics` | `economics` | `leakage` | `reproducibility`.

---

# S0 — INVALIDATING

## S0-1 · The headline TCN result is scored on a different sample from everything it is compared against
`statistics` — `results/extension2_summary.md:15`, `README.md:298,306`, `scripts/run_extension2_fast.py:180,215-216`

`results/forecasts/tcn_partial.parquet` covers **124,216 rows / 2,236 dates / 2003-02-12 →
2012-12-31**. Every other row of that table covers **275,489 rows / 4,897 dates / →
2024-11-27**. `run_extension2_fast.py:180` writes whatever rows exist with no sample check,
and `:215-216` then computes `max(results, key=...)` across the mismatched sets.

Re-scored on TCN's exact rows:

| model | IC on TCN's rows |
|---|---|
| **rolling_vol** — 126-day trailing Σr², zero parameters | **0.7755** |
| transformer_fast | 0.7715 |
| **tcn_partial** ← the headline | **0.7691** |
| lstm | 0.7574 |
| gbm | 0.7273 |
| har_rv | 0.7095 |

TCN is **third**, behind a zero-parameter rolling window. On the strategy side, restricted
to the same 2,236 days, **every one of the ten strategies is positive** and TCN ranks
**eighth of ten** (garch 0.319, mlp 0.304, transformer 0.284, har_rv 0.273, prob_lstm_unc
0.273, gbm 0.256, lstm 0.248, **tcn 0.238**, prob_lstm 0.221, rolling_vol 0.221).

The subsample is easier for *every* model, so this is not a TCN-specific fluke — it is the
decade. Mean IC 2003–2012 vs 2013–2024: rolling_vol +0.070, har_rv +0.068, transformer
+0.046, lstm +0.034.

"TCN shows the highest IC and the only positive Sharpe" and "TCN's gross Sharpe of 0.421 is
the strongest result in the project" are both artefacts of the window mismatch.
`extension2_summary.md:21` does carry a caution note, so this is disclosed-but-uncorrected
rather than concealed.

**Fix:** drop the TCN row from the comparison table, or re-score all ten models on the
2003–2012 subsample. Print `n_rows` and the date range beside every IC in every table.

## S0-2 · Every Diebold-Mariano p-value in the project is computed from a loss function that is identically zero
`statistics` — `scripts/compare_all_models.py:119-121`, `src/eval/tests.py:10-13,79-80`

`compare_all_models.py:119` passes `forecast_log_rv` / `target_log_rv` into
`diebold_mariano_qlike`. `_qlike_loss` clips both arguments to `1e-12`. Log-variances are
negative essentially everywhere (measured: 99.98% of realized, 100.0% of LSTM forecasts),
so both arguments collapse to the same constant and the loss is **exactly** `1 − log(1) − 1
= 0`.

Measured non-zero fraction of the loss differential:

| pair | n | non-zero rows | % | published p |
|---|---|---|---|---|
| rolling_vol vs lstm | 275,472 | 1,187 | 0.43% | 0.1325 |
| har_rv vs gbm | 276,299 | 167 | 0.06% | 0.3173 |
| har_rv vs lstm | 274,945 | **61** | 0.02% | 0.0000 |
| gbm vs lstm | 275,482 | **55** | 0.02% | 0.0000 |
| garch vs gbm | 276,687 | **0** | 0.00% | 1.0000 |

Every p-value in `README.md:32-38` and `results/dm_pvalues.parquet` is driven by the ≤0.5%
tail where a forecast log-variance happens to be positive. Re-run on variance levels the
conclusions invert:

| pair | published | correct |
|---|---|---|
| rolling_vol vs lstm | p = 0.132 "indistinguishable" | **p < 0.001**, t = 51.2 |
| har_rv vs gbm | p = 0.317 "indistinguishable" | **p < 0.001**, t = −4.5 |
| har_rv vs lstm | p < 0.001 | **p = 0.096**, t = 1.67 |
| garch vs gbm | p = 1.000 | **p < 0.001**, t = 4.5 |

The README's most-quoted sentence — *"a sophisticated deep-learning ensemble is
statistically equivalent to a 6-month rolling average"* — is an artefact of the clip.
`README.md:40` also explains the `p = 1.000` cell as a consequence of GARCH's constant
forecast; that explanation is wrong, it is this clip.

**Fix:** pass `forecast_rv`/`target_rv`; make `_qlike_loss` raise on non-positive input
instead of clipping; average the loss differential by date before HAC. Regenerate
`results/dm_pvalues.parquet` and `README.md:30-40`.

## S0-3 · The unscaled-momentum control is scored on a different window, and correcting it reverses a pre-registered verdict
`statistics` — `scripts/compare_all_models.py:99-102`, `README.md:118,126`

The control is built from the raw signal with no restriction to the OOS window: **6,015
days from 2001-02-01**, versus **4,897 days from 2003-02-12** for every scaled strategy.
The 1,118 extra days of 2001–2002 carry the whole difference.

| | Sharpe | n |
|---|---|---|
| unscaled_momentum, as reported | −0.0021 | 6,015 |
| unscaled_momentum, aligned dates | **−0.1343** | 4,897 |

On aligned dates **every** scaled variant beats it (har_rv +0.020, gbm +0.002, rolling_vol
−0.015, lstm −0.041 — all above −0.134). `README.md:118` grades pre-registered Prediction 5
"❌ Falsified — only HAR-RV marginally beat unscaled"; on comparable rows it is **✅
Confirmed**, and `README.md:126`'s narrative ("you cannot scale your way to a positive
Sharpe if the underlying signal has none") loses its evidence.

A single misalignment flipped a pre-registered post-mortem verdict, in the flattering
direction. This is exactly the check `CLAUDE.md` §5 exists to force.

**Fix:** reindex the control to the OOS date set before scoring, and re-grade Prediction 5.

## S0-4 · ~77% of the headline IC is static cross-sectional persistence, not forecasting
`statistics` — `src/eval/tests.py:112-141`, `README.md:15`

The metric is a cross-sectional Spearman correlation of forecast **levels** against realized
**levels**. It rewards knowing that one stock is chronically more volatile than another —
information that does not change and is not a forecast.

Constructing the correct out-of-sample floor (per-ticker mean log realized variance using
only data up to the prior December 31, refreshed once per walk-forward fold — the exact
information set each model gets):

| forecaster | XS-IC |
|---|---|
| **per-ticker constant, zero time-varying information** | **≈ 0.57** |
| har_rv | 0.673 |
| lstm | 0.739 |
| rolling_vol | 0.737 |
| tcn (on its own subsample, where the floor is 0.571) | 0.769 |

The whole modelling effort buys roughly **+0.16 IC over a per-stock constant**. GARCH as
implemented (0.635) adds **+0.06**. Removing the ticker fixed effect from both sides drops
LSTM 0.739 → 0.496 and GARCH 0.642 → 0.334.

Measured on the same rows, IC on **log-vol changes** rather than levels: rolling_vol 0.5368,
transformer 0.5353, tcn 0.5300, lstm 0.5221, har_rv 0.4035. Partial IC controlling for the
already-known trailing 21-day RV: rolling_vol 0.4748, transformer 0.4822, tcn 0.4680,
har_rv 0.2843.

**Fix:** print the per-fold-constant floor as a row in every IC table and report the
increment over it as the headline. `README.md:15` describes IC as measuring "how well each
model ranks stocks by future volatility"; it mostly measures how well the model has learned
which stocks are persistently volatile.

## S0-5 · No model beats a 126-day rolling window, and the winner is smaller than selection noise
`statistics` — `results/extension2_summary.md`, `README.md:295-306`

Paired daily-IC differentials vs `rolling_vol` on TCN's rows, Newey-West(20):

| model | ΔIC | NW se | t | p |
|---|---|---|---|---|
| tcn_partial | **−0.0064** | 0.0037 | −1.76 | 0.079 |
| transformer_fast | −0.0040 | 0.0038 | −1.06 | 0.291 |
| lstm | −0.0182 | 0.0044 | −4.15 | <0.001 |
| mlp_fast | −0.0269 | 0.0050 | −5.40 | <0.001 |
| gbm | −0.0482 | 0.0038 | −12.79 | <0.001 |
| har_rv | −0.0660 | 0.0061 | −10.77 | <0.001 |
| garch | −0.1055 | 0.0080 | −13.21 | <0.001 |

Nothing beats the zero-parameter baseline; four are significantly worse.

On the Sharpe side, `SE(annualised Sharpe) = √(252/2236) = 0.336` on TCN's sample. The
reported +0.238 is **t = 0.71**. Under a pure-noise null, E[max of 4 candidates] = **+0.346**,
E[max of 6] = +0.426, E[max of 10] = +0.516. **The "best" Sharpe in the project is smaller
than what picking the best of four coin flips would produce.**

Candidate count is ~10 architectures/variants and ~40 reported Sharpe cells on one test
sample. Git history shows the test set informed design: `4714e26` and `78a0a51` (prob-LSTM
and uncertainty weighting, 2026-05-22) postdate `5de2465` "post-mortem on all 8 predictions"
(2026-05-22) and Phase 3 completion (2026-05-19), and `README.md:273` states the motivating
hypothesis as an observed test-set result.

**Fix:** report a selection-adjusted interval (White Reality Check / Romano-Wolf stepdown)
over all candidates, and state the candidate count beside every "best" claim.

---

# S1 — MAJOR

## S1-1 · GARCH is a per-ticker constant refreshed annually, not a GARCH forecast
`statistics` — `src/models/baselines.py:179-194`

`res.forecast(horizon=21)` is called once from the last in-sample date, reduced to
`scalar_rv = float(cond_var_sum.iloc[0])` (`:186`), then broadcast to every test date
(`:194`). Measured unique `forecast_rv` values per (ticker, year): **garch 1.0**, lstm 224,
har_rv 224, rolling_vol 224.

The comment at `:175-177` admits the shape was forced to fit the harness. GARCH re-estimates
at the same annual cadence as the ML models but then discards ~250 days of conditioning
information they are allowed to use. Its 0.635 IC measures only cross-sectional level
ordering — it adds +0.06 over a plain constant. Pre-registered Prediction 1 ("HAR-RV beats
GARCH") is therefore **untested**.

**Fix:** keep annual parameter estimation, but re-filter daily — `arch_model(...).fix(params)`
over full history, taking the 21-step-ahead variance sum from each test date.

## S1-2 · The weight pipeline has no scale discipline: unnormalised gross, a √21 unit error, and a 5.9× baseline level bias
`economics` — `src/strategy/scaling.py:35,37`, `src/strategy/portfolio.py:64-67`, `src/models/baselines.py:26`

Three symptoms, one root cause: nothing in the pipeline normalises or checks portfolio
scale, so no invariant exists that any of these would violate.

**(a) Gross exposure is never normalised.** `portfolio.py:64-67` passes weights through
untouched. Measured mean gross: rolling_vol 1.151, har_rv 2.562, gbm 2.687, tcn 2.760,
lstm 2.792, **garch 10.019 (max 1810)**. Within LSTM alone, gross swings 0.68→4.66. Net
exposure std is 0.54 for lstm and **106** for garch — the book is neither dollar- nor
beta-neutral and carries a swinging ±50% market bet.

`mean(r)/std(r)*√252` on this series is a P&L on $1 notional that is not the capital
employed. `results/master_results_table.parquet` proves it: garch shows ann_ret 8695%,
ann_vol 38542%, **max_dd −170%** — arithmetically impossible for a return on capital.
(`README.md:59-61` does flag the garch row as excluded.)

**(b) √21 annualisation error.** `scaling.py:35` and `uncertainty_scale.py:52` compute
`np.sqrt(rv_dt * 252)` where `rv_dt` is a **21-day** variance sum (`targets.py:9`). Correct
factor is 252/21 = 12. Measured: `sqrt(mean_target × 252)` = 2.490 vs correct 0.543 vs
realized annualised vol 0.570 — ratio exactly √21 = 4.583. Because it is a single constant
and `apply_costs` is homogeneous of degree 1 in weights, **Sharpe is exactly unchanged** —
but `README.md:50`'s "this targets 10% annualised portfolio volatility" is false (realized
vols run 8.5%–22.1%).

**(c) rolling_vol's level is 5.94× the target.** `baselines.py:26` sums **126** days of
squared returns and reports it directly as `forecast_rv` against a **21**-day target.
Measured mean(forecast)/mean(target): rolling_vol **5.940**, tcn 1.751, lstm 0.840, gbm
0.679, har_rv 0.628, garch 0.518.

(b) and (c) combine through (a) to corrupt the README risk columns: √5.94 = 2.44, and
rolling_vol's measured gross is 1.151 vs ~2.7 for everything else — ratio 2.35. **Its
flattering −36% max drawdown and 8.5% annualised vol are 2.4× less leverage, not less
risk.** Spearman IC is blind to all of it; QLIKE and Mincer-Zarnowitz are not.

**Fix:** normalise gross inside `vol_scale` (`w /= w.abs().sum()`, or cap gross at a fixed
L) before `build_portfolios`; correct the factor to `252/21`; rescale rolling_vol to
`rv * (21/window)`. Then re-run every level-sensitive metric.

## S1-3 · Effective sample size is ~100–600, not 275,000
`statistics` — `configs/default.yaml:31`, `src/data/targets.py:5`, `src/eval/tests.py:48,98`

`h = 21` and the target is computed at every step, so consecutive targets share 20/21 of
their window. Two different effective-N figures apply to two different statistics, and both
are needed:

**For anything built from the daily IC series** (mean IC > 0, IC_A vs IC_B — the
cross-section is already collapsed inside each daily IC):

```
daily IC observations (TCN sample)   2,236
IC autocorr lag 1/5/21/42/63         0.957 / 0.815 / 0.474 / 0.418 / 0.421
naive SE = std/sqrt(N)               0.00156    t = 493
Newey-West(20) SE                    0.00629    t = 122
implied effective N                    137 dates   (variance inflation 16.3)
Newey-West(60) SE                              t = 82   -> N_eff = 62
non-overlapping 21-day blocks          106
```
The inflation factor is stable at ~16 across models and samples (full sample: 305 effective
dates for lstm, 294 for rolling_vol).

**For row-level pooled statistics** — precisely the DM tests of S0-2, which pool 275,000
rows: 233 time blocks × 2.61 effective names (measured mean pairwise residual correlation
across tickers **0.371**, mean cross-section 55.6) = **≈ 623 independent rows** full-sample,
≈ 277 on the TCN sample. Pooled t-statistics are overstated by ≈ √441 ≈ **21×**.

Consequences: `README.md:245`'s "N = 275,489 OOS predictions" for the prob-LSTM calibration
should read ~600, and ECE = 0.012 sits **below** the sampling floor of ~1/√623 ≈ 0.04 — so
"essentially no miscalibration" (`README.md:243`) is not supported; the estimate cannot
resolve miscalibration at that scale. `tests.py:48`'s `nw_lags = 20` is applied to a
**row-major pooled panel** with ~56 rows per date, so it does not span even one
cross-section and captures none of the 21-day overlap. `tests.py:98`'s Mincer-Zarnowitz uses
default IID covariance on the same panel; its `p_joint` is uninterpretable.

**Fix:** average loss differentials by date before HAC; use lag ≥ 40 on the date series;
report both effective-N figures with their units next to any significance claim.

## S1-4 · No seed distribution exists for any neural result
`reproducibility` — `configs/default.yaml:70`, `scripts/run_extension2_fast.py:43,63`

LSTM uses 5 seeds; every Extension-2 architecture uses 2. No mean/std/min/max across seeds
is reported anywhere in the repo. The `transformer − lstm = +0.0073` gap is compared across
a 2-seed and a 5-seed ensemble, and cannot be attributed to architecture without it. See
`bench/BASELINE.md` §9 for the measured distribution.

## S1-5 · The SHAP attribution is computed entirely in-sample
`statistics` — `scripts/run_phase4.py:84-85`, `src/interp/shap_analysis.py:18`

```python
gbm_shap.fit(train_last)
shap_imp = compute_shap_importance(gbm_shap, train_last, sample_size=5000)
```

The model is fit on `train_last` and attributed on `train_last` — the same rows. The
0.380 Parkinson figure describes what the model **fit**, not what generalises, yet
`README.md:88` calls it "a genuine discovery" and `docs/predictions.md:59` grades a
pre-registered prediction against it. It is also a 5,000-row sample (`shap_analysis.py:18`,
`random_state=42`) of ~1.4M training rows, and "38%" is a share of total mean|SHAP|, not
variance explained.

Two further points nobody has stated: the attribution is the **GBM's** — the
fourth-ranked model — and is presented as a property of the modelling approach generally;
and `vix` carries 0.082 SHAP weight despite having **zero cross-sectional variation** (it is
identical for every ticker on a date, so its standalone cross-sectional IC is undefined —
it cannot rank stocks at all).

**Fix:** compute SHAP on held-out test-window rows, label it as the GBM's, and report it as
a share of |SHAP| rather than as importance.

---

# S2 — MODERATE

## S2-1 · The "chronological" validation split is a ticker holdout with 100% date overlap
`statistics` — `src/models/lstm_model.py:59-85,129-133`; same defect via delegation in `tcn_model.py:89`, `mlp_model.py:54`, `transformer_model.py:82`, and duplicated in `prob_lstm.py:87-88`

`_build_sequences` loops `for tkr in tickers` and appends each ticker's entire series, so
`X` is **ticker-major, date-minor**. `fit()` then slices `X[n-n_val:]` under the comment
*"Chronological 90/10 split — no shuffle"* (`:129`). The positional tail of a ticker-major
array is the last few **tickers**, not the last few **dates**.

Measured on the 2010 training window (56 tickers, 148,530 sequences, n_val = 14,853): the
validation block is exactly the last 7 tickers — BXP, C, CAG, CAH, BBBY, AIZ, AMP — all
spanning 2000-01-03 → 2010-12-31. **Fraction of validation dates that also appear in
training: 1.0000.** No embargo exists at this boundary either.

`gbm.py:33` calls `train.sort_index(level="date")` first, so GBM's tail split is genuinely
temporal — an additional uncontrolled difference in any GBM-vs-neural comparison.

**Not a leak, and I want to be precise about that:** every validation row has
`date ≤ train_end`, and the test window starts `train_end + 43 days`. No OOS information
enters, and no reported OOS number is inflated by this. The `mse_resid_` that feeds the
Jensen correction is a per-fold **scalar**, so it is exactly rank-preserving and Spearman IC
is bit-for-bit unaffected. What it does damage is model selection: early stopping is chosen
against contemporaneous, cross-sectionally correlated names (mean pairwise residual
correlation 0.37), so the stopping epoch is a poor proxy for temporal generalisation — and
the four architectures being compared differ mainly in overfitting propensity.

**Fix:** sort emitted rows by date in `_build_sequences`, then split by date with a 21-day
purge band. Delete the false comments. Regenerate all Extension-1/2 numbers.

## S2-2 · The backtest sizes a momentum bet; it has real power, and the forecasts capture almost none of it
`economics` — `src/strategy/scaling.py:37`, `src/strategy/momentum.py:25`

`w = (target_vol / ann_vol) * z` where `z` is the cross-sectional z-score of a 12-1 momentum
signal. The volatility forecast enters **only** as a per-name multiplier; sign and rank come
entirely from momentum. A reported Sharpe here is momentum's Sharpe modulated by sizing.

I initially concluded from an oracle test that the backtest therefore had no power at all.
**That was wrong** — I had measured it on the S0-1 subsample where all ten strategies
cluster at 0.22–0.32 and nothing discriminates. On the real 2003–2024 window:

| forecast fed to `vol_scale` (identical rows, 4,897 days) | Sharpe |
|---|---|
| **ORACLE — the realized forward RV itself** | **+0.1142** |
| constant RV (zero information) | +0.0288 |
| har_rv | +0.0200 |
| gbm | +0.0019 |
| rolling_vol | −0.0146 |
| lstm | −0.0406 |

Perfect foresight buys **+0.085 over a zero-information constant** and **+0.155 over the
LSTM**. Inverse-vol sizing *is* the economic use of a volatility forecast, and there is
~0.09–0.16 Sharpe of genuine headroom in this design. Every model exploits approximately
none of it, and the LSTM is worse than a constant.

So the IC-vs-Sharpe disconnect is not "the metric is meaningless" — it is that a forecast
can rank stocks by volatility level (S0-4: mostly persistence) while adding nothing to the
conditional information the sizing rule actually needs.

**Fix:** report the oracle and the constant-forecast Sharpe as the ceiling and floor of
every strategy table. To make a claim *about the forecast*, hold gross and dispersion fixed
and vary only the forecast.

## S2-3 · Costs exceed gross alpha for every 22-window model, driven by daily re-sizing
`economics` — `src/strategy/costs.py:16-27`

The cost model itself is correct: 5 bps/side on `w.diff().abs().sum()`, applied to the same
shifted weight series that earns the return. The magnitudes are the problem.

| model | annualised turnover | cost @5bps/side | gross return |
|---|---|---|---|
| lstm | **65.4×** | 3.27% | 2.43% |
| har_rv | 68.0× | 3.40% | 3.81% |
| gbm | 72.3× | 3.61% | 3.66% |
| rolling_vol | 26.6× | 1.33% | 1.21% |
| garch | **281.3×** | 14.07% | — |

65× annual turnover on a **12-month** momentum signal is the tell: the momentum leg barely
moves. The churn is the daily-refreshed volatility forecast re-sizing the entire book every
day, against a 21-day forecast horizon. `configs/default.yaml` specifies
`rebalance: "month_end"`; the code does not implement it.

**Fix:** rebalance at the forecast horizon (monthly), or put a no-trade band on the sizing
multiplier.

## S2-4 · The "S&P 500 cross-section" is the alphabetically-first 80 tickers, fixed at a 2002 as-of date, with present-day sector labels
`statistics` — `src/data/universe.py:174`, and `[:80]` at `run_extension2.py:41`, `run_extension2_fast.py:67`, `compare_all_models.py:32`, `run_phase4.py:44`, `build_data.py:25`, `run_baselines.py:27`, `run_ml_models.py:28`, `run_extension1.py:38`

`get_universe()` returns a **sorted** list; every caller slices `[:80]`. The surviving 63
tickers run `A, AA, AAPL, ABS, …, C, CA, CAG, CAH` and stop dead at CAH. The universe is
fixed at 2002-01-01 for a 2000–2024 study, so no point-in-time rebalancing occurs.

Sectors come from `get_sector(t, Timestamp("2003-01-01"))` against a table whose
`gics_sector` is scraped from **today's** constituent list (`universe.py:64`), back-filled
to historical names. A firm that changed sector carries its 2024 label throughout, and
**70,229 of 353,062 rows (19.9%) have `sector == ""`**, which `gbm.py:28` feeds to LightGBM
as a legitimate category level.

Mitigating: delisted names (ADCT, ABS, ANDV, BEAM, BMC, BMS, CA, BBBY) *are* present, so
survivorship is partially handled. And `sector` is not in `FEATURE_COLS`
(`lstm_model.py:13`), so the sector half touches only GBM and `interp/sector_neutral.py`.

**Fix:** rebuild the universe per window via `get_universe(w.train_end)`; carry sector on
the membership-period row; map missing sectors to explicit `"Unknown"`. At minimum, stop
describing an alphabetic head as "the S&P 500 cross-section".

## S2-5 · Tradable universe at date t is conditioned on the next 21 days of data
`leakage` — `src/data/targets.py:19`, and `panel = features.join(targets).dropna(subset=["target_log_rv"])` at `run_extension2.py:58`, `run_extension2_fast.py:84`, `run_extension1.py:55`, `run_phase4.py:60`

`targets.py:19` drops rows whose forward RV is exactly zero, and the inner join drops rows
with no forward target. Names that halt or go stale over `[t+1, t+21]` are therefore removed
from the cross-section **before `vol_scale` ever sees them**. Genuine lookahead in universe
construction. Small in magnitude — the affected rows are degenerate halted-trading data —
but it is selection on the dependent variable applied to the test set, not just the training
set.

Related: `costs.py:19` `.fillna(0.0)` turns a missing return into a costless zero rather
than closing the position.

**Fix:** determine the tradable set from information available at `t` only; document the
zero-RV drop as a methodology caveat rather than treating it as free.

## S2-6 · The test suite is green and covers none of this
`reproducibility` — `tests/`

136 non-ML tests pass in 74s. No test asserts that the validation split is chronological
(S2-1). No test exercises `diebold_mariano_qlike` at all — `tests/test_stat_tests.py` covers
only the MSE variant, so the degeneracy of S0-2 was never reachable. No test asserts that
rows in a comparison table match (S0-1, S0-3). A green suite here is not evidence of
methodological soundness, and the repo has been treating it as such.

**Fix:** add a test that the validation block's max date is < the training block's min test
date; a test that `_qlike_loss` rejects non-positive input; and a test that any two forecast
series entering one table share an index.

---

# S3 — MINOR

- **`max_drawdown` computed two incompatible ways.** `run_extension2.py:127` and
  `run_extension1.py:119-120` use `max_drawdown(net.cumsum())`; `run_extension2_fast.py:113,179`
  uses `(1+net).cumprod()`. `metrics.py:17-20` divides by `cummax`, which crosses zero for a
  cumsum — measured: lstm cumsum-DD = **inf**, cumprod-DD = 0.7365. This is the `inf%` in
  `results/extension1_summary.md:15-16`, silently repaired to 73.5% in `README.md:280`.
  Both scripts write **the same path** `results/extension2_summary.md`; re-running
  `run_extension2.py` would overwrite the current table with `inf%`. `economics`
- **`scripts/run_baselines.py` cannot run.** `:43` builds `features.join(targets)` with no
  `return` column and passes it to models that index `history["return"]`
  (`baselines.py:25,158`) → `KeyError`. `make baselines` is dead. The cached artefacts came
  from `compare_all_models.py:49`, which builds `rv_panel` correctly. `reproducibility`
- **Forecast caches have no provenance.** Every gate is `if path.exists(): load` with no
  config hash, code SHA, or row count check (`compare_all_models.py:71-77`,
  `run_extension2.py:79-82`, `run_extension2_fast.py:120-123,137-147`, `run_phase4.py:71-74`).
  Artefact mtimes span May 19–25 across a single "comparison". Worse,
  `run_extension2_fast.py:140-147` **prefers** a full-run checkpoint (5 seeds, 50 epochs) over a
  fast one (2 seeds, 10 epochs), so one architecture's series can be a per-window blend of two
  protocols while `:202` writes "Fast mode: 2 seeds, 10 max epochs" into the summary.
  `reproducibility`
- **`requirements.txt` does not resolve on this machine.** `torch==2.3.0` has no CPython 3.14
  wheel; `arch==6.3.0` raises on import under 3.14. `pandas==3.0.1` needs ≥3.11, so the 3.9
  interpreter cannot satisfy the file either. See `bench/BASELINE.md` §0. `reproducibility`
- **MCS and the Sharpe bootstrap are cited in the README but never called.**
  `tests.py:144,180` have zero call sites outside `tests/test_stat_tests.py`, yet
  `README.md:40` reports "the Model Confidence Set at α = 0.10 retains all five models" and
  `README.md:158` cites stationary-bootstrap Sharpe CIs. No such artefact exists in
  `results/`. `statistics`
- **`rolling_vol` gets one extra bar.** `baselines.py:26` has no `.shift(1)` while every
  feature does (`features.py:8,15,28-35`). Not leakage — the target starts at `t+1` — but an
  unequal information set for the baseline that ties the LSTM. Measured cost of fixing:
  ΔIC = 0.0007. `statistics`
- **`metrics.py:46` `icir`** multiplies by `√n` on a series with lag-1 autocorrelation 0.957,
  overstating by ~5×. Not used in any headline number, but it is a loaded gun. `statistics`
- **`uncertainty_scale.py:36-37`** falls back to **raw** momentum values when `sig_std == 0`,
  where `scaling.py:27-28` returns zeros — so the prob-LSTM comparison is not apples-to-apples
  on those dates. `economics`
- **HAR-RV's Jensen σ² is in-sample and per-ticker** (`baselines.py:64`,
  `res.mse_resid`), where every ML model uses a held-out global scalar. A per-ticker σ²
  **reorders the cross-section**; a global one does not. HAR's XS-IC is computed on a
  differently-transformed quantity. `statistics`
- **HAR-RV is fit per-ticker** (`baselines.py:67-70`) while GBM/LSTM are fit on the pooled
  panel — handicapped on estimation sample size relative to the models it benchmarks. A
  pooled panel-HAR with ticker fixed effects is the correct like-for-like comparator.
  `statistics`
- **No EWMA/RiskMetrics baseline exists.** Three lines, no fitting, and it scores 0.7393 on
  TCN's rows — above gbm (0.7273) and har_rv (0.7095). Its absence is the largest single gap
  in the baseline set. `statistics`
- **`targets.py:12-17`** duplicates the same four-sentence comment verbatim. Cosmetic.

---

# REFUTED — claims that did not survive

These were raised and killed. Recording them so they are not re-litigated.

- **"The Parkinson feature overlaps the target window."** REFUTED, and this was attacked
  hardest. `features.py:10-15` is `rolling(5).mean()` then `.shift(1)` → bars **t−5…t−1**.
  `targets.py:9` is `rolling(21).sum().shift(-21)` → bars **t+1…t+21**, verified numerically
  to 1e-16 (`target_rv[t]` = 0.017158451179953606 vs Σr² over t+1…t+21 =
  0.017158451179953564; the t…t+20 window does **not** match). There is a full **one-bar
  dead zone**: bar `t` is consumed by neither. Every other feature verified the same way.
  The sequence models are more conservative still — `feats[t-60:t]` has a newest underlying
  bar of `t−2`. **The SHAP attribution to `pk` is not a leaking feature.** The economic
  explanation holds: measured Spearman correlation with the target is `pk` 0.6997 vs `rv_m`
  0.6896, i.e. a 5-day range estimator legitimately out-predicts a 21-day close-to-close sum
  because Parkinson is ~5× more efficient per observation. (The in-sample computation of
  S1-5 is a separate and real defect.)
- **"No git repository exists."** REFUTED. `git rev-parse --show-toplevel` returns
  `/Users/harry/RL:ML Project/ml-vol-momentum` with 20+ commits. The pre-registration claim
  at `README.md:107` is therefore checkable — commit `5de2465` exists.
- **"`rolling_vol.parquet` and `garch.parquet` cannot be produced by the current code."**
  REFUTED. `compare_all_models.py:49` builds `rv_panel = returns_panel.join(panel)` and
  passes it at `:60`. Only `run_baselines.py` is broken (recorded in S3).
- **"Perfect foresight of the target buys nothing" — my own claim.** REFUTED. I measured the
  oracle on the S0-1 subsample, where every strategy clusters at 0.22–0.32. On the full
  2003–2024 window the oracle earns +0.114 vs the LSTM's −0.041. Corrected in S2-2.
- **"A per-ticker constant scores IC 0.649, so the models add almost nothing."** REFUTED as
  stated — that construction uses the **full-sample** mean, i.e. 2003–2024 targets to
  forecast 2003. It peeks. The competing claim of 0.400 (a constant frozen at 2000–2002 and
  never updated) is a strawman in the other direction. The defensible floor is **≈0.57**,
  from a per-fold expanding constant. Recorded in S0-4.
- **"The validation-split defect inflates the reported OOS numbers."** REFUTED. All
  validation rows are at `date ≤ train_end`, behind the 42-day embargo, and `mse_resid_` is a
  rank-preserving scalar. Downgraded to S2-1, a model-selection defect.
- **"The √21 error explains the poor Sharpes."** REFUTED. It is a single constant and
  `apply_costs` is homogeneous of degree 1 in weights, so Sharpe is exactly invariant.
  Retained in S1-2 only because it falsifies the stated 10%-vol design property.
- **"`p = 1.000` for garch-vs-gbm is caused by GARCH's constant forecast"**
  (`README.md:40`). REFUTED. It is the S0-2 clip — both log-forecast series are entirely
  negative. On levels the same pair gives p < 0.001.

---

# What is clean

Verified and found sound, stated positively because a clean verdict is worth as much as a
defect:

- **Feature/target window arithmetic** — one-bar dead zone, no overlap anywhere, verified
  numerically per feature.
- **Normalisation scope** — every forecaster computes feature/target means and stds on the
  `train` argument only (`lstm_model.py:100-109`, `mlp_model.py:60-64`,
  `transformer_model.py:89-93`, `tcn_model.py:95-99`, `prob_lstm.py:119-125`) and reuses the
  stored statistics in `predict`. No full-sample statistic anywhere in a model.
- **Train/test chronology and embargo** — `CVWindow.__post_init__` (`walk_forward.py:13-20`)
  **raises** if the gap is short; it is a hard constructor gate, not an assertion. 42 calendar
  days ≥ the 21-trading-day target span.
- **`shuffle=True` in the training DataLoaders** is minibatch order within an already-fixed
  train set. Not a leak.
- **No future bars in features** — the only `shift(-k)` in the feature path is the intended
  target; the other is a deliberate leakage canary in `eval/synthetic.py:25`. No `bfill`
  anywhere in `src/`. `vix` is reindexed without `ffill`, so gaps become NaN and drop.
- **Point-in-time weight alignment** — `portfolio.py:72-74` shifts +1, and `costs.py:21`
  pairs `w_shifted[t]` with `r[t]`. No off-by-one in either direction. The cost is charged on
  the same series that earns the return, and the initial ramp is charged.
- **Determinism at fixed seed** — bit-identical across processes (`bench/BASELINE.md` §9a).

---

## Process note

The Phase 3 reconciliation list arrived in the same message as the Phase 0–2 brief, so it
was not possible to run Phase 2 without having read it. I dispatched the four auditors with
no hints from that list, and the reconciliation in `audit/RECONCILIATION.md` is written
against findings the auditors produced independently — but the blindness is procedural, not
absolute, and I am not going to claim otherwise.
