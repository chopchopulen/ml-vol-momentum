# bench/BASELINE.md — frozen reference numbers

Everything here was measured during the Phase 0/2 audit. Nothing in `src/` was modified
to produce it. This file is the reference for the before/after rule in `CLAUDE.md` §3.

Frozen at commit `3e748e8` (`docs: add Extension 3 section — cost sensitivity,
calibration, gradient analysis`), working tree clean except untracked
`scripts/run_extension2_fast.py`.

---

## 0. Environment — the pinned environment does NOT reproduce

`requirements.txt` does not install on this machine. Two hard failures:

| pin | failure |
|---|---|
| `torch==2.3.0` | no CPython 3.14 wheel. Available: 2.9.0+ |
| `arch==6.3.0` | imports fine on 3.11, raises on 3.14: `TypeError: deprecate_kwarg() missing 1 required positional argument: 'new_arg_name'` |

Only CPython 3.14.3 and 3.9.6 are installed. `pandas==3.0.1` needs ≥3.11, so 3.9 cannot
satisfy the rest of the file either. **There is no Python on this machine on which
`requirements.txt` resolves.**

Audit environment (`/tmp/mlvm-venv`, deviations marked):

```
python 3.14.3    (Makefile venv path /tmp/ml-vol-momentum-venv did not exist)
torch  2.9.0     ← DEVIATION from pinned 2.3.0
arch   8.0.0     ← DEVIATION from pinned 6.3.0
pandas 3.0.1     numpy 2.4.3      scipy 1.17.1
statsmodels 0.14.6                lightgbm 4.3.0
```

Reproduce the environment:
```bash
python3 -m venv /tmp/mlvm-venv
/tmp/mlvm-venv/bin/pip install -r <(grep -v '^torch==' requirements.txt) torch==2.9.0
/tmp/mlvm-venv/bin/pip install -U arch
cd "/Users/harry/RL:ML Project/ml-vol-momentum" && /tmp/mlvm-venv/bin/pip install -e .
```

All 80 tickers 2000-01-01→2024-12-31 are present in `data/cache/`, so every number below
was produced **offline** — no network, no yfinance call.

---

## 1. The frozen evaluation panel

```bash
/tmp/mlvm-venv/bin/python <scratchpad>/build_panel.py
```
353,062 rows, MultiIndex `(date, ticker)`, 2000-01-03 → 2024-11-27, 80 tickers.
Columns: `rv_d rv_w rv_m pk skew kurt vix log_dv ret_21 target_rv target_log_rv sector`.

This is byte-identical to the panel `scripts/run_extension2_fast.py` constructs at its
lines 70–86; it is cached only so audit harnesses do not rebuild features on every run.

---

## 2. What the headline number actually is

`IC = 0.7691` (`results/extension2_summary.md:15`, `README.md:298`) is:

- the mean over **dates** of the daily cross-sectional **Spearman** correlation between
  `forecast_rv` and `target_rv` (`src/eval/tests.py:112`)
- on **volatility levels**, cross-sectionally — not on changes, not time-series
- from a **2-seed ensemble** (seeds `[0, 1]`), `max_epochs=10`, `patience=3`
  (`scripts/run_extension2_fast.py:43,63`)
- over **10 of 22** walk-forward windows: test years **2003–2012 only**
  (`results/forecasts/tcn_partial.parquet`, 124,216 rows, 2,236 dates, 57 tickers)

Every other architecture in that same table is scored over **22** windows and 275,489
rows. **The rows are not the same. The numbers were never comparable.**

---

## 3. Frozen table A — every model on ITS OWN sample vs on the TCN's sample

`n = 124,216` rows / 2,236 dates for the right-hand column. Same dates, same tickers,
same rows, for every row of the table.

| model | IC, own sample | n (own) | **IC on TCN's sample** | n |
|---|---|---|---|---|
| **rolling_vol** (126d trailing Σr², 0 params) | 0.7373 | 276,830 | **0.7755** | 124,210 |
| transformer_fast | 0.7463 | 275,489 | **0.7715** | 124,216 |
| **tcn_partial** ← headline | **0.7691** | 124,216 | **0.7691** | 124,216 |
| prob_lstm | 0.7379 | 275,489 | 0.7581 | 124,216 |
| lstm | 0.7389 | 275,489 | 0.7574 | 124,216 |
| mlp_fast | 0.7226 | 275,489 | 0.7487 | 124,216 |
| naive EWMA(λ=0.94) | 0.6954 | 352,929 | 0.7393 | 124,216 |
| gbm | 0.6935 | 277,288 | 0.7273 | 124,214 |
| har_rv | 0.6729 | 276,510 | 0.7095 | 123,955 |
| naive RW (trailing 21d RV) | 0.6565 | 351,527 | 0.7002 | 124,216 |
| garch | 0.6354 | 276,978 | 0.6700 | 124,111 |
| naive per-ticker constant | 0.5708 | 337,420 | 0.6116 | 123,778 |

On identical rows the TCN ranks **third**, behind a zero-parameter 126-day rolling
window. Its top-of-table position came entirely from being scored on a different,
easier sample.

The RW and EWMA rows are audit additions — no EWMA baseline exists in the repo.

## 4. Frozen table B — the sample really is easier

Mean cross-sectional IC by era, same model, same code:

| model | 2003–2012 (TCN's sample) | 2013–2024 | gap |
|---|---|---|---|
| rolling_vol | 0.7757 | 0.7052 | **+0.070** |
| har_rv | 0.7098 | 0.6421 | **+0.068** |
| lstm | 0.7574 | 0.7237 | **+0.034** |
| transformer_fast | 0.7716 | 0.7253 | **+0.046** |

Every model gains on 2003–2012. The subsample, not the architecture, is doing the work.

## 5. Frozen table C — levels vs changes, and increment over the naive benchmark

All on the TCN sample.

| model | IC (levels) — what is reported | IC (log-vol changes vs known trailing 21d RV) | partial IC controlling for trailing 21d RV |
|---|---|---|---|
| rolling_vol | **0.7755** | **0.5368** | **0.4748** |
| transformer_fast | 0.7715 | 0.5353 | 0.4822 |
| tcn_partial | 0.7691 | 0.5300 | 0.4680 |
| lstm | 0.7574 | 0.5221 | 0.4479 |
| garch | 0.6700 | 0.4618 | 0.3754 |
| gbm | 0.7273 | 0.4448 | 0.3382 |
| har_rv | 0.7095 | 0.4035 | 0.2843 |

A per-ticker constant forecast — which by construction contains **no** time-varying
information — scores **0.6116** on the reported metric. That is the floor the 0.769
should be read against, not zero.

## 6. Frozen table D — paired IC differences vs rolling_vol, Newey-West(20)

Daily IC differentials on the TCN sample, 2,236 dates, HAC lag 20 (= h−1).

| model | ΔIC vs rolling_vol | NW se | t | p |
|---|---|---|---|---|
| tcn_partial | **−0.0064** | 0.0037 | −1.76 | 0.079 |
| transformer_fast | −0.0040 | 0.0038 | −1.06 | 0.291 |
| lstm | −0.0182 | 0.0044 | −4.15 | <0.001 |
| mlp_fast | −0.0269 | 0.0050 | −5.40 | <0.001 |
| gbm | −0.0482 | 0.0038 | −12.79 | <0.001 |
| har_rv | −0.0660 | 0.0061 | −10.77 | <0.001 |
| garch | −0.1055 | 0.0080 | −13.21 | <0.001 |

**No model beats the 126-day rolling window.** Four are significantly worse.

## 7. Frozen table E — effective sample size

For the headline IC:

```
raw OOS rows                 124,216      ← the figure quoted as "OOS rows"
daily IC observations          2,236
mean daily IC                 0.7691
std of daily IC               0.0737
naive SE = std/sqrt(2236)     0.00156     t = 493
IC autocorr, lag 1/5/21/42    0.957 / 0.815 / 0.474 / 0.418
Newey-West(20) SE             0.00629     t = 122
implied effective N (dates)      137      ← vs 2,236 raw
non-overlapping 21d blocks       106
```

`h = 21`, so consecutive targets share 20/21 of their window. The effective number of
independent observations is **~106–137**, not 124,216. Any t-statistic in the repo built
on the raw row count is overstated by roughly √(124216/137) ≈ **30×**.

## 8. Frozen table F — backtest anatomy

Same forecast rows in all three cases; only the `forecast_rv` fed to `vol_scale` changes.

| sizing input | Sharpe | ann_vol | mean gross | net exp. std |
|---|---|---|---|---|
| TCN forecast (the headline path) | **0.238** | 0.187 | 2.76 | 0.517 |
| constant forecast | 0.127 | 0.447 | 2.98 | 0.024 |
| **oracle = the realized forward RV itself** | **0.235** | 0.175 | 3.19 | 0.702 |
| repo's `unscaled_momentum`, same dates | −0.024 | — | — | — |

**Perfect foresight of the target scores 0.235 — no better than the TCN's 0.238.** The
Sharpe in this backtest carries no information about forecast quality; there is nothing
for a better forecast to win. Gross exposure floats between 0.00 and 4.36 with no
normalisation, so the series is not a return on a defined capital base.

`max_drawdown` is computed two different ways in two scripts that write the same summary
file: `(1+net).cumprod()` in `run_extension2_fast.py:113,179` and `net.cumsum()` in
`run_extension2.py:127`. The second is what produces the `inf%` in
`results/extension1_summary.md`.

---

## 9. PHASE 0 — reproducibility

### 9a. Determinism at fixed seed: PASS

Same seed, same data, two independent processes, window 0 (test year 2003), single-seed
TCN, fast-mode config:

```
run 1:  IC = 0.7522386942
run 2:  IC = 0.7522386942
max |Δforecast_rv| over 12,096 rows = 0.000e+00
```

Bit-identical. CPU-only PyTorch on this machine reproduces exactly at a fixed seed.
`TCNForecaster.fit` seeds `torch.manual_seed` and `np.random.seed`
(`src/models/tcn_model.py:92-93`) but **not** `random`, and does not set
`torch.use_deterministic_algorithms(True)` — determinism here is observed, not enforced,
and is not guaranteed to survive a move to GPU.

### 9b. Seed-to-seed distribution

PENDING — sweep in progress. Protocol below is frozen regardless of outcome.

**Protocol (reduced, and the reduction is stated because it matters):** windows 0–2 (test
years 2003–2005), **single-seed** TCN (not the 2-seed ensemble), `max_epochs=10`,
`patience=3`, 12 distinct seeds `{0..9, 42, 43}` plus a repeat of 42 for the determinism
check.

The headline protocol (10 windows) costs ~3 h/seed on this machine; 12 seeds over 10
windows is ~36 h. Three windows is what fits. This makes the measured σ **conservative
in the right direction**: averaging over 3 windows leaves more seed noise than averaging
over 10, so the true 10-window σ is somewhat smaller than what this reports. It does not
bias the mean level, but the 3-window mean is not comparable to the 10-window 0.769 and
must not be quoted as if it were.

```bash
PHASE0_NWIN=3 PHASE0_THREADS=3 PHASE0_TAG=w3 \
  /tmp/mlvm-venv/bin/python -u <scratchpad>/phase0.py run <seeds...>
```

The harness verifies, before the sweep, that truncating `history` to
`test_start − 150 days` yields **bit-identical** test-window predictions to passing the
full history (max abs diff 0.000e+00), so the speedup does not change the arithmetic.

| seed | mean IC (3 windows) |
|---|---|
| _pending_ | |

**mean / std / min / max: PENDING**

Until this table is filled, `IC = 0.769` remains a single draw from an unmeasured
distribution and must not be compared to any other model's number.
