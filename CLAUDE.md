# ml-vol-momentum — working rules

Cross-sectional equity volatility forecasting. 9 features, 6 forecaster architectures,
expanding-window walk-forward CV, and a momentum backtest whose positions are sized by
the volatility forecast.

This file is binding. It overrides convenience, habit, and anything a results file claims.

---

## 0. `docs/FINAL_NUMBERS.md` is the single source of truth

**No number leaves this repo unless it appears in `docs/FINAL_NUMBERS.md` first.** Results here
live on two disjoint panels and are easy to misattribute; FINAL_NUMBERS is the reconciled record.

**Always name the panel.** Attaching one panel's descriptors to the other panel's finding is
defect S0-1:

| | rows | dates | tickers | window | forecasters |
|---|---|---|---|---|---|
| **Panel A** | 123,749 | 2,236 | 56 | 2003-02-12 → 2012-12-31 | incl. TCN |
| **Panel B** | 274,632 | 4,897 | 61 | 2003-02-12 → 2024-11-27 | TCN absent |

Standing conventions:

- **Never quote an IC level against a paired ΔIC.** `rolling_vol` 0.7754 vs Transformer
  **0.7714** (Δ = −0.0040) — not "0.7754 vs −0.0040."
- **No Sharpe is a return on capital** (OPEN-3). Relative brackets only.
- **N is not 275,489.** Effective N is 139–201 dates (Panel B), 61–96 (Panel A), or ~623
  independent panel rows. Pooled t-statistics were overstated ~21×.
- **The 0.380 SHAP figure is the GBM's, computed in-sample.** The `pk` *feature* result stands;
  the attribution number does not.
- **`const_floor` 0.5505 → 0.1934 are Spearman ICs, not R².**

---

## 1. Never tune to a metric

A change is permitted **only** if it corrects one of:

- **leakage** — target information reaching features, training, or model selection
- **methodology** — split construction, embargo, normalisation scope, evaluation protocol
- **measurement** — how a number is computed, aggregated, or compared

Adjusting a hyperparameter, window length, threshold, architecture, seed, or sample
range *because a reported number improved* is forbidden. If you notice yourself doing
it, stop and report it as a rule violation rather than committing it.

A correction that makes a headline number **worse** is a successful correction. Report it
plainly. Do not go looking for an offsetting change.

## 2. Every reported number cites its seed AND its seed distribution

Neural training here is not deterministic across seeds. A single-seed IC is one draw.

Any result quoted in a README, a results file, a commit message, or a conversation must
carry:

- the seed (or seed list, for an ensemble)
- the mean, std, min and max across **at least 10 seeds** under the same protocol
- the number of walk-forward windows and the test date range it was computed on

`IC = 0.769` is not a result. `IC = 0.769 (seeds [0,1] ensemble; 10/22 windows,
2003-02..2012-12)` is a result, and it still needs its seed distribution before it can be
compared to anything.

## 3. No metric changes without before/after against `bench/BASELINE.md`

`bench/BASELINE.md` holds the frozen reference numbers and the exact commands that
reproduce them. Any change that can move a reported number must ship with a before/after
table against that file, both sides citing seed and seed distribution.

If `bench/BASELINE.md` does not exist or is stale, re-freeze it before changing anything.

## 4. Auditors are read-only

The audit agents in `.claude/agents/` — `leakage-auditor`, `statistics-auditor`,
`baseline-auditor`, `backtest-auditor`, `adversarial-reviewer` — have **no Edit and no
Write tool**, by design. They describe fixes; they never apply them. `architect` writes
only `audit/`, `bench/`, `.claude/agents/`, and this file.

Only `builder` edits model or feature code, and only against a specific finding ID in
`audit/FINDINGS.md` that the user has explicitly approved.

## 5. Numbers are only comparable on identical rows

Before putting two numbers in the same table, verify they were computed on the same
dates, the same tickers, and the same row count — and print the row count next to them.
A model scored on a different subsample is not comparable to one scored on the full
sample, no matter how the table is laid out.

This has already bitten this repo once. See `audit/FINDINGS.md`.

## 6. Comments are not evidence

Several comments in `src/` assert point-in-time correctness that the surrounding code
does not deliver. When auditing window arithmetic, write out the bar ranges yourself from
the operations, and ignore what the comment claims.

---

## Environment

The project's parent directory contains a colon (`/Users/harry/RL:ML Project/`), which
Python's `venv` module treats as a `PATH` separator — the venv must live outside the
project tree.

`requirements.txt` does not currently resolve on this machine: `torch==2.3.0` has no
CPython 3.14 wheel, and `arch==6.3.0` fails at import on 3.14
(`TypeError: deprecate_kwarg() missing 1 required positional argument`). The audit
environment deviates as recorded in `bench/BASELINE.md`. That deviation is itself an open
reproducibility finding — do not silently rewrite the pins to match whatever happens to
install.

Test suite must be split across three pytest invocations; LightGBM (OpenMP) and PyTorch
are both fork-unsafe and SIGABRT when mixed in one process on macOS. See `Makefile`.
