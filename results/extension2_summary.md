> # ⚠️ SUPERSEDED — see [`aligned_comparison.md`](aligned_comparison.md) and [`../docs/FINAL_NUMBERS.md`](../docs/FINAL_NUMBERS.md)
>
> The TCN row below is **not comparable** to any other row in this table: it was scored on
> 124,216 rows / 2,236 dates / 2003–2012, every other row on 275,489 rows / 4,897 dates /
> 2003–2024. On the aligned sample the TCN is **third on IC** and **8th of 11 on Sharpe**, and
> **no model beats a zero-parameter 126-day rolling estimator**. No Sharpe in this table is a
> return on capital — gross exposure is never normalised.

# Extension 2: Architecture Comparison Results

All architectures use: same 9 features, seq_len=60, Huber loss,
same walk-forward CV (42-day embargo, 2003-2024).
Fast-mode architectures: 2 seeds, 10 max epochs.
LSTM uses the existing 5-seed full-run result for reference.

## Results

| Architecture | Mean XS-IC | IC Std | Sharpe | Max DD | Windows | Notes |
|---|---|---|---|---|---|---|
| LSTM | 0.7389 | 0.0905 | -0.041 | 73.7% | 22 | 5-seed, 50 epochs, full run |
| Transformer | 0.7463 | 0.0895 | -0.026 | 73.6% | 22 | d_model=16, 2-seed, 10 epochs |
| MLP | 0.7226 | 0.0961 | -0.048 | 75.3% | 22 | 540-dim flat input, 2-seed, 10 epochs |
| TCN | 0.7691† | 0.0737 | +0.238† | 36.7% | 10 | 2-seed, 10 epochs, 2003–2012 only |

† **RETIRED — different sample from every other row.** Aligned (123,749 rows, 2003–2012, all
10 forecasters): TCN IC **0.7688**, ranking **third** behind rolling_vol 0.7754 and
transformer 0.7714 — **ΔIC = −0.0066 (t = −1.51)** against a zero-parameter baseline. Aligned
Sharpe **+0.2358**, ranking **8th of 11**, where every strategy is positive.
| Prob LSTM (plain) | 0.7379 | 0.0931 | -0.046 | 73.5% | 22 | Gaussian head, 5-seed |
| Prob LSTM (unc-weighted) | 0.7379 | 0.0931 | -0.036 | 73.1% | 22 | Uncertainty-weighted sizing |

## Key findings

- ~~**TCN has the highest IC (0.769) and the only positive Sharpe (+0.238)** — but only on 10/22
  windows (2003–2012). The partial sample may be favourable; treat with caution.~~
  **RETIRED.** The partial sample *was* favourable — to every model, not just the TCN
  (rolling_vol +0.070, har_rv +0.068, transformer +0.046, lstm +0.034 on 2003–2012 vs
  2013–2024). Aligned, the TCN is third on IC and 8th of 11 on Sharpe, and **every** strategy
  on that sample is positive, so "the only positive Sharpe" was an artefact of comparing a
  2003–2012 figure against 2003–2024 figures.
- **Transformer slightly beats LSTM on IC (0.746 vs 0.739)** and has a better Sharpe (-0.026 vs -0.041),
  consistent with attention capturing longer-range dependencies — but was too slow on CPU for production use
  (avg ~30 min/window vs ~7 min for LSTM).
- **MLP is the weakest on IC (0.723)**, confirming that temporal structure does help vs a flat feature vector.
- **Probabilistic LSTM: uncertainty weighting gave a marginal Sharpe improvement** (-0.036 vs -0.046 plain,
  vs -0.041 point LSTM). The IC is unchanged — distributional output doesn't improve rank ordering,
  only position sizing.
- **The IC ≠ Sharpe disconnect persists across all architectures.** LSTM has higher IC than MLP but worse
  Sharpe. The mapping from forecast quality to strategy performance is noisy in this 2003–2024 period.
