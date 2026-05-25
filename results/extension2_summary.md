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
| TCN | 0.7691 | 0.0737 | +0.238 | 36.7% | 10 | 2-seed, 10 epochs, 2003–2012 only |
| Prob LSTM (plain) | 0.7379 | 0.0931 | -0.046 | 73.5% | 22 | Gaussian head, 5-seed |
| Prob LSTM (unc-weighted) | 0.7379 | 0.0931 | -0.036 | 73.1% | 22 | Uncertainty-weighted sizing |

## Key findings

- **TCN has the highest IC (0.769) and the only positive Sharpe (+0.238)** — but only on 10/22 windows
  (2003–2012). The partial sample may be favourable; treat with caution.
- **Transformer slightly beats LSTM on IC (0.746 vs 0.739)** and has a better Sharpe (-0.026 vs -0.041),
  consistent with attention capturing longer-range dependencies — but was too slow on CPU for production use
  (avg ~30 min/window vs ~7 min for LSTM).
- **MLP is the weakest on IC (0.723)**, confirming that temporal structure does help vs a flat feature vector.
- **Probabilistic LSTM: uncertainty weighting gave a marginal Sharpe improvement** (-0.036 vs -0.046 plain,
  vs -0.041 point LSTM). The IC is unchanged — distributional output doesn't improve rank ordering,
  only position sizing.
- **The IC ≠ Sharpe disconnect persists across all architectures.** LSTM has higher IC than MLP but worse
  Sharpe. The mapping from forecast quality to strategy performance is noisy in this 2003–2024 period.
