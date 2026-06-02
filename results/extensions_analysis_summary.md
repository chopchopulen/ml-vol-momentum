# Extensions Analysis Summary

## 1. Cost Sensitivity

Sharpe ratio at different round-trip transaction costs:

| Model | 0 bps (gross) | 5 bps | 10 bps | 20 bps | Breakeven |
|-------|--------------|-------|--------|--------|-----------|
| HAR-RV | 0.187 | 0.103 | 0.020 | -0.147 | 11.2 bps |
| GBM | 0.165 | 0.084 | 0.002 | -0.161 | 10.1 bps |
| LSTM | 0.117 | 0.038 | -0.041 | -0.198 | 7.4 bps |
| Prob LSTM | 0.125 | 0.045 | -0.036 | -0.197 | 7.8 bps |
| Transformer | 0.135 | 0.055 | -0.026 | -0.188 | 8.4 bps |
| TCN (partial) | 0.421 | 0.330 | 0.238 | 0.056 | 23.1 bps |

**Key finding:** Most forecasters have positive gross Sharpe but go negative after 5–10 bps costs.
The vol-scaling signal itself has value; transaction costs are the binding constraint.

## 2. Prob LSTM Calibration

- N = 275,489 OOS predictions
- Standardised residual mean = 0.033 (ideal: 0) — well-centred
- Standardised residual std = 1.056 (ideal: 1.0) — well-scaled
- ECE = 0.0123 (0 = perfect; 0.05 = 5pp avg miscalibration)

**Interpretation:** The model is reasonably calibrated.

## 3. LSTM Gradient Sensitivity

Top features by mean |∂output/∂input|:

| Feature | Sensitivity | Rank |
|---------|------------|------|
| pk | 0.023446 | 1 |
| rv_m | 0.020707 | 2 |
| rv_w | 0.009289 | 3 |
| rv_d | 0.009008 | 4 |
| vix | 0.008750 | 5 |
| ret_21 | 0.006524 | 6 |
| kurt | 0.004123 | 7 |
| log_dv | 0.003292 | 8 |
| skew | 0.002828 | 9 |

**Temporal sensitivity:** Peak gradient at lag ~1 days.
The LSTM assigns most weight to recent lags.
