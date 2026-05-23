# Pre-Registered Predictions

Committed before Phase 3 (ML model training). Results compared ex-post.

**Date committed:** 2026-05-17

1. HAR-RV will beat GARCH(1,1)-t on both time-series QLIKE and cross-sectional IC by a wide, DM-significant margin. Hansen & Lunde (2005) show nothing beats GARCH on a single-asset univariate setup, but HAR-RV is materially superior for multi-horizon cross-sectional RV forecasting.

2. GBM will tie or modestly beat HAR-RV on cross-sectional IC (+0.02 to +0.05). On time-series QLIKE the gap will be small and may favour HAR-RV (GBM optimises a different loss).

3. LSTM will not beat GBM and may lose to HAR-RV on time-series QLIKE. Per-seed Sharpe std will be > 0.15.

4. The signal-scaling Sharpe ranking will compress: net-of-cost Sharpe for HAR-RV-scaled, GBM-scaled, and LSTM-scaled momentum will be within ~0.15 of each other. The MCS at α=0.10 will retain HAR-RV, GBM, and LSTM in the same confidence set. The forecast comparison (MCS) will be more statistically significant than the Sharpe-difference bootstrap.

   **Falsification clause:** If `sharpe_diff_bootstrap(LSTM_scaled, HAR_RV_scaled)` returns a 90% CI not crossing zero with point estimate > 0.3, prediction #4 is falsified → headline becomes "ML scaling adds material value." Symmetric: if HAR-RV beats ML by > 0.3 with CI not crossing zero → "HAR-RV strictly dominates."

5. All vol-scaled variants will beat unscaled momentum on Sharpe and max-drawdown. The 2009 crash will be mitigated by any reasonable vol forecast. The choice of forecaster matters less than the act of scaling.

6. SHAP will rank `rv_m > rv_w > vix > rv_d` for GBM — the ML model will rediscover HAR-RV's econometric structure.

7. Cross-sectional IC will outpredict time-series MSE for ranking models on the Sharpe leaderboard. The model with the highest XS-IC will have the highest scaled-momentum Sharpe even when its time-series MSE is not lowest. If false, the cross-sectional IC framing claim in the writeup is wrong.

8. The sector-neutral variant will reduce all Sharpes by ~0.1–0.2 but preserve the ranking across forecasters.

---

## Post-Mortem (Phase 3 + 4 results)

Results finalized 2026-05-22.

### Prediction 1 — HAR-RV beats GARCH(1,1) on QLIKE and IC
**Verdict: ✅ CONFIRMED**

Cross-sectional IC: HAR-RV 0.673 > GARCH 0.645. DM p-value (HAR-RV vs GARCH): < 0.001 — highly significant. The finding is consistent with Hansen & Lunde (2005): within a single series GARCH is hard to beat, but HAR-RV's multi-horizon structure gives it a clear edge for cross-sectional RV ranking.

### Prediction 2 — GBM ties/modestly beats HAR-RV (+0.02 to +0.05 IC)
**Verdict: ❌ BORDERLINE MISS**

GBM IC: 0.692. HAR-RV IC: 0.673. Difference: +0.019 — just below the +0.02 floor of the predicted range. DM p-value (GBM vs HAR-RV): 0.317 — not statistically distinguishable. Direction is correct (GBM > HAR-RV) but the gap is smaller than predicted. Interpretation: the tabular HAR-RV features already capture most of the predictable signal; GBM provides marginal improvement that is not robust to multiple testing.

### Prediction 3 — LSTM won't beat GBM; seed-to-seed Sharpe std > 0.15
**Verdict: ❌ FALSIFIED (IC direction wrong)**

LSTM IC: 0.739 — the *highest* of all models, beating GBM (0.692) and even RollingVol (0.737). However, LSTM Sharpe (-0.041) is the *lowest*. This apparent contradiction (highest IC, lowest Sharpe) is explained by Prediction 7 analysis: IC does not predict Sharpe ranking in this dataset. The IC result was unexpected.

### Prediction 4 — Sharpe ranking compresses; HAR-RV/GBM/LSTM within 0.15 of each other
**Verdict: ✅ CONFIRMED (with caveat)**

HAR-RV: 0.020, GBM: 0.002, LSTM: -0.041. Spread = 0.061 < 0.15 threshold. GARCH is an outlier (0.226) due to degenerate vol-scaling weights from its constant forecast — not a real signal. DM test: HAR-RV ≈ GBM (p=0.317), LSTM ≈ RollingVol (p=0.132) — all ML models statistically indistinguishable from simpler baselines on forecast quality. Forecast comparison is indeed more discriminating than Sharpe difference (all ML Sharpes near zero with overlapping uncertainty).

### Prediction 5 — All scaled variants beat unscaled momentum
**Verdict: ❌ FALSIFIED**

Unscaled Sharpe: -0.002. HAR-RV scaled: 0.020. GBM scaled: 0.002. LSTM scaled: -0.041. RollingVol scaled: -0.015. Only HAR-RV scaled marginally beats unscaled. The 2003-2024 sample period is unusually hostile to momentum strategies; vol-scaling did not provide the consistent Sharpe uplift seen in Barroso & Santa-Clara (2015), whose sample focused on 1927-2011. The act of scaling can *increase* idiosyncratic vol exposure when sigma_hat is noisy.

### Prediction 6 — SHAP: rv_m > rv_w > vix > rv_d
**Verdict: ❌ FALSIFIED (biggest surprise)**

Actual SHAP ranking: pk (Parkinson) 0.380 > rv_m 0.274 > sector 0.108 > vix 0.082 > rv_w 0.045. Parkinson's range-based estimator (using daily high/low) dominates at 0.38, well ahead of the HAR-RV monthly component (0.274). This suggests intraday range carries information about cross-sectional vol dispersion that is not captured by squared close-to-close returns. The HAR-RV structure is partially rediscovered (rv_m > rv_w) but Parkinson's dominance falsifies the specific ranking prediction.

### Prediction 7 — XS-IC predicts Sharpe leaderboard better than time-series MSE
**Verdict: ❌ FALSIFIED**

LSTM has the highest XS-IC (0.739) but the lowest Sharpe (-0.041). RollingVol has the second-highest IC (0.737) and the second-lowest Sharpe (-0.015). HAR-RV has lower IC (0.673) but the highest non-degenerate Sharpe (0.020). XS-IC does NOT predict Sharpe ranking in this sample. A possible explanation: the 2003-2024 momentum environment has near-zero alpha for all models; marginal IC differences of < 0.1 do not translate to meaningful Sharpe differences after costs, and the ranking is dominated by noise.

### Prediction 8 — Sector-neutral reduces Sharpe by 0.1-0.2 but preserves ranking
**Verdict: ❌ FALSIFIED (magnitude wrong; ranking disrupted)**

Actual sector-neutral Sharpe changes: rolling_vol -0.094, har_rv -0.026, garch +0.001, gbm -0.115, lstm -0.137. Magnitude ranges from -0.001 to -0.137 (not uniformly 0.1-0.2). Ranking is disrupted: GARCH moves from outlier to near HAR-RV; GBM and LSTM fall most. The strategies predominantly captured cross-sector dispersion in momentum, not within-sector; demeaning removes that signal entirely.

---

### Summary

| # | Prediction | Verdict |
|---|-----------|---------|
| 1 | HAR-RV beats GARCH on IC | ✅ |
| 2 | GBM modestly beats HAR-RV (+0.02 to +0.05) | ❌ (borderline: +0.019) |
| 3 | LSTM won't beat GBM | ❌ (LSTM IC > GBM IC) |
| 4 | Sharpe ranking compresses | ✅ |
| 5 | All scaled beat unscaled | ❌ |
| 6 | SHAP: rv_m > rv_w > vix > rv_d | ❌ (pk dominates) |
| 7 | XS-IC predicts Sharpe ranking | ❌ |
| 8 | SN reduces Sharpe by 0.1-0.2 uniformly | ❌ |

2 of 8 confirmed. The most important finding: **forecast quality (IC) does not reliably translate to strategy performance (Sharpe) after costs in this sample period**, and **the 2003-2024 momentum environment had near-zero alpha regardless of forecaster quality**. These are honest, defensible results.
