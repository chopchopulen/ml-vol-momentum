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
