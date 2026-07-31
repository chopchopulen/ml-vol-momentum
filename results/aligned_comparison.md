# Aligned comparison — every forecaster on identical rows and dates

Regenerated from cached forecasts by `scripts/aligned_comparison.py`.
No model retrained, no hyperparameter touched.

Newey-West truncation = 42 lags (h = 21, targets overlap 20/21).
`*` marks a baseline added during the audit — not previously in the repo.

## PANEL A — all 10 forecasters incl. TCN

**123,749 rows · 2,236 dates · 56 tickers · 2003-02-12 → 2012-12-31**

| model | XS-IC (levels) | t(NW) | N_eff | XS-IC (changes) | XS-IC (ticker-demeaned) | pooled panel IC | Δ vs rolling_vol | Δ t(NW) | Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| rolling_vol | 0.7754 | 84.6292 | 79 | 0.5365 | 0.5016 | 0.7436 | — | — | 0.2115 |
| transformer_fast | 0.7714 | 90.1653 | 80 | 0.5351 | 0.5111 | 0.7450 | -0.0040 | -0.8992 | 0.2813 |
| tcn_partial | 0.7688 | 93.2408 | 80 | 0.5297 | 0.4963 | 0.7639 | -0.0066 | -1.5061 | 0.2358 |
| prob_lstm | 0.7579 | 77.7464 | 75 | 0.5224 | 0.4980 | 0.7586 | -0.0175 | -3.0388 | 0.2183 |
| lstm | 0.7572 | 82.1935 | 78 | 0.5220 | 0.4899 | 0.7586 | -0.0182 | -3.2689 | 0.2458 |
| mlp_fast | 0.7484 | 80.1489 | 79 | 0.5231 | 0.4852 | 0.7428 | -0.0270 | -4.4434 | 0.3013 |
| ewma_094* | 0.7391 | 71.4468 | 75 | 0.4123 | 0.4584 | 0.7524 | -0.0363 | -9.5711 | 0.2935 |
| gbm | 0.7274 | 79.4569 | 96 | 0.4452 | 0.4467 | 0.7407 | -0.0480 | -11.8038 | 0.2447 |
| har_rv | 0.7091 | 56.3384 | 72 | 0.4034 | 0.4067 | 0.7162 | -0.0663 | -8.6697 | 0.2679 |
| garch | 0.6716 | 44.9448 | 67 | 0.4621 | 0.3258 | 0.5157 | -0.1038 | -9.7634 | 0.3119 |
| const_floor* | 0.6118 | 33.8760 | 61 | 0.4237 | 0.3086 | 0.4946 | -0.1637 | -9.9949 | 0.1550 |

**Momentum control (S0-3):** unrestricted Sharpe -0.0021 over 6,015 days; reindexed to this panel **-0.0752** over 2,236 days.

**Sizing bracket:** oracle (realized forward RV) Sharpe +0.2313 · constant forecast +0.1263


### DM-QLIKE p-values, variance levels, panel A

```
                   garch     gbm  har_rv  lstm  mlp_fast  prob_lstm  rolling_vol  tcn_partial  transformer_fast  ewma_094*  const_floor*
garch             1.0000  0.0000     0.0   0.0       0.0        0.0          0.0          0.0            0.0000     0.0000        0.0556
gbm               0.0000  1.0000     0.0   0.0       0.0        0.0          0.0          0.0            0.7729     0.3524        0.0000
har_rv            0.0000  0.0000     1.0   0.0       0.0        0.0          0.0          0.0            0.0000     0.0000        0.0000
lstm              0.0000  0.0000     0.0   1.0       0.0        0.0          0.0          0.0            0.0000     0.0000        0.0000
mlp_fast          0.0000  0.0000     0.0   0.0       1.0        0.0          0.0          0.0            0.0000     0.0000        0.0000
prob_lstm         0.0000  0.0000     0.0   0.0       0.0        1.0          0.0          0.0            0.0000     0.0000        0.0000
rolling_vol       0.0000  0.0000     0.0   0.0       0.0        0.0          1.0          0.0            0.0000     0.0000        0.0000
tcn_partial       0.0000  0.0000     0.0   0.0       0.0        0.0          0.0          1.0            0.0000     0.0000        0.0000
transformer_fast  0.0000  0.7729     0.0   0.0       0.0        0.0          0.0          0.0            1.0000     0.1613        0.0000
ewma_094*         0.0000  0.3524     0.0   0.0       0.0        0.0          0.0          0.0            0.1613     1.0000        0.0000
const_floor*      0.0556  0.0000     0.0   0.0       0.0        0.0          0.0          0.0            0.0000     0.0000        1.0000
```

## PANEL B — 9 full-period forecasters, TCN absent

**274,632 rows · 4,897 dates · 61 tickers · 2003-02-12 → 2024-11-27**

| model | XS-IC (levels) | t(NW) | N_eff | XS-IC (changes) | XS-IC (ticker-demeaned) | pooled panel IC | Δ vs rolling_vol | Δ t(NW) | Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| transformer_fast | 0.7455 | 113.5752 | 186 | 0.5717 | 0.5066 | 0.7260 | +0.0085 | +2.7989 | -0.0277 |
| lstm | 0.7382 | 112.2620 | 190 | 0.5644 | 0.4960 | 0.7347 | +0.0011 | +0.3136 | -0.0420 |
| prob_lstm | 0.7372 | 108.5056 | 188 | 0.5624 | 0.4970 | 0.7315 | +0.0002 | +0.0456 | -0.0471 |
| rolling_vol | 0.7370 | 102.3951 | 178 | 0.5568 | 0.4903 | 0.7111 | — | — | -0.0253 |
| mlp_fast | 0.7217 | 101.6055 | 183 | 0.5553 | 0.4809 | 0.7226 | -0.0153 | -3.8762 | -0.0495 |
| gbm | 0.6933 | 96.8778 | 201 | 0.4799 | 0.4262 | 0.7098 | -0.0437 | -12.9643 | -0.0498 |
| ewma_094* | 0.6898 | 81.6937 | 174 | 0.4216 | 0.4309 | 0.7046 | -0.0473 | -15.2493 | -0.0018 |
| har_rv | 0.6723 | 79.0885 | 175 | 0.4181 | 0.3857 | 0.6835 | -0.0647 | -15.0208 | 0.0086 |
| garch | 0.6410 | 65.8448 | 152 | 0.4958 | 0.3336 | 0.5092 | -0.0960 | -12.8987 | 0.0861 |
| const_floor* | 0.5505 | 46.2600 | 139 | 0.4185 | 0.1934 | 0.4523 | -0.1865 | -19.8465 | 0.0312 |

**Momentum control (S0-3):** unrestricted Sharpe -0.0021 over 6,015 days; reindexed to this panel **-0.1343** over 4,897 days.

**Sizing bracket:** oracle (realized forward RV) Sharpe +0.1120 · constant forecast +0.0281


### DM-QLIKE p-values, variance levels, panel B

```
                   garch     gbm  har_rv    lstm  mlp_fast  prob_lstm  rolling_vol  transformer_fast  ewma_094*  const_floor*
garch             1.0000  0.0000  0.0000  0.0000    0.0000     0.0000          0.0            0.0000        0.0        0.0004
gbm               0.0000  1.0000  0.4228  0.0000    0.4048     0.0000          0.0            0.7151        0.0        0.0000
har_rv            0.0000  0.4228  1.0000  0.1441    0.7108     0.0025          0.0            0.4356        0.0        0.0000
lstm              0.0000  0.0000  0.1441  1.0000    0.0000     0.0000          0.0            0.0000        0.0        0.0000
mlp_fast          0.0000  0.4048  0.7108  0.0000    1.0000     0.0000          0.0            0.7770        0.0        0.0000
prob_lstm         0.0000  0.0000  0.0025  0.0000    0.0000     1.0000          0.0            0.0000        0.0        0.0000
rolling_vol       0.0000  0.0000  0.0000  0.0000    0.0000     0.0000          1.0            0.0000        0.0        0.0000
transformer_fast  0.0000  0.7151  0.4356  0.0000    0.7770     0.0000          0.0            1.0000        0.0        0.0000
ewma_094*         0.0000  0.0000  0.0000  0.0000    0.0000     0.0000          0.0            0.0000        1.0        0.0000
const_floor*      0.0004  0.0000  0.0000  0.0000    0.0000     0.0000          0.0            0.0000        0.0        1.0000
```
