# ML Volatility Forecasting for Momentum Signal Scaling

A from-scratch quantitative research project comparing ML-based volatility forecasters (GBM, LSTM) against econometric baselines (HAR-RV, GARCH) for momentum signal scaling on S&P 500 constituents (2003–2024).

**Research Question:** *Under what conditions, if any, does ML-based volatility forecasting translate to improved risk-adjusted momentum returns after transaction costs, relative to econometric baselines?*

---

## Key Results

### Forecast Quality (Cross-Sectional IC, 2003–2024)

| Model | Mean XS-IC | DM vs HAR-RV (p) |
|-------|-----------|-----------------|
| LSTM (5-seed ensemble) | **0.739** | < 0.001 |
| RollingVol (6-month) | 0.737 | — |
| GBM (LightGBM panel) | 0.692 | 0.317 |
| HAR-RV (per-stock OLS) | 0.673 | baseline |
| GARCH(1,1)-t (per-stock) | 0.645 | < 0.001 |

LSTM and RollingVol are statistically indistinguishable on forecast quality (DM p=0.132). GBM and HAR-RV are indistinguishable (p=0.317). The Model Confidence Set at α=0.10 retains all five models.

### Strategy Performance (Vol-Scaled Momentum, net 10bps round-trip, 2003–2024)

| Strategy | Sharpe | Max DD |
|----------|--------|--------|
| HAR-RV scaled | 0.020 | -66% |
| GBM scaled | 0.002 | -72% |
| Unscaled momentum | -0.002 | -91% |
| RollingVol scaled | -0.015 | -36% |
| LSTM scaled | -0.041 | -74% |
| GARCH scaled* | 0.226 | -170% |

*GARCH produces a constant forecast broadcast across all dates. The vol-scaling formula amplifies this into extreme weights — the 0.226 Sharpe and -170% max drawdown are artefacts of degenerate scaling, not a signal result.

**Main finding:** Forecast quality does not translate to strategy performance in this sample. LSTM has the highest IC (0.739) but the lowest non-degenerate Sharpe (-0.041). HAR-RV has the lowest IC of the ML-comparable models (0.673) but the highest Sharpe (0.020). The 2003-2024 period is structurally hostile to momentum strategies — net-of-cost Sharpes are near zero for all forecasters. The choice of volatility model matters far less than the sample period.

### SHAP Feature Importance (GBM)

| Feature | SHAP Importance |
|---------|----------------|
| pk (Parkinson range) | **0.380** |
| rv_m (21-day RV) | 0.274 |
| sector | 0.108 |
| vix | 0.082 |
| rv_w (5-day RV) | 0.045 |

Parkinson's range estimator dominates — not rv_m as pre-registered. Intraday high/low range carries cross-sectional dispersion information beyond squared close-to-close returns.

### VIX Regime IC

All models forecast better in high-VIX environments. LSTM is most regime-sensitive (low-VIX IC: 0.709, high-VIX: 0.788). GBM shows the largest uplift in high-VIX years (2007–2009).

---

## Honest Evaluation

8 predictions were pre-registered before Phase 3 training commenced (see `docs/predictions.md`). 2 of 8 were confirmed:

| # | Prediction | Verdict |
|---|-----------|---------|
| 1 | HAR-RV beats GARCH on IC | ✅ |
| 2 | GBM modestly beats HAR-RV (+0.02 to +0.05) | ❌ (borderline: +0.019) |
| 3 | LSTM won't beat GBM | ❌ (LSTM IC > GBM IC) |
| 4 | Sharpe ranking compresses | ✅ |
| 5 | All scaled beat unscaled | ❌ |
| 6 | SHAP: rv_m > rv_w > vix > rv_d | ❌ (pk dominates) |
| 7 | XS-IC predicts Sharpe ranking | ❌ |
| 8 | SN reduces Sharpe by 0.1–0.2 uniformly | ❌ |

All 5 models × all strategy variants are reported in the master results table — no cherry-picking.

---

## Methodology

### Data
- **Universe:** S&P 500 point-in-time membership (Wikipedia change-log; ~80 tickers for tractability)
- **Prices:** yfinance adjusted OHLCV, 2000–2024
- **Target:** Forward 21-day realized variance, modelled in log space
- **Cross-validation:** Expanding window, annual retrain, 42-day embargo (López de Prado §7), 22 OOS windows (2003–2024)

### Models
- **RollingVol:** 6-month rolling realized variance (Barroso & Santa-Clara 2015 baseline)
- **GARCH(1,1)-t:** Per-stock, Student-t innovations, annual refit
- **HAR-RV:** Per-stock OLS on daily/weekly/monthly log-RV components (Corsi 2009)
- **GBM:** Panel LightGBM, sector as categorical feature, no ticker ID (prevents memorisation)
- **LSTM:** 1-layer, hidden=48, Huber loss (δ=1.0 in log-RV space), 5-seed ensemble

### Signal Scaling
Momentum: 12-1 (skip last month). Vol scaling: `weight_i = (target_vol / σ̂_i) * z_score(signal_i)`. Monthly rebalance, 10bps round-trip costs. PIT-conservative: signals use data through close of rebalance date; trades execute at open of next day.

### Forecast Comparison
Diebold-Mariano test with QLIKE loss (Patton 2011). Mincer-Zarnowitz calibration test. Stationary bootstrap (Politis & Romano 1994) for Sharpe difference confidence intervals.

---

## Project Structure

```
ml-vol-momentum/
├── src/
│   ├── data/          # universe (PIT S&P 500), loaders, features, targets
│   ├── models/        # RollingVol, GARCH, HAR-RV, GBM, LSTM
│   ├── strategy/      # momentum, vol-scaling, portfolio, costs
│   ├── eval/          # walk-forward CV, metrics, DM test, IC
│   └── interp/        # SHAP, regime analysis, sector-neutral
├── scripts/           # pipeline runners
├── tests/             # unit, integration, synthetic, replication tests
├── results/           # parquet outputs (forecasts, strategies, tables)
├── docs/
│   ├── figures/       # equity curves, SHAP bar, IC-by-year, regime heatmap
│   └── predictions.md # pre-registered predictions + post-mortem
└── configs/
    └── default.yaml   # all hyperparameters (never retuned per OOS window)
```

---

## Reproduction

**Prerequisites:** Python 3.14, ~8 GB RAM (GBM peak), ~2 h wall-clock (GARCH ~45 min, LSTM ~90 min with per-window checkpointing).

```bash
# 1. Create venv
make venv          # creates /tmp/ml-vol-momentum-venv

# 2. Run full pipeline
make data          # download OHLCV, build features/targets
make baselines     # RollingVol, GARCH, HAR-RV walk-forward
make ml            # GBM + LSTM walk-forward (checkpointed — safe to interrupt)
make analysis      # comparison tables, SHAP, regime, sector-neutral figures

# 3. Tests
make test          # 128 tests; split into 3 invocations (macOS OpenMP/torch isolation)
```

Each step is idempotent: cached results are skipped on rerun.

**Venv note:** The project path contains a colon (`RL:ML Project/`), which breaks Python's venv module. The Makefile defaults to `/tmp/ml-vol-momentum-venv`. Override with `make VENV=/your/path/venv`.

---

## Design Decisions

Every modelling choice is documented in `docs/superpowers/plans/`. Key decisions:

- **Why log-RV target?** RV is approximately log-normal (ABDL 2003); log-transformation makes OLS HAR-RV appropriate and stabilises ML loss surfaces.
- **Why 42-day embargo?** López de Prado: embargo must exceed `2 × max(feature_window, target_horizon) = 2 × 21 = 42` to prevent label overlap.
- **Why Huber loss for LSTM?** Equity RV is fat-tailed; MSE over-weights 2008/2020 tail events. Huber δ=1.0 (≈1 std in log-RV space) bounds gradient on outliers.
- **Why no ticker as GBM feature?** Prevents memorisation of idiosyncratic stock histories including delistings.
- **Why cross-sectional IC?** Vol scaling is a cross-sectional operation — Spearman rank IC directly measures what matters. (Though Prediction 7 shows IC → Sharpe translation was weaker than expected.)

---

## References

- Corsi (2009) — HAR-RV
- Barroso & Santa-Clara (2015) — volatility-managed portfolios
- Gu, Kelly & Xiu (2020) — ML in cross-sectional asset pricing
- López de Prado (2018) — purged walk-forward, embargo
- Patton (2011) — QLIKE loss robustness to noisy proxies
- Hansen, Lunde & Nason (2011) — Model Confidence Set
- Diebold & Mariano (1995) — forecast comparison
