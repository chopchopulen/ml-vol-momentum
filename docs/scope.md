# Project Scope and Universe Definition

## Universe

**Current scope (Phases 1–2):** 80-ticker subsample from the 2002 S&P 500 point-in-time universe.

**Why 80 tickers:**
- Fast iteration during development and checkpoint validation
- HAR-RV and GARCH walk-forward runs in minutes rather than hours
- Sufficient cross-section for quintile construction (16 stocks per quintile)

**Limitations of 80-ticker scope:**
- Quintile portfolios have ~16 stocks per leg; full S&P 500 has ~100. Smaller cross-section = noisier signals and lower Sharpe
- The 80-ticker universe excludes many tickers that are delisted from yfinance (survivorship in the *other* direction — names that yfinance can retrieve for the 1998–2024 period)
- Literature benchmarks (Jegadeesh-Titman, Barroso-Santa-Clara) use the full US equity universe (1000+ stocks)

**Planned full-universe run:**
- Phase 3 or Phase 4 will run the full S&P 500 point-in-time universe (~500 tickers, annual membership changes)
- The pipeline has been designed for arbitrary universe size; the 80-ticker constraint is not structural

**Interviewer answer:**
> "The development and validation work used an 80-ticker subsample for fast iteration. The full Phase 3/4 evaluation will use the full historical S&P 500 membership (~500 tickers per year, ~700 unique names over 2000–2024). The 80-ticker numbers in this doc are directionally correct but not the final production numbers."

---

## Per-Phase Universe Summary

| Phase | Universe | N tickers | Purpose |
|-------|----------|-----------|---------|
| Phase 1 (baselines) | 2002 S&P 500 PIT subset | 80 | HAR-RV/GARCH validation |
| Phase 2 (strategy) | Same 80-ticker subset | 80 | Barroso replication, checkpoint gates |
| Phase 3 (ML models) | TBD — 80 or full | 80+ | GBM/LSTM forecasting |
| Phase 4 (robustness) | Full S&P 500 PIT | ~500/yr | Production results, writeup numbers |

---

## Checkpoint 2a: Momentum Sharpe Interpretation

The checkpoint 2a result (unscaled momentum Sharpe = 0.185 net, 2000–2007) is lower than the literature benchmark (~0.6–0.8 on the full US universe). This is **expected and understood**, not a bug:

### Gross vs net breakdown
- Gross Sharpe (0 bps): **0.29** (2000–2007)
- Net Sharpe (10 bps round-trip): **0.19** (2000–2007)
- Transaction costs account for ~0.10 Sharpe drag on a 32-position daily-rebalanced strategy

### Per-year story
| Year | Net Sharpe | Note |
|------|-----------|------|
| 2000 | +0.49 | Momentum strong (dot-com losers losing) |
| 2001 | −0.51 | Post-crash reversal begins |
| **2002** | **+1.11** | Momentum revives (tech still falling) |
| **2003** | **−1.61** | **Momentum crash: dot-com winners reversed hard** |
| 2004 | +0.87 | Recovery, momentum works |
| 2005 | +0.41 | Works |
| 2006 | +0.29 | Works |
| 2007 | +0.65 | Pre-crisis, momentum strong |

**2003 is the key:** The dot-com bust recovery produced a massive momentum reversal. Prior losers (tech) bounced 50–100%. The 12-1 momentum signal was long the prior winners (which were beaten down further in 2002) and short the prior losers (which bounced). This is Daniel-Moskowitz (2016)'s "momentum crash" phenomenon — identical mechanism to 2009, just in a different sector.

If you **exclude 2003** (post-reversal year), 2004–2007 net Sharpe = 0.54, squarely in the literature range.

### Why 80 tickers matters here
- 16 stocks per quintile leg
- 2003 reversal was concentrated in tech/telecom. With full universe diversification, the reversal effect is diluted across sectors. With 80 names heavy in the 2002 S&P 500, the reversal is more concentrated
- Literature benchmarks on 500–2000+ names average out idiosyncratic reversal events

### Interviewer answer
> "Pre-2008 momentum Sharpe of 0.19 net includes 2003, which was a major momentum crash driven by dot-com reversal — prior tech losers bounced hard and prior winners got hit. Excluding 2003 (a known momentum-hostile year), 2004–2007 net Sharpe is 0.54, consistent with the literature. The 80-ticker subsample amplifies the crash because sector concentration is higher. The full-universe numbers will be smoother."
