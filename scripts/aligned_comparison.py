"""Aligned comparison table — every forecaster on identical rows and dates.

Regenerates the headline comparison from CACHED forecasts only. No model is
retrained, no hyperparameter is touched. Fixes applied here:

  S0-1  every forecaster scored on the same (date, ticker) rows
  S0-2  Diebold-Mariano QLIKE on variance levels, not log-variances
  S0-3  the momentum control reindexed to the OOS date set
  #3    cross-sectional IC (per-date, averaged) reported next to the pooled
        panel IC and the ticker-demeaned IC, so the fixed-effect share is visible
  #4    every t-statistic uses effective N under h=21 overlapping targets

Two panels are produced, because "identical rows" has two defensible readings
and they answer different questions:

  PANEL A  all 10 forecasters incl. TCN, on the intersection -> 2003-2012.
           The only sample on which the TCN's 0.769 can be compared to anything.
  PANEL B  the 9 forecasters that cover 2003-2024, on their intersection.
           The full-period comparison, TCN necessarily absent.

Outputs: results/aligned_comparison.md, results/aligned_ic_table.parquet,
         results/aligned_dm_pvalues_panelA.parquet (and _panelB)

Usage: /tmp/mlvm-venv/bin/python scripts/aligned_comparison.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from scipy.stats import spearmanr, rankdata

from src.data.universe import get_universe, get_sector
from src.data.loaders import load_ohlcv, load_vix
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.strategy.momentum import momentum_signal
from src.strategy.scaling import vol_scale
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.eval.metrics import sharpe
from src.eval.comparison import build_dm_matrix

H = 21                 # forecast horizon; consecutive targets share (H-1)/H
NW_LAG = 2 * H         # Newey-West truncation on a DAILY series, >= h

# ── panel ─────────────────────────────────────────────────────────────────
start, end = pd.Timestamp("2000-01-01"), pd.Timestamp("2024-12-31")
tickers = get_universe(pd.Timestamp("2002-01-01"))[:80]
ohlcv = load_ohlcv(tickers, start, end)
vix = load_vix(start, end)

frames = []
for t, grp in ohlcv.groupby(level="ticker"):
    c = grp.droplevel("ticker")["close"]
    r = np.log(c / c.shift(1)); r.name = "return"
    frames.append(r.set_axis(pd.MultiIndex.from_arrays(
        [r.index, [t] * len(r)], names=["date", "ticker"])))
returns_panel = pd.concat(frames).to_frame()

panel = (build_feature_panel(ohlcv, vix)
         .join(forward_rv(returns_panel), how="inner")
         .dropna(subset=["target_log_rv"]))
realized = panel["target_rv"]
rv_m = panel["rv_m"]
signal = momentum_signal(ohlcv[["close"]], lookback=252, skip=21)

# ── cached forecasts + audit-added naive baselines ────────────────────────
models = {p.stem: pd.read_parquet(p)
          for p in sorted(Path("results/forecasts").glob("*.parquet"))}

r_wide = returns_panel["return"].unstack("ticker")
def _stack(df):
    s = df.stack(future_stack=True); s.index.names = ["date", "ticker"]
    return s.dropna()

# EWMA / RiskMetrics is absent from the repo; a referee expects it.
_ew = _stack((r_wide ** 2).fillna(0.0).ewm(alpha=0.06, adjust=False).mean().shift(1) * H)
models["ewma_094*"] = _ew[_ew > 0].rename("forecast_rv").to_frame()
# Per-ticker constant refreshed each fold: the zero-information floor.
_cst = _stack((r_wide ** 2).expanding(252).mean().shift(1) * H)
models["const_floor*"] = _cst[_cst > 0].rename("forecast_rv").to_frame()

# ── metrics ───────────────────────────────────────────────────────────────
def daily_xs_ic(f, r, idx):
    """Per-date cross-sectional Spearman IC. THE quant-standard definition."""
    d = pd.concat([f.rename("f"), r.rename("r")], axis=1, join="inner").dropna()
    d = d[d.index.isin(idx)]
    out = {dt: spearmanr(g["f"], g["r"])[0]
           for dt, g in d.groupby(level="date") if len(g) >= 5}
    return pd.Series(out).dropna().sort_index()

def pooled_ic(f, r, idx):
    """Spearman over the whole (date,ticker) panel at once. Dominated by the
    permanent ticker ordering — reported only to show the contrast."""
    d = pd.concat([f.rename("f"), r.rename("r")], axis=1, join="inner").dropna()
    d = d[d.index.isin(idx)]
    return float(spearmanr(d["f"], d["r"])[0])

def demeaned_xs_ic(f, r, idx):
    """Per-date XS IC after removing each ticker's own mean log level from both
    sides — strips the fixed effect, leaving only timing information."""
    d = pd.concat([np.log(f).rename("f"), np.log(r).rename("r")],
                  axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d.index.isin(idx)]
    g = d.groupby(level="ticker")
    d = d.assign(f=d["f"] - g["f"].transform("mean"), r=d["r"] - g["r"].transform("mean"))
    out = {dt: spearmanr(x["f"], x["r"])[0]
           for dt, x in d.groupby(level="date") if len(x) >= 5}
    return pd.Series(out).dropna().sort_index()

def changes_xs_ic(f, r, base, idx):
    """XS IC on log-vol CHANGES relative to the trailing RV already known at
    forecast time — removes shared persistence from both sides."""
    d = pd.concat([f.rename("f"), r.rename("r"), base.rename("b")],
                  axis=1, join="inner").dropna()
    d = d[d.index.isin(idx)]
    d = d[(d > 0).all(axis=1)]
    d = d.assign(df=np.log(d["f"]) - np.log(d["b"]), dr=np.log(d["r"]) - np.log(d["b"]))
    out = {dt: spearmanr(x["df"], x["dr"])[0]
           for dt, x in d.groupby(level="date") if len(x) >= 5}
    return pd.Series(out).dropna().sort_index()

def nw_mean(series, lag=NW_LAG):
    """Mean, Newey-West SE, t and implied effective N for a daily series whose
    observations overlap by (h-1)/h."""
    v = np.asarray(series.dropna(), dtype=float)
    n = len(v)
    res = sm.OLS(v, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    se_nw = float(res.bse[0])
    se_naive = v.std(ddof=1) / np.sqrt(n)
    n_eff = n * (se_naive / se_nw) ** 2 if se_nw > 0 else np.nan
    return dict(mean=v.mean(), se_nw=se_nw, t=float(res.tvalues[0]),
                p=float(res.pvalues[0]), n_dates=n, n_eff=n_eff)

def paired_nw(a, b, lag=NW_LAG):
    c = a.index.intersection(b.index)
    return nw_mean((a[c] - b[c]).dropna(), lag)

def run_strategy(fc_df, idx):
    fc = fc_df.reindex(idx).dropna(subset=["forecast_rv"])
    w = build_portfolios(None, weights=vol_scale(signal, fc, target_vol=0.10),
                         mode="vol_targeted_gross")
    return apply_costs(w, returns_panel, cost_bps=10.0).dropna()

# ── panels ────────────────────────────────────────────────────────────────
ALL = list(models)
FULL_PERIOD = [m for m in ALL if m != "tcn_partial"]

def intersect(names):
    idx = None
    for n in names:
        idx = models[n].index if idx is None else idx.intersection(models[n].index)
    return idx.intersection(realized.dropna().index)

panels = {"A": ("all 10 forecasters incl. TCN", ALL, intersect(ALL)),
          "B": ("9 full-period forecasters, TCN absent", FULL_PERIOD, intersect(FULL_PERIOD))}

out_md = ["# Aligned comparison — every forecaster on identical rows and dates\n",
          "Regenerated from cached forecasts by `scripts/aligned_comparison.py`.",
          "No model retrained, no hyperparameter touched.\n",
          f"Newey-West truncation = {NW_LAG} lags (h = {H}, targets overlap {H-1}/{H}).",
          "`*` marks a baseline added during the audit — not previously in the repo.\n"]

ic_records, strat_cache = [], {}

for key, (desc, names, idx) in panels.items():
    dates = idx.get_level_values("date")
    hdr = (f"## PANEL {key} — {desc}\n\n"
           f"**{len(idx):,} rows · {dates.nunique():,} dates · "
           f"{idx.get_level_values('ticker').nunique()} tickers · "
           f"{dates.min().date()} → {dates.max().date()}**\n")
    print(hdr)
    out_md.append(hdr)

    rows = []
    base_name = "rolling_vol"
    base_ic = daily_xs_ic(models[base_name]["forecast_rv"], realized, idx)

    for n in names:
        f = models[n]["forecast_rv"]
        ic = daily_xs_ic(f, realized, idx)
        st = nw_mean(ic)
        ch = changes_xs_ic(f, realized, rv_m, idx)
        dm = demeaned_xs_ic(f, realized, idx)
        net = run_strategy(models[n], idx); strat_cache[(key, n)] = net
        d = paired_nw(ic, base_ic) if n != base_name else None
        rows.append({
            "model": n,
            "XS-IC (levels)": st["mean"],
            "t(NW)": st["t"],
            "N_eff": st["n_eff"],
            "XS-IC (changes)": ch.mean(),
            "XS-IC (ticker-demeaned)": dm.mean(),
            "pooled panel IC": pooled_ic(f, realized, idx),
            f"Δ vs {base_name}": np.nan if d is None else d["mean"],
            "Δ t(NW)": np.nan if d is None else d["t"],
            "Sharpe": sharpe(net),
        })
        ic_records.append({"panel": key, **rows[-1]})

    df = pd.DataFrame(rows).sort_values("XS-IC (levels)", ascending=False)
    fmt = df.copy()
    for c in fmt.columns:
        if c == "model":
            continue
        fmt[c] = fmt[c].map(lambda v: "—" if pd.isna(v) else
                            (f"{v:,.0f}" if c == "N_eff" else f"{v:+.4f}"
                             if c.startswith("Δ") else f"{v:.4f}"))
    print(fmt.to_string(index=False))
    out_md += ["| " + " | ".join(fmt.columns) + " |",
               "|" + "---|" * len(fmt.columns)]
    out_md += ["| " + " | ".join(str(v) for v in r) + " |"
               for r in fmt.itertuples(index=False)]

    # momentum control, reindexed to THIS panel's dates (S0-3)
    w_un = build_portfolios(signal, mode="long_short_quintile")
    net_un_raw = apply_costs(w_un, returns_panel, cost_bps=10.0).dropna()
    any_net = strat_cache[(key, names[0])]
    net_un = net_un_raw.reindex(any_net.index).dropna()
    ctl = (f"\n**Momentum control (S0-3):** unrestricted "
           f"Sharpe {sharpe(net_un_raw):+.4f} over {len(net_un_raw):,} days; "
           f"reindexed to this panel **{sharpe(net_un):+.4f}** over {len(net_un):,} days.\n")
    print(ctl); out_md.append(ctl)

    # oracle / floor bracket
    orc = models[names[0]].copy()
    orc["forecast_rv"] = realized.reindex(orc.index)
    orc = orc.dropna(subset=["forecast_rv"])
    cst = models[names[0]].copy()
    cst["forecast_rv"] = float(models[names[0]]["forecast_rv"].median())
    br = (f"**Sizing bracket:** oracle (realized forward RV) "
          f"Sharpe {sharpe(run_strategy(orc, idx)):+.4f} · "
          f"constant forecast {sharpe(run_strategy(cst, idx)):+.4f}\n")
    print(br); out_md.append(br)

    # DM on VARIANCE LEVELS (S0-2)
    stats, pv = build_dm_matrix({n: models[n]["forecast_rv"].reindex(idx).dropna()
                                 for n in names}, realized.reindex(idx))
    pv.to_parquet(f"results/aligned_dm_pvalues_panel{key}.parquet")
    out_md += [f"\n### DM-QLIKE p-values, variance levels, panel {key}\n",
               "```", pv.round(4).to_string(), "```\n"]
    print(f"\nDM p-values (variance levels), panel {key}:\n{pv.round(4).to_string()}\n")

pd.DataFrame(ic_records).to_parquet("results/aligned_ic_table.parquet")
Path("results/aligned_comparison.md").write_text("\n".join(out_md))
print("\nwrote results/aligned_comparison.md, results/aligned_ic_table.parquet")
