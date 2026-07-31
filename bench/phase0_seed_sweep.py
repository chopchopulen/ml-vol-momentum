"""PHASE 0 — reproducibility gate for the TCN.

Protocol (frozen, documented in bench/BASELINE.md):
  * data     : identical to scripts/run_extension2_fast.py — 80 tickers,
               2000-01-01..2024-12-31, 9 features, target = forward 21d RV.
  * windows  : windows[0:10]  (test years 2003..2012) — the exact sample the
               headline IC 0.769 was computed on (results/forecasts/tcn_partial.parquet).
  * model    : TCNForecaster, SINGLE seed (not the 2-seed ensemble), max_epochs=10,
               patience=3 — the fast-mode config that produced the headline.
  * metric   : mean over dates of the daily cross-sectional Spearman IC between
               forecast_rv and target_rv (src/eval/tests.cross_sectional_ic).

Two modes:
  equiv   — verify that truncating `history` to (test_start - 150 calendar days)
            gives bit-identical test-window predictions to passing full history.
  run N.. — fit/predict for each given seed over the 10 windows, report IC.

Usage: phase0.py equiv | phase0.py run 42 42 0 1 2 ...
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, time, json, platform
os.chdir("/Users/harry/RL:ML Project/ml-vol-momentum"); sys.path.insert(0, os.getcwd())

import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.stats import spearmanr

# fast-mode config patch, exactly as scripts/run_extension2_fast.py does it
from src import config as _cm
_orig = _cm.load_config
def _patched():
    c = _orig(); c["models"]["lstm"].update({"max_epochs": 10, "patience": 3, "seeds": [0, 1]}); return c
_cm.load_config = _patched
import importlib, src.models.tcn_model as _t; importlib.reload(_t)
from src.eval.walk_forward import generate_windows

SP = Path("/private/tmp/claude-501/-Users-harry-RL-ML-Project/94eb5fe3-922a-47a6-a7a5-50f293eaa2b8/scratchpad")
panel = pd.read_parquet(SP / "panel.parquet")
realized = panel["target_rv"]
DATES = panel.index.get_level_values("date")
_NW = int(os.environ.get("PHASE0_NWIN", "10"))
WINDOWS = generate_windows(pd.Timestamp("2000-01-01"), pd.Timestamp("2024-12-31"),
                           first_test_year=2003)[:_NW]
if os.environ.get("PHASE0_THREADS"):
    torch.set_num_threads(int(os.environ["PHASE0_THREADS"]))
_TAG = os.environ.get("PHASE0_TAG", f"w{_NW}")
TRUNC_DAYS = 150   # calendar; > seq_len=60 trading days of lookback


def fit_predict(w, seed, full_history: bool):
    tr = panel[(DATES >= w.train_start) & (DATES <= w.train_end)]
    m = _t.TCNForecaster()
    m.fit(tr, seed=seed)
    if full_history:
        hist = panel[DATES <= w.test_end]
    else:
        lo = w.test_start - pd.Timedelta(days=TRUNC_DAYS)
        hist = panel[(DATES >= lo) & (DATES <= w.test_end)]
    preds = m.predict(hist)
    pd_ = preds.index.get_level_values("date")
    return preds[(pd_ >= w.test_start) & (pd_ <= w.test_end)]


def mean_ic(oos):
    d = pd.concat([oos["forecast_rv"].rename("f"), realized.rename("r")],
                  axis=1, join="inner").dropna()
    vals = {}
    for dt, g in d.groupby(level="date"):
        if len(g) >= 5:
            vals[dt] = spearmanr(g["f"], g["r"])[0]
    s = pd.Series(vals)
    return float(s.mean()), len(s), len(d)


if sys.argv[1] == "equiv":
    w = WINDOWS[0]
    t0 = time.time(); a = fit_predict(w, 42, True);  t1 = time.time()
    b = fit_predict(w, 42, False); t2 = time.time()
    common = a.index.intersection(b.index)
    print(f"full-history : {len(a)} rows, {t1-t0:.0f}s")
    print(f"truncated    : {len(b)} rows, {t2-t1:.0f}s")
    print(f"index equal  : {a.index.equals(b.index)}   common={len(common)}")
    diff = (a.loc[common, "forecast_rv"] - b.loc[common, "forecast_rv"]).abs()
    print(f"max abs diff : {diff.max():.3e}")
    print(f"IC full={mean_ic(a)[0]:.10f}  IC trunc={mean_ic(b)[0]:.10f}")
    sys.exit(0)

seeds = [int(s) for s in sys.argv[2:]]
print(f"platform={platform.platform()} torch={torch.__version__} threads={torch.get_num_threads()}")
print(f"windows={len(WINDOWS)}  test years {WINDOWS[0].test_end.year}..{WINDOWS[-1].test_end.year}")
out_path = SP / f"phase0_results_{_TAG}.jsonl"
for seed in seeds:
    t0 = time.time()
    frames = []
    for i, w in enumerate(WINDOWS):
        tw = time.time()
        frames.append(fit_predict(w, seed, full_history=False))
        print(f"  seed={seed} win{i:02d} ({w.test_end.year}) rows={len(frames[-1])} {time.time()-tw:.0f}s", flush=True)
    oos = pd.concat(frames).sort_index()
    ic, n_dates, n_rows = mean_ic(oos)
    rec = {"tag": _TAG, "n_windows": len(WINDOWS), "seed": seed, "mean_ic": ic,
           "n_dates": n_dates, "n_rows": n_rows, "secs": round(time.time() - t0, 1)}
    print("RESULT " + json.dumps(rec), flush=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
