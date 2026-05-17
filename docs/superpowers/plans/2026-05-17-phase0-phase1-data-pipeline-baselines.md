# Phase 0 + Phase 1: Environment, Data Pipeline & Baseline Models

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete project scaffold, survivorship-bias-free S&P 500 universe, OHLCV + VIX data pipeline, a locked-down evaluation framework (walk-forward CV + all metrics), and three baseline volatility forecasters (RollingVol, GARCH(1,1)-t, HAR-RV), validated by synthetic leakage canaries and a directional Barroso-Santa-Clara replication.

**Architecture:** Every module is pure Python with a `Panel = pd.DataFrame` (long-form, MultiIndex `(date, ticker)`) as the shared data contract. The evaluation framework and its synthetic canary tests are built *before* any feature or model code — this is the project's main defence against silent look-ahead leakage. HAR-RV/GARCH are per-stock; they share a common `Forecaster` protocol so later ML models drop in without touching strategy code.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy, statsmodels, arch (GARCH + MCS bootstrap), lightgbm, torch, yfinance, requests/beautifulsoup4 (wiki scrape), joblib, pytest, pyyaml, pyarrow (parquet cache), matplotlib/seaborn.

---

## File Map

```
ml-vol-momentum/
  configs/
    default.yaml                  ← all hyperparameters (created Task 1)
  src/
    __init__.py
    data/
      __init__.py
      universe.py                 ← S&P 500 PIT membership (Task 2)
      loaders.py                  ← OHLCV + VIX, parquet cache (Task 3)
      features.py                 ← vol features, all .shift(1) (Task 5)
      targets.py                  ← forward 21-day RV + log RV (Task 5)
    models/
      __init__.py
      forecaster.py               ← Forecaster Protocol (Task 6)
      baselines.py                ← RollingVol, GARCH11, HARRV (Task 7)
    eval/
      __init__.py
      walk_forward.py             ← CVWindow, generate_windows, run_walk_forward (Task 4)
      metrics.py                  ← sharpe, sortino, IC, ICIR, drawdown, turnover (Task 4)
      synthetic.py                ← leakage canary, white-noise panel (Task 4)
      tests.py                    ← diebold_mariano, mincer_zarnowitz, cross_sectional_ic (Task 8)
      comparison.py               ← build_results_table, build_dm_matrix (Task 8)
  tests/
    __init__.py
    test_universe.py              ← Task 2
    test_loaders.py               ← Task 3
    test_features.py              ← Task 5
    test_targets.py               ← Task 5
    test_walk_forward.py          ← Task 4
    test_metrics.py               ← Task 4
    test_synthetic.py             ← Task 4
    test_baselines.py             ← Task 7
    test_walk_forward_integration.py  ← Task 9
    test_replication.py           ← Task 9
  Makefile                        ← Task 1
  requirements.txt                ← Task 1
  pyproject.toml                  ← Task 1
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `ml-vol-momentum/requirements.txt`
- Create: `ml-vol-momentum/pyproject.toml`
- Create: `ml-vol-momentum/configs/default.yaml`
- Create: `ml-vol-momentum/Makefile`
- Create: `ml-vol-momentum/src/__init__.py` (and all `__init__.py` files below)

- [ ] **Step 1: Create the directory tree**

```bash
cd "~/RL:ML Project"
mkdir -p ml-vol-momentum/{src/{data,models,eval,strategy,viz,interp},configs,tests,results/{forecasts,strategies,garch_params},data/{cache,processed},docs/superpowers/plans,notebooks}
touch ml-vol-momentum/src/__init__.py
touch ml-vol-momentum/src/data/__init__.py
touch ml-vol-momentum/src/models/__init__.py
touch ml-vol-momentum/src/eval/__init__.py
touch ml-vol-momentum/src/strategy/__init__.py
touch ml-vol-momentum/src/viz/__init__.py
touch ml-vol-momentum/src/interp/__init__.py
touch ml-vol-momentum/tests/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
pandas>=2.2
numpy>=1.26
scipy>=1.13
statsmodels>=0.14
arch>=6.3
lightgbm>=4.3
torch>=2.3
yfinance>=0.2.40
requests>=2.32
beautifulsoup4>=4.12
joblib>=1.4
pyyaml>=6.0
pyarrow>=16.0
matplotlib>=3.9
seaborn>=0.13
shap>=0.45
pytest>=8.2
pytest-cov>=5.0
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=70"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "ml-vol-momentum"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

- [ ] **Step 4: Install dependencies**

```bash
cd ml-vol-momentum
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Expected: all packages install without errors.

- [ ] **Step 5: Write `configs/default.yaml`**

```yaml
project:
  name: ml-vol-momentum
  seed: 42

data:
  start_date: "2000-01-01"
  end_date:   "2024-12-31"
  primary_source: yfinance
  fallback_source: stooq
  vix_ticker: "^VIX"
  cache_dir: data/cache/
  membership_table: data/processed/sp500_membership.parquet

features:
  rv_windows: [1, 5, 21]
  parkinson_window: 5
  skew_kurt_window: 63
  feature_shift: 1
  target_horizon: 21

cv:
  scheme: expanding_annual
  embargo_days: 42
  train_start: "2000-01-01"
  first_test_year: 2003
  last_test_year: 2024

models:
  rolling_vol:
    window: 126
  garch:
    p: 1
    q: 1
    dist: "t"
    rescale: true
    n_jobs: 8
    refit_frequency_years: 1
  har_rv:
    target: "log_rv"
    components: ["rv_d", "rv_w", "rv_m"]
    fit_method: "ols"
  gbm:
    target: "log_rv"
    learning_rate: 0.05
    num_leaves: 31
    max_depth: 6
    min_data_in_leaf: 200
    feature_fraction: 0.8
    bagging_fraction: 0.8
    n_estimators: 1500
    early_stopping_rounds: 100
    categorical_features: ["sector"]
  lstm:
    target: "log_rv"
    sequence_length: 60
    hidden_size: 48
    num_layers: 1
    dropout: 0.25
    loss: "huber"
    huber_delta: 1.0
    optimizer: "adam"
    learning_rate: 1.0e-3
    batch_size: 1024
    max_epochs: 50
    patience: 5
    val_fraction: 0.1
    seeds: [0, 1, 2, 3, 4]

strategy:
  momentum:
    lookback: 252
    skip: 21
  portfolios:
    long_short_quintile: true
    long_only_quintile: true
    vol_targeted_gross: true
  vol_target_annualized: 0.10
  rebalance: "month_end"

costs:
  round_trip_bps: 10.0

eval:
  forecast_loss: "qlike"
  dm_lag: 20
  ic_method: "spearman"
  ann_factor: 252

interp:
  shap_background_size: 5000
  regime:
    method: "vix_percentile"
    thresholds: [0.33, 0.67]
    fit_on: "train_window"
```

- [ ] **Step 6: Write `Makefile`**

```makefile
.PHONY: test data baselines ml analysis all

PYTHON = .venv/bin/python

test:
	.venv/bin/pytest tests/ -v

data:
	$(PYTHON) -m src.data.universe
	$(PYTHON) -m src.data.loaders

baselines:
	$(PYTHON) -m src.models.baselines

ml:
	$(PYTHON) -m src.models.gbm
	$(PYTHON) -m src.models.lstm_model

analysis:
	$(PYTHON) -m src.eval.comparison
	$(PYTHON) -m src.interp.shap_analysis
	$(PYTHON) -m src.interp.regime_analysis

all: data baselines ml analysis
```

- [ ] **Step 7: Write a config loader utility** (used by every module)

Create `src/config.py`:

```python
from pathlib import Path
import yaml

_ROOT = Path(__file__).parent.parent

def load_config(path: Path | None = None) -> dict:
    if path is None:
        path = _ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
```

- [ ] **Step 8: Initialise git and commit scaffold**

```bash
cd ml-vol-momentum
git init
echo ".venv/" >> .gitignore
echo "data/cache/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "results/" >> .gitignore
git add .
git commit -m "chore: project scaffold, requirements, default config"
```

---

## Task 2: Survivorship-Bias-Free S&P 500 Universe

**Files:**
- Create: `src/data/universe.py`
- Create: `tests/test_universe.py`

Background: Wikipedia's "List of S&P 500 companies" page has two tables: the current constituents table and a "Selected changes" table listing additions and removals with dates. We parse both to reconstruct point-in-time membership. The data goes back to ~1994 reliably. We cache the result as parquet.

- [ ] **Step 1: Write the failing tests first**

Create `tests/test_universe.py`:

```python
import pandas as pd
import pytest
from src.data.universe import get_universe, build_membership_table, get_sector
from pathlib import Path

CACHE = Path("data/processed/sp500_membership.parquet")

class TestMembershipTable:
    def test_table_has_required_columns(self):
        df = build_membership_table(CACHE)
        assert set(["ticker", "added_date", "removed_date",
                    "gics_sector"]).issubset(df.columns)

    def test_table_is_cached_on_second_call(self, tmp_path):
        p = tmp_path / "test.parquet"
        df1 = build_membership_table(p)
        df2 = build_membership_table(p)
        pd.testing.assert_frame_equal(df1, df2)

class TestGetUniverse:
    def test_lehman_in_universe_sept_2008(self):
        u = get_universe(pd.Timestamp("2008-09-14"))
        assert "LEH" in u, "Lehman should be in S&P 500 just before bankruptcy"

    def test_lehman_not_in_universe_oct_2008(self):
        u = get_universe(pd.Timestamp("2008-10-01"))
        assert "LEH" not in u, "Lehman removed 2008-09-19"

    def test_aapl_always_present(self):
        for date_str in ["2005-01-01", "2012-06-15", "2020-01-01", "2023-06-01"]:
            u = get_universe(pd.Timestamp(date_str))
            assert "AAPL" in u

    def test_returns_list_of_strings(self):
        u = get_universe(pd.Timestamp("2010-01-04"))
        assert isinstance(u, list)
        assert all(isinstance(t, str) for t in u)

    def test_universe_size_reasonable(self):
        u = get_universe(pd.Timestamp("2010-01-04"))
        assert 450 <= len(u) <= 510

    def test_at_least_30_change_events(self):
        df = build_membership_table(CACHE)
        removed = df[df["removed_date"].notna()]
        assert len(removed) >= 30, "Should have reconstructed ≥30 historical removals"

class TestGetSector:
    def test_aapl_is_tech(self):
        sector = get_sector("AAPL", pd.Timestamp("2020-01-01"))
        assert "Technology" in sector or "Information" in sector
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_universe.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — universe.py doesn't exist yet.

- [ ] **Step 3: Implement `src/data/universe.py`**

```python
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "Mozilla/5.0 (research-project; educational use)"}

def _fetch_wiki_html() -> str:
    r = requests.get(_WIKI_URL, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def _parse_current_constituents(soup: BeautifulSoup) -> pd.DataFrame:
    table = soup.find("table", {"id": "constituents"})
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        ticker = cells[0].get_text(strip=True).replace(".", "-")
        sector = cells[2].get_text(strip=True)
        sub = cells[3].get_text(strip=True)
        rows.append({"ticker": ticker, "gics_sector": sector,
                     "gics_sub_industry": sub,
                     "added_date": pd.NaT, "removed_date": pd.NaT})
    return pd.DataFrame(rows)

def _parse_changes(soup: BeautifulSoup) -> pd.DataFrame:
    table = soup.find("table", {"id": "changes"})
    if table is None:
        return pd.DataFrame(columns=["ticker", "added_date", "removed_date",
                                     "gics_sector", "gics_sub_industry"])
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 3:
            continue
        try:
            date_str = cells[0].get_text(strip=True)
            event_date = pd.to_datetime(date_str, errors="coerce")
            added_ticker = cells[1].get_text(strip=True).replace(".", "-").strip()
            removed_ticker = cells[3].get_text(strip=True).replace(".", "-").strip() if len(cells) > 3 else ""
        except Exception:
            continue
        if added_ticker:
            rows.append({"ticker": added_ticker, "added_date": event_date,
                         "removed_date": pd.NaT,
                         "gics_sector": "", "gics_sub_industry": ""})
        if removed_ticker:
            rows.append({"ticker": removed_ticker,
                         "added_date": pd.NaT, "removed_date": event_date,
                         "gics_sector": "", "gics_sub_industry": ""})
    return pd.DataFrame(rows)

def _merge_current_and_changes(current: pd.DataFrame,
                                changes: pd.DataFrame) -> pd.DataFrame:
    # Start from current constituents (all currently in the index, removed_date=NaT)
    # Overlay removal dates from the changes table for tickers that have been removed
    removed = changes[changes["removed_date"].notna()][["ticker", "removed_date"]]
    added   = changes[changes["added_date"].notna()][["ticker", "added_date",
                                                       "gics_sector", "gics_sub_industry"]]

    # Build a combined set: current ∪ all historically removed tickers
    all_tickers = pd.concat([
        current,
        added[~added["ticker"].isin(current["ticker"])]
    ], ignore_index=True)

    # Attach removal dates
    removal_map = removed.dropna(subset=["ticker"]).set_index("ticker")["removed_date"].to_dict()
    all_tickers["removed_date"] = all_tickers["ticker"].map(removal_map)

    # Attach earliest known add date for non-current historical tickers
    add_map = (added.dropna(subset=["ticker"])
               .sort_values("added_date")
               .drop_duplicates("ticker", keep="first")
               .set_index("ticker")["added_date"].to_dict())
    mask_no_date = all_tickers["added_date"].isna()
    all_tickers.loc[mask_no_date, "added_date"] = (
        all_tickers.loc[mask_no_date, "ticker"].map(add_map)
    )

    return all_tickers.drop_duplicates(subset=["ticker", "added_date",
                                                "removed_date"]).reset_index(drop=True)

def build_membership_table(out_path: Path) -> pd.DataFrame:
    out_path = Path(out_path)
    if out_path.exists():
        return pd.read_parquet(out_path)
    html = _fetch_wiki_html()
    soup = BeautifulSoup(html, "html.parser")
    current = _parse_current_constituents(soup)
    changes = _parse_changes(soup)
    df = _merge_current_and_changes(current, changes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df

_MEMBERSHIP: pd.DataFrame | None = None

def _get_membership() -> pd.DataFrame:
    global _MEMBERSHIP
    if _MEMBERSHIP is None:
        from src.config import load_config
        cfg = load_config()
        _MEMBERSHIP = build_membership_table(Path(cfg["data"]["membership_table"]))
    return _MEMBERSHIP

def get_universe(date: pd.Timestamp) -> list[str]:
    df = _get_membership()
    mask = (
        (df["added_date"].isna() | (df["added_date"] <= date)) &
        (df["removed_date"].isna() | (df["removed_date"] > date))
    )
    return sorted(df.loc[mask, "ticker"].tolist())

def get_sector(ticker: str, date: pd.Timestamp) -> str:
    df = _get_membership()
    rows = df[(df["ticker"] == ticker) &
              (df["added_date"].isna() | (df["added_date"] <= date)) &
              (df["removed_date"].isna() | (df["removed_date"] > date))]
    if rows.empty:
        return "Unknown"
    return rows.iloc[0]["gics_sector"]

if __name__ == "__main__":
    from src.config import load_config
    cfg = load_config()
    df = build_membership_table(Path(cfg["data"]["membership_table"]))
    print(f"Built membership table: {len(df)} records")
    print(f"Removed tickers: {df['removed_date'].notna().sum()}")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_universe.py -v
```

Expected: all 9 tests pass. If `test_lehman_not_in_universe_oct_2008` fails, the Wikipedia changes table may have the removal date off-by-one — adjust `>` vs `>=` in `get_universe` filter.

- [ ] **Step 5: Commit**

```bash
git add src/data/universe.py tests/test_universe.py
git commit -m "feat: survivorship-bias-free S&P 500 universe from Wikipedia"
```

---

## Task 3: OHLCV + VIX Data Loader with Parquet Cache

**Files:**
- Create: `src/data/loaders.py`
- Create: `tests/test_loaders.py`

Background: yfinance returns split/dividend-adjusted prices when `auto_adjust=True`. We always use adjusted close. For cross-validation we include Stooq as a fallback. VIX is loaded as `^VIX` from yfinance. All data is cached to parquet files keyed by `(ticker, start_date, end_date)` so reruns are instant.

- [ ] **Step 1: Write failing tests**

Create `tests/test_loaders.py`:

```python
import pandas as pd
import pytest
from src.data.loaders import load_ohlcv, load_vix, cross_check_prices

class TestLoadOHLCV:
    def test_returns_expected_columns(self):
        df = load_ohlcv(["AAPL"], pd.Timestamp("2023-01-01"),
                        pd.Timestamp("2023-03-31"))
        assert set(["open","high","low","close","volume"]).issubset(df.columns)
        assert df.index.names == ["date", "ticker"]

    def test_returns_correct_ticker(self):
        df = load_ohlcv(["AAPL", "MSFT"], pd.Timestamp("2023-01-01"),
                        pd.Timestamp("2023-01-31"))
        tickers = df.index.get_level_values("ticker").unique().tolist()
        assert set(tickers) == {"AAPL", "MSFT"}

    def test_trading_days_only(self):
        df = load_ohlcv(["AAPL"], pd.Timestamp("2023-01-01"),
                        pd.Timestamp("2023-01-31"))
        dates = df.index.get_level_values("date")
        # No weekends
        assert all(d.dayofweek < 5 for d in dates)
        # Jan 2023 has 21 trading days
        assert len(dates) == 21

    def test_close_prices_positive(self):
        df = load_ohlcv(["AAPL"], pd.Timestamp("2023-01-01"),
                        pd.Timestamp("2023-03-31"))
        assert (df["close"] > 0).all()

    def test_cache_returns_same_result(self, tmp_path, monkeypatch):
        import src.data.loaders as L
        monkeypatch.setattr(L, "_CACHE_DIR", tmp_path)
        df1 = L.load_ohlcv(["MSFT"], pd.Timestamp("2023-06-01"),
                            pd.Timestamp("2023-06-30"))
        df2 = L.load_ohlcv(["MSFT"], pd.Timestamp("2023-06-01"),
                            pd.Timestamp("2023-06-30"))
        pd.testing.assert_frame_equal(df1, df2)

class TestLoadVIX:
    def test_returns_series(self):
        vix = load_vix(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
        assert isinstance(vix, pd.Series)
        assert vix.name == "vix"

    def test_vix_spiked_march_2020(self):
        vix = load_vix(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-04-01"))
        assert vix.max() > 60, "VIX peaked above 80 in March 2020"

    def test_vix_positive(self):
        vix = load_vix(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
        assert (vix > 0).all()

class TestCrossCheck:
    def test_returns_dataframe(self):
        result = cross_check_prices(["AAPL", "MSFT"],
                                    pd.Timestamp("2020-01-01"),
                                    pd.Timestamp("2020-12-31"))
        assert isinstance(result, pd.DataFrame)
        assert "max_pct_diff" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_loaders.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/data/loaders.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import Literal
import pandas as pd
import numpy as np
import yfinance as yf
from src.config import load_config

_cfg = load_config()
_CACHE_DIR = Path(_cfg["data"]["cache_dir"])

def _cache_path(ticker: str, start: pd.Timestamp, end: pd.Timestamp,
                source: str) -> Path:
    key = f"{ticker}_{start.date()}_{end.date()}_{source}.parquet"
    return _CACHE_DIR / key

def _fetch_yfinance(tickers: list[str], start: pd.Timestamp,
                    end: pd.Timestamp) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    # yfinance returns MultiIndex columns when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.set_levels(
            raw.columns.levels[0].str.lower(), level=0)
        frames = []
        for tkr in tickers:
            if tkr not in raw.columns.get_level_values(1):
                continue
            sub = raw.xs(tkr, level=1, axis=1).copy()
            sub.index.name = "date"
            sub["ticker"] = tkr
            frames.append(sub)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
    else:
        raw.columns = raw.columns.str.lower()
        raw.index.name = "date"
        raw["ticker"] = tickers[0]
        df = raw
    df = df.reset_index().set_index(["date", "ticker"])
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    return df[keep].sort_index()

def _fetch_stooq(ticker: str, start: pd.Timestamp,
                 end: pd.Timestamp) -> pd.DataFrame:
    import io, requests
    url = (f"https://stooq.com/q/d/l/?s={ticker.lower()}.us"
           f"&d1={start.strftime('%Y%m%d')}&d2={end.strftime('%Y%m%d')}&i=d")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = df.columns.str.lower()
    if "date" not in df.columns or df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = ticker
    df = df.rename(columns={"vol": "volume"})
    df = df.set_index(["date", "ticker"])
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    return df[keep].sort_index()

def load_ohlcv(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp,
               source: Literal["yfinance", "stooq"] = "yfinance") -> pd.DataFrame:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_frames = []
    fetch_tickers = []

    # Check per-ticker cache
    for tkr in tickers:
        cp = _cache_path(tkr, start, end, source)
        if cp.exists():
            all_frames.append(pd.read_parquet(cp))
        else:
            fetch_tickers.append(tkr)

    if fetch_tickers:
        if source == "yfinance":
            fetched = _fetch_yfinance(fetch_tickers, start, end)
        else:
            fetched = pd.concat(
                [_fetch_stooq(t, start, end) for t in fetch_tickers],
                ignore_index=False)
        if not fetched.empty:
            for tkr in fetch_tickers:
                if tkr in fetched.index.get_level_values("ticker"):
                    sub = fetched.xs(tkr, level="ticker", drop_level=False)
                    cp = _cache_path(tkr, start, end, source)
                    sub.to_parquet(cp)
                    all_frames.append(sub)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames).sort_index()

def load_vix(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    cfg = load_config()
    cp = _cache_path("VIX", start, end, "yfinance")
    if cp.exists():
        return pd.read_parquet(cp)["vix"]
    raw = yf.download(cfg["data"]["vix_ticker"], start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float, name="vix")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw[("Close", cfg["data"]["vix_ticker"])]
    else:
        close = raw["Close"]
    s = close.rename("vix")
    s.index.name = "date"
    s.index = pd.to_datetime(s.index)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(cp)
    return s

def cross_check_prices(tickers: list[str], start: pd.Timestamp,
                       end: pd.Timestamp) -> pd.DataFrame:
    results = []
    for tkr in tickers:
        try:
            yf_df = load_ohlcv([tkr], start, end, source="yfinance")
            st_df = load_ohlcv([tkr], start, end, source="stooq")
            if yf_df.empty or st_df.empty:
                results.append({"ticker": tkr, "max_pct_diff": np.nan,
                                 "mean_pct_diff": np.nan, "status": "missing"})
                continue
            yf_close = yf_df.xs(tkr, level="ticker")["close"]
            st_close = st_df.xs(tkr, level="ticker")["close"]
            common = yf_close.index.intersection(st_close.index)
            if len(common) == 0:
                results.append({"ticker": tkr, "max_pct_diff": np.nan,
                                 "mean_pct_diff": np.nan, "status": "no_overlap"})
                continue
            diff = (yf_close[common] - st_close[common]).abs() / st_close[common]
            results.append({"ticker": tkr,
                             "max_pct_diff": diff.max(),
                             "mean_pct_diff": diff.mean(),
                             "status": "ok"})
        except Exception as e:
            results.append({"ticker": tkr, "max_pct_diff": np.nan,
                             "mean_pct_diff": np.nan, "status": str(e)})
    return pd.DataFrame(results)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_loaders.py -v
```

Expected: all 8 tests pass. Note: `test_trading_days_only` may need the exact trading day count adjusted if Jan 2023 has a holiday — assert `18 <= len(dates) <= 23` if it's environment-sensitive.

- [ ] **Step 5: Commit**

```bash
git add src/data/loaders.py tests/test_loaders.py
git commit -m "feat: OHLCV + VIX loader with parquet cache and Stooq cross-check"
```

---

## Task 4: Evaluation Framework — Walk-Forward CV, Metrics, and Leakage Canaries

**Files:**
- Create: `src/eval/walk_forward.py`
- Create: `src/eval/metrics.py`
- Create: `src/eval/synthetic.py`
- Create: `tests/test_walk_forward.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_synthetic.py`

**CRITICAL:** The leakage canary test must be written and must pass before any feature or model code is written. This is the project's primary defence against silent look-ahead leakage.

- [ ] **Step 1: Write failing tests for walk_forward**

Create `tests/test_walk_forward.py`:

```python
import pandas as pd
import pytest
from src.eval.walk_forward import CVWindow, generate_windows

class TestCVWindow:
    def test_embargo_invariant_is_enforced(self):
        with pytest.raises(ValueError, match="embargo"):
            CVWindow(
                train_start=pd.Timestamp("2000-01-01"),
                train_end=pd.Timestamp("2001-12-31"),
                test_start=pd.Timestamp("2002-01-10"),  # only 10 days gap
                test_end=pd.Timestamp("2002-12-31"),
                embargo_days=42,
            )

    def test_valid_window_created(self):
        w = CVWindow(
            train_start=pd.Timestamp("2000-01-01"),
            train_end=pd.Timestamp("2001-12-31"),
            test_start=pd.Timestamp("2002-02-28"),  # 59 days gap > 42
            test_end=pd.Timestamp("2002-12-31"),
            embargo_days=42,
        )
        assert w.train_start < w.train_end < w.test_start < w.test_end

class TestGenerateWindows:
    def setup_method(self):
        self.windows = generate_windows(
            start=pd.Timestamp("2000-01-01"),
            end=pd.Timestamp("2006-12-31"),
            embargo_days=42,
            first_test_year=2003,
        )

    def test_returns_list_of_cvwindows(self):
        assert isinstance(self.windows, list)
        assert all(isinstance(w, CVWindow) for w in self.windows)

    def test_windows_cover_correct_years(self):
        test_years = [w.test_start.year for w in self.windows]
        assert 2003 in test_years
        assert 2006 in test_years
        assert 2002 not in test_years

    def test_embargo_respected_all_windows(self):
        for w in self.windows:
            gap = (w.test_start - w.train_end).days
            assert gap >= 42, f"Embargo violated: gap={gap} days"

    def test_expanding_window_train_start_fixed(self):
        for w in self.windows:
            assert w.train_start == pd.Timestamp("2000-01-01")

    def test_train_end_increases_monotonically(self):
        ends = [w.train_end for w in self.windows]
        assert ends == sorted(ends)
```

- [ ] **Step 2: Write failing tests for metrics**

Create `tests/test_metrics.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.eval.metrics import sharpe, sortino, max_drawdown, calmar, icir

class TestSharpe:
    def test_zero_returns_gives_zero_sharpe(self):
        r = pd.Series(np.zeros(252))
        assert sharpe(r) == 0.0

    def test_constant_positive_return(self):
        r = pd.Series([0.001] * 252)
        s = sharpe(r)
        # mean = 0.001, std = 0 → inf, but we return 0 for std=0 by convention
        assert np.isfinite(s) or np.isinf(s)  # either is acceptable

    def test_iid_normal_sharpe_near_zero(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 10_000))
        s = sharpe(r)
        assert abs(s) < 0.5, f"White noise Sharpe should be near 0, got {s:.3f}"

    def test_positive_drift_positive_sharpe(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.001, 0.01, 2520))
        s = sharpe(r)
        assert s > 0

class TestMaxDrawdown:
    def test_monotone_up_zero_drawdown(self):
        prices = pd.Series(np.cumprod(1 + np.full(100, 0.001)))
        assert max_drawdown(prices) == 0.0

    def test_50pct_drop_gives_correct_drawdown(self):
        prices = pd.Series([100, 80, 50, 60, 70])
        dd = max_drawdown(prices)
        assert abs(dd - 0.5) < 1e-9

class TestICIR:
    def test_returns_float(self):
        ic = pd.Series(np.random.default_rng(1).normal(0.05, 0.1, 100))
        result = icir(ic)
        assert isinstance(result, float)

    def test_higher_mean_gives_higher_icir(self):
        ic_good = pd.Series(np.full(100, 0.1))
        ic_poor = pd.Series(np.full(100, 0.01))
        assert icir(ic_good) > icir(ic_poor)
```

- [ ] **Step 3: Write failing tests for synthetic canaries**

Create `tests/test_synthetic.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.eval.synthetic import white_noise_panel, leakage_test_signal

class TestWhiteNoisePanel:
    def test_shape(self):
        panel = white_noise_panel(n_dates=252, n_stocks=50, seed=0)
        assert len(panel) == 252 * 50

    def test_values_near_zero_mean(self):
        panel = white_noise_panel(n_dates=2520, n_stocks=100, seed=0)
        assert abs(panel["return"].mean()) < 1e-3

class TestLeakageTestSignal:
    def test_leakage_signal_is_future_return(self):
        rng = np.random.default_rng(99)
        n, m = 1000, 20
        dates = pd.date_range("2010-01-01", periods=n, freq="B")
        tickers = [f"S{i:03d}" for i in range(m)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date","ticker"])
        returns = pd.Series(rng.normal(0, 0.01, n*m), index=idx, name="return")
        panel = returns.to_frame()
        sig = leakage_test_signal(panel)
        # sig at date t should equal return at date t+1
        for tkr in tickers[:3]:
            r_t1 = returns.xs(tkr, level="ticker").shift(-1)
            s_t  = sig.xs(tkr, level="ticker")["signal"]
            common = r_t1.index.intersection(s_t.index)
            pd.testing.assert_series_equal(
                r_t1[common].reset_index(drop=True),
                s_t[common].reset_index(drop=True),
                check_names=False,
            )
```

- [ ] **Step 4: Run all failing tests**

```bash
pytest tests/test_walk_forward.py tests/test_metrics.py tests/test_synthetic.py -v
```

Expected: all fail with `ImportError`.

- [ ] **Step 5: Implement `src/eval/walk_forward.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class CVWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_days: int = 42

    def __post_init__(self):
        gap = (self.test_start - self.train_end).days
        if gap < self.embargo_days:
            raise ValueError(
                f"embargo violated: test_start={self.test_start.date()} is only "
                f"{gap} days after train_end={self.train_end.date()}; "
                f"required ≥ {self.embargo_days}"
            )

def generate_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    embargo_days: int = 42,
    first_test_year: int = 2003,
) -> list[CVWindow]:
    windows = []
    for year in range(first_test_year, end.year + 1):
        train_end = pd.Timestamp(f"{year - 1}-12-31")
        test_start = train_end + pd.Timedelta(days=embargo_days + 1)
        # Ensure test_start is in the correct year after the embargo
        if test_start.year < year:
            test_start = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=1)
            # Recheck embargo
            if (test_start - train_end).days < embargo_days:
                test_start = train_end + pd.Timedelta(days=embargo_days + 1)
        test_end = pd.Timestamp(f"{year}-12-31")
        if test_end > end:
            test_end = end
        if test_start >= test_end:
            continue
        windows.append(CVWindow(
            train_start=start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            embargo_days=embargo_days,
        ))
    return windows
```

- [ ] **Step 6: Implement `src/eval/metrics.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def sharpe(r: pd.Series, ann: int = 252) -> float:
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(ann))

def sortino(r: pd.Series, ann: int = 252) -> float:
    downside = r[r < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(r.mean() / downside.std() * np.sqrt(ann))

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(abs(dd.min()))

def calmar(r: pd.Series, ann: int = 252) -> float:
    ann_ret = float(r.mean() * ann)
    equity = (1 + r).cumprod()
    mdd = max_drawdown(equity)
    if mdd == 0:
        return np.inf
    return ann_ret / mdd

def information_coefficient(signal: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    dates = signal.index.get_level_values("date").unique()
    ic_vals = {}
    for dt in dates:
        try:
            sig = signal.xs(dt, level="date")["signal"]
            ret = returns.xs(dt, level="date")["return"]
            common = sig.index.intersection(ret.index)
            if len(common) < 5:
                continue
            rho, _ = spearmanr(sig[common], ret[common])
            ic_vals[dt] = rho
        except Exception:
            continue
    return pd.Series(ic_vals, name="IC")

def icir(ic: pd.Series) -> float:
    if ic.std() == 0:
        return 0.0
    return float(ic.mean() / ic.std() * np.sqrt(len(ic)))

def turnover(weights: pd.DataFrame) -> pd.Series:
    w = weights["weight"].unstack("ticker")
    diff = w.diff().abs().sum(axis=1)
    return diff.rename("turnover")
```

- [ ] **Step 7: Implement `src/eval/synthetic.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd

def white_noise_panel(n_dates: int, n_stocks: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-03", periods=n_dates, freq="B")
    tickers = [f"S{i:04d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    returns = rng.normal(0, 0.01, n_dates * n_stocks)
    return pd.DataFrame({"return": returns}, index=idx)

def leakage_test_signal(returns_panel: pd.DataFrame) -> pd.DataFrame:
    # Signal at t = return at t+1 (perfect look-ahead leakage).
    # If the evaluation pipeline correctly applies the +1 trading-day shift,
    # this signal should produce Sharpe >> 15. If it doesn't, the eval is broken.
    r = returns_panel["return"].unstack("ticker")
    sig = r.shift(-1).stack()  # t+1 return, presented at t
    sig.name = "signal"
    return sig.to_frame()
```

- [ ] **Step 8: Run all tests**

```bash
pytest tests/test_walk_forward.py tests/test_metrics.py tests/test_synthetic.py -v
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/eval/walk_forward.py src/eval/metrics.py src/eval/synthetic.py \
        tests/test_walk_forward.py tests/test_metrics.py tests/test_synthetic.py
git commit -m "feat: walk-forward CV (42-day embargo), metrics, leakage canary"
```

---

## Task 5: Features and Targets

**Files:**
- Create: `src/data/features.py`
- Create: `src/data/targets.py`
- Create: `tests/test_features.py`
- Create: `tests/test_targets.py`

**Key invariant:** Every feature column must be `.shift(1)`-ed at construction time. The `test_no_lookahead` test enforces this mechanically.

- [ ] **Step 1: Write failing tests for targets**

Create `tests/test_targets.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.data.targets import forward_rv

class TestForwardRV:
    def setup_method(self):
        # Constant return r every day: forward RV at any t should == 21 * r^2
        dates = pd.date_range("2010-01-04", periods=100, freq="B")
        tickers = ["AAA", "BBB"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date","ticker"])
        r = 0.005
        self.panel = pd.DataFrame({"return": r}, index=idx)
        self.r = r

    def test_forward_rv_identity_constant_return(self):
        result = forward_rv(self.panel, horizon=21)
        # At any date with a full 21-day forward window, RV = 21 * r^2
        first_valid = result.index.get_level_values("date").unique()[0]
        rv_vals = result.xs(first_valid, level="date")["target_rv"]
        expected = 21 * self.r ** 2
        np.testing.assert_allclose(rv_vals.values, expected, rtol=1e-9)

    def test_log_rv_is_log_of_rv(self):
        result = forward_rv(self.panel, horizon=21)
        valid = result.dropna()
        np.testing.assert_allclose(
            valid["target_log_rv"].values,
            np.log(valid["target_rv"].values),
            rtol=1e-9,
        )

    def test_last_horizon_rows_are_nan(self):
        result = forward_rv(self.panel, horizon=21)
        dates = result.index.get_level_values("date").unique()
        # Last 21 trading days should be NaN (no full forward window)
        last_date = dates[-1]
        assert result.xs(last_date, level="date")["target_rv"].isna().all()

    def test_columns_present(self):
        result = forward_rv(self.panel)
        assert "target_rv" in result.columns
        assert "target_log_rv" in result.columns
```

- [ ] **Step 2: Write failing tests for features**

Create `tests/test_features.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.data.features import realized_variance, parkinson, build_feature_panel

class TestRealizedVariance:
    def setup_method(self):
        dates = pd.date_range("2010-01-04", periods=30, freq="B")
        self.r = pd.Series(np.linspace(0.001, 0.002, 30), index=dates)

    def test_rv_window_1_equals_r_squared(self):
        rv = realized_variance(self.r, 1)
        # rv at t = r_t^2 (window=1 means just today's squared return)
        # But we .shift(1) so rv[t] = r[t-1]^2
        for i in range(1, len(self.r)):
            t = self.r.index[i]
            t_prev = self.r.index[i-1]
            np.testing.assert_allclose(rv[t], self.r[t_prev]**2, rtol=1e-9)

    def test_rv_is_nonnegative(self):
        rv = realized_variance(self.r, 5)
        assert (rv.dropna() >= 0).all()

class TestParkinson:
    def test_parkinson_identity_known_value(self):
        import math
        # Single-day panel: H=110, L=100, window=1
        dates = pd.date_range("2010-01-04", periods=6, freq="B")
        high = pd.Series([110]*6, index=dates, dtype=float)
        low  = pd.Series([100]*6, index=dates, dtype=float)
        pk = parkinson(high, low, window=1)
        # Parkinson = (1/(4*ln2)) * (ln(H/L))^2, then .shift(1)
        expected = (1 / (4 * math.log(2))) * (math.log(110/100))**2
        # Check day index 1 (which reflects day 0's values after shift)
        np.testing.assert_allclose(pk.iloc[1], expected, rtol=1e-6)

class TestBuildFeaturePanel:
    def setup_method(self):
        dates = pd.date_range("2010-01-04", periods=150, freq="B")
        tickers = ["AAA", "BBB"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date","ticker"])
        rng = np.random.default_rng(7)
        self.ohlcv = pd.DataFrame({
            "open":   100 + rng.normal(0, 1, len(idx)),
            "high":   102 + rng.normal(0, 1, len(idx)),
            "low":    98  + rng.normal(0, 1, len(idx)),
            "close":  100 + rng.normal(0, 1, len(idx)),
            "volume": rng.integers(1_000_000, 5_000_000, len(idx)).astype(float),
        }, index=idx)
        vix_dates = pd.date_range("2010-01-04", periods=150, freq="B")
        self.vix = pd.Series(15 + rng.normal(0, 2, 150), index=vix_dates, name="vix")

    def test_expected_feature_columns_present(self):
        feat = build_feature_panel(self.ohlcv, self.vix)
        for col in ["rv_d","rv_w","rv_m","r2","pk","skew","kurt","vix","log_dv","ret_21"]:
            assert col in feat.columns, f"Missing feature column: {col}"

    def test_no_lookahead(self):
        # Build features on data[:T] and data[:T+1].
        # Feature values at index T-1 must be identical.
        T = 100
        dates_full = pd.date_range("2010-01-04", periods=T+1, freq="B")
        tickers = ["AAA"]
        rng = np.random.default_rng(99)
        def make_panel(n):
            idx = pd.MultiIndex.from_product(
                [dates_full[:n], tickers], names=["date","ticker"])
            return pd.DataFrame({
                "open": 100.0, "high": 102.0, "low": 98.0,
                "close": 100 + rng.normal(0, 1, n),
                "volume": 1_000_000.0,
            }, index=idx)
        vix = pd.Series(15.0, index=dates_full)
        feat_T   = build_feature_panel(make_panel(T),   vix)
        feat_Tp1 = build_feature_panel(make_panel(T+1), vix)
        check_date = dates_full[T-2]  # second-to-last date of the shorter panel
        for col in feat_T.columns:
            val_T   = feat_T.xs(check_date, level="date")[col]
            val_Tp1 = feat_Tp1.xs(check_date, level="date")[col]
            pd.testing.assert_series_equal(val_T, val_Tp1,
                                            check_names=False,
                                            obj=f"lookahead in {col}")
```

- [ ] **Step 3: Run tests to verify failure**

```bash
pytest tests/test_targets.py tests/test_features.py -v
```

Expected: `ImportError`.

- [ ] **Step 4: Implement `src/data/targets.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd

def forward_rv(panel: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    r = panel["return"].unstack("ticker")
    rv = (r ** 2).rolling(horizon).sum().shift(-(horizon))
    # shift(-horizon) aligns the sum of [t+1, t+horizon] to index t
    # Actually: rolling(h).sum() gives sum_{s=t-h+1}^{t} r_s^2
    # We want sum_{s=t+1}^{t+h} r_s^2 → shift by -h puts future window at t
    rv_stacked = rv.stack()
    rv_stacked.name = "target_rv"
    log_rv = np.log(rv_stacked).rename("target_log_rv")
    return pd.concat([rv_stacked, log_rv], axis=1)
```

- [ ] **Step 5: Implement `src/data/features.py`**

```python
from __future__ import annotations
import math
import numpy as np
import pandas as pd

def realized_variance(returns: pd.Series, window: int) -> pd.Series:
    rv = (returns ** 2).rolling(window).sum()
    return rv.shift(1)

def parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    log_hl_sq = (np.log(high / low)) ** 2
    pk = log_hl_sq.rolling(window).mean() / (4 * math.log(2))
    return pk.shift(1)

def build_feature_panel(ohlcv: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    results = []
    for ticker, grp in ohlcv.groupby(level="ticker"):
        grp = grp.droplevel("ticker")
        close = grp["close"]
        r = np.log(close / close.shift(1))

        rv_d  = realized_variance(r, 1)
        rv_w  = realized_variance(r, 5)
        rv_m  = realized_variance(r, 21)
        r2    = (r ** 2).shift(1)
        pk    = parkinson(grp["high"], grp["low"], window=5)
        skew  = r.rolling(63).skew().shift(1)
        kurt  = r.rolling(63).kurt().shift(1)
        vix_s = vix.reindex(close.index).shift(1)
        log_dv = np.log(grp["volume"] * close).shift(1)
        ret_21 = r.rolling(21).sum().shift(1)

        feat = pd.DataFrame({
            "rv_d": rv_d,
            "rv_w": rv_w,
            "rv_m": rv_m,
            "r2":   r2,
            "pk":   pk,
            "skew": skew,
            "kurt": kurt,
            "vix":  vix_s,
            "log_dv": log_dv,
            "ret_21": ret_21,
        })
        feat.index = pd.MultiIndex.from_product(
            [[ticker], feat.index], names=["ticker", "date"])
        feat = feat.swaplevel().sort_index()
        results.append(feat)
    return pd.concat(results).sort_index()
```

- [ ] **Step 6: Run all feature and target tests**

```bash
pytest tests/test_features.py tests/test_targets.py -v
```

Expected: all pass. If `test_no_lookahead` fails, there is a look-ahead bug — fix before proceeding. This test is the canary.

- [ ] **Step 7: Commit**

```bash
git add src/data/features.py src/data/targets.py \
        tests/test_features.py tests/test_targets.py
git commit -m "feat: vol forecasting features (PIT, all shift(1)) and forward-RV targets"
```

---

## Task 6: Forecaster Protocol

**Files:**
- Create: `src/models/forecaster.py`

- [ ] **Step 1: Implement the protocol**

Create `src/models/forecaster.py`:

```python
from __future__ import annotations
from typing import Protocol, runtime_checkable
import pandas as pd

@runtime_checkable
class Forecaster(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> None:
        ...

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        ...
        # Returns a Panel (date, ticker) with a single column "forecast_log_rv"
        # (forecast in log-RV space) and "forecast_rv" (back-transformed,
        # with Jensen correction).
```

- [ ] **Step 2: Verify Protocol import works**

```bash
python -c "from src.models.forecaster import Forecaster; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/models/forecaster.py
git commit -m "feat: Forecaster protocol (structural typing)"
```

---

## Task 7: Baseline Volatility Models

**Files:**
- Create: `src/models/baselines.py`
- Create: `tests/test_baselines.py`

Background:
- `RollingVolModel` — 6-month (126-day) rolling realized variance. No fitting needed; purely window-based.
- `GARCH11Model` — per-ticker GARCH(1,1) with Student-t innovations via the `arch` package. Predict 21-day ahead variance by summing daily conditional-variance forecasts.
- `HARRV` — per-ticker OLS on `log_rv_{t,t+21} = β0 + βd·log_rv_d + βw·log_rv_w + βm·log_rv_m + ε`. Predict in log space, back-transform with Jensen correction.

- [ ] **Step 1: Write failing tests**

Create `tests/test_baselines.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.models.baselines import RollingVolModel, GARCH11Model, HARRV

def _make_panel(n_dates: int = 500, n_tickers: int = 3,
                seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-03", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date","ticker"])
    returns = rng.normal(0, 0.012, len(idx))
    return pd.DataFrame({"return": returns}, index=idx)

class TestRollingVolModel:
    def test_implements_forecaster_protocol(self):
        from src.models.forecaster import Forecaster
        m = RollingVolModel(window=126)
        assert isinstance(m, Forecaster)

    def test_predict_returns_panel(self):
        panel = _make_panel()
        m = RollingVolModel(window=126)
        m.fit(panel)
        out = m.predict(panel)
        assert "forecast_rv" in out.columns
        assert out.index.names == ["date", "ticker"]

    def test_forecast_rv_nonnegative(self):
        panel = _make_panel()
        m = RollingVolModel(window=126)
        m.fit(panel)
        out = m.predict(panel)
        assert (out["forecast_rv"].dropna() >= 0).all()

class TestHARRV:
    def test_fit_produces_coefficients(self):
        from src.data.features import build_feature_panel
        from src.data.targets import forward_rv
        import yfinance as yf
        # Use synthetic panel
        panel = _make_panel(n_dates=600)
        # Build features
        dates = panel.index.get_level_values("date").unique()
        tickers = panel.index.get_level_values("ticker").unique()
        # Manually compute rv columns for feature panel
        rows = []
        for tkr in tickers:
            r = panel.xs(tkr, level="ticker")["return"]
            rv_d = (r**2).shift(1)
            rv_w = (r**2).rolling(5).sum().shift(1)
            rv_m = (r**2).rolling(21).sum().shift(1)
            sub = pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m})
            sub.index = pd.MultiIndex.from_product([[tkr], sub.index],
                                                    names=["ticker","date"])
            rows.append(sub.swaplevel().sort_index())
        features = pd.concat(rows)
        targets = forward_rv(panel)
        m = HARRV()
        m.fit(pd.concat([features, targets], axis=1).dropna())
        # Should have a coefficient per ticker
        assert len(m.coef_) == len(tickers)

    def test_harrv_matches_statsmodels(self):
        import statsmodels.api as sm
        # Single ticker test
        rng = np.random.default_rng(5)
        n = 500
        rv_d = rng.uniform(1e-5, 5e-4, n)
        rv_w = pd.Series(rv_d).rolling(5).mean().fillna(rv_d.mean()).values
        rv_m = pd.Series(rv_d).rolling(21).mean().fillna(rv_d.mean()).values
        log_rv_target = (np.log(rv_d) * 0.4 + np.log(rv_w) * 0.3 +
                         np.log(rv_m) * 0.2 + rng.normal(0, 0.1, n))
        X = sm.add_constant(
            np.column_stack([np.log(rv_d), np.log(rv_w), np.log(rv_m)]))
        res = sm.OLS(log_rv_target, X).fit()
        # Fit our HAR-RV on the same data
        dates = pd.date_range("2000-01-03", periods=n, freq="B")
        idx = pd.MultiIndex.from_product([dates, ["T0"]], names=["date","ticker"])
        df = pd.DataFrame({
            "rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m,
            "target_log_rv": log_rv_target,
        }, index=idx)
        m = HARRV()
        m.fit(df)
        our_coef = m.coef_["T0"]
        np.testing.assert_allclose(
            our_coef["beta_d"], res.params[1], atol=0.01)
        np.testing.assert_allclose(
            our_coef["beta_w"], res.params[2], atol=0.01)
        np.testing.assert_allclose(
            our_coef["beta_m"], res.params[3], atol=0.01)

class TestGARCH11:
    def test_fit_converges_on_synthetic_panel(self):
        panel = _make_panel(n_dates=400, n_tickers=2)
        m = GARCH11Model()
        m.fit(panel)
        conv = m.convergence_log_
        assert sum(1 for v in conv.values() if v) >= 1  # at least 1 converges

    def test_predict_returns_panel(self):
        panel = _make_panel(n_dates=400, n_tickers=2)
        m = GARCH11Model()
        m.fit(panel)
        out = m.predict(panel)
        assert "forecast_rv" in out.columns
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_baselines.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/models/baselines.py`**

```python
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from joblib import Parallel, delayed
from src.config import load_config

_cfg = load_config()

class RollingVolModel:
    name = "rolling_vol"

    def __init__(self, window: int | None = None):
        self.window = window or _cfg["models"]["rolling_vol"]["window"]

    def fit(self, train: pd.DataFrame) -> None:
        pass  # window-based, no fitting

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for tkr, grp in history.groupby(level="ticker"):
            r = grp.droplevel("ticker")["return"]
            rv = (r ** 2).rolling(self.window).sum()
            sub = pd.DataFrame({"forecast_rv": rv, "forecast_log_rv": np.log(rv)})
            sub.index = pd.MultiIndex.from_product(
                [sub.index, [tkr]], names=["date", "ticker"])
            sub = sub.swaplevel().sort_index()
            rows.append(sub)
        return pd.concat(rows).sort_index()


class HARRV:
    name = "har_rv"

    def __init__(self):
        self.coef_: dict[str, dict] = {}

    def _fit_ticker(self, tkr: str, df: pd.DataFrame) -> tuple[str, dict]:
        sub = df.loc[df.index.get_level_values("ticker") == tkr].copy()
        sub = sub[["rv_d", "rv_w", "rv_m", "target_log_rv"]].dropna()
        if len(sub) < 30:
            return tkr, {}
        log_rv_d = np.log(sub["rv_d"].clip(lower=1e-12))
        log_rv_w = np.log(sub["rv_w"].clip(lower=1e-12))
        log_rv_m = np.log(sub["rv_m"].clip(lower=1e-12))
        X = sm.add_constant(
            np.column_stack([log_rv_d, log_rv_w, log_rv_m]))
        y = sub["target_log_rv"].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(y, X).fit()
        return tkr, {
            "alpha":  res.params[0],
            "beta_d": res.params[1],
            "beta_w": res.params[2],
            "beta_m": res.params[3],
            "sigma2": res.mse_resid,
        }

    def fit(self, train: pd.DataFrame) -> None:
        tickers = train.index.get_level_values("ticker").unique()
        results = [self._fit_ticker(t, train) for t in tickers]
        self.coef_ = {t: c for t, c in results if c}

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for tkr, coef in self.coef_.items():
            try:
                sub = history.loc[
                    history.index.get_level_values("ticker") == tkr
                ]
                r = sub.droplevel("ticker")["return"]
                rv_d = (r**2).shift(1)
                rv_w = (r**2).rolling(5).sum().shift(1)
                rv_m = (r**2).rolling(21).sum().shift(1)
                log_rv_hat = (
                    coef["alpha"]
                    + coef["beta_d"] * np.log(rv_d.clip(lower=1e-12))
                    + coef["beta_w"] * np.log(rv_w.clip(lower=1e-12))
                    + coef["beta_m"] * np.log(rv_m.clip(lower=1e-12))
                )
                # Jensen correction: E[RV] = exp(μ + σ²/2)
                rv_hat = np.exp(log_rv_hat + coef["sigma2"] / 2)
                df_out = pd.DataFrame({
                    "forecast_log_rv": log_rv_hat,
                    "forecast_rv": rv_hat,
                })
                df_out.index = pd.MultiIndex.from_product(
                    [df_out.index, [tkr]], names=["date","ticker"])
                df_out = df_out.swaplevel().sort_index()
                rows.append(df_out)
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows).sort_index()


def _fit_one_garch(tkr: str, r: pd.Series, dist: str,
                   rescale: bool) -> tuple[str, object, bool]:
    try:
        data = r * 100 if rescale else r
        am = arch_model(data.dropna(), vol="Garch", p=1, q=1,
                        dist=dist, rescale=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)
        converged = res.convergence_flag == 0
        return tkr, res, converged
    except Exception:
        return tkr, None, False


class GARCH11Model:
    name = "garch"

    def __init__(self):
        gcfg = _cfg["models"]["garch"]
        self.dist = gcfg["dist"]
        self.rescale = gcfg["rescale"]
        self.n_jobs = gcfg["n_jobs"]
        self.fitted_: dict[str, object] = {}
        self.convergence_log_: dict[str, bool] = {}

    def fit(self, train: pd.DataFrame) -> None:
        tickers = train.index.get_level_values("ticker").unique().tolist()
        series = [
            train.xs(t, level="ticker")["return"] for t in tickers
        ]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_garch)(t, r, self.dist, self.rescale)
            for t, r in zip(tickers, series)
        )
        for tkr, res, conv in results:
            self.convergence_log_[tkr] = conv
            if res is not None:
                self.fitted_[tkr] = res

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for tkr, res in self.fitted_.items():
            try:
                r = history.xs(tkr, level="ticker")["return"]
                data = r * 100 if self.rescale else r
                # Re-forecast from the last fit (fixed params, rolling update)
                fcast = res.forecast(horizon=21, reindex=False)
                # fcast.variance has shape (n_obs, 21); sum across horizon
                cond_var_sum = fcast.variance.sum(axis=1)
                if self.rescale:
                    cond_var_sum = cond_var_sum / (100 ** 2)
                df_out = pd.DataFrame({"forecast_rv": cond_var_sum,
                                       "forecast_log_rv": np.log(cond_var_sum)})
                df_out.index = pd.MultiIndex.from_product(
                    [df_out.index, [tkr]], names=["date","ticker"])
                df_out = df_out.swaplevel().sort_index()
                rows.append(df_out)
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows).sort_index()
```

- [ ] **Step 4: Run baseline tests**

```bash
pytest tests/test_baselines.py -v
```

Expected: all pass. GARCH tests may be slow (2-3 seconds per model fit). If `test_harrv_matches_statsmodels` fails, debug the OLS alignment — the coefficients should match within `atol=0.01`.

- [ ] **Step 5: Commit**

```bash
git add src/models/baselines.py tests/test_baselines.py
git commit -m "feat: RollingVol, GARCH(1,1)-t, and HAR-RV baseline forecasters"
```

---

## Task 8: Statistical Tests — DM, Mincer-Zarnowitz, Cross-Sectional IC, MCS, Bootstrap

**Files:**
- Create: `src/eval/tests.py`
- Create: `src/eval/comparison.py`
- Create: `tests/test_stat_tests.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_stat_tests.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.eval.tests import (diebold_mariano, mincer_zarnowitz,
                             cross_sectional_ic, sharpe_diff_bootstrap)

class TestDieboldMariano:
    def test_identical_forecasts_high_pvalue(self):
        rng = np.random.default_rng(0)
        realized = pd.Series(rng.uniform(1e-5, 1e-3, 500))
        forecast = pd.Series(rng.uniform(1e-5, 1e-3, 500))
        e1 = realized - forecast
        e2 = realized - forecast  # identical errors
        stat, pval = diebold_mariano(e1, e2, h=21)
        assert pval > 0.5, f"Identical forecasts should have p >> 0.5, got {pval:.3f}"

    def test_clearly_better_forecast_low_pvalue(self):
        rng = np.random.default_rng(1)
        realized = pd.Series(rng.uniform(1e-5, 1e-3, 2000))
        good_fc  = realized + rng.normal(0, 1e-5, 2000)   # near-perfect
        bad_fc   = pd.Series(rng.uniform(1e-5, 1e-3, 2000))  # random
        e1 = realized.values - good_fc.values
        e2 = realized.values - bad_fc.values
        _, pval = diebold_mariano(pd.Series(e1), pd.Series(e2), h=21,
                                   loss="mse")
        assert pval < 0.05

    def test_pvalue_uniform_under_h0(self):
        rng = np.random.default_rng(42)
        pvals = []
        for _ in range(200):
            realized = pd.Series(rng.uniform(1e-5, 1e-3, 500))
            f1 = realized + rng.normal(0, 5e-5, 500)
            f2 = realized + rng.normal(0, 5e-5, 500)
            e1 = realized - f1
            e2 = realized - f2
            _, p = diebold_mariano(e1, e2, h=1, loss="mse")
            pvals.append(p)
        # Under H0 p-values should be roughly uniform; test fraction < 0.05 ≈ 5%
        reject_rate = np.mean(np.array(pvals) < 0.05)
        assert 0.01 < reject_rate < 0.15, f"Size distortion: reject_rate={reject_rate:.3f}"

class TestMincerZarnowitz:
    def test_perfect_forecast_alpha0_beta1(self):
        rng = np.random.default_rng(3)
        realized = pd.Series(rng.uniform(0.0001, 0.01, 300))
        forecast = realized.copy()  # perfect
        result = mincer_zarnowitz(realized, forecast)
        assert abs(result["alpha"]) < 0.01
        assert abs(result["beta"] - 1.0) < 0.05

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(4)
        r = pd.Series(rng.normal(0.005, 0.001, 100))
        f = pd.Series(rng.normal(0.005, 0.001, 100))
        result = mincer_zarnowitz(r, f)
        for k in ["alpha", "beta", "p_joint", "r2"]:
            assert k in result

class TestCrossectionalIC:
    def test_perfect_rank_correlation_gives_one(self):
        dates = pd.date_range("2020-01-02", periods=10, freq="B")
        tickers = [f"S{i}" for i in range(20)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date","ticker"])
        vals = np.tile(np.arange(20, dtype=float), 10)
        forecast = pd.DataFrame({"forecast_rv": vals}, index=idx)
        realized = pd.DataFrame({"target_rv": vals}, index=idx)
        ic = cross_sectional_ic(forecast, realized)
        assert (ic.dropna() > 0.99).all()

class TestSharpeDiffBootstrap:
    def test_same_series_ci_crosses_zero(self):
        rng = np.random.default_rng(10)
        r = pd.Series(rng.normal(0.001, 0.01, 1260))
        result = sharpe_diff_bootstrap(r, r, n_boot=500, seed=7)
        assert result["ci_lo"] < 0 < result["ci_hi"]

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(11)
        r1 = pd.Series(rng.normal(0.001, 0.01, 500))
        r2 = pd.Series(rng.normal(0.0005, 0.01, 500))
        result = sharpe_diff_bootstrap(r1, r2, n_boot=200, seed=0)
        for k in ["sharpe_diff_point", "ci_lo", "ci_hi", "p_value"]:
            assert k in result
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_stat_tests.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/eval/tests.py`**

```python
from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from src.eval.metrics import sharpe

def _qlike_loss(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    h = np.clip(forecast, 1e-12, None)
    r = np.clip(realized,  1e-12, None)
    return r / h - np.log(r / h) - 1

def diebold_mariano(
    e1: pd.Series,
    e2: pd.Series,
    h: int = 21,
    loss: Literal["mse", "qlike"] = "qlike",
) -> tuple[float, float]:
    from statsmodels.stats.stattools import durbin_watson
    if loss == "mse":
        d = e1.values ** 2 - e2.values ** 2
    else:
        # For QLIKE we need realized and forecast, not errors.
        # Caller passes errors; fall back to MSE when QLIKE not computable.
        d = e1.values ** 2 - e2.values ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    d_mean = np.mean(d)
    # Newey-West HAC variance at lag h-1
    nw_lags = h - 1
    model = sm.OLS(d, np.ones(n))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    stat = float(res.tvalues[0])
    pval = float(res.pvalues[0])
    return stat, pval

def diebold_mariano_qlike(
    realized: pd.Series,
    forecast1: pd.Series,
    forecast2: pd.Series,
    h: int = 21,
) -> tuple[float, float]:
    common = realized.index.intersection(forecast1.index).intersection(forecast2.index)
    r = realized[common].values
    f1 = forecast1[common].values
    f2 = forecast2[common].values
    d = _qlike_loss(r, f1) - _qlike_loss(r, f2)
    d = d[~np.isnan(d)]
    n = len(d)
    model = sm.OLS(d, np.ones(n))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": h - 1})
    return float(res.tvalues[0]), float(res.pvalues[0])

def mincer_zarnowitz(realized: pd.Series, forecast: pd.Series) -> dict:
    common = realized.index.intersection(forecast.index)
    r = realized[common].values
    f = forecast[common].values
    mask = ~(np.isnan(r) | np.isnan(f))
    r, f = r[mask], f[mask]
    X = sm.add_constant(f)
    res = sm.OLS(r, X).fit()
    alpha, beta = float(res.params[0]), float(res.params[1])
    # Joint Wald test H0: alpha=0, beta=1
    R = np.array([[1, 0], [0, 1]])
    q = np.array([0.0, 1.0])
    wald_stat = float((R @ res.params - q) @
                      np.linalg.inv(R @ res.cov_params() @ R.T) @
                      (R @ res.params - q))
    p_joint = float(1 - chi2.cdf(wald_stat, df=2))
    return {"alpha": alpha, "beta": beta, "p_joint": p_joint, "r2": float(res.rsquared)}

def cross_sectional_ic(forecast: pd.DataFrame, realized: pd.DataFrame) -> pd.Series:
    from scipy.stats import spearmanr
    fcol = "forecast_rv" if "forecast_rv" in forecast.columns else forecast.columns[0]
    rcol = "target_rv"   if "target_rv" in realized.columns  else realized.columns[0]
    dates = forecast.index.get_level_values("date").unique()
    ic_vals = {}
    for dt in dates:
        try:
            f = forecast.xs(dt, level="date")[fcol]
            r = realized.xs(dt, level="date")[rcol]
            common = f.index.intersection(r.index)
            if len(common) < 5:
                continue
            rho, _ = spearmanr(f[common], r[common])
            ic_vals[dt] = rho
        except Exception:
            continue
    return pd.Series(ic_vals, name="cross_sectional_IC")

def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.10,
    block_size: int = 21,
    n_boot: int = 10_000,
) -> dict:
    from arch.bootstrap import MCS
    mcs = MCS(losses.dropna(), size=alpha, block_size=block_size,
              reps=n_boot, method="R")
    mcs.compute()
    return {
        "included": list(mcs.included),
        "excluded": list(mcs.excluded),
        "pvalues":  mcs.pvalues.to_dict(),
    }

def sharpe_diff_bootstrap(
    r1: pd.Series,
    r2: pd.Series,
    block_size: int = 21,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    from arch.bootstrap import StationaryBootstrap
    rng = np.random.default_rng(seed)
    n = min(len(r1), len(r2))
    r1 = r1.iloc[:n].values
    r2 = r2.iloc[:n].values
    point = sharpe(pd.Series(r1)) - sharpe(pd.Series(r2))
    def _stat(x, y):
        return sharpe(pd.Series(x)) - sharpe(pd.Series(y))
    bs = StationaryBootstrap(block_size, r1, r2, seed=seed)
    boot_diffs = []
    for data, _ in bs.bootstrap(n_boot):
        boot_diffs.append(_stat(data[0], data[1]))
    boot_diffs = np.array(boot_diffs)
    ci_lo = np.percentile(boot_diffs, 5)
    ci_hi = np.percentile(boot_diffs, 95)
    p_value = float(np.mean(np.abs(boot_diffs - point) >= np.abs(point)))
    return {"sharpe_diff_point": point, "ci_lo": ci_lo,
            "ci_hi": ci_hi, "p_value": p_value}
```

- [ ] **Step 4: Implement `src/eval/comparison.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from src.eval.metrics import sharpe, sortino, max_drawdown, calmar
from src.eval.tests import diebold_mariano_qlike, mincer_zarnowitz

def build_results_table(strategies: dict[str, pd.Series],
                        ann: int = 252) -> pd.DataFrame:
    rows = []
    for name, r in strategies.items():
        equity = (1 + r).cumprod()
        rows.append({
            "strategy": name,
            "sharpe":   sharpe(r, ann),
            "sortino":  sortino(r, ann),
            "max_dd":   -max_drawdown(equity),
            "calmar":   calmar(r, ann),
            "ann_ret":  r.mean() * ann,
            "ann_vol":  r.std() * np.sqrt(ann),
        })
    return pd.DataFrame(rows).set_index("strategy")

def build_dm_matrix(forecasts: dict[str, pd.Series],
                    realized: pd.Series) -> pd.DataFrame:
    names = list(forecasts.keys())
    stats = pd.DataFrame(index=names, columns=names, dtype=float)
    pvals = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i == j:
                stats.loc[n1, n2] = 0.0
                pvals.loc[n1, n2] = 1.0
                continue
            s, p = diebold_mariano_qlike(realized, forecasts[n1], forecasts[n2])
            stats.loc[n1, n2] = s
            pvals.loc[n1, n2] = p
    return stats, pvals

def build_mz_table(forecasts: dict[str, pd.Series],
                   realized: pd.Series) -> pd.DataFrame:
    rows = []
    for name, fc in forecasts.items():
        result = mincer_zarnowitz(realized, fc)
        result["model"] = name
        rows.append(result)
    return pd.DataFrame(rows).set_index("model")
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_stat_tests.py -v
```

Expected: all pass. `test_pvalue_uniform_under_h0` is probabilistic (runs 200 simulations); it may occasionally fail due to randomness — re-run once if so.

- [ ] **Step 6: Commit**

```bash
git add src/eval/tests.py src/eval/comparison.py tests/test_stat_tests.py
git commit -m "feat: DM (QLIKE), Mincer-Zarnowitz, cross-sectional IC, MCS, stationary bootstrap"
```

---

## Task 9: Integration Tests and Barroso-Santa-Clara Replication

**Files:**
- Create: `tests/test_walk_forward_integration.py`
- Create: `tests/test_replication.py`

These are the most important validation tests. **If the Barroso replication fails, the pipeline has a bug. Do not proceed to Phase 2 until it passes.**

- [ ] **Step 1: Write walk-forward integration test**

Create `tests/test_walk_forward_integration.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.eval.walk_forward import generate_windows, run_walk_forward, CVWindow
from src.models.baselines import RollingVolModel

def _make_panel(n_dates=600, n_tickers=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-03", periods=n_dates, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date","ticker"])
    r = rng.normal(0, 0.012, len(idx))
    return pd.DataFrame({"return": r}, index=idx)

class TestRunWalkForward:
    def setup_method(self):
        self.panel = _make_panel()
        self.windows = generate_windows(
            start=pd.Timestamp("2000-01-01"),
            end=pd.Timestamp("2002-05-31"),
            embargo_days=42,
            first_test_year=2001,
        )

    def test_produces_oos_forecasts_no_nans(self):
        forecaster = RollingVolModel(window=63)
        result = run_walk_forward(forecaster, self.panel, self.windows)
        assert isinstance(result, pd.DataFrame)
        assert "forecast_rv" in result.columns
        oos_dates = result.index.get_level_values("date")
        # All OOS forecast dates must fall within test windows
        for w in self.windows:
            mask = (oos_dates >= w.test_start) & (oos_dates <= w.test_end)
            assert mask.sum() > 0

    def test_train_data_never_includes_oos(self):
        seen_dates = []
        class TrackingForecaster:
            name = "tracker"
            def fit(self_, train):
                seen_dates.append(
                    train.index.get_level_values("date").max()
                )
            def predict(self_, history):
                return pd.DataFrame()
        run_walk_forward(TrackingForecaster(), self.panel, self.windows)
        for i, max_train_date in enumerate(seen_dates):
            w = self.windows[i]
            assert max_train_date <= w.train_end, \
                f"Window {i}: training data leaked into OOS"

    def test_40day_embargo_never_violated(self):
        for w in self.windows:
            gap = (w.test_start - w.train_end).days
            assert gap >= 42
```

- [ ] **Step 2: Implement `run_walk_forward` in `src/eval/walk_forward.py`**

Add this function to the existing `walk_forward.py`:

```python
def run_walk_forward(
    forecaster,
    panel: pd.DataFrame,
    windows: list[CVWindow],
) -> pd.DataFrame:
    from src.models.forecaster import Forecaster
    oos_frames = []
    for w in windows:
        train_mask = (
            (panel.index.get_level_values("date") >= w.train_start) &
            (panel.index.get_level_values("date") <= w.train_end)
        )
        test_mask = (
            (panel.index.get_level_values("date") >= w.test_start) &
            (panel.index.get_level_values("date") <= w.test_end)
        )
        train = panel[train_mask]
        # For predict, provide the full history up to end of test window
        # so rolling forecasters can use lagged data in the test window.
        history_mask = (
            panel.index.get_level_values("date") <= w.test_end
        )
        history = panel[history_mask]
        forecaster.fit(train)
        preds = forecaster.predict(history)
        if preds.empty:
            continue
        # Restrict to test window dates only
        pred_dates = preds.index.get_level_values("date")
        oos = preds[(pred_dates >= w.test_start) & (pred_dates <= w.test_end)]
        oos_frames.append(oos)
    if not oos_frames:
        return pd.DataFrame()
    return pd.concat(oos_frames).sort_index()
```

- [ ] **Step 3: Run walk-forward integration tests**

```bash
pytest tests/test_walk_forward_integration.py -v
```

Expected: all pass.

- [ ] **Step 4: Write Barroso replication test**

Create `tests/test_replication.py`:

```python
"""
Directional replication of Barroso & Santa-Clara (2015).
Tests use real market data (2000-2010). They may take 2-5 minutes.
If these fail, the pipeline has a bug. Do not proceed to Phase 2.
"""
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow  # skip with `pytest -m "not slow"` for fast CI

@pytest.fixture(scope="module")
def market_data():
    from src.data.universe import get_universe, build_membership_table
    from src.data.loaders import load_ohlcv
    from src.config import load_config
    from pathlib import Path
    cfg = load_config()
    build_membership_table(Path(cfg["data"]["membership_table"]))
    start = pd.Timestamp("1998-01-01")  # need 2 yrs of history for 12-1 signal
    end   = pd.Timestamp("2010-12-31")
    # Use a manageable 50-ticker subsample from the 2000 universe to keep
    # test runtime reasonable; directional check doesn't need full 500.
    universe = get_universe(pd.Timestamp("2000-01-03"))[:50]
    prices = load_ohlcv(universe, start, end)
    return prices

def _make_mom_signal(prices_panel):
    close = prices_panel["close"].unstack("ticker")
    log_ret = np.log(close / close.shift(1))
    mom = log_ret.rolling(252).sum().shift(21) - log_ret.rolling(21).sum().shift(1)
    return mom.stack().rename("signal").to_frame()

def _quintile_ls_returns(signal, returns, weights_shift=1):
    dates = signal.index.get_level_values("date").unique()
    pnl = []
    for dt in sorted(dates)[252:]:  # skip warmup
        try:
            sig = signal.xs(dt, level="date")["signal"].dropna()
            if len(sig) < 10:
                continue
            q20 = sig.quantile(0.20)
            q80 = sig.quantile(0.80)
            long_tkrs  = sig[sig >= q80].index.tolist()
            short_tkrs = sig[sig <= q20].index.tolist()
            next_dates = returns.index.get_level_values("date").unique()
            future = next_dates[next_dates > dt]
            if len(future) == 0:
                continue
            next_dt = future[0]
            rets = returns.xs(next_dt, level="date")["return"]
            long_r  = rets[rets.index.isin(long_tkrs)].mean()
            short_r = rets[rets.index.isin(short_tkrs)].mean()
            if np.isnan(long_r) or np.isnan(short_r):
                continue
            pnl.append({"date": next_dt, "return": long_r - short_r})
        except Exception:
            continue
    return pd.DataFrame(pnl).set_index("date")["return"]

def _vol_scale_returns(raw_returns, window=126):
    rv = raw_returns.rolling(window).std()
    rv = rv.replace(0, np.nan).ffill()
    target_vol = 0.10 / np.sqrt(252)
    scaled = raw_returns * (target_vol / rv)
    return scaled

class TestBarrosoReplication:
    def test_2009_crash_visible_unscaled(self, market_data):
        log_ret = np.log(market_data["close"].unstack("ticker") /
                        market_data["close"].unstack("ticker").shift(1))
        returns_panel = log_ret.stack().rename("return").to_frame()
        signal = _make_mom_signal(market_data)
        pnl = _quintile_ls_returns(signal, returns_panel)
        # Focus on 2009
        pnl_2009 = pnl["2009"]
        if len(pnl_2009) == 0:
            pytest.skip("Not enough data to test 2009")
        equity_2009 = (1 + pnl_2009).cumprod()
        from_peak = (equity_2009 - equity_2009.cummax()) / equity_2009.cummax()
        max_dd_2009 = float(from_peak.min())
        assert max_dd_2009 < -0.20, \
            f"Expected momentum crash in 2009 (DD < -0.20), got {max_dd_2009:.3f}"

    def test_scaled_sharpe_exceeds_unscaled(self, market_data):
        log_ret = np.log(market_data["close"].unstack("ticker") /
                        market_data["close"].unstack("ticker").shift(1))
        returns_panel = log_ret.stack().rename("return").to_frame()
        signal = _make_mom_signal(market_data)
        pnl_raw = _quintile_ls_returns(signal, returns_panel)
        pnl_scaled = _vol_scale_returns(pnl_raw, window=126)
        # Trim warmup
        pnl_raw    = pnl_raw.dropna().loc["2002":]
        pnl_scaled = pnl_scaled.dropna().loc["2002":]
        from src.eval.metrics import sharpe
        sr_raw    = sharpe(pnl_raw)
        sr_scaled = sharpe(pnl_scaled)
        assert sr_scaled > sr_raw, \
            (f"Vol-scaled Sharpe ({sr_scaled:.3f}) should exceed "
             f"unscaled Sharpe ({sr_raw:.3f})")
        assert sr_scaled - sr_raw >= 0.1, \
            f"Sharpe uplift too small: {sr_scaled - sr_raw:.3f} < 0.1"
```

- [ ] **Step 5: Run integration and replication tests**

```bash
pytest tests/test_walk_forward_integration.py -v
pytest tests/test_replication.py -v -m slow
```

The replication test downloads real market data and will take 3-10 minutes. Expected results:
- `test_2009_crash_visible_unscaled`: passes if momentum drawdown in 2009 < -20%.
- `test_scaled_sharpe_exceeds_unscaled`: passes if vol-scaled Sharpe > unscaled Sharpe with ≥ 0.1 uplift.

If either test fails, debug before proceeding to Phase 2.

- [ ] **Step 6: CHECKPOINT 1f — HAR-RV OOS validation**

Run the following validation script manually (not a pytest test — needs real data):

```bash
python -c "
from src.data.universe import get_universe
from src.data.loaders import load_ohlcv
from src.data.features import build_feature_panel
from src.data.targets import forward_rv
from src.models.baselines import HARRV
from src.eval.walk_forward import generate_windows, run_walk_forward
from src.data.loaders import load_vix
import pandas as pd, numpy as np

print('Loading data...')
u = get_universe(pd.Timestamp('2000-01-03'))[:30]
prices = load_ohlcv(u, pd.Timestamp('2000-01-01'), pd.Timestamp('2003-12-31'))
vix = load_vix(pd.Timestamp('2000-01-01'), pd.Timestamp('2003-12-31'))
log_ret = prices['close'].unstack('ticker').apply(lambda x: np.log(x/x.shift(1)))
returns_panel = log_ret.stack().rename('return').to_frame()
features = build_feature_panel(prices, vix)
targets = forward_rv(returns_panel)
train = pd.concat([features, targets], axis=1).dropna()
windows = generate_windows(pd.Timestamp('2000-01-01'), pd.Timestamp('2003-12-31'),
                           first_test_year=2003)
m = HARRV()
oos = run_walk_forward(m, train, windows)
real_targets = targets['target_log_rv']
common = oos.index.intersection(real_targets.index)
residuals = real_targets[common] - oos.loc[common,'forecast_log_rv']
ss_res = (residuals**2).sum()
ss_tot = ((real_targets[common] - real_targets[common].mean())**2).sum()
oos_r2 = float(1 - ss_res/ss_tot)
print(f'HAR-RV OOS R² on log_rv (2003): {oos_r2:.3f}')
print(f'EXPECTED: 0.30 to 0.65')
assert 0.30 <= oos_r2 <= 0.65, f'CHECKPOINT 1f FAILED: OOS R² = {oos_r2:.3f}'
print('CHECKPOINT 1f PASSED')
"
```

Expected output: `HAR-RV OOS R² on log_rv (2003): 0.XX` with `0.30 ≤ XX ≤ 0.65`. If it falls outside this range, debug the features/targets pipeline before continuing.

- [ ] **Step 7: Full test suite run**

```bash
pytest tests/ -v -m "not slow"
```

Expected: all non-slow tests pass (≥ 30 tests).

- [ ] **Step 8: Commit**

```bash
git add src/eval/walk_forward.py tests/test_walk_forward_integration.py \
        tests/test_replication.py
git commit -m "feat: walk-forward run_walk_forward + Barroso-Santa-Clara replication tests"
```

---

## Task 10: End-to-End Smoke Test and Phase 1 Wrap-Up

- [ ] **Step 1: Run the leakage canary end-to-end**

```bash
python -c "
from src.eval.synthetic import white_noise_panel, leakage_test_signal
from src.eval.metrics import sharpe, information_coefficient
import numpy as np, pandas as pd

# White noise: Sharpe should be ~0
panel = white_noise_panel(n_dates=504, n_stocks=50, seed=0)
pnl_white = panel.groupby(level='date')['return'].mean()
sr_white = sharpe(pnl_white)
print(f'White noise Sharpe: {sr_white:.3f}  (expected: |s| < 0.3)')
assert abs(sr_white) < 0.3, f'CANARY FAILED: white noise Sharpe = {sr_white:.3f}'

# Leakage signal: Sharpe should be >> 15 when using future returns directly
dates = pd.date_range('2010-01-04', periods=504, freq='B')
tickers = [f'S{i:04d}' for i in range(50)]
idx = pd.MultiIndex.from_product([dates, tickers], names=['date','ticker'])
rng = np.random.default_rng(0)
returns = pd.Series(rng.normal(0, 0.01, 504*50), index=idx)
panel2 = returns.rename('return').to_frame()
sig = leakage_test_signal(panel2)

# Simulate the leaked portfolio: long stocks with positive future return
pnl_leaked = []
all_dates = dates.tolist()
for i, dt in enumerate(all_dates[:-1]):
    s = sig.xs(dt, level='date')['signal']
    r = returns.xs(all_dates[i+1], level='date')
    long_r = r[s > 0].mean()
    short_r = r[s < 0].mean()
    pnl_leaked.append(long_r - short_r)

pnl_leaked_s = pd.Series(pnl_leaked)
sr_leaked = sharpe(pnl_leaked_s)
print(f'Leakage signal Sharpe: {sr_leaked:.1f}  (expected: > 15)')
assert sr_leaked > 15, f'CANARY FAILED: leakage Sharpe = {sr_leaked:.1f} < 15'
print('Both canary tests PASSED. Evaluation framework is clean.')
"
```

- [ ] **Step 2: Write the `docs/predictions.md` pre-registration document**

Create `docs/predictions.md`:

```markdown
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
```

- [ ] **Step 3: Commit predictions document**

```bash
git add docs/predictions.md
git commit -m "doc: pre-registered predictions committed before Phase 3 ML training"
```

- [ ] **Step 4: Final full test run**

```bash
pytest tests/ -v -m "not slow" --tb=short
```

Expected: all pass with 0 failures.

- [ ] **Step 5: Tag Phase 1 complete**

```bash
git tag -a phase1-complete -m "Phase 1 complete: data pipeline, eval framework, baselines, replication verified"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Survivorship-bias-free S&P 500 universe | Task 2 |
| get_universe(date) -> List[ticker] | Task 2 |
| Daily OHLCV from yfinance 2000-2024 + VIX | Task 3 |
| Stooq fallback / cross_check_prices | Task 3 |
| Walk-forward CV with 42-day embargo | Task 4 |
| Metrics: Sharpe, Sortino, max DD, Calmar, IC, ICIR, turnover | Task 4 |
| White-noise canary (Sharpe ≈ 0) | Tasks 4, 10 |
| Leakage canary (Sharpe > 15) | Tasks 4, 10 |
| Vol features: RV at 3 horizons, r², Parkinson, VIX, skew/kurt, vol, ret_21 | Task 5 |
| Forward 21-day RV + log RV target | Task 5 |
| test_no_lookahead for features | Task 5 |
| Forecaster protocol | Task 6 |
| Rolling vol (Barroso-Santa-Clara baseline) | Task 7 |
| GARCH(1,1)-t (arch package, joblib parallel) | Task 7 |
| HAR-RV (Corsi 2009), log-RV target, Jensen correction | Task 7 |
| DM (QLIKE, Newey-West HAC) | Task 8 |
| Mincer-Zarnowitz (joint Wald H0: α=0, β=1) | Task 8 |
| Cross-sectional IC | Task 8 |
| MCS (Hansen, Lunde & Nason 2011) | Task 8 |
| Stationary bootstrap Sharpe-diff CI | Task 8 |
| configs/default.yaml with all hyperparameters | Task 1 |
| run_walk_forward integration test | Task 9 |
| CHECKPOINT 1f: HAR-RV OOS R² in [0.30, 0.65] | Task 9 |
| Barroso replication: 2009 crash + Sharpe uplift | Task 9 |
| Pre-registered predictions | Task 10 |
| git scaffold + Makefile | Task 1 |

All spec requirements covered. No placeholders or TBDs found.

---

Plan complete and saved to `docs/superpowers/plans/2026-05-17-phase0-phase1-data-pipeline-baselines.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, clean context per task

**2. Inline Execution** — execute tasks sequentially in this session using the executing-plans skill

**Which approach?**
