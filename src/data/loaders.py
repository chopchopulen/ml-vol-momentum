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
    if raw.empty:
        return pd.DataFrame()

    # yfinance >= 0.2 always returns MultiIndex columns with levels ['Price', 'Ticker']
    # even for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        # Determine level positions: Price level and Ticker level
        level_names = list(raw.columns.names)
        if "Price" in level_names and "Ticker" in level_names:
            price_level = level_names.index("Price")
            ticker_level = level_names.index("Ticker")
        else:
            # Fallback: assume level 0 = field, level 1 = ticker
            price_level = 0
            ticker_level = 1

        frames = []
        for tkr in tickers:
            try:
                # xs on the ticker level
                sub = raw.xs(tkr, level=ticker_level, axis=1).copy()
            except KeyError:
                continue
            sub.columns = sub.columns.str.lower()
            sub.index.name = "date"
            sub.index = pd.to_datetime(sub.index)
            sub["ticker"] = tkr
            frames.append(sub)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
    else:
        raw.columns = raw.columns.str.lower()
        raw.index.name = "date"
        raw.index = pd.to_datetime(raw.index)
        raw["ticker"] = tickers[0]
        df = raw

    df = df.reset_index().set_index(["date", "ticker"])
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].sort_index()
    # Normalize datetime precision to ms for consistent parquet roundtrip
    df.index = df.index.set_levels(
        df.index.levels[0].astype("datetime64[ms]"), level=0)
    return df


def _fetch_stooq(ticker: str, start: pd.Timestamp,
                 end: pd.Timestamp) -> pd.DataFrame:
    import io
    import requests
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
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].sort_index()


def load_ohlcv(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp,
               source: Literal["yfinance", "stooq"] = "yfinance") -> pd.DataFrame:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_frames = []
    fetch_tickers = []

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
            parts = [_fetch_stooq(t, start, end) for t in fetch_tickers]
            parts = [p for p in parts if not p.empty]
            fetched = pd.concat(parts, ignore_index=False) if parts else pd.DataFrame()

        if not fetched.empty:
            for tkr in fetch_tickers:
                tkr_mask = fetched.index.get_level_values("ticker") == tkr
                if tkr_mask.any():
                    sub = fetched[tkr_mask]
                    cp = _cache_path(tkr, start, end, source)
                    sub.to_parquet(cp)
                    all_frames.append(sub)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames).sort_index()


def load_vix(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    cfg = load_config()
    vix_ticker = cfg["data"]["vix_ticker"]
    cp = _cache_path("VIX", start, end, "yfinance")
    if cp.exists():
        return pd.read_parquet(cp)["vix"]

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = yf.download(vix_ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float, name="vix")

    # Handle MultiIndex columns (yfinance >= 0.2 always returns MultiIndex)
    if isinstance(raw.columns, pd.MultiIndex):
        # Try standard column names first
        try:
            close = raw["Close"][vix_ticker]
        except KeyError:
            try:
                close = raw["Close"].iloc[:, 0]
            except Exception:
                close = raw.iloc[:, 0]
    else:
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]

    s = close.rename("vix")
    s.index.name = "date"
    s.index = pd.to_datetime(s.index)
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
                             "max_pct_diff": float(diff.max()),
                             "mean_pct_diff": float(diff.mean()),
                             "status": "ok"})
        except Exception as e:
            results.append({"ticker": tkr, "max_pct_diff": np.nan,
                             "mean_pct_diff": np.nan, "status": str(e)})
    return pd.DataFrame(results)
