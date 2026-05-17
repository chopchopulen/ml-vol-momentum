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
    removed = changes[changes["removed_date"].notna()][["ticker", "removed_date"]]
    added   = changes[changes["added_date"].notna()][["ticker", "added_date",
                                                       "gics_sector", "gics_sub_industry"]]

    # Tickers only in removed list (never re-added, no longer current) need to be included
    # so we can correctly exclude them in historical point-in-time queries.
    removed_only_tickers = set(removed["ticker"].dropna()) - set(current["ticker"]) - set(added["ticker"].dropna())
    removed_only_rows = []
    for ticker in removed_only_tickers:
        removed_only_rows.append({"ticker": ticker, "gics_sector": "",
                                   "gics_sub_industry": "",
                                   "added_date": pd.NaT, "removed_date": pd.NaT})
    removed_only_df = pd.DataFrame(removed_only_rows) if removed_only_rows else pd.DataFrame(
        columns=["ticker", "gics_sector", "gics_sub_industry", "added_date", "removed_date"])

    all_tickers = pd.concat([
        current,
        added[~added["ticker"].isin(current["ticker"])],
        removed_only_df,
    ], ignore_index=True)

    removal_map = removed.dropna(subset=["ticker"]).set_index("ticker")["removed_date"].to_dict()
    all_tickers["removed_date"] = all_tickers["ticker"].map(removal_map)

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
