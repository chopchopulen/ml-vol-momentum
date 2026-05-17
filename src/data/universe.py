from __future__ import annotations
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
    """
    Build a membership table where each row represents ONE membership period.
    Tickers that were removed and re-added get multiple rows.
    """
    # Sector lookup from current constituents (authoritative for active members)
    sector_map = (current.set_index("ticker")[["gics_sector", "gics_sub_industry"]]
                  .to_dict("index"))

    # Collect all addition and removal events from the changes table
    additions = (changes[changes["added_date"].notna()]
                 [["ticker", "added_date"]]
                 .dropna(subset=["ticker"])
                 .copy())
    removals  = (changes[changes["removed_date"].notna()]
                 [["ticker", "removed_date"]]
                 .dropna(subset=["ticker"])
                 .copy())

    # Sort events chronologically
    additions = additions.sort_values("added_date").reset_index(drop=True)
    removals  = removals.sort_values("removed_date").reset_index(drop=True)

    rows = []

    # Current constituents: they are active NOW (removed_date = NaT)
    for _, row in current.iterrows():
        tkr = row["ticker"]
        # Find the most recent addition event for this ticker (if any)
        tkr_adds = additions[additions["ticker"] == tkr].sort_values("added_date")
        if tkr_adds.empty:
            # No recorded addition — member since before our history
            added = pd.NaT
        else:
            added = tkr_adds.iloc[-1]["added_date"]
        rows.append({
            "ticker": tkr,
            "added_date": added,
            "removed_date": pd.NaT,
            "gics_sector": row["gics_sector"],
            "gics_sub_industry": row["gics_sub_industry"],
        })

    current_tickers = set(current["ticker"].tolist())

    # Historical (removed) tickers: match each removal with the preceding addition
    # For each ticker in the removals table that is NOT currently active,
    # pair up: (add_1 → remove_1), (add_2 → remove_2), ...
    # If there are more removals than additions, the first removal's add_date = NaT
    historical_tickers = set(removals["ticker"].tolist()) - current_tickers
    for tkr in sorted(historical_tickers):
        tkr_adds = sorted(additions[additions["ticker"] == tkr]["added_date"].tolist())
        tkr_rems = sorted(removals[removals["ticker"] == tkr]["removed_date"].tolist())
        # Pair up periods: excess removals get NaT as add_date (member from beginning).
        # Align from the end so that later events pair correctly:
        # e.g. 1 addition, 2 removals -> (NaT, rem[0]), (add[0], rem[1])
        num_periods = max(len(tkr_adds), len(tkr_rems))
        add_pad = [pd.NaT] * (num_periods - len(tkr_adds)) + tkr_adds
        rem_pad = [pd.NaT] * (num_periods - len(tkr_rems)) + tkr_rems
        for i in range(num_periods):
            added   = add_pad[i]
            removed = rem_pad[i]
            # Back-fill sector from current map if available, else empty
            sec_info = sector_map.get(tkr, {})
            rows.append({
                "ticker": tkr,
                "added_date": added,
                "removed_date": removed,
                "gics_sector": sec_info.get("gics_sector", ""),
                "gics_sub_industry": sec_info.get("gics_sub_industry", ""),
            })

    df = pd.DataFrame(rows)

    # Validate: no row should have added_date > removed_date
    inverted = df[(df["added_date"].notna()) &
                  (df["removed_date"].notna()) &
                  (df["added_date"] > df["removed_date"])]
    if not inverted.empty:
        import warnings
        warnings.warn(
            f"Universe table has {len(inverted)} rows with inverted dates "
            f"(added > removed). Tickers: {inverted['ticker'].tolist()[:10]}"
        )

    return df.reset_index(drop=True)

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
