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
        assert len(removed) >= 30, "Should have reconstructed >=30 historical removals"

class TestGetSector:
    def test_aapl_is_tech(self):
        sector = get_sector("AAPL", pd.Timestamp("2020-01-01"))
        assert "Technology" in sector or "Information" in sector
