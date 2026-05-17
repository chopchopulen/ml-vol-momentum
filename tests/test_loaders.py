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
        # Jan 2023 has 20 trading days (Jan 2 is a holiday — New Year's observed)
        assert 18 <= len(dates) <= 23

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
