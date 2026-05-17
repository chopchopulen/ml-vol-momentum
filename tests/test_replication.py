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

def _dedup_panel(prices_panel):
    """Remove duplicate (date, ticker) entries that may arise from data download."""
    if prices_panel.index.duplicated().any():
        prices_panel = prices_panel[~prices_panel.index.duplicated(keep="first")]
    return prices_panel

def _make_mom_signal(prices_panel):
    prices_panel = _dedup_panel(prices_panel)
    close = prices_panel["close"].unstack("ticker")
    log_ret = np.log(close / close.shift(1))
    mom = log_ret.rolling(252).sum().shift(21) - log_ret.rolling(21).sum().shift(1)
    return mom.stack(future_stack=True).rename("signal").to_frame()

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
        deduped = _dedup_panel(market_data)
        close_wide = deduped["close"].unstack("ticker")
        log_ret = np.log(close_wide / close_wide.shift(1))
        returns_panel = log_ret.stack(future_stack=True).rename("return").to_frame()
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
        deduped = _dedup_panel(market_data)
        close_wide = deduped["close"].unstack("ticker")
        log_ret = np.log(close_wide / close_wide.shift(1))
        returns_panel = log_ret.stack(future_stack=True).rename("return").to_frame()
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
