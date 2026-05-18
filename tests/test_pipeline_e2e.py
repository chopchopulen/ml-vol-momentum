import numpy as np
import pandas as pd
import pytest
from src.eval.synthetic import white_noise_panel, leakage_test_signal
from src.strategy.momentum import momentum_signal
from src.strategy.portfolio import build_portfolios
from src.strategy.costs import apply_costs
from src.eval.metrics import sharpe


def _build_panel_with_prices(n_dates=600, n_stocks=50, seed=42):
    """White-noise returns panel plus fake close prices."""
    panel = white_noise_panel(n_dates, n_stocks, seed)
    returns = panel["return"].unstack("ticker")
    prices = (1 + returns).cumprod()
    prices_long = prices.stack(future_stack=True).rename("close").to_frame()
    prices_long.index.names = ["date", "ticker"]
    return panel, prices_long


class TestPipelineEndToEnd:
    def test_white_noise_unscaled_sharpe_near_zero(self):
        """White-noise returns → momentum signal → LS quintile → Sharpe should be near 0."""
        panel, prices = _build_panel_with_prices(n_dates=600, n_stocks=50, seed=7)
        signal = momentum_signal(prices, lookback=252, skip=21)
        weights = build_portfolios(signal, mode="long_short_quintile")
        net = apply_costs(weights, panel, cost_bps=10.0)
        sr = sharpe(net.dropna())
        assert abs(sr) < 2.0, f"White-noise Sharpe = {sr:.2f}, expected |Sharpe| < 2.0"

    def test_leakage_signal_gives_high_sharpe(self):
        """Signal = return at t+1 (perfect leakage) → Sharpe >> 10 after portfolio shift."""
        panel, prices = _build_panel_with_prices(n_dates=600, n_stocks=50, seed=7)
        # leakage_test_signal returns signal at t = return at t+1
        signal = leakage_test_signal(panel)
        weights = build_portfolios(signal, mode="long_short_quintile")
        net = apply_costs(weights, panel, cost_bps=0.0)
        sr = sharpe(net.dropna())
        assert sr > 10, f"Leakage Sharpe = {sr:.2f}, expected > 10"

    def test_pipeline_runs_without_error(self):
        """The full pipeline from momentum through costs runs without raising."""
        from src.strategy.scaling import vol_scale
        from src.models.baselines import RollingVolModel
        panel, prices = _build_panel_with_prices(n_dates=400, n_stocks=20, seed=9)
        signal = momentum_signal(prices, lookback=252, skip=21)
        # Make a simple mock forecast panel using rolling variance
        rv_model = RollingVolModel(window=126)
        rv_model.fit(panel)
        oos_forecasts = rv_model.predict(panel)
        if oos_forecasts.empty:
            pytest.skip("No forecasts generated on synthetic panel")
        w_scaled = vol_scale(signal, oos_forecasts, target_vol=0.10)
        w_portfolio = build_portfolios(None, weights=w_scaled, mode="vol_targeted_gross")
        net_scaled = apply_costs(w_portfolio, panel, cost_bps=10.0)
        assert isinstance(net_scaled, pd.Series)
        assert len(net_scaled.dropna()) > 50
