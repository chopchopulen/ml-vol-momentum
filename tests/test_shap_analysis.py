import pytest
import numpy as np
import pandas as pd

pytest.importorskip("shap")
pytestmark = pytest.mark.shap


def _make_panel(n=200):
    dates = pd.date_range("2010-01-01", periods=n // 10)
    tickers = [f"T{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(0)
    cols = ["rv_d", "rv_w", "rv_m", "pk", "skew", "kurt", "vix", "log_dv", "ret_21"]
    data = {c: rng.standard_normal(len(idx)) ** 2 for c in cols}
    data["sector"] = "Tech"
    data["target_log_rv"] = rng.standard_normal(len(idx))
    return pd.DataFrame(data, index=idx)


@pytest.fixture(scope="module")
def fitted_model_and_panel():
    from src.models.gbm import GBMForecaster
    panel = _make_panel(200)
    model = GBMForecaster()
    model.fit(panel)
    return model, panel


def test_compute_shap_returns_series(fitted_model_and_panel):
    from src.interp.shap_analysis import compute_shap_importance
    model, panel = fitted_model_and_panel
    result = compute_shap_importance(model, panel)
    assert isinstance(result, pd.Series)


def test_compute_shap_index_is_feature_names(fitted_model_and_panel):
    from src.interp.shap_analysis import compute_shap_importance
    model, panel = fitted_model_and_panel
    result = compute_shap_importance(model, panel)
    assert "rv_m" in result.index
    assert "rv_w" in result.index
    assert "vix" in result.index


def test_compute_shap_values_nonnegative(fitted_model_and_panel):
    from src.interp.shap_analysis import compute_shap_importance
    model, panel = fitted_model_and_panel
    result = compute_shap_importance(model, panel)
    assert (result >= 0).all()


def test_compute_shap_works_with_small_sample():
    from src.interp.shap_analysis import compute_shap_importance
    from src.models.gbm import GBMForecaster
    panel = _make_panel(500)
    model = GBMForecaster()
    model.fit(panel)
    result = compute_shap_importance(model, panel, sample_size=10)
    expected_features, _ = model._prepare_X(panel)
    expected_features = list(expected_features.columns)
    assert list(result.index) == sorted(expected_features, key=lambda f: result[f], reverse=True)
