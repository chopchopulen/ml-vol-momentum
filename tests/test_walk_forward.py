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
