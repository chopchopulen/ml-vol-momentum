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
                f"required >= {self.embargo_days}"
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
