from __future__ import annotations
from typing import Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class Forecaster(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> None:
        ...

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        ...
        # Returns a Panel (date, ticker) with columns:
        #   "forecast_log_rv" — forecast in log-RV space
        #   "forecast_rv"     — back-transformed with Jensen correction
