from __future__ import annotations
import numpy as np
import pandas as pd

# PIT CONVENTION (enforced throughout this module):
#   Signal at month-end t uses prices through CLOSE of t (no future data).
#   Trades execute at OPEN of t+1. portfolio.py shifts weights by +1 day
#   before multiplying realized returns — this is the project-wide convention.
#   Any code that uses t+1 prices to compute the signal at t is a bug.


def momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """12-1 momentum: cumulative return from t-lookback to t-skip."""
    raise NotImplementedError("Phase 2 implementation")
