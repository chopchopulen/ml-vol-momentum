from __future__ import annotations
import pandas as pd

# PIT CONVENTION (enforced throughout this module):
#   sigma_hat_{i,t} is the forecast produced from data through CLOSE of t.
#   The resulting weight is held from open t+1 onward.
#   Signal and forecast use the SAME timestamp t; portfolio.py applies the
#   +1 shift before computing returns. Never use t+1 vol to scale weight at t.


def vol_scale(
    signal: pd.DataFrame,
    sigma_hat: pd.DataFrame,
    target_vol: float,
) -> pd.DataFrame:
    """Cross-sectionally scale signal by inverse forecast vol to target_vol."""
    raise NotImplementedError("Phase 2 implementation")
