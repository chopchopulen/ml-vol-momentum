from __future__ import annotations
import pandas as pd


def demean_signal(signal: pd.Series, sector_map: dict[str, str]) -> pd.Series:
    """Within each (date, sector) group, subtract the group mean."""
    df = signal.to_frame("signal").copy()
    df["sector"] = df.index.get_level_values("ticker").map(sector_map)
    df["signal"] = df.groupby(
        [df.index.get_level_values("date"), "sector"]
    )["signal"].transform(lambda x: x - x.mean())
    return df["signal"].rename(signal.name)
