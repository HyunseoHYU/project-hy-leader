"""Retrospective regime labels used ONLY as router supervision.

Uses the forward `horizon`-day return scaled by trailing volatility, so labels
look ahead by design; they must never be computed for the test period's inputs.
"""
import numpy as np
import pandas as pd


def label_regimes(df: pd.DataFrame, horizon: int = 20, k: float = 0.5, vol_window: int = 30) -> pd.Series:
    """0=bear, 1=sideways, 2=bull. Threshold = k * trailing_vol * sqrt(horizon)."""
    logc = np.log(df["close"])
    fwd = logc.shift(-horizon) - logc
    vol = logc.diff().rolling(vol_window).std()
    z = fwd / (vol * np.sqrt(horizon))
    lab = pd.Series(1.0, index=df.index)
    lab[z > k] = 2
    lab[z < -k] = 0
    lab[z.isna()] = np.nan
    return lab.rename("regime")
