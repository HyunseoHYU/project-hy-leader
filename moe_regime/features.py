"""Causal features (use only information available at time t)."""
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for w in (5, 20, 60):
        f[f"ret_{w}"] = np.log(c).diff(w)
    for w in (10, 30):
        f[f"vol_{w}"] = r.rolling(w).std()
    f["ma_ratio_20"] = c / c.rolling(20).mean() - 1
    f["ma_ratio_60"] = c / c.rolling(60).mean() - 1
    f["drawdown_60"] = c / c.rolling(60).max() - 1
    up, dn = r.clip(lower=0).rolling(14).mean(), (-r.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = up / (up + dn + 1e-12) - 0.5
    s = df["sentiment"]
    f["sent"] = s
    f["sent_ma_10"] = s.rolling(10).mean()
    f["sent_chg_5"] = s.diff(5)
    f["sent_z_60"] = (s - s.rolling(60).mean()) / (s.rolling(60).std() + 1e-9)
    return f


def next_day_return(df: pd.DataFrame) -> pd.Series:
    """Target for experts: return from t to t+1 (label only; never a feature)."""
    return np.log(df["close"]).diff().shift(-1).rename("target")
