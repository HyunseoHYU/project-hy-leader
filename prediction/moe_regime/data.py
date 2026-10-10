"""Data loading and a synthetic regime-switching generator for offline runs."""
import numpy as np
import pandas as pd

from . import REGIMES


def load_csv(path: str) -> pd.DataFrame:
    """CSV needs columns: date, close, and optionally volume, sentiment."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").set_index("date")
    if "close" not in df:
        raise ValueError("CSV must contain a 'close' column")
    if "sentiment" not in df:
        df["sentiment"] = 0.0
    return df


def make_synthetic(n: int = 2500, seed: int = 0) -> pd.DataFrame:
    """Markov-switching price with a sentiment series that leads/co-moves with drift."""
    rng = np.random.default_rng(seed)
    drift = {0: -0.0045, 1: 0.0, 2: 0.0050}
    vol = {0: 0.030, 1: 0.012, 2: 0.020}
    stay = 0.985
    state, states = 1, []
    for _ in range(n):
        if rng.random() > stay:
            state = int(rng.choice([s for s in range(3) if s != state]))
        states.append(state)
    states = np.array(states)
    ret = np.array([rng.normal(drift[s], vol[s]) for s in states])
    close = 100 * np.exp(np.cumsum(ret))
    raw = np.array([drift[s] / 0.005 for s in states]) * 0.6
    sent = pd.Series(raw + rng.normal(0, 0.5, n)).ewm(span=5).mean().to_numpy()
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"close": close, "volume": rng.lognormal(10, 0.3, n), "sentiment": sent,
         "true_regime": [REGIMES[s] for s in states]}, index=idx)
