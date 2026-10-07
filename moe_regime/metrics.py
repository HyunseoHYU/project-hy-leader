import numpy as np


def confusion(y, p, k=3):
    m = np.zeros((k, k), int)
    for a, b in zip(y.astype(int), p.astype(int)):
        m[a, b] += 1
    return m


def macro_f1(y, p, k=3):
    m, f = confusion(y, p, k), []
    for i in range(k):
        tp = m[i, i]; pr = tp / max(m[:, i].sum(), 1); rc = tp / max(m[i].sum(), 1)
        f.append(0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return float(np.mean(f))


def trading_stats(pred, ret):
    """Long/short by sign of prediction; daily returns, crypto 365d/yr."""
    pos = np.sign(pred)
    pnl = pos * ret
    sharpe = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(365)
    ic = np.corrcoef(pred, ret)[0, 1] if pred.std() > 0 else 0.0
    return {"dir_acc": float((pos == np.sign(ret)).mean()), "ic": float(ic),
            "sharpe": float(sharpe), "cum_ret": float(np.exp(pnl.sum()) - 1)}
