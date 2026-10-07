"""End-to-end: data -> features -> regime labels -> chronological split -> fit -> evaluate."""
import json
import numpy as np
import pandas as pd

from . import REGIMES
from .data import load_csv, make_synthetic
from .experts import RidgeExpert
from .features import build_features, next_day_return
from .labels import label_regimes
from .metrics import confusion, macro_f1, trading_stats
from .moe import RegimeMoE


def run(cfg: dict) -> dict:
    df = load_csv(cfg["csv"]) if cfg.get("csv") else make_synthetic(cfg.get("n", 2500), cfg.get("seed", 0))
    X = build_features(df)
    lab = label_regimes(df, **cfg.get("label", {}))
    y = next_day_return(df)
    data = pd.concat([X, lab, y], axis=1).dropna()
    h = cfg.get("label", {}).get("horizon", 20)
    n = len(data); n_tr = int(n * cfg.get("train_frac", 0.7))
    # purge `horizon` rows so forward-looking labels at the train tail don't see test prices
    tr, te = data.iloc[: n_tr - h], data.iloc[n_tr:]
    feats = list(X.columns)
    moe = RegimeMoE(cfg.get("router"), cfg.get("expert_alpha", 10.0)).fit(
        tr[feats].to_numpy(), tr["regime"].to_numpy(), tr["target"].to_numpy())
    pred, gate = moe.predict(te[feats].to_numpy())
    hard = gate.argmax(1)
    base = RidgeExpert(cfg.get("expert_alpha", 10.0)).fit(tr[feats].to_numpy(), tr["target"].to_numpy())
    bpred = base.predict(te[feats].to_numpy())
    ret = te["target"].to_numpy()
    out = {
        "n_train": len(tr), "n_test": len(te),
        "router_acc": float((hard == te["regime"].to_numpy()).mean()),
        "router_macro_f1": macro_f1(te["regime"].to_numpy(), hard),
        "confusion(rows=true,cols=pred;" + ",".join(REGIMES) + ")": confusion(te["regime"].to_numpy(), hard).tolist(),
        "test_regime_share": {r: float((te["regime"] == i).mean()) for i, r in enumerate(REGIMES)},
        "moe": trading_stats(pred, ret),
        "single_model_baseline": trading_stats(bpred, ret),
        "buy_and_hold_cum_ret": float(np.exp(ret.sum()) - 1),
    }
    res = te[["regime", "target"]].copy()
    res["pred_regime"] = [REGIMES[i] for i in hard]
    for i, r in enumerate(REGIMES):
        res[f"p_{r}"] = gate[:, i]
    res["moe_pred"] = pred
    if cfg.get("out_dir"):
        import os
        os.makedirs(cfg["out_dir"], exist_ok=True)
        res.to_csv(f"{cfg['out_dir']}/routing_test.csv")
        json.dump(out, open(f"{cfg['out_dir']}/metrics.json", "w"), indent=2)
    return out
