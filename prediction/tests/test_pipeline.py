import numpy as np
from moe_regime.data import make_synthetic
from moe_regime.features import build_features
from moe_regime.labels import label_regimes
from moe_regime.router import Router
from moe_regime.pipeline import run


def test_features_are_causal():
    df = make_synthetic(400)
    f1 = build_features(df)
    df2 = df.copy(); df2.iloc[300:, df2.columns.get_loc("close")] *= 5
    f2 = build_features(df2)
    assert np.allclose(f1.iloc[:300].fillna(0), f2.iloc[:300].fillna(0))


def test_labels_cover_three_regimes():
    lab = label_regimes(make_synthetic(2000)).dropna()
    assert set(lab.unique()) == {0.0, 1.0, 2.0}


def test_router_topk_and_simplex():
    X = np.random.default_rng(0).normal(size=(200, 5)); y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1)
    p = Router(top_k=2, epochs=50).fit(X, y).predict_proba(X)
    assert np.allclose(p.sum(1), 1) and ((p > 0).sum(1) <= 2).all()


def test_pipeline_beats_chance():
    import yaml
    from pathlib import Path
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"  # 실행 위치와 무관하게
    cfg = yaml.safe_load(open(config_path)); cfg["out_dir"] = None
    accs = [run({**cfg, "seed": s})["router_acc"] for s in range(4)]
    assert np.mean(accs) > 0.42  # 3 classes -> chance ~0.33
