"""Mixture of regime experts: output = sum_e gate_e(x) * expert_e(x)."""
import numpy as np

from . import REGIMES
from .experts import RidgeExpert
from .router import Router


class RegimeMoE:
    def __init__(self, router_kw=None, expert_alpha: float = 10.0, min_samples: int = 30):
        self.router = Router(**(router_kw or {}))
        self.alpha, self.min_samples = expert_alpha, min_samples

    def fit(self, X, regime, y):
        self.router.fit(X, regime)
        self.fallback = RidgeExpert(self.alpha).fit(X, y)
        self.experts = []
        for e in range(len(REGIMES)):
            # soft responsibilities from labels: own regime = 1, others small
            w = np.where(regime == e, 1.0, 0.05)
            ok = (regime == e).sum() >= self.min_samples
            self.experts.append(RidgeExpert(self.alpha).fit(X, y, w) if ok else self.fallback)
        return self

    def route(self, X):
        return self.router.predict_proba(X)

    def predict(self, X):
        g = self.route(X)
        preds = np.stack([ex.predict(X) for ex in self.experts], axis=1)
        return (g * preds).sum(1), g
