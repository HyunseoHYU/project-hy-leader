"""Gating network: softmax regression (optionally 1 hidden layer) in pure numpy."""
import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Router:
    def __init__(self, n_experts: int = 3, hidden: int = 16, lr: float = 0.05, epochs: int = 400,
                 l2: float = 1e-3, temperature: float = 1.0, top_k: int = 0, seed: int = 0):
        self.E, self.H, self.lr, self.epochs, self.l2 = n_experts, hidden, lr, epochs, l2
        self.T, self.top_k, self.seed = temperature, top_k, seed

    def _forward(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        return h, h @ self.W2 + self.b2

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        self.W1 = rng.normal(0, 1 / np.sqrt(d), (d, self.H)); self.b1 = np.zeros(self.H)
        self.W2 = rng.normal(0, 1 / np.sqrt(self.H), (self.H, self.E)); self.b2 = np.zeros(self.E)
        Y = np.eye(self.E)[y.astype(int)]
        cw = n / (self.E * np.maximum(Y.sum(0), 1))          # inverse-frequency class weights
        sw = (Y * cw).sum(1, keepdims=True)
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        for t in range(1, self.epochs + 1):
            h, logits = self._forward(Xs)
            g = (softmax(logits) - Y) * sw / n
            dW2 = h.T @ g + self.l2 * self.W2; db2 = g.sum(0)
            dh = (g @ self.W2.T) * (1 - h ** 2)
            grads = [Xs.T @ dh + self.l2 * self.W1, dh.sum(0), dW2, db2]
            for i, (p, gr) in enumerate(zip(params, grads)):   # Adam
                m[i] = 0.9 * m[i] + 0.1 * gr; v[i] = 0.999 * v[i] + 0.001 * gr ** 2
                p -= self.lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, logits = self._forward((X - self.mu) / self.sd)
        p = softmax(logits / self.T)
        if 0 < self.top_k < self.E:                           # sparse top-k gating
            thr = np.sort(p, axis=1)[:, -self.top_k][:, None]
            p = np.where(p >= thr, p, 0.0)
            p /= p.sum(1, keepdims=True)
        return p
