"""One ridge regressor per regime, trained on that regime's samples."""
import numpy as np


class RidgeExpert:
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def fit(self, X, y, w=None):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        w = np.ones(len(X)) if w is None else w
        A = Xs.T @ (Xs * w[:, None]) + self.alpha * np.diag([0.0] + [1.0] * (Xs.shape[1] - 1))
        self.beta = np.linalg.solve(A, Xs.T @ (w * y))
        return self

    def predict(self, X):
        return np.c_[np.ones(len(X)), (X - self.mu) / self.sd] @ self.beta
