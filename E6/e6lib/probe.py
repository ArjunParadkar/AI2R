"""Test 3 — streaming ridge probes (closed form from accumulated sufficient statistics).

RidgeAccumulator: fit W (d_in+1 → d_out) from Σ ZᵀZ and Σ ZᵀY accumulated over
batches, then evaluate held-out R² and top-k accuracy streaming as well, so a
50k-dim target (logit delta) is never materialised.
"""
from __future__ import annotations
import numpy as np
import torch


class RidgeAccumulator:
    def __init__(self, d_in: int, d_out: int, lam: float = 1.0, device="cpu", dtype=torch.float64):
        self.d_in, self.d_out, self.lam = d_in, d_out, lam
        self.dev, self.dt = device, dtype
        self.ZtZ = torch.zeros(d_in + 1, d_in + 1, device=device, dtype=dtype)
        self.ZtY = torch.zeros(d_in + 1, d_out, device=device, dtype=dtype)
        self.n = 0
        self.W = None

    @staticmethod
    def _aug(Z):
        return torch.cat([Z, torch.ones(len(Z), 1, device=Z.device, dtype=Z.dtype)], 1)

    def add(self, Z: torch.Tensor, Y: torch.Tensor):
        Za = self._aug(Z.to(self.dev, self.dt)); Y = Y.to(self.dev, self.dt)
        self.ZtZ += Za.T @ Za; self.ZtY += Za.T @ Y; self.n += len(Za)

    def fit(self):
        reg = self.lam * torch.eye(self.d_in + 1, device=self.dev, dtype=self.dt); reg[-1, -1] = 0.0
        self.W = torch.linalg.solve(self.ZtZ + reg, self.ZtY)
        return self

    def predict(self, Z: torch.Tensor) -> torch.Tensor:
        return self._aug(Z.to(self.dev, self.dt)) @ self.W

    def train_mean(self) -> torch.Tensor:
        return self.ZtY[-1] / max(self.n, 1)


class R2Evaluator:
    """Streaming held-out R² = 1 − Σ‖Y−Ŷ‖² / Σ‖Y−Ȳ_train‖²."""
    def __init__(self, ybar: torch.Tensor):
        self.ybar = ybar; self.ss_res = 0.0; self.ss_tot = 0.0; self.n = 0

    def add(self, Y: torch.Tensor, Yhat: torch.Tensor):
        Y = Y.to(Yhat); self.ss_res += float(((Y - Yhat) ** 2).sum()); self.ss_tot += float(((Y - self.ybar) ** 2).sum()); self.n += len(Y)

    @property
    def r2(self) -> float:
        return 1.0 - self.ss_res / max(self.ss_tot, 1e-30)


def topk_accuracy(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    top = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean([y[i] in top[i] for i in range(len(y))]))


def ridge_onehot_probe(Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray, yte: np.ndarray, n_classes: int,
                       lam: float = 1.0) -> dict:
    """Ridge-to-one-hot classifier; returns held-out top-1/top-5 accuracy."""
    acc = RidgeAccumulator(Ztr.shape[1], n_classes, lam)
    Y = torch.zeros(len(ytr), n_classes, dtype=torch.float64); Y[torch.arange(len(ytr)), torch.as_tensor(ytr)] = 1.0
    acc.add(torch.as_tensor(Ztr), Y); acc.fit()
    S = acc.predict(torch.as_tensor(Zte)).numpy()
    return dict(top1=topk_accuracy(S, yte, 1), top5=topk_accuracy(S, yte, 5))
