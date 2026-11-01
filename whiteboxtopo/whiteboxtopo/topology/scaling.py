"""Persistence under gauge scalings (handbook Z5).

E2 (whitebox-v2, 2026-08-30) measured that raw activation-cloud persistence is
NOT invariant under per-unit ReLU scalings: total persistence Σ(d-b)² is
degree-2 covariant under a *uniform* rescale (TC(αX) = α² TC(X)), and per-unit
scalings move it with no single exponent (CV 1.4–3.0). The exact repair E2
verified is per-unit standardisation of the cloud (each hidden unit divided by
its own scale), which makes every diagram-derived statistic exactly invariant
under G. This module ships that as ``standardized_cloud`` and
``normalized_persistence``; the F10 audit re-measures the covariance law and Z5
records the empirical exponent for each statistic.
"""
from __future__ import annotations
import numpy as np
from ..core.persistence import compute_diagrams


def standardized_cloud(A: np.ndarray, how: str = "std", eps: float = 0.0) -> np.ndarray:
    """Divide every column (unit) of an activation cloud by its own scale: ``std`` (sample
    std over points), ``rms`` (root-mean-square), or ``max``. Columns with scale 0 are left as 0.
    Each is exactly equivariant under per-unit positive scalings, so the standardised cloud is
    a gauge-invariant object (up to unit relabelling, which persistence ignores)."""
    A = np.asarray(A, dtype=np.float64)
    if how == "std":
        s = A.std(0)
    elif how == "rms":
        s = np.sqrt((A ** 2).mean(0))
    elif how == "max":
        s = np.abs(A).max(0)
    else:
        raise ValueError(how)
    s = np.where(s <= eps, 1.0, s)
    return A / s[None, :]


def normalized_persistence(A: np.ndarray, maxdim: int = 1, how: str = "std", backend: str = "ripser", **kw):
    """Persistence summary of the standardised cloud: dict(beta0, beta1, total_persistence_H0,
    total_persistence_H1, persistent_entropy_H1) — each invariant under per-unit scalings."""
    Z = standardized_cloud(A, how)
    dg = compute_diagrams(Z, maxdim=maxdim, backend=backend, **kw)
    out = {}
    for k in range(maxdim + 1):
        D = np.asarray(dg.dgms[k]).reshape(-1, 2)
        fin = np.isfinite(D[:, 1]) if len(D) else np.zeros(0, bool)
        life = D[fin, 1] - D[fin, 0]
        out[f"n_bars_H{k}"] = int(len(D)); out[f"total_persistence_H{k}"] = float(np.sum(life ** 2))
        p = life / life.sum() if life.sum() > 0 else np.zeros(0)
        out[f"persistent_entropy_H{k}"] = float(-(p[p > 0] * np.log(p[p > 0])).sum()) if len(p) else 0.0
    return out
