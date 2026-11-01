"""Definition 3 — Structural Velocity SV^l(t) = W_2(D^l_k(t), D^l_k(t-1))."""
from __future__ import annotations
from typing import Sequence
import numpy as np
from persim import wasserstein


def _as_dgm(D) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    return D if D.size else np.empty((0, 2))


def diagram_distance(D_a, D_b) -> float:
    """2-Wasserstein distance between two finite diagrams (persim). Points may
    be matched to the diagonal; empty-vs-empty is 0."""
    A, B = _as_dgm(D_a), _as_dgm(D_b)
    if A.size == 0 and B.size == 0:
        return 0.0
    return float(wasserstein(A, B))


def structural_velocity(D_t, D_tm1) -> float:
    """SV between consecutive frames. Symmetric; SV(D, D) == 0."""
    return diagram_distance(D_t, D_tm1)


def sv_trajectory(diagrams: Sequence[np.ndarray]) -> np.ndarray:
    """SV over a film of one layer. Index 0 is 0.0 by convention (original code)."""
    out = [0.0]
    for i in range(1, len(diagrams)):
        out.append(structural_velocity(diagrams[i], diagrams[i - 1]))
    return np.asarray(out)
