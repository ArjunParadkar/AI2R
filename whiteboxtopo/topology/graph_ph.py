"""Clique-complex persistent homology of a weighted graph (handbook S5 / Z3).

Edge-weight filtration: strongest edges enter first. With weights w >= 0, the
filtration value of edge (u, v) is 1 - w_uv / max w (so t = 0 is the strongest
edge, t = 1 the weakest); vertices enter at 0; the flag (clique) complex is
expanded up to ``maxdim + 1``. Standard tool (cite Horak–Maletić–Rajković 2009,
Petri et al. 2013); what is new downstream is relating β1 to discovery cost.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class GraphPHResult:
    dgms: List[np.ndarray]      # per degree (n, 2) in filtration coordinates
    betti1_finite: int          # number of finite H1 bars (cycles that get filled)
    betti1_essential: int       # H1 classes never filled (infinite bars)
    total_persistence: np.ndarray   # per degree, finite bars
    n_edges: int


def graph_ph(A, maxdim: int = 1) -> GraphPHResult:
    import gudhi
    A = np.asarray(A, dtype=np.float64); n = A.shape[0]
    W = np.abs(np.triu(A, 1)); wmax = W.max() if W.size else 0.0
    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i], 0.0)
    iu = np.nonzero(W)
    for i, j in zip(*iu):
        st.insert([int(i), int(j)], float(1.0 - W[i, j] / wmax) if wmax > 0 else 0.0)
    st.expansion(maxdim + 1)
    st.compute_persistence(persistence_dim_max=True)
    dgms = []
    for k in range(maxdim + 1):
        D = np.asarray(st.persistence_intervals_in_dimension(k), dtype=np.float64).reshape(-1, 2)
        dgms.append(D)
    d1 = dgms[1] if maxdim >= 1 else np.zeros((0, 2))
    fin = np.isfinite(d1[:, 1]) if len(d1) else np.zeros(0, bool)
    tp = np.array([float(np.sum(D[np.isfinite(D[:, 1]), 1] - D[np.isfinite(D[:, 1]), 0])) if len(D) else 0.0 for D in dgms])
    return GraphPHResult(dgms, int(fin.sum()), int((~fin).sum()) if len(d1) else 0, tp, int(len(iu[0])))
