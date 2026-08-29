"""Definition 7 — BiasTop^l(t) = W_2(D^l_k(t; s=0), D^l_k(t; s=1)).

Topological distance between the activation clouds of two subgroups
(e.g. male- vs female-referring probe sentences) under the SAME reducer.
This is the *measurement*; the "topology leads output bias" precedence
claim built on it is in AUDIT.md bin 2 (shaky) and is NOT part of this
module.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from ..core.persistence import compute_diagrams
from ..core.reduce import Reducer
from .sv import diagram_distance


def bias_top(D_a, D_b) -> float:
    return diagram_distance(D_a, D_b)


def bias_top_permutation_test(
    X_a: np.ndarray, X_b: np.ndarray, reducer: Optional[Reducer],
    n_permutations: int = 500, maxdim: int = 1, rng=None, backend: str = "auto",
):
    """Permutation null for BiasTop: shuffle subgroup labels, recompute.
    Uses H1 if both diagrams have H1 bars, else H0 (original behaviour).
    Returns (observed, p_value, null_samples)."""
    rng = np.random.default_rng(0) if rng is None else rng

    def _bt(Xa, Xb):
        Da = compute_diagrams(Xa, reducer, maxdim=maxdim, backend=backend)
        Db = compute_diagrams(Xb, reducer, maxdim=maxdim, backend=backend)
        k = 1 if (maxdim >= 1 and Da[1].size and Db[1].size) else 0
        return bias_top(Da[k], Db[k])

    obs = _bt(X_a, X_b)
    pooled = np.vstack([X_a, X_b]); n_a = len(X_a)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(len(pooled))
        null[i] = _bt(pooled[perm[:n_a]], pooled[perm[n_a:]])
    p = float((1 + np.sum(null >= obs)) / (1 + n_permutations))
    return float(obs), p, null
