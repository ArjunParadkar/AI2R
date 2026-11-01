"""Zigzag persistence across a sequence of point clouds (handbook S5 / Z1).

Union-of-consecutive construction: for clouds X_0, ..., X_{T-1} at a fixed Rips scale r,

    X_0 ⊆ X_0∪X_1 ⊇ X_1 ⊆ X_1∪X_2 ⊇ X_2 ⊆ ...      (positions 0, 1, 2, ..., 2T-2)

Every point is tagged with its cloud index, so a simplex of Rips(∪X_t, r) is
present exactly on one contiguous position interval: vertices all in X_i ->
[2i-1, 2i+1]; vertices in X_i ∪ X_{i+1} (both) -> {2i+1}; otherwise never.
Dionysus's zigzag algorithm then gives, per homology degree, bars
[birth, death) in *position* coordinates; ``to_cloud_index`` halves them so a
bar [b, d) reads "this cycle exists from cloud ⌈b/2⌉ to cloud ⌊(d-1)/2⌋".

Contrast with the per-epoch "film" (v1): computing a diagram per cloud and
matching bars across clouds can mislabel a persistent feature as a death +
birth whenever its representative cycle moves; the zigzag module tracks the
*same* homology class through the inclusions and needs no matching.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np


@dataclass
class ZigzagResult:
    bars: List[np.ndarray]           # per degree: (n_bars, 2) [birth, death) in position coordinates (death may be inf)
    n_positions: int
    scale: float
    n_points: int
    n_simplices: int

    def bars_in_cloud_index(self, dim: int) -> np.ndarray:
        """Bars as (first_cloud, last_cloud) inclusive; a bar alive only at a union position
        (odd) is reported with first > last."""
        B = self.bars[dim]
        if len(B) == 0:
            return np.zeros((0, 2))
        b, d = B[:, 0], B[:, 1]
        d = np.where(np.isfinite(d), d, self.n_positions)
        return np.stack([np.ceil(b / 2), np.floor((d - 1) / 2)], 1)

    def betti_curve(self, dim: int) -> np.ndarray:
        """β_dim at every position 0..n_positions-1."""
        out = np.zeros(self.n_positions, dtype=int)
        for b, d in self.bars[dim]:
            d = self.n_positions if not np.isfinite(d) else d
            out[int(b):int(d)] += 1
        return out


def zigzag(clouds: Sequence[np.ndarray], scale: float, maxdim: int = 1, max_points: Optional[int] = None,
           seed: int = 0) -> ZigzagResult:
    """Zigzag persistence of ``clouds`` (list of (n_t, d) arrays) at Rips ``scale``.
    ``max_points`` subsamples each cloud (uniformly, fixed seed) to bound the complex size."""
    import dionysus as d
    rng = np.random.default_rng(seed)
    Xs = []
    for X in clouds:
        X = np.asarray(X, dtype=np.float64)
        if max_points is not None and len(X) > max_points:
            X = X[rng.choice(len(X), max_points, replace=False)]
        Xs.append(X)
    T = len(Xs)
    offsets = np.cumsum([0] + [len(X) for X in Xs])
    P = np.vstack(Xs)
    # simplices are built per consecutive union only (never across non-adjacent clouds), which
    # keeps the complex to what the zigzag actually sees
    present = {}   # vertex tuple (global ids) -> [appear, disappear)
    for i in range(T):
        pair = P[offsets[i]:offsets[i + 2]] if i + 1 < T else P[offsets[i]:offsets[i + 1]]
        loc = np.arange(offsets[i], offsets[i + 2] if i + 1 < T else offsets[i + 1])
        f = d.fill_rips(pair, maxdim + 1, float(scale))
        for s in f:
            key = tuple(sorted(int(loc[v]) for v in s))
            tag = np.unique([0 if k < offsets[i + 1] else 1 for k in key])
            if len(tag) == 1 and tag[0] == 0:
                present.setdefault(key, [max(2 * i - 1, 0), 2 * i + 2])            # all in X_i: [2i-1, 2i+1]
            elif len(tag) == 1:
                present.setdefault(key, [2 * i + 1, 2 * i + 4])                    # all in X_{i+1}: [2i+1, 2i+3]
            else:
                present.setdefault(key, [2 * i + 1, 2 * i + 2])                    # mixed: {2i+1}
    n_pos = 2 * T - 1
    keys = sorted(present, key=lambda k: (len(k), k))
    f2 = d.Filtration([d.Simplex(list(k)) for k in keys])
    times2 = [[present[k][0], min(present[k][1], n_pos)] if present[k][1] <= n_pos else [present[k][0]] for k in keys]
    zz, dgms, cells = d.zigzag_homology_persistence(f2, times2)
    bars = []
    for k in range(maxdim + 1):
        if k < len(dgms):
            B = np.array([[p.birth, p.death] for p in dgms[k]], dtype=np.float64).reshape(-1, 2)
            B[:, 1] = np.where(B[:, 1] >= n_pos, np.inf, B[:, 1])
        else:
            B = np.zeros((0, 2))
        bars.append(B)
    return ZigzagResult(bars=bars, n_positions=n_pos, scale=float(scale), n_points=len(P), n_simplices=len(f2))
