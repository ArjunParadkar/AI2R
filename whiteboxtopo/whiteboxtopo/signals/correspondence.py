"""whiteboxtopo.representation_correspondence — does the model's internal
representation preserve the topology of the data manifold? (Sub-H4:
understanding as structure-preserving mapping.)

Primary instrument: **RTD, Representation Topology Divergence** (Barannikov,
Trofimov, Sotnikov, Trimbach, Filippov, Burnaev; ICML 2022). Two point clouds
P (data) and P~ (activations) with a one-to-one correspondence of points
(same inputs). Each distance matrix is normalised by its 0.9-quantile, then
the R-Cross-Barcode is the persistence barcode of the 2n-vertex filtration

        [[ 0      , r_P            ],
         [ r_P    , min(r_P, r_P~) ]]

and RTD_1 = sum of H1 lifetimes, symmetrised: (RTD(P,P~) + RTD(P~,P))/2,
averaged over random batches of 500 points. RTD = 0 iff the two clouds have
the same neighbourhood structure at every scale; large RTD = the map
creates/destroys topological features. This follows the reference
implementation (github.com/IlyaTrofimov/RTD, ``calc_embed_dist``) exactly,
with giotto-ph/ripser in place of Ripser++.

Calibrated output: RTD is a divergence in arbitrary units, so the signal
reports  correspondence = 1 - RTD(P, P~) / RTD(P, shuffle(P~))  clipped to
[0, 1]: 1 = the representation is topologically identical to the data, 0 =
no better than a random pairing of the same activation cloud (this null keeps
the representation's own topology and destroys only the correspondence).
Bootstrap over batches gives the interval.

Secondary (pairing-free) instrument: 2-Wasserstein distance between the H1
diagrams of the two clouds after the same normalisation — "same shape",
regardless of which point maps where.

Pre-registered hypotheses (April proposal), tested in benchmarks/PHASE3.md:
 (a) correspondence high for generalising models,
 (b) near zero for memorising models,
 (c) predictive of held-out generalisation without using held-out data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import numpy as np

from ..core.persistence import available_backends, pairwise_distances, _finite
from ..core.reduce import fit_reducer
from ..core.bootstrap import summarize, BootEstimate
from ..metrics.sv import diagram_distance
from ..metrics.tc import topological_complexity


def _ph_lower(Dm: np.ndarray, maxdim: int, backend: str):
    """Persistence of an arbitrary symmetric non-negative matrix (not necessarily a metric)."""
    if backend == "auto":
        backend = available_backends()[0]
    if backend == "giotto-ph":
        from gph import ripser_parallel
        return ripser_parallel(Dm, maxdim=maxdim, metric="precomputed", n_threads=-1)["dgms"]
    from ripser import ripser
    return ripser(Dm, maxdim=maxdim, distance_matrix=True)["dgms"]


def normalized_distances(X: np.ndarray, q: float = 0.9, device: str = "cpu") -> np.ndarray:
    D = pairwise_distances(np.asarray(X, dtype=np.float64), device)
    return D / max(np.quantile(D, q), 1e-12)


def cross_barcode(r1: np.ndarray, r2: np.ndarray, dim: int = 1, backend: str = "auto") -> list:
    """R-Cross-Barcode(P, P~) from normalised distance matrices r1 (P) and r2 (P~)."""
    n = r1.shape[0]; d = np.zeros((2 * n, 2 * n))
    d[n:, :n] = r1; d[:n, n:] = r1.T; d[n:, n:] = np.minimum(r1, r2)
    d[d < r1.mean() * 1e-6] = 0.0
    return [_finite(D) for D in _ph_lower(d, dim, backend)]


def rtd_once(r1, r2, backend="auto") -> float:
    b = cross_barcode(r1, r2, 1, backend)[1]
    return float(np.sum(b[:, 1] - b[:, 0])) if b.size else 0.0


def rtd(P: np.ndarray, Q: np.ndarray, n_trials: int = 10, batch: int = 500, seed: int = 0, backend: str = "auto",
        device: str = "cpu", symmetric: bool = True) -> BootEstimate:
    """Representation Topology Divergence between paired clouds P and Q (rows correspond)."""
    P, Q = np.asarray(P, dtype=np.float64), np.asarray(Q, dtype=np.float64)
    assert len(P) == len(Q), "P and Q must be paired (same number of rows, same order)"
    rng = np.random.default_rng(seed); n = len(P); m = min(batch, n); vals = []
    for _ in range(n_trials):
        idx = rng.choice(n, m, replace=False)
        r1, r2 = normalized_distances(P[idx], device=device), normalized_distances(Q[idx], device=device)
        v = rtd_once(r1, r2, backend)
        if symmetric:
            v = 0.5 * (v + rtd_once(r2, r1, backend))
        vals.append(v)
    return summarize(vals, float(np.mean(vals)), m)


def diagram_similarity(P: np.ndarray, Q: np.ndarray, device: str = "cpu", backend: str = "auto") -> float:
    """Pairing-free: W2 between the H1 diagrams of the two normalised clouds."""
    r1, r2 = normalized_distances(P, device=device), normalized_distances(Q, device=device)
    D1 = _finite(_ph_lower(r1, 1, backend)[1]); D2 = _finite(_ph_lower(r2, 1, backend)[1])
    return diagram_distance(D1, D2)


@dataclass
class Correspondence:
    score: float           # 1 - RTD/RTD_null, clipped to [0,1]
    ci_low: float
    ci_high: float
    rtd: BootEstimate
    rtd_null: BootEstimate
    diagram_w2: float
    n: int
    layer: object = None

    def __repr__(self):
        return (f"Correspondence(layer={self.layer}, score={self.score:.3f} CI95=[{self.ci_low:.3f},{self.ci_high:.3f}], "
                f"RTD={self.rtd.point:.3g} null={self.rtd_null.point:.3g}, W2(H1)={self.diagram_w2:.3g}, n={self.n})")


def correspondence(P: np.ndarray, Q: np.ndarray, n_trials: int = 10, batch: int = 500, seed: int = 0,
                   backend: str = "auto", device: str = "cpu", layer=None) -> Correspondence:
    """Calibrated correspondence between paired clouds (data P, representation Q)."""
    P, Q = np.asarray(P, dtype=np.float64), np.asarray(Q, dtype=np.float64)
    r = rtd(P, Q, n_trials, batch, seed, backend, device)
    rng = np.random.default_rng(seed + 1); Qp = Q[rng.permutation(len(Q))]
    r0 = rtd(P, Qp, n_trials, batch, seed, backend, device)
    ratio = r.samples / np.maximum(r0.samples, 1e-12)
    s = np.clip(1.0 - ratio, 0, 1); point = float(np.clip(1.0 - r.point / max(r0.point, 1e-12), 0, 1))
    lo, hi = np.percentile(s, [2.5, 97.5])
    w2 = diagram_similarity(P[:batch], Q[:batch], device, backend)
    return Correspondence(point, float(lo), float(hi), r, r0, w2, len(P), layer)


def representation_correspondence(model=None, data=None, layers: Optional[Sequence] = None, activations: Optional[Dict] = None,
                                  data_points: Optional[np.ndarray] = None, n_components: Optional[int] = None,
                                  n_trials: int = 10, batch: int = 500, seed: int = 0, device: str = "cpu", backend: str = "auto") -> Dict[object, Correspondence]:
    """Correspondence between the data manifold and each layer's representation.

    Pass ``model``, ``data`` (iterable of batches whose first element is the
    input tensor; the flattened inputs are the data cloud unless
    ``data_points`` is given, e.g. a known manifold parameterisation), and
    ``layers`` (module names); or pass ``activations`` = {layer: (n, d)} and
    ``data_points`` = (n, p) directly. ``n_components``: optional PCA of the
    activations before RTD (RTD itself is dimension-free; PCA only for speed).
    """
    if activations is None:
        if model is None or data is None or layers is None:
            raise ValueError("pass activations= and data_points=, or model=, data=, layers=")
        import torch
        from ..activations import ActivationExtractor
        batches = list(data); ex = ActivationExtractor(model, list(layers)); activations = ex(batches); ex.remove()
        if data_points is None:
            data_points = np.concatenate([(b[0] if isinstance(b, (tuple, list)) else b).reshape(len(b[0] if isinstance(b, (tuple, list)) else b), -1).cpu().numpy() for b in batches])
    if data_points is None:
        raise ValueError("data_points is required with activations=")
    out = {}
    for l, A in activations.items():
        A = np.asarray(A, dtype=np.float64)
        if n_components:
            A = fit_reducer(A, n_components).transform(A)
        out[l] = correspondence(data_points, A, n_trials, batch, seed, backend, device, layer=l)
    return out
