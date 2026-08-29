"""Vietoris–Rips persistent homology of an activation point cloud.

Definition 1/2 of the paper: the activation point cloud A^l(t) and its
persistence diagrams D^l_k(t). Follows the original pipeline exactly:

  * optional fixed linear reduction (see ``reduce``),
  * VR filtration truncated at the ``thresh_percentile`` of pairwise distances
    (95th by default — the "sparse Rips" of the original code),
  * infinite bars are DROPPED (the original code masked non-finite deaths),
  * maxdim=1 by default (H0 and H1).

Backends (all produce the same diagrams up to floating point, except
``landmark`` which is an approximation and is validated in
``benchmarks/phase1_validate_approx.py``):

  ripser      exact, single-threaded C++ (Bauer 2021). Reference backend.
  giotto-ph   exact, multi-threaded Ripser core (Burella Pérez et al. 2021)
              with optional edge collapse (Boissonnat & Pritam 2020), also
              exact. Default when installed. Edge collapse is OFF by default:
              on dense 30-dim activation clouds it was 10-25x SLOWER
              (benchmarks/PHASE1.md), it only pays off on sparse/low-dim data.
  landmark    maxmin (farthest-point) landmark subsample of size ``n_landmarks``
              then exact PH on the landmarks (de Silva & Carlsson 2004).
              Approximate; cost is independent of n once n > n_landmarks.

Ripser++ (Zhang et al. 2020) — a CUDA backend — is NOT used: it ships wheels
only up to cp37 and needs nvcc to build. The GPU is used for the pairwise
distance matrix only (``distance_device='cuda'``).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.spatial.distance import pdist, squareform

from .reduce import Reducer


@dataclass
class Diagrams:
    """Finite persistence diagrams for one point cloud, one per homology dim."""
    dgms: list  # list[np.ndarray (n_k, 2)]
    thresh: float
    n_points: int
    backend: str
    meta: dict = field(default_factory=dict)

    def __getitem__(self, k: int) -> np.ndarray:
        return self.dgms[k]

    def __len__(self) -> int:
        return len(self.dgms)

    @property
    def maxdim(self) -> int:
        return len(self.dgms) - 1

    def lifetimes(self, k: int) -> np.ndarray:
        D = self.dgms[k]
        return D[:, 1] - D[:, 0] if D.size else np.empty(0)


def available_backends() -> list[str]:
    out = ["ripser", "landmark"]
    try:
        import gph  # noqa: F401
        out.insert(0, "giotto-ph")
    except ImportError:
        pass
    return out


def _finite(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    if D.size == 0:
        return np.empty((0, 2))
    return D[np.isfinite(D[:, 1])]


def pairwise_distances(X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Full (n, n) Euclidean distance matrix; on GPU via torch if asked."""
    if device != "cpu":
        import torch
        T = torch.as_tensor(X, dtype=torch.float32, device=device)
        D = torch.cdist(T, T).double().cpu().numpy()
        np.fill_diagonal(D, 0.0)
        return np.maximum(D, D.T)  # enforce symmetry after fp32 cdist
    return squareform(pdist(X))


def landmarks_maxmin(D: np.ndarray, n_landmarks: int, seed: int = 0) -> np.ndarray:
    """Farthest-point (maxmin) landmark selection on a distance matrix."""
    n = D.shape[0]
    if n_landmarks >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(n))]
    mind = D[idx[0]].copy()
    for _ in range(n_landmarks - 1):
        j = int(np.argmax(mind)); idx.append(j)
        mind = np.minimum(mind, D[j])
    return np.asarray(idx)


def _run_ripser(D, maxdim, thresh):
    from ripser import ripser
    return ripser(D, maxdim=maxdim, thresh=thresh, distance_matrix=True)["dgms"]


def _run_gph(D, maxdim, thresh, n_threads=-1, collapse_edges=True):
    from gph import ripser_parallel
    return ripser_parallel(D, maxdim=maxdim, thresh=thresh, metric="precomputed",
                           n_threads=n_threads, collapse_edges=collapse_edges)["dgms"]


def compute_diagrams(
    X: np.ndarray,
    reducer: Optional[Reducer] = None,
    maxdim: int = 1,
    thresh_percentile: float = 95.0,
    backend: str = "auto",
    thresh: Optional[float] = None,
    n_landmarks: int = 200,
    n_threads: int = -1,
    collapse_edges: bool = False,
    distance_device: str = "cpu",
    seed: int = 0,
) -> Diagrams:
    """Persistence diagrams of a point cloud (rows = points).

    Parameters
    ----------
    X : (n, d) activations. Reduced with ``reducer`` if given.
    maxdim : highest homology dimension to compute (1 = loops).
    thresh_percentile : VR filtration is truncated at this percentile of the
        pairwise distances, unless ``thresh`` is given explicitly. 100 = full.
    backend : 'auto' | 'ripser' | 'giotto-ph' | 'landmark'.
    n_landmarks : landmark count for the 'landmark' backend.
    distance_device : 'cpu' or a torch device for the distance matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    if reducer is not None:
        X = reducer.transform(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_points, n_features); got {X.shape}")
    n = X.shape[0]
    if n < 3:
        return Diagrams([np.empty((0, 2)) for _ in range(maxdim + 1)], 0.0, n, backend)
    if backend == "auto":
        backend = available_backends()[0]
    Dm = pairwise_distances(X, distance_device)
    meta = {}
    if backend == "landmark":
        li = landmarks_maxmin(Dm, n_landmarks, seed)
        Dm = Dm[np.ix_(li, li)]; meta["landmarks"] = li
    if thresh is None:
        thresh = float(np.percentile(Dm[np.triu_indices(Dm.shape[0], 1)], thresh_percentile))
    if backend == "ripser":
        dgms = _run_ripser(Dm, maxdim, thresh)
    elif backend in ("giotto-ph", "landmark"):
        if "giotto-ph" in available_backends():
            dgms = _run_gph(Dm, maxdim, thresh, n_threads, collapse_edges)
        else:
            dgms = _run_ripser(Dm, maxdim, thresh)
    else:
        raise ValueError(f"unknown backend {backend!r}; available: {available_backends()}")
    return Diagrams([_finite(D) for D in dgms], float(thresh), n, backend, meta)
