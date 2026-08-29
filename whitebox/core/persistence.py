"""Vietoris–Rips persistent homology of an activation point cloud.

Definition 1/2 of the paper: the activation point cloud A^l(t) and its
persistence diagrams D^l_k(t). Follows the original pipeline exactly:

  * optional fixed linear reduction (see ``reduce``),
  * VR filtration truncated at the ``thresh_percentile`` of pairwise distances
    (95th by default — the "sparse Rips" of the original code),
  * infinite bars are DROPPED (the original code masked non-finite deaths),
  * maxdim=1 by default (H0 and H1).

Backends:
  ripser      exact, single-threaded C++ (the original backend; reference)
  giotto-ph   exact, multi-threaded (Bauer's Ripser core + parallel apparent pairs)
  auto        giotto-ph if importable, else ripser

Approximate/landmark backends land in Phase 1 and are validated against these.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
from scipy.spatial.distance import pdist

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
    out = ["ripser"]
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


def _run_ripser(X, maxdim, thresh):
    from ripser import ripser
    return ripser(X, maxdim=maxdim, thresh=thresh)["dgms"]


def _run_gph(X, maxdim, thresh, n_threads=-1):
    from gph import ripser_parallel
    return ripser_parallel(X, maxdim=maxdim, thresh=thresh, n_threads=n_threads)["dgms"]


def compute_diagrams(
    X: np.ndarray,
    reducer: Optional[Reducer] = None,
    maxdim: int = 1,
    thresh_percentile: float = 95.0,
    backend: str = "auto",
    thresh: Optional[float] = None,
) -> Diagrams:
    """Persistence diagrams of a point cloud (rows = points).

    Parameters
    ----------
    X : (n, d) activations. Reduced with ``reducer`` if given.
    maxdim : highest homology dimension to compute (1 = loops).
    thresh_percentile : VR filtration is truncated at this percentile of the
        pairwise distances, unless ``thresh`` is given explicitly.
    backend : 'auto' | 'ripser' | 'giotto-ph'.
    """
    X = np.asarray(X, dtype=np.float64)
    if reducer is not None:
        X = reducer.transform(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_points, n_features); got {X.shape}")
    n = X.shape[0]
    if n < 3:
        return Diagrams([np.empty((0, 2)) for _ in range(maxdim + 1)], 0.0, n, backend)
    if thresh is None:
        thresh = float(np.percentile(pdist(X), thresh_percentile))
    if backend == "auto":
        backend = available_backends()[0]
    if backend == "ripser":
        dgms = _run_ripser(X, maxdim, thresh)
    elif backend == "giotto-ph":
        dgms = _run_gph(X, maxdim, thresh)
    else:
        raise ValueError(f"unknown backend {backend!r}; available: {available_backends()}")
    return Diagrams([_finite(D) for D in dgms], float(thresh), n, backend)
