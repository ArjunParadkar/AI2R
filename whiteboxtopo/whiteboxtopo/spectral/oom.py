"""P4, "what it does not know": out-of-manifold distance in the diffusion
embedding, and its SPEC 6.5 null (random Gaussian projection).

Procedure (SPEC 6.5). Split the in-distribution cloud into a reference part
and a held-out part; build the diffusion embedding on the reference part only;
Nystrom-extend held-out and new points into it; score each point by
``d_M(h*) = min_{h in reference} d_t(h*, h)``; the decision threshold is the
99th percentile of the *held-out* d_M, so it is calibrated on points the
embedding never saw.

Null it was validated against: the same scoring on a random Gaussian
projection of the raw cloud (:func:`random_projection_null`) - the diffusion
embedding is only doing work if it separates better than that. Test case:
a noisy unit circle vs points at radius 3 (AUC > 0.9 for held-out vs OOD).

Failure boundary. (1) ``d_M`` is a distance to the *reference sample*, not to
the manifold: with few reference points held-out d_M is large too and the
threshold goes up with it. (2) The score saturates - every point far enough
out has a Nystrom image near 0 and therefore nearly the same d_M, so d_M
ranks near-OOD inputs but does not measure how far out a far point is.
(3) Nothing here is calibrated as a probability; "flagged" means "beyond the
99th percentile of held-out in-distribution", nothing more.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Sequence
import numpy as np
from scipy.spatial.distance import cdist

from .laplacian import laplacian, diffusion_coords, nystrom, default_eps


@dataclass
class OOMResult:
    """Out-of-manifold scores for the query points (SPEC 6.5)."""
    d_M: np.ndarray            # (n_new,) min diffusion distance to a reference point
    threshold: float           # 99th percentile of the held-out in-distribution d_M
    flags: np.ndarray          # (n_new,) bool, d_M > threshold
    d_M_in: np.ndarray         # (n_held,) the held-out in-distribution scores
    embedding: str             # 'diffusion' or 'random_projection'
    n_ref: int = 0
    n_eigs: int = 0
    t: int = 1
    k: int = 15
    seed: int = 0
    lam: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_dict(self) -> Dict[str, object]:
        return {"embedding": self.embedding, "threshold": float(self.threshold),
                "n_new": int(self.d_M.size), "n_ref": int(self.n_ref), "n_held": int(self.d_M_in.size),
                "n_eigs": int(self.n_eigs), "t": int(self.t), "k": int(self.k), "seed": int(self.seed),
                "n_flagged": int(self.flags.sum()), "frac_flagged": float(self.flags.mean()) if self.flags.size else 0.0,
                "d_M": self.d_M.tolist(), "flags": self.flags.astype(bool).tolist(),
                "d_M_in_median": float(np.median(self.d_M_in)) if self.d_M_in.size else float("nan"),
                "lam": np.asarray(self.lam).tolist()}


def _split(n: int, seed: int, frac_ref: float = 0.8):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_ref = int(round(frac_ref * n))
    n_ref = int(np.clip(n_ref, 3, n - 2))
    return perm[:n_ref], perm[n_ref:]


def out_of_manifold(cloud, h_new, k: int = 15, t: int = 1, n_eigs: int = 10, seed: int = 0,
                    frac_ref: float = 0.8) -> OOMResult:
    """SPEC 6.5 out-of-manifold score for ``h_new`` against the cloud ``cloud``.

    Deterministic given ``seed`` (the only randomness is the reference/held-out
    split). Returns :class:`OOMResult` with ``d_M``, ``threshold``, ``flags``
    and the held-out in-distribution scores ``d_M_in`` (so the caller can
    compute the SPEC 6.5 AUC without re-scoring).
    """
    X = np.asarray(cloud, dtype=np.float64)
    Q = np.atleast_2d(np.asarray(h_new, dtype=np.float64))
    if X.ndim != 2 or Q.shape[1] != X.shape[1]:
        raise ValueError("cloud (n,d) and h_new (m,d) must share the ambient dimension")
    ref_i, held_i = _split(X.shape[0], seed, frac_ref)
    ref = X[ref_i]
    k = int(min(k, ref.shape[0] - 1))
    eps = default_eps(ref, k)
    L, W = laplacian(ref, k=k, eps=eps)
    n_eigs = int(min(n_eigs, ref.shape[0] - 2))
    Psi_ref, lam, phi = diffusion_coords(L, t=t, n_eigs=n_eigs)
    params = {"k": k, "eps": eps, "degrees": np.asarray(W.sum(axis=1)).ravel()}
    scale = np.maximum(1.0 - lam, 0.0) ** float(t)

    def _dM(points):
        phi_new = nystrom(ref, params, points, phi, lam)
        return cdist(phi_new * scale[None, :], Psi_ref).min(axis=1)

    d_in = _dM(X[held_i])
    d_new = _dM(Q)
    thr = float(np.percentile(d_in, 99))
    return OOMResult(d_M=d_new, threshold=thr, flags=d_new > thr, d_M_in=d_in, embedding="diffusion",
                     n_ref=ref.shape[0], n_eigs=n_eigs, t=t, k=k, seed=seed, lam=lam)


def random_projection_null(cloud, h_new, d: int = 10, seed: int = 0, frac_ref: float = 0.8) -> OOMResult:
    """SPEC 6.5 null: identical OOD scoring on a random Gaussian projection.

    Same split, same "min distance to a reference point, threshold at the 99th
    percentile of held-out" rule, but the embedding is ``X @ G / sqrt(d)`` with
    ``G ~ N(0,1)^{D x d}`` instead of the diffusion map. A diffusion-embedding
    result is only evidence for SPEC 2.10 if it beats this at the same ``d``.

    Returns the same :class:`OOMResult` shape (``embedding='random_projection'``,
    ``lam`` empty). Deterministic given ``seed``.
    """
    X = np.asarray(cloud, dtype=np.float64)
    Q = np.atleast_2d(np.asarray(h_new, dtype=np.float64))
    rng = np.random.default_rng(seed)
    ref_i, held_i = _split(X.shape[0], seed, frac_ref)
    d = int(min(d, X.shape[1])) if X.shape[1] > 0 else 1
    G = rng.standard_normal((X.shape[1], d)) / np.sqrt(d)
    P_ref = X[ref_i] @ G
    d_in = cdist(X[held_i] @ G, P_ref).min(axis=1)
    d_new = cdist(Q @ G, P_ref).min(axis=1)
    thr = float(np.percentile(d_in, 99))
    return OOMResult(d_M=d_new, threshold=thr, flags=d_new > thr, d_M_in=d_in,
                     embedding="random_projection", n_ref=P_ref.shape[0], n_eigs=d, t=1, k=0, seed=seed)
