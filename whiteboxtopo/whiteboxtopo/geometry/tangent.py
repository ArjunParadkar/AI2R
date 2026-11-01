"""Local tangent spaces and second fundamental form of an activation cloud (SPEC 2.11).

Tangent space ``T_h M`` by PCA of the k-NN neighbourhood; the normal component
of the neighbours is fitted by a quadratic form ``Q_h`` (the second
fundamental form estimate) and the principal curvatures are its eigenvalues.

Nulls these were validated against (tests/test_geometry.py): a flat plane,
whose fitted ``Q`` must be ~0 (this is the null - a curvature estimator that
reports structure on a plane is reporting noise), and a unit circle embedded
in R^3, whose principal curvature must come out at 1 within 25 %.

Failure boundary.
(1) Both estimators are *local sample* statistics. With ``k`` too small the
quadratic fit is underdetermined (it needs k > 1 + m + m(m+1)/2 points, m the
tangent dimension) and the min-norm least-squares solution silently shrinks Q
toward 0; with ``k`` too large the neighbourhood is no longer a graph over the
tangent plane and the fit is meaningless.
(2) Noise of size sigma over a neighbourhood of radius r biases curvature
upward by roughly sigma/r^2 - on a plane with noise the estimate is not 0 but
a noise floor, which is why the plane test asserts "small relative to the
circle", not "zero".
(3) For codimension > 1 the true second fundamental form is vector-valued.
We project the normal displacements onto their dominant direction and fit a
scalar-valued form there, so ``norm`` is a lower bound on the full
vector-valued norm. The normal direction is oriented toward the mean normal
displacement of the neighbours, which fixes the otherwise arbitrary sign: a
locally convex neighbourhood (sphere, circle) gets positive curvature.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors

EXPLAINED_TARGET = 0.90
MAX_INTRINSIC_DIM = 5


def _neighborhoods(X: np.ndarray, k: int):
    """(idx, V) with idx (n, k) the k nearest *other* points and V (n, k, d) their offsets from the point."""
    n = X.shape[0]
    k = int(min(k, n - 1))
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    idx = idx[:, 1:]
    return idx, X[idx] - X[:, None, :]


def tangent_pca(cloud, k: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """Local PCA tangent spaces (SPEC 2.11). Returns ``(bases, explained)``.

    For each point, PCA of the point together with its ``k`` nearest
    neighbours (centred on the neighbourhood mean). The intrinsic dimension
    ``m`` is the smallest number of components reaching 90 % explained
    variance, **capped at 5** and at ``min(n_dims, k)``; because the returned
    ``bases`` is one array, a single global ``m`` is used - the median of the
    per-point dimensions (documented deviation from a strictly per-point
    dimension, which would be ragged).

    * ``bases`` (n, n_dims, m): orthonormal columns spanning the estimated tangent space.
    * ``explained`` (n, m): explained-variance ratio of each retained component.

    Deterministic: no randomness (PCA signs are not fixed and are irrelevant -
    only the span is used downstream).
    """
    X = np.asarray(cloud, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError("cloud must be (n_points, n_dims) with n_points >= 3")
    n, D = X.shape
    k = int(min(k, n - 1))
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    P = X[idx]                                              # (n, k+1, D), self included
    P = P - P.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)        # batched
    var = S ** 2
    tot = np.where(var.sum(axis=1, keepdims=True) > 0, var.sum(axis=1, keepdims=True), 1.0)
    ratio = var / tot
    cum = np.cumsum(ratio, axis=1)
    dims = np.argmax(cum >= EXPLAINED_TARGET, axis=1) + 1
    dims = np.minimum(dims, MAX_INTRINSIC_DIM)
    m = int(np.clip(int(np.median(dims)), 1, min(MAX_INTRINSIC_DIM, D, k, Vt.shape[1])))
    bases = np.transpose(Vt[:, :m, :], (0, 2, 1))           # (n, D, m)
    return bases, ratio[:, :m]


@dataclass
class CurvatureResult:
    """Second fundamental form estimate per point (SPEC 2.11)."""
    principal: np.ndarray       # (n, m) principal curvatures = eigenvalues of Q_h, ascending
    norm: np.ndarray            # (n,) spectral norm ||Q_h||_2
    mean_curvature: np.ndarray  # (n,) trace(Q_h)/m
    dim: int                    # intrinsic (tangent) dimension m used
    k: int
    residual: np.ndarray        # (n,) rms residual of the quadratic fit, relative to the normal displacements

    def to_dict(self) -> Dict[str, object]:
        return {"dim": int(self.dim), "k": int(self.k), "n_points": int(self.norm.size),
                "norm_median": float(np.median(self.norm)), "norm_mean": float(self.norm.mean()),
                "principal_median": float(np.median(self.principal)),
                "residual_median": float(np.median(self.residual)),
                "norm": self.norm.tolist(), "principal": self.principal.tolist(),
                "mean_curvature": self.mean_curvature.tolist()}


def _quadratic_design(U: np.ndarray) -> np.ndarray:
    """(n, k, p) design [1, u_a, u_a^2, u_a u_b (a<b)] for s = c + l.u + 1/2 u^T Q u."""
    n, k, m = U.shape
    cols = [np.ones((n, k)), *[U[:, :, a] for a in range(m)]]
    for a in range(m):
        cols.append(U[:, :, a] ** 2)
    for a in range(m):
        for b in range(a + 1, m):
            cols.append(U[:, :, a] * U[:, :, b])
    return np.stack(cols, axis=-1)


def curvature(cloud, k: int = 15) -> CurvatureResult:
    """Per-point curvature of the cloud (SPEC 2.11): eigenvalues of the fitted ``Q_h``.

    For each point: tangent basis ``T`` from :func:`tangent_pca`; neighbour
    offsets split into tangent coordinates ``u = T^T v`` and normal residual
    ``r = v - T T^T v``; the scalar normal displacement ``s = r . e`` along the
    dominant normal direction ``e`` (oriented toward the mean residual) is fitted
    by ordinary least squares with ``s = c + l.u + 1/2 u^T Q u``. The linear and
    constant terms absorb the tilt/offset of the PCA plane, which is estimated
    from the same finite sample. Principal curvatures are ``eigvalsh(Q)``,
    ``norm`` is ``max |eig(Q)|``.

    Deterministic: no randomness. Validated on a plane (Q ~ 0) and on a unit
    circle in R^3 (principal curvature 1). See the module docstring for the
    failure boundary - in particular ``k`` must exceed ``1 + m + m(m+1)/2``.
    """
    X = np.asarray(cloud, dtype=np.float64)
    n, D = X.shape
    k = int(min(k, n - 1))
    T, _ = tangent_pca(X, k=k)                              # (n, D, m)
    m = T.shape[2]
    idx, V = _neighborhoods(X, k)                           # V (n, k, D)
    U = np.einsum("nkd,ndm->nkm", V, T)                     # tangent coords
    R = V - np.einsum("nkm,ndm->nkd", U, T)                 # normal residual in ambient coords
    e = R.mean(axis=1)                                      # mean normal displacement direction
    ne = np.linalg.norm(e, axis=1)
    weak = ne <= 1e-14 * (np.linalg.norm(R, axis=(1, 2)) + 1e-300)
    if weak.any():                                          # degenerate: fall back to the top PC of R
        _, _, Vt = np.linalg.svd(R[weak], full_matrices=False)
        e[weak] = Vt[:, 0, :]
        ne[weak] = 1.0
    e = e / np.where(ne > 0, ne, 1.0)[:, None]
    s = np.einsum("nkd,nd->nk", R, e)                       # (n, k)

    Dm = _quadratic_design(U)                               # (n, k, p)
    coef = np.einsum("nkp,nk->np", np.transpose(np.linalg.pinv(Dm), (0, 2, 1)), s)
    fit = np.einsum("nkp,np->nk", Dm, coef)
    denom = np.sqrt((s ** 2).mean(axis=1))
    resid = np.sqrt(((s - fit) ** 2).mean(axis=1)) / np.where(denom > 0, denom, 1.0)

    Q = np.zeros((n, m, m))
    for a in range(m):
        Q[:, a, a] = 2.0 * coef[:, 1 + m + a]               # u_a^2 coefficient is q_aa / 2
    c = 1 + 2 * m
    for a in range(m):
        for b in range(a + 1, m):
            Q[:, a, b] = Q[:, b, a] = coef[:, c]
            c += 1
    ev = np.linalg.eigvalsh(Q)
    return CurvatureResult(principal=ev, norm=np.abs(ev).max(axis=1), mean_curvature=np.trace(Q, axis1=1, axis2=2) / m,
                           dim=m, k=k, residual=resid)
