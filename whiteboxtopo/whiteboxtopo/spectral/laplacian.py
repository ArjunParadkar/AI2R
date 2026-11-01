"""Graph Laplacian and diffusion coordinates of an activation cloud (SPEC 2.10).

Construction, exactly as SPEC 2.10 states it: k-NN graph on the cloud with
Gaussian weights ``w_ij = exp(-||h_i - h_j||^2 / eps)``, symmetrised, normalized
Laplacian ``L = I - D^{-1/2} W D^{-1/2}``.

Nulls these were validated against (tests/test_spectral.py): a noisy circle,
whose diffusion embedding must recover the circular coordinate (this is the
Belkin-Niyogi limit behaving), and isotropic Gaussian blobs, whose small-
eigenvalue count must equal the number of separated components and *not* the
number of points.

Failure boundary. (1) ``eps`` set from the median squared k-NN distance is a
scale estimate, not a limit: with n small or eps far from the median the
operator is no longer close to the Laplace-Beltrami operator and nothing here
is a manifold statement. (2) The graph is built from pairwise distances, so it
is permutation-invariant but NOT invariant under per-unit scalings of the
cloud - standardise the cloud first (``whiteboxtopo.topology.scaling``) if you
want a gauge-invariant answer, and record that as gauge-fixing. (3) If the
k-NN graph is disconnected the eigenvalue 0 has multiplicity equal to the
number of components and the "second" eigenvector is a component indicator,
not a coordinate.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

DENSE_MAX = 1500          # above this many points the eigensolver switches to ARPACK


def default_eps(cloud: np.ndarray, k: int = 15) -> float:
    """SPEC 2.10 default kernel width: the median squared k-NN distance (self excluded)."""
    X = np.asarray(cloud, dtype=np.float64)
    k = int(min(k, max(1, X.shape[0] - 1)))
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    d2 = dist[:, 1:] ** 2
    eps = float(np.median(d2))
    if not np.isfinite(eps) or eps <= 0:                     # duplicated points
        pos = d2[d2 > 0]
        eps = float(np.median(pos)) if pos.size else 1.0
    return eps


def _knn_kernel(ref: np.ndarray, query: np.ndarray, k: int, eps: float, drop_self: bool):
    """(idx, w) of the k nearest reference points of each query point and their Gaussian weights."""
    n_ref = ref.shape[0]
    k = int(min(k, n_ref - 1 if drop_self else n_ref))
    nn = NearestNeighbors(n_neighbors=k + (1 if drop_self else 0)).fit(ref)
    dist, idx = nn.kneighbors(query)
    if drop_self:
        dist, idx = dist[:, 1:], idx[:, 1:]
    return idx, np.exp(-(dist ** 2) / eps)


def laplacian(cloud: np.ndarray, k: int = 15, eps: Optional[float] = None):
    """Normalized graph Laplacian of an activation cloud (SPEC 2.10).

    Returns ``(L, W)``: ``L = I - D^{-1/2} W D^{-1/2}`` and the symmetrised
    weight matrix ``W``, both ``scipy.sparse`` CSR. ``w_ij = exp(-||h_i-h_j||^2/eps)``
    on the k-NN graph (self-loops excluded), symmetrised with ``max(W, W^T)`` so
    that an edge survives if *either* endpoint chose the other (the usual
    diffusion-map convention; ``(W+W^T)/2`` would halve one-sided edges).
    ``eps`` defaults to the median squared k-NN distance.

    Deterministic: no randomness. ``k`` is clipped to ``n-1``.
    """
    X = np.asarray(cloud, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError("cloud must be (n_points, n_dims) with n_points >= 3")
    n = X.shape[0]
    k = int(min(k, n - 1))
    eps = float(eps) if eps is not None else default_eps(X, k)
    if eps <= 0:
        raise ValueError("eps must be positive")
    idx, w = _knn_kernel(X, X, k, eps, drop_self=True)
    rows = np.repeat(np.arange(n), idx.shape[1])
    W = sp.coo_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n)).tocsr()
    W = W.maximum(W.T)
    W.setdiag(0.0)
    W.eliminate_zeros()
    deg = np.asarray(W.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0))
    Dm = sp.diags(dinv)
    S = (Dm @ W @ Dm).tocsr()
    L = (sp.identity(n, format="csr") - S).tocsr()
    L = ((L + L.T) * 0.5).tocsr()            # kill the 1e-17 asymmetry
    return L, W.tocsr()


def _fix_signs(V: np.ndarray) -> np.ndarray:
    """Deterministic eigenvector sign: the largest-|.| entry of each column is positive."""
    V = np.asarray(V, dtype=np.float64)
    piv = np.argmax(np.abs(V), axis=0)
    s = np.sign(V[piv, np.arange(V.shape[1])])
    s[s == 0] = 1.0
    return V * s[None, :]


def smallest_eigs(L, n_eigs: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """``n_eigs`` smallest eigenpairs of the normalized Laplacian, ascending.

    Computed as the *largest* eigenpairs of ``S = I - L`` (same eigenvectors,
    ``lam_S = 1 - lam_L``), which is the numerically well-conditioned end.
    Dense for ``n <= DENSE_MAX``; ARPACK with a fixed start vector above it, so
    the result is deterministic either way.
    """
    n = L.shape[0]
    n_eigs = int(min(n_eigs, n))
    S = sp.identity(n, format="csr") - sp.csr_matrix(L)
    if n <= DENSE_MAX or n_eigs >= n - 1:
        vals, vecs = sla.eigh(S.toarray())
        order = np.argsort(vals)[::-1][:n_eigs]
        lam_s, V = vals[order], vecs[:, order]
    else:
        lam_s, V = eigsh(S, k=n_eigs, which="LA", v0=np.full(n, 1.0 / np.sqrt(n)))
        order = np.argsort(lam_s)[::-1]
        lam_s, V = lam_s[order], V[:, order]
    return np.clip(1.0 - lam_s, 0.0, 2.0), _fix_signs(V)


def diffusion_coords(L, t: int = 1, n_eigs: int = 10):
    """Diffusion coordinates of the cloud ``L`` was built from (SPEC 2.10).

    Returns ``(Psi, lam, phi)``:

    * ``lam`` (n_eigs,)  - the smallest *Laplacian* eigenvalues with the
      trivial one (lam_0 = 0, the degree eigenvector) dropped;
    * ``phi`` (n, n_eigs) - the matching eigenvectors, sign-fixed;
    * ``Psi`` (n, n_eigs) - ``Psi_t(h)_i = (1 - lam_i)^t * phi_i(h)``.

    Convention. SPEC 2.10 writes ``Psi_t = (lam_i^t phi_i)`` with ``lam_i`` the
    eigenvalues of the *diffusion operator* ``S = D^{-1/2} W D^{-1/2}``. Here
    ``lam`` always denotes eigenvalues of ``L = I - S``, so the multiplier is
    ``(1-lam_i)^t``; the two agree. Euclidean distance in ``Psi`` is the
    diffusion distance ``d_t``.

    Failure boundary: on a disconnected graph the dropped "trivial" eigenvector
    is only one of the several 0-eigenvectors, so the first retained
    coordinates are component indicators rather than within-component
    coordinates - check ``lam[0]`` (it should be well above 0).
    """
    lam_all, V_all = smallest_eigs(L, n_eigs + 1)
    lam, phi = lam_all[1:], V_all[:, 1:]                       # drop the trivial pair
    Psi = phi * (np.maximum(1.0 - lam, 0.0) ** float(t))[None, :]
    return Psi, lam, phi


def kernel_params(cloud: np.ndarray, k: int = 15, eps: Optional[float] = None) -> Dict[str, object]:
    """The ``W_params`` bundle ``nystrom`` needs: ``{'k', 'eps', 'degrees'}``.

    ``degrees`` are the row sums of the *symmetrised* k-NN weight matrix, i.e.
    exactly the ``D`` used in the Laplacian, so the extension uses the same
    normalisation as the eigenproblem it extends.
    """
    X = np.asarray(cloud, dtype=np.float64)
    k = int(min(k, X.shape[0] - 1))
    eps = float(eps) if eps is not None else default_eps(X, k)
    _, W = laplacian(X, k=k, eps=eps)
    return {"k": k, "eps": eps, "degrees": np.asarray(W.sum(axis=1)).ravel()}


def nystrom(cloud: np.ndarray, W_params, h_new: np.ndarray, phi: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Nystrom extension of the eigenvectors ``phi`` to new points (SPEC 2.10 / 6.5).

    ``S phi_i = (1-lam_i) phi_i`` with ``S(x,y) = w(x,y)/sqrt(d(x) d(y))``, so for
    a new point ``x*``::

        phi_i(x*) = 1/(1-lam_i) * sum_j  w(x*,h_j)/sqrt(d(x*) d(h_j)) * phi_i(h_j)

    with the sum over the ``k`` nearest reference points (the same k-NN
    truncation the graph itself uses) and ``d(x*) = sum_j w(x*,h_j)``.
    ``W_params`` is the mapping from :func:`kernel_params` (``k``, ``eps`` and
    optionally precomputed ``degrees``); a plain ``{'k':..,'eps':..}`` works and
    the degrees are recomputed.

    Returns ``phi_new`` (n_new, n_eigs). Deterministic.

    Failure boundary: for a point far outside the cloud every ``w(x*,h_j)``
    underflows, ``d(x*) -> 0`` and the extension tends to 0 - which is what
    makes it usable as an out-of-manifold score, but it is NOT an estimate of
    "where on the manifold" such a point sits. Eigenvectors with ``lam`` near 1
    are divided by a small number and are the noisiest to extend.
    """
    X = np.asarray(cloud, dtype=np.float64)
    Q = np.atleast_2d(np.asarray(h_new, dtype=np.float64))
    phi = np.asarray(phi, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    k = int(W_params["k"]) if "k" in W_params else 15
    eps = float(W_params["eps"])
    deg = W_params.get("degrees")
    if deg is None:
        deg = kernel_params(X, k=k, eps=eps)["degrees"]
    deg = np.asarray(deg, dtype=np.float64)
    idx, w = _knn_kernel(X, Q, min(k, X.shape[0]), eps, drop_self=False)
    d_new = w.sum(axis=1)
    d_new = np.where(d_new > 0, d_new, np.finfo(float).tiny)
    dj = np.where(deg[idx] > 0, deg[idx], 1.0)
    s = w / (np.sqrt(d_new)[:, None] * np.sqrt(dj))            # (n_new, k)
    num = np.einsum("nk,nkm->nm", s, phi[idx])                 # (n_new, n_eigs)
    scale = np.maximum(1.0 - lam, 1e-12)
    return num / scale[None, :]
