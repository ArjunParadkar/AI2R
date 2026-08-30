"""Overlapping cover of examples from an embedding: PCA -> a cover method -> the
nerve. Ported (numpy/scikit-learn only) from whitebox-v2 E3's `e3lib/cluster.py`
plus the `cover_from_embedding` orchestration in `e3lib/pipeline.py`.

E3's own Amendment 1 (SPEC.md) found the originally pre-registered method
(fuzzy c-means + a relative-membership threshold, `method="fcm"`) degenerate on
real embeddings (every centre collapses onto the global mean). The method that
actually worked end-to-end on a real model is the **2-nearest-centre cover**
(`method="knn2"`, the default here): k-means++ centres, each example belongs to
the clusters of its two nearest centres. `method="fcm"` is kept for the record
and for callers who want the original construction on their own data (it is
not degenerate on every dataset, only on the E3 embeddings), with the same
adaptive-theta fallback E3 used.
"""
from __future__ import annotations

import numpy as np


def preprocess(emb, n_pca=20):
    """Standardise each dimension, then PCA to n_pca components (SPEC S2)."""
    emb = np.asarray(emb, dtype=float)
    n_pca = min(n_pca, emb.shape[0] - 1, emb.shape[1])
    X = (emb - emb.mean(0)) / (emb.std(0) + 1e-8)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_pca].T


def fcm(X, k, m=1.5, n_init=10, max_iter=300, tol=1e-6, seed=0):
    """Fuzzy c-means (Bezdek). Returns memberships U [N,k] (rows sum to 1),
    centres, objective. Centres are initialised from k random data points (SPEC
    Amendment 1: a near-uniform membership initialisation collapses every
    centre onto the global mean, a degenerate fixed point with U == 1/k)."""
    N = X.shape[0]
    best = None
    for init in range(n_init):
        rng = np.random.default_rng(seed + init)
        C = X[rng.choice(N, k, replace=False)]
        obj_prev = np.inf
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1) + 1e-12   # [N,k]
            inv = d2 ** (-1.0 / (m - 1.0))
            U = inv / inv.sum(1, keepdims=True)
            Um = U ** m
            C = (Um.T @ X) / (Um.sum(0)[:, None] + 1e-12)
            obj = float((Um * d2).sum())
            if abs(obj_prev - obj) < tol * max(1.0, obj):
                break
            obj_prev = obj
        if best is None or obj < best[2]:
            best = (U, C, obj)
    return best


def soft_cover(U, theta=0.5):
    """prompt p in U_i iff u_ip >= theta * max_j u_jp (argmax always included). -> bool [N,k]"""
    return U >= theta * U.max(1, keepdims=True) - 1e-12


def hard_cover(U):
    """Argmax-only cover (no overlaps) -- the degenerate check (kill ii): a
    nerve built on this has zero edges by construction."""
    M = np.zeros_like(U, dtype=bool)
    M[np.arange(len(U)), U.argmax(1)] = True
    return M


def knn_centre_cover(X, k, n_nearest=2, seed=0, n_init=10):
    """SPEC Amendment 1: k-means++ centres; U_i = examples for which centre i is
    among the n_nearest nearest centres. Returns (members bool [N,k], squared
    distances [N,k], centres)."""
    from sklearn.cluster import KMeans
    n_clusters = min(k, len(X))
    km = KMeans(n_clusters, n_init=n_init, random_state=seed).fit(X)
    d2 = ((X[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
    order = np.argsort(d2, 1)
    M = np.zeros((len(X), n_clusters), bool)
    for r in range(min(n_nearest, n_clusters)):
        M[np.arange(len(X)), order[:, r]] = True
    return M, d2, km.cluster_centers_


def composition(members, metas, keys):
    """Per cluster: histogram of each meta key (list of dicts). `metas` is a
    list of per-example dicts (e.g. {"template": ..., "order": ...})."""
    out = []
    for i in range(members.shape[1]):
        idx = np.where(members[:, i])[0]
        row = {"cluster": i, "size": int(len(idx))}
        for key in keys:
            vals = [metas[j][key] for j in idx]
            u, c = np.unique(vals, return_counts=True)
            order = np.argsort(-c)
            row[key] = {str(u[o]): int(c[o]) for o in order}
        out.append(row)
    return out


def _cyclomatic(n_v, edges):
    from .core import _UF
    uf = _UF(n_v)
    for a, b in edges:
        uf.union(a, b)
    return len(edges) - n_v + len({uf.find(v) for v in range(n_v)})


def cover_from_embedding(emb, k, theta=0.5, n_pca=20, seed=0, n_min=10, adaptive=True, m=1.5,
                         method="knn2"):
    """Overlapping cover from a real-valued embedding of the examples.

    method="knn2" (default; SPEC Amendment 1): 2-nearest-centre cover on
        k-means++ centres. Every example lies in exactly two clusters; no
        fuzzifier or threshold needed. Triple overlaps are empty by
        construction (the triangle diagnostic is therefore vacuous under this
        method -- reported, not hidden, via `tris=[]`).
    method="fcm": the originally pre-registered fuzzy c-means + relative-
        membership rule. Degenerate on the E3 embeddings (every centre
        collapsed onto the global mean); kept for callers whose data does not
        hit that degeneracy, with the same adaptive theta-lowering fallback
        (0.5 -> 0.33 if the nerve has no cycle) E3 used.

    Returns dict(U, members, edges, tris, overlap, theta, fcm_obj).
    """
    from . import core as sh
    X = preprocess(emb, n_pca)
    if method == "knn2":
        members, d2, _ = knn_centre_cover(X, k, 2, seed)
        U = -d2                                   # "membership" proxy: argmax = nearest centre
        edges, tris, ov = sh.nerve(members, n_min)
        return dict(U=U, members=members, edges=edges, tris=tris, overlap=ov, theta=None, fcm_obj=None)
    if method != "fcm":
        raise ValueError(f"method must be 'knn2' or 'fcm', got {method!r}")
    U, _, obj = fcm(X, k, m=m, seed=seed)
    members = soft_cover(U, theta)
    edges, tris, ov = sh.nerve(members, n_min)
    used_theta = theta
    if adaptive and _cyclomatic(k, edges) < 1 and theta > 0.33:
        members = soft_cover(U, 0.33)
        edges, tris, ov = sh.nerve(members, n_min)
        used_theta = 0.33
    return dict(U=U, members=members, edges=edges, tris=tris, overlap=ov, theta=used_theta, fcm_obj=obj)
