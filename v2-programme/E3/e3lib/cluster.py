"""Overlapping cover of prompt space: PCA + fuzzy c-means + relative-membership rule."""
from __future__ import annotations

import numpy as np


def preprocess(emb, n_pca=20):
    X = (emb - emb.mean(0)) / (emb.std(0) + 1e-8)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_pca].T


def fcm(X, k, m=1.5, n_init=10, max_iter=300, tol=1e-6, seed=0):
    """Fuzzy c-means (Bezdek). Returns memberships U [N,k] (rows sum to 1), centers, objective.
    Centres are initialised from k random data points (SPEC amendment 1: a near-uniform
    membership initialisation collapses every centre onto the global mean, a degenerate
    fixed point with U ≡ 1/k)."""
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
    M = np.zeros_like(U, dtype=bool)
    M[np.arange(len(U)), U.argmax(1)] = True
    return M


def composition(members, metas, keys):
    """Per cluster: histogram of each meta key (dict of dicts)."""
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


def knn_centre_cover(X, k, n_nearest=2, seed=0, n_init=10):
    """SPEC amendment 1: k-means++ centres; U_i = prompts for which centre i is among the
    n_nearest nearest centres. Returns (members bool [N,k], distances [N,k], centres)."""
    from sklearn.cluster import KMeans
    km = KMeans(k, n_init=n_init, random_state=seed).fit(X)
    d2 = ((X[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
    order = np.argsort(d2, 1)
    M = np.zeros((len(X), k), bool)
    for r in range(n_nearest):
        M[np.arange(len(X)), order[:, r]] = True
    return M, d2, km.cluster_centers_
