"""WHITEBOX: persistent homology of activation dynamics.

Implements (per WHITEBOX, NeurIPS 2026):
  Definition 1: Activation point cloud A^l(t)
  Definition 2: Topological film F^l = (D^l_k(t))_t
  Definition 3: Structural Velocity SV^l(t) = W_2(D^l_k(t), D^l_k(t-1))
  Definition 4: Topological Complexity TC^l(t) = sum (d_i - b_i)^2
  Definition 5: Feature Persistence Lifetime FPL(phi)
  Definition 6: Concept Crystallization Epoch t_c*
  Definition 7: BiasTop^l(t) = W_2(D^l_k(t; s=0), D^l_k(t; s=1))
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from ripser import ripser
from persim import wasserstein
from scipy.spatial.distance import pdist


def fit_pca(X: np.ndarray, n_components: int = 30, random_state: int = 0) -> PCA:
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)
    return pca


def persistence_diagrams(X, pca=None, maxdim=1, thresh_percentile=95.0):
    """VR persistent homology with sparse threshold. Returns [D0, D1, ...]."""
    if pca is not None:
        X = pca.transform(X)
    pd = pdist(X)
    thresh = float(np.percentile(pd, thresh_percentile))
    res = ripser(X, maxdim=maxdim, thresh=thresh)
    out = []
    for D in res["dgms"]:
        D = np.asarray(D, dtype=float)
        if D.size == 0:
            out.append(np.empty((0, 2)))
        else:
            mask = np.isfinite(D[:, 1])
            out.append(D[mask])
    return out


def structural_velocity(D_t, D_tm1) -> float:
    A = D_t if D_t.size else np.empty((0, 2))
    B = D_tm1 if D_tm1.size else np.empty((0, 2))
    if A.size == 0 and B.size == 0:
        return 0.0
    return float(wasserstein(A, B))


def topological_complexity(D) -> float:
    if D.size == 0:
        return 0.0
    return float(np.sum((D[:, 1] - D[:, 0]) ** 2))


def betti_at_scale(D, scale) -> int:
    if D.size == 0:
        return 0
    return int(np.sum((D[:, 0] <= scale) & (D[:, 1] > scale)))


def bias_top(D_a, D_b) -> float:
    return structural_velocity(D_a, D_b)


def feature_persistence_lifetimes(diagram_film):
    """Return distribution of consecutive-epoch run lengths above median."""
    counts = []
    medians = []
    for D in diagram_film:
        if D.size:
            lts = D[:, 1] - D[:, 0]
            counts.append(len(lts))
            medians.append(float(np.median(lts)))
        else:
            counts.append(0)
            medians.append(0.0)
    if not medians:
        return np.array([0])
    global_med = float(np.median([m for m in medians if m > 0])) if any(medians) else 0.0
    above = []
    for D in diagram_film:
        if D.size:
            lts = D[:, 1] - D[:, 0]
            above.append(int(np.sum(lts > global_med)))
        else:
            above.append(0)
    above = np.array(above)
    thresh = np.median(above)
    runs, cur = [], 0
    for v in above:
        if v > thresh:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return np.array(runs) if runs else np.array([0])


def permutation_pvalue_bias(X_a, X_b, pca, n_permutations=500, rng=None, maxdim=1):
    rng = rng or np.random.default_rng(0)
    Da = persistence_diagrams(X_a, pca, maxdim=maxdim)
    Db = persistence_diagrams(X_b, pca, maxdim=maxdim)
    k = 1 if (Da[1].size and Db[1].size) else 0
    obs = bias_top(Da[k], Db[k])
    pooled = np.vstack([X_a, X_b])
    n_a = len(X_a)
    null = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(pooled))
        Xa = pooled[perm[:n_a]]
        Xb = pooled[perm[n_a:]]
        Da_p = persistence_diagrams(Xa, pca, maxdim=maxdim)
        Db_p = persistence_diagrams(Xb, pca, maxdim=maxdim)
        k_p = 1 if (Da_p[1].size and Db_p[1].size) else 0
        null.append(bias_top(Da_p[k_p], Db_p[k_p]))
    null = np.array(null)
    p = float((1 + np.sum(null >= obs)) / (1 + len(null)))
    return obs, p, null
