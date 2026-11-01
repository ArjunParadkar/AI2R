"""Section 2.1-2.3 of the spec.

The cloud A_l = {h_l(x_i)} samples M_l.  We estimate it with a kNN graph, local tangent
spaces by PCA, local intrinsic dimension by the two-NN estimator, and we always report the
manifold score m_l -- the honesty mechanism.  Edges are weighted by the pullback metric, so
"distance" everywhere below means "how much the network's output moves", never raw Euclidean.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from sklearn.neighbors import NearestNeighbors


def knn_graph(X, k=15, J=None, metric='euclidean', candidates=80, out=None):
    """Symmetric kNN graph on the cloud.

    metric='euclidean' takes neighbours in the ambient space.  metric='g' takes them under the
    pullback metric instead: a candidate pool of the `candidates` nearest Euclidean points is
    re-ranked by ||J_i (h_j - h_i)||, i.e. by how much the network's output actually moves.
    That is what "nearest on the Riemannian manifold" means, and it matters -- in activation
    space two points can be close while the network treats them completely differently.

    metric='secant' is the Jacobian-free pullback: neighbours are re-ranked by
    ||psi(h_j) - psi(h_i)||, the chord in output space.  By the mean-value theorem this is the
    integral of J along the segment, so for neighbouring points it agrees with the pullback
    length to first order -- and it needs only the cached outputs `out`, so it works at layers
    far too wide to differentiate (conv feature maps, LLM residual streams).

    Returns (indices, dists, csr adjacency).
    """
    n = len(X)
    k = min(k, n - 1)
    if metric == 'secant':
        if out is None:
            raise ValueError("metric='secant' needs the cached outputs")
        c = min(candidates, n - 1)
        nn = NearestNeighbors(n_neighbors=c + 1).fit(X)
        _, cand = nn.kneighbors(X)
        cand = cand[:, 1:]
        gd = np.linalg.norm(out[cand] - out[:, None, :], axis=2)
        sel = np.argsort(gd, axis=1)[:, :k]
        idx = np.take_along_axis(cand, sel, 1)
        dist = np.take_along_axis(gd, sel, 1)
    elif metric == 'g':
        if J is None:
            raise ValueError("metric='g' needs the Jacobians")
        c = min(candidates, n - 1)
        nn = NearestNeighbors(n_neighbors=c + 1).fit(X)
        _, cand = nn.kneighbors(X)
        cand = cand[:, 1:]                                   # (n, c)
        delta = X[cand] - X[:, None, :]                      # (n, c, D)
        gd = np.linalg.norm(np.einsum('nod,ncd->nco', J, delta), axis=2)   # (n, c)
        sel = np.argsort(gd, axis=1)[:, :k]
        idx = np.take_along_axis(cand, sel, 1)
        dist = np.take_along_axis(gd, sel, 1)
    else:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        dist, idx = nn.kneighbors(X)
        dist, idx = dist[:, 1:], idx[:, 1:]
    rows = np.repeat(np.arange(n), k)
    A = csr_matrix((np.maximum(dist.ravel(), 1e-12), (rows, idx.ravel())), shape=(n, n))
    A = A.maximum(A.T)
    return idx, dist, A


def tangent_pca(X, idx, dim_cap=None):
    """Local tangent space at every point by PCA on its neighbourhood.

    Batched: one SVD call over an (n, k, D) stack rather than a Python loop, which is the
    difference between 40 s and 2 s on a 2,000-point probe and so decides whether a
    checkpoint sweep is affordable at all.

    Returns (bases, spectra): bases[i] is (D, d_cap) orthonormal, spectra[i] the eigenvalues.
    """
    n, D = X.shape
    k = idx.shape[1]
    d_cap = min(dim_cap or k, k, D)
    N = X[idx]                                   # (n, k, D)
    N = N - N.mean(1, keepdims=True)
    U, sv, Vt = np.linalg.svd(N, full_matrices=False)
    lam = (sv ** 2) / max(k - 1, 1)              # (n, min(k,D))
    bases = np.ascontiguousarray(np.transpose(Vt[:, :d_cap, :], (0, 2, 1))).astype(np.float32)
    spectra = lam.astype(np.float32)
    return bases, spectra


def intrinsic_dim(X, idx=None, k=15, ranks=None):
    """Local intrinsic dimension by the two-NN estimator, evaluated at the graph scale.

    The textbook TwoNN uses the ratio r2/r1 of the first two neighbour distances.  On real
    activation clouds -- and on any noisy sample of a manifold -- the r1/r2 shell sits inside
    the noise ball, so that estimator reports the ambient noise dimension rather than the
    dimension of the manifold.  (Measured, tests/test_manifold.py: a 1-D circle in R^10 with
    sigma=0.01 noise, n=785: d = 6.9 at ranks (1,2); at the graph scale d = 2.28 for k=15,
    1.30 for k=30, 1.07 for k=60.  The graph-scale estimate is only as good as k is large
    relative to the noise ball -- at the default k=15 it still over-reads a noisy circle.)

    We therefore use the general-rank MLE at the same scale the kNN graph and the tangent PCA
    use, ranks (k//2, k).  For a d-dimensional manifold r_k ~ k^(1/d), and exactly

        E[log(r_k2 / r_k1)] = (H_{k2-1} - H_{k1-1}) / d,

    so d_hat = (H_{k2-1} - H_{k1-1}) / mean(log(r_k2/r_k1)).  At ranks (1,2) this reduces to
    the familiar d = n / sum(log mu).  Both scales are returned; the graph-scale one is used
    downstream and the fine-scale one is reported as a diagnostic.
    """
    n = len(X)
    k1, k2 = ranks if ranks is not None else (max(1, k // 2), k)
    need = max(k2, 2) + 1
    nn = NearestNeighbors(n_neighbors=min(need, n)).fit(X)
    dist, ids = nn.kneighbors(X)
    eps = 1e-12
    H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, need + 2))])   # H[m] = sum_{j<=m} 1/j
    def ratio_dim(a, b):
        lr = np.log(np.maximum(dist[:, b], eps) / np.maximum(dist[:, a], eps))
        norm = H[b - 1] - H[a - 1]
        return lr, norm
    lr, norm = ratio_dim(k1, k2)
    lr_fine, norm_fine = ratio_dim(1, 2)
    if idx is None:
        idx = ids[:, 1:k + 1]
    nb_mean = (lr[idx].sum(1) + lr) / (idx.shape[1] + 1)      # mean over {i} u N(i)
    d_local = np.where(nb_mean > eps, norm / np.maximum(nb_mean, eps), np.nan)
    d_global = norm / max(lr.mean(), eps)
    d_global_fine = norm_fine / max(lr_fine.mean(), eps)
    return d_local, dict(d_global=float(d_global), d_global_fine_scale=float(d_global_fine),
                         ranks=(int(k1), int(k2)))


def manifold_score(spectra, d_local, D, gap_ratio=3.0, dim_frac=0.2, rel_floor=1e-8):
    """m_l in [0,1]: the fraction of points that look like they sit on a low-dim manifold.

    A point counts iff (i) its local PCA spectrum has a gap of at least `gap_ratio` between
    consecutive eigenvalues and (ii) the dimension implied by that gap, and the TwoNN
    estimate, are both <= dim_frac * D  (the spec's "d << d_l").

    The gap is searched only among eigenvalues above rel_floor * lambda_1.  A neighbourhood of
    k points spans at most k-1 dimensions, so the tail of the spectrum is exactly zero; left in,
    that numerical zero produces an enormous spurious gap at the last index and every point
    fails.  Every downstream answer is annotated with the m this returns.
    """
    n = len(spectra)
    lam = np.asarray(spectra, dtype=np.float64)
    lam1 = np.maximum(lam[:, :1], 1e-300)
    valid = lam > rel_floor * lam1                     # numerically meaningful eigenvalues
    rank = valid.sum(1)
    gap_at = np.ones(n, dtype=int)
    gap_sz = np.zeros(n)
    for i in range(n):
        r = int(rank[i])
        if r < 2:
            gap_at[i], gap_sz[i] = 1, np.inf           # a genuine 1-D direction and nothing else
            continue
        l = lam[i, :r]
        ratios = l[:-1] / np.maximum(l[1:], 1e-300)
        a = int(ratios.argmax())
        gap_at[i], gap_sz[i] = a + 1, float(ratios[a])
    dim_lim = max(1.0, dim_frac * D)
    ok = (gap_sz >= gap_ratio) & (gap_at <= dim_lim) & (np.nan_to_num(d_local, nan=1e9) <= dim_lim)
    return dict(m=float(ok.mean()), pca_dim=gap_at, gap=gap_sz,
                d_twonn_median=float(np.nanmedian(d_local)), D=int(D),
                gap_ratio=gap_ratio, dim_frac=dim_frac)


def pullback_metric_lengths(X, J, A):
    """Re-weight every graph edge by the pullback metric (spec 2.2).

        ||h_j - h_i||_g^2 = delta^T Gbar delta,  Gbar = 1/2 (J_i^T J_i + J_j^T J_j)

    which equals 1/2(||J_i delta||^2 + ||J_j delta||^2) -- computed that way, so we never
    form a d_l x d_l matrix.  J has shape (n, d_out, d_l).
    """
    A = A.tocoo()
    i, j = A.row, A.col
    delta = X[j] - X[i]
    Ji_d = np.einsum('nod,nd->no', J[i], delta)
    Jj_d = np.einsum('nod,nd->no', J[j], delta)
    w = np.sqrt(0.5 * ((Ji_d ** 2).sum(1) + (Jj_d ** 2).sum(1)) + 1e-30)
    return csr_matrix((w, (i, j)), shape=A.shape)


def secant_metric_lengths(out, A):
    """Edge lengths ||psi(h_j) - psi(h_i)||: the Jacobian-free pullback metric."""
    A = A.tocoo()
    w = np.linalg.norm(out[A.col] - out[A.row], axis=1) + 1e-12
    return csr_matrix((w, (A.row, A.col)), shape=A.shape)


def geodesic_distances(W, sources=None, return_pred=False):
    """Discrete geodesics = Dijkstra under the g-weighted graph (spec 2.3)."""
    out = dijkstra(W, directed=False, indices=sources, return_predecessors=return_pred)
    return out


def geodesic(W, i, j):
    """The discrete geodesic from i to j: the node path and its g-length."""
    d, pred = dijkstra(W, directed=False, indices=i, return_predecessors=True)
    if not np.isfinite(d[j]):
        return None, np.inf
    path, cur = [j], j
    while cur != i:
        cur = pred[cur]
        if cur < 0:
            return None, np.inf
        path.append(cur)
    return np.array(path[::-1]), float(d[j])


@dataclass
class ManifoldFit:
    """Everything section 2 asks for, computed once and reused by all seven problems."""
    X: np.ndarray
    idx: np.ndarray
    A: csr_matrix                 # Euclidean-weighted kNN graph
    W: csr_matrix                 # g-weighted kNN graph (the one that matters)
    bases: np.ndarray
    spectra: np.ndarray
    d_local: np.ndarray
    d_global: float
    score: dict
    J: np.ndarray = None
    timings: dict = field(default_factory=dict)

    @property
    def m(self):
        return self.score['m']

    def n_components(self):
        return connected_components(self.W, directed=False)[0]


def fit_manifold(X, J=None, k=15, dim_cap=8, metric='euclidean', out=None, score_k=None):
    """One call: graph, tangents, dimension, manifold score, pullback-weighted graph.

    The manifold score is computed on its own neighbourhood of size score_k = clip(4 * d, k, 96),
    where d is the two-NN dimension.  A neighbourhood of k points spans at most k-1 directions,
    so k = 15 cannot show a spectral gap for a 12-dimensional manifold no matter how clean it
    is -- it scored a trained ResNet's penultimate layer at m = 0.0005 for exactly that reason.
    """
    import time
    t = {}
    t0 = time.time(); idx, _, A = knn_graph(X, k, J=J, metric=metric, out=out); t['knn'] = time.time() - t0
    t0 = time.time(); bases, spectra = tangent_pca(X, idx, dim_cap); t['tangent_pca'] = time.time() - t0
    t0 = time.time(); d_local, dinfo = intrinsic_dim(X, idx, k); t['intrinsic_dim'] = time.time() - t0
    if score_k is None:
        score_k = int(np.clip(4 * np.ceil(dinfo['d_global']), k, min(96, len(X) - 2)))
    if score_k > k:
        t0 = time.time()
        idx_s = NearestNeighbors(n_neighbors=score_k + 1).fit(X).kneighbors(X)[1][:, 1:]
        sub = np.random.default_rng(0).choice(len(X), size=min(len(X), 600), replace=False)
        _, spec_s = tangent_pca(X, idx_s[sub], 2)
        score = manifold_score(spec_s, d_local[sub], X.shape[1])
        t['score'] = time.time() - t0
    else:
        score = manifold_score(spectra, d_local, X.shape[1])
    score.update(dinfo); score['score_k'] = int(score_k)
    t0 = time.time()
    if metric == 'secant':
        W = secant_metric_lengths(out, A)
    elif J is not None:
        W = pullback_metric_lengths(X, J, A)
    else:
        W = A
    t['pullback'] = time.time() - t0
    return ManifoldFit(X=X, idx=idx, A=A, W=W, bases=bases, spectra=spectra,
                       d_local=d_local, d_global=dinfo['d_global'], score=score, J=J, timings=t)
