"""Invisible displacement and neighbourhood persistence (run 2, block B).

Run 1's ignorance score d_g pushed the off-manifold displacement THROUGH the Jacobian of the
remaining network.  That keeps only what the output can see and deletes the rest -- and an
out-of-distribution input is displaced mostly along directions the head ignores.  This module
measures the deleted part instead.

    delta      = h - hbar                 hbar = nearest training activation (Euclidean)
    delta_par  = J^+ J delta              the part the remaining network reads (row space of J)
    delta_perp = (I - J^+ J) delta        the part it cannot see (kernel of J)
    iota       = ||delta_perp|| / median in-distribution ||delta_perp||
    kappa      = mean Jaccard overlap of the k nearest-training-activation index sets between
                 consecutive layers (1 = the same neighbours at every depth)
    combined   = iota * (1 - kappa)

The pseudo-inverse is truncated at rcond * sigma_max (rcond = 1e-3 as pre-registered), so the
"kernel" is the numerical kernel at that tolerance.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

__all__ = ['row_space', 'kernel_split', 'invisible_displacement', 'iota', 'knn_sets', 'jaccard_rows',
           'kappa', 'combined', 'auroc', 'summary']


def row_space(J, rcond=1e-3):
    """Orthonormal basis of the numerical row space of J.

    J : (o, d) or a stack (m, o, d).  Returns (V, rank) with V of shape (..., r_max, d) whose
    rows beyond each matrix's own rank are zeroed (so V^T V is the projector J^+ J), and rank an
    int or an (m,) int array.  Singular values <= rcond * sigma_max are treated as zero.
    """
    J = np.asarray(J, dtype=np.float64)
    single = J.ndim == 2
    if single:
        J = J[None]
    _, S, Vt = np.linalg.svd(J, full_matrices=False)               # Vt: (m, min(o,d), d)
    keep = S > rcond * S[:, :1]
    keep &= S > 0
    V = Vt * keep[:, :, None]
    rank = keep.sum(1).astype(int)
    return (V[0], int(rank[0])) if single else (V, rank)


def kernel_split(delta, J, rcond=1e-3):
    """Split displacements into the part J reads and the part in its kernel.

    delta : (n, d).  J : one (o, d) matrix shared by every row, or (n, o, d) one per row.
    Returns (delta_par, delta_perp, rank): delta_par = J^+ J delta, delta_perp = delta - delta_par.
    """
    delta = np.asarray(delta, dtype=np.float64)
    J = np.asarray(J)
    if J.ndim == 2:
        V, rank = row_space(J, rcond)
        par = (delta @ V.T) @ V
        rank = np.full(len(delta), rank, int)
    else:
        V, rank = row_space(J, rcond)
        par = np.einsum('nr,nrd->nd', np.einsum('nrd,nd->nr', V, delta), V)
    return par, delta - par, rank


def invisible_displacement(H, X_train, J, rcond=1e-3, chunk=256):
    """B1 for a batch of activations.

    H        (n, d)  activations to score
    X_train  (N, d)  training activations; hbar is the Euclidean nearest one
    J        (o, d) one Jacobian for the whole layer (linear head), or (N, o, d) one per training
             activation, or a callable ids -> (len(ids), o, d) evaluated only at the nearest
             training activations that are actually used.
    Returns dict: perp, par, total (norms, (n,)), ratio = perp/total, nn_idx, rank (per point),
    kernel_dim (per point).
    """
    H = np.asarray(H, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=1).fit(X_train)
    dist, idx = nn.kneighbors(H)
    idx = idx[:, 0]
    delta = H - X_train[idx]
    d = H.shape[1]
    if not callable(J) and np.asarray(J).ndim == 2:
        par, perp, rank = kernel_split(delta, J, rcond)
    else:
        uniq, inv = np.unique(idx, return_inverse=True)
        par = np.zeros_like(delta); rank = np.zeros(len(H), int)
        Ju = J(uniq) if callable(J) else None
        for s in range(0, len(uniq), chunk):
            u = uniq[s:s + chunk]
            Jc = Ju[s:s + chunk] if Ju is not None else np.asarray(J)[u]
            V, r = row_space(Jc, rcond)
            rows = np.where((inv >= s) & (inv < s + len(u)))[0]
            Vr = V[inv[rows] - s]
            par[rows] = np.einsum('nr,nrd->nd', np.einsum('nrd,nd->nr', Vr, delta[rows]), Vr)
            rank[rows] = r[inv[rows] - s]
        perp = delta - par
    tot = np.linalg.norm(delta, axis=1)
    pn, an = np.linalg.norm(perp, axis=1), np.linalg.norm(par, axis=1)
    return dict(perp=pn, par=an, total=tot, ratio=pn / np.maximum(tot, 1e-30), nn_idx=idx,
                rank=rank, kernel_dim=d - rank)


def iota(perp_norm, perp_norm_in):
    """||delta_perp|| in units of the median in-distribution ||delta_perp||."""
    return np.asarray(perp_norm) / max(float(np.median(perp_norm_in)), 1e-30)


def knn_sets(X_train, H, k=15):
    """(n, k) indices of the k nearest training activations of each row of H (Euclidean)."""
    k = min(k, len(X_train))
    return NearestNeighbors(n_neighbors=k).fit(X_train).kneighbors(H)[1]


def jaccard_rows(A, B):
    """Row-wise Jaccard index |A_i n B_i| / |A_i u B_i| of two (n, k) integer index arrays."""
    A, B = np.asarray(A), np.asarray(B)
    out = np.empty(len(A))
    for i in range(len(A)):
        a, b = set(A[i].tolist()), set(B[i].tolist())
        out[i] = len(a & b) / max(len(a | b), 1)
    return out


def kappa(sets_per_layer):
    """Mean Jaccard of the neighbour index sets between CONSECUTIVE layers.

    sets_per_layer : list (length L >= 2) of (n, k) index arrays, in depth order, all indexing
    the same training set.  Returns (kappa (n,), per_pair (L-1, n)).
    """
    if len(sets_per_layer) < 2:
        raise ValueError('kappa needs at least two layers')
    per = np.stack([jaccard_rows(sets_per_layer[i], sets_per_layer[i + 1])
                    for i in range(len(sets_per_layer) - 1)])
    return per.mean(0), per


def combined(iota_v, kappa_v):
    return np.asarray(iota_v) * (1.0 - np.asarray(kappa_v))


def auroc(score_in, score_out):
    """AUROC of 'higher score = out-of-distribution'."""
    from sklearn.metrics import roc_auc_score
    s = np.r_[score_in, score_out].astype(np.float64)
    s = np.nan_to_num(s, nan=0.0, posinf=np.finfo(np.float64).max, neginf=-np.finfo(np.float64).max)
    return float(roc_auc_score(np.r_[np.zeros(len(score_in)), np.ones(len(score_out))], s))


def summary(v):
    v = np.asarray(v, dtype=float)
    q = np.percentile(v, [25, 50, 75])
    return dict(mean=float(v.mean()), q25=float(q[0]), median=float(q[1]), q75=float(q[2]))
