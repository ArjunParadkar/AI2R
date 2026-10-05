"""Discrete Morse-Smale complex of a scalar function on the pullback-weighted kNN graph (SPEC2, block C).

C1  f lives on the nodes.  Every node flows to its highest-f neighbour (to itself if no neighbour is higher);
    following the flow ends at a local maximum, and the basin of a node is that terminal maximum.  A SEPARATRIX
    EDGE is a graph edge whose endpoints lie in different basins AND carry different labels (argmax).
C2  d_sep(x) = min over separatrix edges e=(u,v) of [geodesic(x,u) + w(e)/2]  -- the geodesic distance to the
    edge's midpoint, the midpoint being a virtual node half-way along the edge.
C3  at the midpoint of each separatrix edge, a quadratic model of g = psi_y - psi_y' in tangent coordinates,
    g ~ c + b.u + u^T Q u.  The zero set of g is the decision boundary between y and y'; its second
    fundamental form is the Hessian restricted to the boundary's tangent space over the gradient norm, so
    ||II|| = ||P (2Q) P||_F / ||b||,  P = I - b b^T / |b|^2.
    rho'(x) = d_sep(x) / (1 + ||II|| at the nearest separatrix edge).

Nothing here depends on torch; psi enters only through cached outputs at the probe points.
"""
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, triu
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

__all__ = ['ascending_flow', 'morse_smale', 'separatrix_distance', 'boundary_curvature', 'rho_prime',
           'quadratic_design']


def ascending_flow(W, f):
    """(succ, basin): succ[i] = highest-f neighbour of i if it is strictly higher than i, else i;
    basin[i] = the local maximum reached by iterating succ."""
    W = csr_matrix(W)
    f = np.asarray(f, dtype=np.float64)
    n = W.shape[0]
    succ = np.arange(n)
    ip, ix = W.indptr, W.indices
    for i in range(n):
        nb = ix[ip[i]:ip[i + 1]]
        if len(nb) == 0:
            continue
        j = nb[np.argmax(f[nb])]
        if f[j] > f[i]:
            succ[i] = j
    basin = succ.copy()
    for _ in range(n):                                   # pointer jumping; f strictly increases along succ, no cycles
        nxt = basin[basin]
        if np.array_equal(nxt, basin):
            break
        basin = nxt
    return succ, basin


def morse_smale(W, f, labels):
    """The complex: flow, basins, maxima, and separatrix edges (different basin AND different label).

    Returns a dict; `sep_edges` is (E, 2) with u < v, `sep_w` their graph lengths.  The two relaxed edge counts
    (basin-only, label-only) are returned so a reader can see what each half of the condition removes.
    """
    W = csr_matrix(W)
    labels = np.asarray(labels)
    succ, basin = ascending_flow(W, f)
    U = triu(W, k=1).tocoo()
    u, v, w = U.row, U.col, U.data
    diff_basin = basin[u] != basin[v]
    diff_label = labels[u] != labels[v]
    sep = diff_basin & diff_label
    maxima = np.unique(basin)
    return dict(succ=succ, basin=basin, maxima=maxima, n_maxima=int(len(maxima)),
                sep_edges=np.stack([u[sep], v[sep]], 1).astype(int), sep_w=w[sep].astype(np.float64),
                n_edges=int(len(u)), n_edges_diff_basin=int(diff_basin.sum()),
                n_edges_diff_label=int(diff_label.sum()), n_sep=int(sep.sum()),
                basin_sizes=np.bincount(np.searchsorted(maxima, basin)),
                n_label_pure_basins=int(sum(len(np.unique(labels[basin == m])) == 1 for m in maxima)))


def separatrix_distance(W, sep_edges, sep_w):
    """d_sep at every node and the index of the separatrix edge that attains it.

    One virtual midpoint node per separatrix edge, joined to both endpoints by w(e)/2; a multi-source Dijkstra
    from the midpoints then gives min_e [geodesic(x, endpoint) + w(e)/2] over both endpoints of every edge.  The
    midpoints add no shortcut: u-mid-v has the same length as the edge u-v that already exists.
    """
    W = csr_matrix(W)
    n = W.shape[0]
    E = len(sep_edges)
    if E == 0:
        return np.full(n, np.inf), np.full(n, -1)
    C = W.tocoo()
    mid = n + np.arange(E)
    r = np.concatenate([C.row, mid, mid, sep_edges[:, 0], sep_edges[:, 1]])
    c = np.concatenate([C.col, sep_edges[:, 0], sep_edges[:, 1], mid, mid])
    h = 0.5 * np.asarray(sep_w)
    d = np.concatenate([C.data, h, h, h, h])
    A = coo_matrix((d, (r, c)), shape=(n + E, n + E)).tocsr()
    dist, _, src = dijkstra(A, directed=False, indices=mid, min_only=True, return_predecessors=True)
    dist, src = dist[:n], src[:n]
    nearest = np.where(np.isfinite(dist), src - n, -1)
    return dist, nearest


def quadratic_design(u):
    """Columns [1, u_1..u_d, u_i u_j (i<=j)] for coordinates u of shape (..., k, d)."""
    d = u.shape[-1]
    iu, ju = np.triu_indices(d)
    return np.concatenate([np.ones(u.shape[:-1] + (1,)), u, u[..., iu] * u[..., ju]], -1), iu, ju


def boundary_curvature(X, logits, sep_edges, labels, d, k=30, chunk=512):
    """||II|| of the decision boundary at each separatrix edge midpoint (C3).

    For edge (u, v) with labels (y, y'): the k nearest probe points (Euclidean, in the space X lives in) to the
    midpoint (X_u + X_v)/2; tangent coordinates = the top-d local PCA directions of those k points, origin at the
    midpoint; least-squares fit of g = logits[:, y] - logits[:, y'].  When 1 + d + d(d+1)/2 > k the system is
    under-determined and the minimum-norm least-squares solution is returned (flagged in `underdetermined`).

    Returns dict(II, bnorm, Qnorm, resid, underdetermined, n_params).
    """
    X = np.asarray(X, dtype=np.float64)
    E = len(sep_edges)
    d = int(min(d, X.shape[1]))
    k = int(min(k, len(X)))
    n_params = 1 + d + d * (d + 1) // 2
    out = dict(II=np.full(E, np.nan), bnorm=np.full(E, np.nan), Qnorm=np.full(E, np.nan),
               resid=np.full(E, np.nan), underdetermined=bool(n_params > k), n_params=int(n_params), k=k, d=d)
    if E == 0:
        return out
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    y, y2 = labels[sep_edges[:, 0]], labels[sep_edges[:, 1]]
    for s in range(0, E, chunk):
        e = slice(s, min(E, s + chunk))
        mid = 0.5 * (X[sep_edges[e, 0]] + X[sep_edges[e, 1]])             # (m, D)
        ids = nn.kneighbors(mid, return_distance=False)                   # (m, k)
        N = X[ids]                                                        # (m, k, D)
        Nc = N - N.mean(1, keepdims=True)
        _, _, Vt = np.linalg.svd(Nc, full_matrices=False)
        T = np.transpose(Vt[:, :d, :], (0, 2, 1))                         # (m, D, d)
        u = np.einsum('mkD,mDd->mkd', N - mid[:, None, :], T)
        A, iu, ju = quadratic_design(u)                                   # (m, k, p)
        g = (logits[ids, y[e][:, None]] - logits[ids, y2[e][:, None]]).astype(np.float64)   # (m, k); never (m, k, C)
        coef = np.einsum('mpk,mk->mp', np.linalg.pinv(A), g)              # min-norm least squares
        res = np.linalg.norm(np.einsum('mkp,mp->mk', A, coef) - g, axis=1) / np.maximum(np.linalg.norm(g, axis=1), 1e-300)
        b = coef[:, 1:1 + d]
        Q = np.zeros((len(b), d, d))
        q = coef[:, 1 + d:]
        Q[:, iu, ju] = np.where(iu == ju, q, 0.5 * q)
        Q[:, ju, iu] = np.where(iu == ju, q, 0.5 * q)
        bn = np.linalg.norm(b, axis=1)
        bh = b / np.maximum(bn, 1e-300)[:, None]
        P = np.eye(d)[None] - bh[:, :, None] * bh[:, None, :]
        H = np.einsum('mab,mbc,mcd->mad', P, 2 * Q, P)
        II = np.linalg.norm(H, axis=(1, 2)) / np.maximum(bn, 1e-300)
        II[bn < 1e-12] = np.nan
        out['II'][e], out['bnorm'][e], out['Qnorm'][e], out['resid'][e] = II, bn, np.linalg.norm(2 * Q, axis=(1, 2)), res
    return out


def rho_prime(d_sep, nearest, II):
    """rho'(x) = d_sep(x) / (1 + ||II|| at the nearest separatrix edge); inf where no separatrix is reachable,
    nan where the nearest edge's fit is degenerate."""
    d_sep = np.asarray(d_sep, dtype=np.float64)
    r = np.full(len(d_sep), np.inf)
    ok = nearest >= 0
    r[ok] = d_sep[ok] / (1.0 + np.asarray(II)[nearest[ok]])
    return r
