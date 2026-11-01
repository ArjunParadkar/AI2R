"""Section 2.7 and problems P4/P5: how the manifold bends and how far away a new point is."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def second_fundamental_form(X, idx=None, k=30, dim_cap=4, bases=None, ridge=1e-8):
    """||II_h|| at every point: fit the normal component of the neighbours as a quadratic
    form in tangent coordinates (spec 2.7).

    For each point we take a neighbourhood, split it into a tangent part (top `dim_cap` local
    PCA directions) and a normal part, and least-squares fit

        normal_component ~= Q(u, u),   u = tangent coordinates,

    over the d(d+1)/2 quadratic monomials.  Reported is the Frobenius norm of the fitted
    coefficient tensor; for a d-sphere of radius R that equals sqrt(d)/(2R), so the estimator
    is proportional to 1/R with a fixed constant -- verified on spheres of three radii.
    `k` is larger than the graph's k because the fit needs more neighbours than monomials.

    Batched through the normal equations so a 2,000-point probe costs seconds.
    """
    n, D = X.shape
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, ids = nn.kneighbors(X)
    ids = ids[:, 1:]
    N = X[ids] - X[:, None, :]                       # (n, k, D)
    Nc = N - N.mean(1, keepdims=True)
    U, sv, Vt = np.linalg.svd(Nc, full_matrices=False)
    d = int(min(dim_cap, k - 1, D))
    T = np.transpose(Vt[:, :d, :], (0, 2, 1))        # (n, D, d)
    u = np.einsum('nkD,nDd->nkd', N, T)              # tangent coordinates
    nrm = N - np.einsum('nkd,nDd->nkD', u, T)        # normal residual
    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    M = np.stack([u[:, :, i] * u[:, :, j] for i, j in pairs], -1)   # (n, k, m)
    if M.shape[1] <= M.shape[2]:
        return np.zeros(n)
    G = np.einsum('nkm,nkl->nml', M, M)
    R = np.einsum('nkm,nkD->nmD', M, nrm)
    G = G + ridge * np.trace(G, axis1=1, axis2=2)[:, None, None] * np.eye(G.shape[1])
    C = np.linalg.solve(G, R)                        # (n, m, D)
    return np.linalg.norm(C, axis=(1, 2))


def lipschitz_field(J):
    """Lip(h) = ||J_l(h)||_2, the spectral norm of the Jacobian of the remaining network."""
    return np.linalg.norm(J, ord=2, axis=(1, 2))


def _tangent_normal_distance(h_star, X, bases, nn):
    """d_g(h*, M): push the component of h* - hbar normal to the tangent space through the
    Jacobian at the nearest probe point (spec 3.6)."""
    _, ids = nn.kneighbors(h_star)
    ids = ids[:, 0]
    delta = h_star - X[ids]
    P = bases[ids]                                       # (n*, D, d)
    proj = np.einsum('nDd,nD->nd', P, delta)
    normal = delta - np.einsum('nDd,nd->nD', P, proj)
    return ids, normal


def reach(X, bases, J, holdout_acts, percentile=99):
    """eps_l: the 99th percentile of d_g(h, M) over held-out in-distribution inputs."""
    nn = NearestNeighbors(n_neighbors=1).fit(X)
    ids, normal = _tangent_normal_distance(holdout_acts, X, bases, nn)
    dg = np.linalg.norm(np.einsum('nod,nd->no', J[ids], normal), axis=1)
    return float(np.percentile(dg, percentile)), dg, nn


def ignorance(h_star, X, bases, J, eps, nn=None):
    """Distance from the tubular neighbourhood, in units of the reach.  <= 1 means known."""
    if nn is None:
        nn = NearestNeighbors(n_neighbors=1).fit(X)
    ids, normal = _tangent_normal_distance(h_star, X, bases, nn)
    dg = np.linalg.norm(np.einsum('nod,nd->no', J[ids], normal), axis=1)
    return dg / max(eps, 1e-30), dg


def failure_risk(curv, lip):
    """rho(x) = ||II_h||_2 * Lip(h): the manifold bends and the map is steep."""
    return np.asarray(curv) * np.asarray(lip)


def confidence_saddles(W, f):
    """Saddle points of the max-logit function on the kNN graph.

    Discrete Morse theory: a vertex is a saddle when its upper link -- the neighbours with a
    higher value of f -- falls into two or more connected components.  Those are the points
    where the confidence landscape splits, and they sit on decision ridges.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    W = W.tocsr()
    n = W.shape[0]
    f = np.asarray(f)
    saddles, ncomp = [], np.zeros(n, dtype=int)
    for v in range(n):
        nb = W.indices[W.indptr[v]:W.indptr[v + 1]]
        up = nb[f[nb] > f[v]]
        if len(up) < 2:
            ncomp[v] = len(up)
            continue
        sub = W[up][:, up]
        c = connected_components(sub, directed=False)[0]
        ncomp[v] = c
        if c >= 2:
            saddles.append(v)
    return np.array(saddles, dtype=int), ncomp


def manifold_margin(W, pred):
    """For every probe point, the g-geodesic distance to the nearest point the network labels
    differently -- the P2 margin, computed for the whole cloud at once (one multi-source
    Dijkstra per predicted class).  Small margin = the decision boundary is close *along the
    manifold*: this is the framework's own answer to "where will it fail"."""
    from scipy.sparse.csgraph import dijkstra
    pred = np.asarray(pred)
    out = np.full(len(pred), np.inf)
    for c in np.unique(pred):
        src = np.where(pred != c)[0]
        if len(src) == 0:
            continue
        d = dijkstra(W, directed=False, indices=src, min_only=True)
        sel = pred == c
        out[sel] = d[sel]
    return out


def tubular_distance(h_star, X, bases, nn=None, k_scale=10):
    """Ambient tubular distance: the norm of the component of h* - hbar NORMAL to the tangent
    space at the nearest probe point, in units of the local normal thickness of the manifold
    (the same residual measured for the probe's own neighbours).

    The spec's d_g pushes that normal component through the Jacobian first.  That measures only
    the part of the displacement the output can see -- but an out-of-distribution input is
    displaced precisely in directions the head ignores (a linear head on 512 features keeps 10
    directions and discards 502), so d_g calls it "known".  Measured on ResNet/SVHN: d_g AUC
    0.64, below softmax.  Off-manifold-ness has to be measured in the activation space itself.
    """
    if nn is None:
        nn = NearestNeighbors(n_neighbors=k_scale + 1).fit(X)
    dist, ids = nn.kneighbors(h_star, n_neighbors=1)
    ids = ids[:, 0]
    delta = h_star - X[ids]
    P = bases[ids]
    normal = delta - np.einsum('nDd,nd->nD', P, np.einsum('nDd,nD->nd', P, delta))
    # local normal thickness at each probe point
    _, nb = nn.kneighbors(X, n_neighbors=min(k_scale + 1, len(X)))
    dl = X[nb[:, 1:]] - X[:, None, :]
    nl = dl - np.einsum('nDd,nkd->nkD', bases, np.einsum('nDd,nkD->nkd', bases, dl))
    thick = np.linalg.norm(nl, axis=2).mean(1) + 1e-12
    return np.linalg.norm(normal, axis=1) / thick[ids]
