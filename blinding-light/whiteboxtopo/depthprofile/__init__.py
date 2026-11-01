"""Depth-scale profile of class fibers (SPEC2 block A).

A fiber F_y^l is the set of probe activations at layer l whose label is y.  Everything here is
H0 only, and H0 of a Rips filtration is single linkage: the number of connected components of the
graph {d(i,j) <= eps} equals n minus the number of single-linkage merge heights <= eps.  So one
linkage per fiber gives beta0 at EVERY scale exactly -- no graph is ever built per eps.

  single_linkage(D)            merge heights + cluster sizes of a distance matrix
  mst_largest_edge(D)          the scale at which the whole cloud becomes one component
  robust_scale(D, 0.99)        the scale at which 99% of the points are in one component
  beta0_at / beta0_curve       components of a fiber at one eps / on a grid
  fiber_beta0_profile          per-class beta0 at a fixed eps and on a grid
  depth_of_connection, untangling_index     d*(y) and U = 1 - E[d*/L]
  pullback_distance_matrix     g-geodesic (or secant) all-pairs distances on the k-NN graph
  gw_fibers                    Gromov-Wasserstein between two median-normalised fibers (POT)
"""
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import pdist, squareform

__all__ = ['euclidean_distance_matrix', 'single_linkage', 'mst_largest_edge', 'robust_scale',
           'median_pairwise', 'beta0_at', 'beta0_curve', 'fiber_beta0_profile', 'fiber_connection_scales', 'gw_product_coupling_loss',
           'depth_of_connection', 'untangling_index', 'pullback_distance_matrix', 'gw_fibers']


def euclidean_distance_matrix(X):
    """Dense symmetric Euclidean distance matrix (float64)."""
    return squareform(pdist(np.asarray(X, dtype=np.float64)))


def _condensed(D):
    D = np.asarray(D, dtype=np.float64)
    return squareform(D, checks=False) if D.ndim == 2 else D


def single_linkage(D):
    """(heights, sizes): the n-1 single-linkage merge heights in ascending order and the size of
    the cluster created by each merge.  The heights are exactly the MST edge lengths."""
    n = len(D) if np.ndim(D) == 2 else int(round((1 + np.sqrt(1 + 8 * len(D))) / 2))
    if n < 2:
        return np.zeros(0), np.zeros(0, int)
    Z = linkage(_condensed(D), method='single')
    return Z[:, 2].copy(), Z[:, 3].astype(int)


def mst_largest_edge(D):
    """Largest edge of the minimum spanning tree = the smallest eps at which beta0 = 1."""
    h, _ = single_linkage(D)
    return float(h.max()) if len(h) else 0.0


def robust_scale(D, frac=0.99):
    """Smallest eps at which one component holds at least `frac` of the points."""
    h, s = single_linkage(D)
    n = len(h) + 1
    hit = np.where(s >= frac * n)[0]
    return float(h[hit[0]]) if len(hit) else float(h.max())


def median_pairwise(D):
    """Median of the off-diagonal distances."""
    return float(np.median(_condensed(D)))


def beta0_curve(D_sub, eps_grid):
    """beta0 of the Rips graph {d <= eps} on the points of D_sub, for every eps in the grid."""
    n = len(D_sub)
    eps_grid = np.atleast_1d(np.asarray(eps_grid, dtype=np.float64))
    if n == 0:
        return np.zeros(len(eps_grid), int)
    if n == 1:
        return np.ones(len(eps_grid), int)
    h, _ = single_linkage(D_sub)
    return (n - np.searchsorted(h, eps_grid, side='right')).astype(int)


def beta0_at(D_sub, eps):
    return int(beta0_curve(D_sub, [eps])[0])


def fiber_beta0_profile(D, labels, eps_list, eps_grid=None, classes=None):
    """For every class y: beta0 of the fiber at each eps in eps_list, and (optionally) on a grid.

    Returns (classes, B[n_classes, len(eps_list)], C[n_classes, len(eps_grid)] or None)."""
    labels = np.asarray(labels)
    classes = np.unique(labels[labels >= 0]) if classes is None else np.asarray(classes)
    eps_list = np.atleast_1d(np.asarray(eps_list, dtype=np.float64))
    B = np.zeros((len(classes), len(eps_list)), int)
    C = None if eps_grid is None else np.zeros((len(classes), len(eps_grid)), int)
    for i, y in enumerate(classes):
        sel = np.where(labels == y)[0]
        n = len(sel)
        if n < 2:
            B[i] = n
            if C is not None:
                C[i] = n
            continue
        h, _ = single_linkage(D[np.ix_(sel, sel)])
        B[i] = n - np.searchsorted(h, eps_list, side='right')
        if C is not None:
            C[i] = n - np.searchsorted(h, eps_grid, side='right')
    return classes, B, C


def fiber_connection_scales(D, labels, classes=None):
    """The exact scale at which each fiber becomes one component (its own largest MST edge).
    beta0(F_y) = 1 at eps  <=>  fiber_connection_scale(y) <= eps."""
    labels = np.asarray(labels)
    classes = np.unique(labels[labels >= 0]) if classes is None else np.asarray(classes)
    out = np.zeros(len(classes))
    for i, y in enumerate(classes):
        sel = np.where(labels == y)[0]
        out[i] = mst_largest_edge(D[np.ix_(sel, sel)]) if len(sel) > 1 else 0.0
    return out


def depth_of_connection(beta0_by_layer):
    """d* = first layer index in 1..L with beta0 == 1, and L if there is none.
    `beta0_by_layer` has shape (L,) or (n_classes, L).  Also returns the mask 'never connected'."""
    b = np.atleast_2d(np.asarray(beta0_by_layer))
    L = b.shape[1]
    one = b == 1
    never = ~one.any(1)
    d = np.where(never, L, one.argmax(1) + 1)
    return d.astype(int), never


def untangling_index(dstar, L):
    """U = 1 - E_y[d*/L].  Range [0, 1 - 1/L]: 0 when no fiber connects before the last layer."""
    return float(1.0 - np.mean(np.asarray(dstar, dtype=np.float64) / L))


def _pullback_lengths_chunked(X, J, A, chunk_bytes=256e6):
    """Same numbers as manifold.core.pullback_metric_lengths, edge-chunked: that function gathers
    J[i] and J[j] for ALL edges at once (2 x n_edges x d_out x d_l floats -- 12 GB for 3,830 points
    of a 113 x 256 Jacobian)."""
    from scipy.sparse import csr_matrix
    A = A.tocoo()
    i, j = A.row, A.col
    w = np.empty(len(i), dtype=np.float64)
    step = max(1, int(chunk_bytes // (J.shape[1] * J.shape[2] * J.itemsize)))
    for s in range(0, len(i), step):
        ii, jj = i[s:s + step], j[s:s + step]
        delta = X[jj] - X[ii]
        a = np.einsum('nod,nd->no', J[ii], delta)
        b = np.einsum('nod,nd->no', J[jj], delta)
        w[s:s + step] = np.sqrt(0.5 * ((a ** 2).sum(1) + (b ** 2).sum(1)) + 1e-30)
    return csr_matrix((w, (i, j)), shape=A.shape)


def pullback_distance_matrix(X, J=None, out=None, k=15):
    """All-pairs geodesic distance on the k-NN graph whose neighbours AND edge lengths come from
    the pullback metric (J given) or from the Jacobian-free secant metric (`out` given).
    Pairs in different graph components are put at twice the largest finite distance, as the
    library's `geodesic_distance_matrix` does.  Returns (D, n_graph_components)."""
    from scipy.sparse.csgraph import connected_components
    from whiteboxtopo.manifold import knn_graph
    from whiteboxtopo.manifold.core import secant_metric_lengths
    if J is not None:
        _, _, A = knn_graph(X, k, J=J, metric='g')
        W = _pullback_lengths_chunked(X, J, A)
    elif out is not None:
        _, _, A = knn_graph(X, k, metric='secant', out=out)
        W = secant_metric_lengths(out, A)
    else:
        raise ValueError('need Jacobians or cached outputs')
    ncomp = int(connected_components(W, directed=False)[0])
    D = dijkstra(W, directed=False)
    fin = np.isfinite(D)
    if not fin.all():
        D[~fin] = 2.0 * D[fin].max()
    D = np.maximum(D, D.T)
    np.fill_diagonal(D, 0.0)
    return D, ncomp


def gw_fibers(D1, D2, max_iter=1000, tol=1e-9):
    """Gromov-Wasserstein discrepancy (POT gromov_wasserstein2, square loss, uniform weights)
    between two metric spaces given by distance matrices, each divided by its own median
    off-diagonal distance.  Returns (value, converged_flag_or_None)."""
    import ot
    C1 = np.asarray(D1, dtype=np.float64); C2 = np.asarray(D2, dtype=np.float64)
    m1, m2 = median_pairwise(C1), median_pairwise(C2)
    C1 = C1 / (m1 if m1 > 0 else 1.0); C2 = C2 / (m2 if m2 > 0 else 1.0)
    p = np.full(len(C1), 1.0 / len(C1)); q = np.full(len(C2), 1.0 / len(C2))
    val, log = ot.gromov.gromov_wasserstein2(C1, C2, p, q, 'square_loss', max_iter=max_iter,
                                             tol_rel=tol, tol_abs=tol, log=True)
    return float(max(val, 0.0)), log


def gw_product_coupling_loss(D1, D2):
    """The GW objective at the independent coupling p x q (the solver's starting point): the value
    any two median-normalised spaces get with no matching at all -- the reference for 'large'."""
    from ot.gromov import init_matrix, gwloss
    C1 = np.asarray(D1, dtype=np.float64); C2 = np.asarray(D2, dtype=np.float64)
    m1, m2 = median_pairwise(C1), median_pairwise(C2)
    C1 = C1 / (m1 if m1 > 0 else 1.0); C2 = C2 / (m2 if m2 > 0 else 1.0)
    p = np.full(len(C1), 1.0 / len(C1)); q = np.full(len(C2), 1.0 / len(C2))
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, 'square_loss')
    return float(gwloss(constC, hC1, hC2, np.outer(p, q)))
