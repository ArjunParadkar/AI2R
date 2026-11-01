"""N2: graph-geodesic distances on a g-weighted kNN graph.

Edge i~j weight = sqrt(dx^T g(m) dx), dx = x_j - x_i, g evaluated at the midpoint m (error O(|dx|^3) per edge,
exact for constant g).  Neighbours: the 3k nearest in coordinates are re-ranked by the g-length, k kept.
Graph is symmetrised (union).  Dijkstra from every source.

Complexity: kNN O(n log n) (KD-tree) + O(n k) metric evaluations + Dijkstra O(n (n k + n log n)) all-pairs
(O(s (n k + n log n)) from s sources).  Memory O(n k) sparse, O(n^2) for the dense output.
Error model: path-length inflation from direction quantisation of the k neighbours is ~ 1/cos(pi/k) - 1 (worst
case) and does NOT vanish as n -> infinity at fixed k; it vanishes as k grows (k ~ log n for connectivity).
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def knn_graph(X, k, g_fn=None, periodic=None):
    X = np.asarray(X, float)
    n, d = X.shape
    kk = min(3 * k + 1, n)
    tree = cKDTree(X)
    _, idx = tree.query(X, kk)
    idx = idx[:, 1:]
    I = np.repeat(np.arange(n), idx.shape[1]); J = idx.ravel()
    dx = X[J] - X[I]
    if g_fn is None:
        w = np.linalg.norm(dx, axis=1)
    else:
        G = np.asarray(g_fn(0.5 * (X[I] + X[J])))
        w = np.sqrt(np.maximum(np.einsum('ei,eij,ej->e', dx, G, dx), 0))
    W = w.reshape(n, -1)
    keep = np.argsort(W, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = np.take_along_axis(idx, keep, 1).ravel()
    vals = np.take_along_axis(W, keep, 1).ravel()
    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return A.maximum(A.T)


def graph_geodesics(X, k, g_fn=None, sources=None):
    A = knn_graph(X, k, g_fn)
    return dijkstra(A, directed=False, indices=sources)


def relative_errors(D_graph, D_true, mask):
    return (D_graph[mask] - D_true[mask]) / D_true[mask]
