"""Structural measures on a thresholded attribution graph (ported from whitebox-v2 E4 ``e4lib.graph``).

``E`` is an (N, N) array of directed attribution scores u -> v (any sign); it is
symmetrised as |E| + |E|^T with zero diagonal before thresholding. Requires the
optional ``legibility`` extra (networkx, python-igraph, leidenalg).
"""
from __future__ import annotations
import numpy as np


def _need(mod):
    try:
        return __import__(mod)
    except ImportError as e:
        raise ImportError(f"whitebox.legibility needs {mod}: pip install 'whitebox-topo[legibility]'") from e


def symmetrise(E) -> np.ndarray:
    A = np.abs(np.asarray(E, dtype=float)); A = A + A.T; np.fill_diagonal(A, 0); return A


def threshold_density(A, density: float) -> np.ndarray:
    """Keep the top ``density`` fraction of the existing (nonzero) undirected edges by weight."""
    A = np.asarray(A, float); iu = np.triu_indices_from(A, 1); w = A[iu]; nz = w > 0
    if nz.sum() == 0: return np.zeros_like(A)
    if density >= 1: return A.copy()
    k = max(1, int(round(density * nz.sum())))
    thr = np.sort(w[nz])[::-1][k - 1]
    B = np.zeros_like(A); m = (w >= thr) & nz
    B[iu[0][m], iu[1][m]] = w[m]; return B + B.T


def modularity_q(A, n_restarts: int = 10, seed: int = 0):
    """Leiden modularity (best of ``n_restarts``) of a symmetric weighted adjacency.
    Returns (Q, membership)."""
    ig = _need("igraph"); la = _need("leidenalg")
    A = np.asarray(A, float); iu = np.triu_indices_from(A, 1); m = A[iu] > 0
    if m.sum() == 0: return 0.0, np.zeros(len(A), int)
    g = ig.Graph(n=len(A), edges=list(zip(iu[0][m].tolist(), iu[1][m].tolist())))
    g.es["weight"] = A[iu][m].tolist()
    best, bestp = -1.0, None
    for r in range(n_restarts):
        p = la.find_partition(g, la.ModularityVertexPartition, weights="weight", n_iterations=-1, seed=seed + r)
        q = p.quality()
        if q > best: best, bestp = q, np.array(p.membership)
    return float(best), bestp


def treewidth_bound(A) -> int:
    """Min-fill-in UPPER bound on the treewidth of the graph of nonzero entries
    (networkx ``treewidth_min_fill_in``). Exact on trees (1), cycles (2), cliques (n-1);
    an over-estimate is possible on grids and denser graphs."""
    nx = _need("networkx")
    from networkx.algorithms.approximation import treewidth_min_fill_in
    G = nx.from_numpy_array((np.asarray(A) > 0).astype(int))
    if G.number_of_edges() == 0: return 0
    tw, _ = treewidth_min_fill_in(G); return int(tw)


DENSITIES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def modularity_sweep(E, densities=DENSITIES, with_tw: bool = True, seed: int = 0):
    """Q (and treewidth bound) of the graph thresholded at each edge density."""
    A = symmetrise(E); rows = []
    for d in densities:
        B = threshold_density(A, d); q, _ = modularity_q(B, seed=seed)
        tw = treewidth_bound(B) if with_tw else np.nan
        rows.append(dict(density=float(d), Q=q, tw=tw, n_edges=int((B > 0).sum() // 2)))
    return rows


def shuffled_q_null(E, density: float = 0.1, n: int = 20, seed: int = 0):
    """Q of graphs with the edge weights permuted among the same edge slots (the layered
    skeleton is kept). Returns (mean, sd). E4: real attribution graphs sit BELOW this
    null for 90 % of models — strong edges concentrate on hubs."""
    A = symmetrise(E); iu = np.triu_indices_from(A, 1); w = A[iu]; nz = w > 0
    rng = np.random.default_rng(seed); qs = []
    for i in range(n):
        w2 = w.copy(); w2[nz] = rng.permutation(w[nz])
        B = np.zeros_like(A); B[iu] = w2; B = B + B.T
        qs.append(modularity_q(threshold_density(B, density), n_restarts=3, seed=seed + i)[0])
    return float(np.mean(qs)), float(np.std(qs))


def sparsity(node_importance) -> float:
    """Effective sparsity 1 - PR/N with participation ratio PR = (sum s)^2 / sum s^2 of |s|."""
    s = np.abs(np.asarray(node_importance, float))
    if s.sum() == 0: return 1.0
    pr = s.sum() ** 2 / (s ** 2).sum(); return float(1 - pr / len(s))


def weight_sparsity(model, rel: float = 1e-3) -> float:
    """Fraction of weight-matrix entries below ``rel`` x the matrix max, averaged over matrices."""
    fr = []
    for p in model.parameters():
        if p.dim() < 2: continue
        w = p.detach().abs(); fr.append((w < rel * w.max()).float().mean().item())
    return float(np.mean(fr)) if fr else float("nan")
