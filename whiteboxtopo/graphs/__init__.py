"""whiteboxtopo.graphs — attribution graphs and their structure (handbook S4).

    A = attribution_graph(model, (x, targets), threshold=0.1)   # symmetric weighted adjacency (N x N)
    modularity(A)      -> (Q, membership)   Leiden, 10 restarts
    treewidth_ub(A)    -> int               min-fill upper bound
    sparsity(A)        -> 1 - PR/N of edge weights (or of a node-importance vector)

``attribution_graph`` uses edge attribution patching (EAP, mean corruption) on any
node-instrumented model (``whiteboxtopo.legibility.models`` protocol; E4's MLP/TinyTF
in ``legibility.e4models``). ``threshold`` is an edge *density* (fraction of the
strongest |E| kept), as in E4, so graphs of different models are comparable.
Frozen once E4/V starts: no per-model tuning.
"""
from __future__ import annotations
import numpy as np
from ..legibility.graph import (symmetrise, threshold_density, modularity_q, treewidth_bound, modularity_sweep,
                                shuffled_q_null, sparsity as _sparsity, weight_sparsity, DENSITIES)
from ..legibility.discovery import logprob_metric


def attribution_graph(model, data, threshold: float = 0.1, metric=logprob_metric, raw: bool = False):
    """EAP edge scores of ``model`` on ``data = (x, targets)`` -> symmetric adjacency at edge
    density ``threshold`` (``raw=True`` returns the signed directed score matrix instead)."""
    x, targets = data
    model.eval()
    means = model.node_means(x)
    E, node = model.edge_scores(x, targets, means) if "metric" not in model.edge_scores.__code__.co_varnames \
        else model.edge_scores(x, targets, means, metric=metric)
    E = E.numpy() if hasattr(E, "numpy") else np.asarray(E)
    if raw:
        return E
    return threshold_density(symmetrise(E), threshold)


def modularity(A, n_restarts: int = 10, seed: int = 0):
    """Leiden modularity of a (weighted, symmetric) adjacency: ``(Q, membership)``."""
    return modularity_q(np.asarray(A), n_restarts=n_restarts, seed=seed)


def treewidth_ub(A) -> int:
    """Min-fill treewidth upper bound of the graph with edges where A != 0."""
    return treewidth_bound(np.asarray(A))


def sparsity(A_or_importance) -> float:
    """Effective sparsity 1 - PR/N (PR = participation ratio) of the upper-triangle edge weights
    of a square matrix, or of a 1-D importance vector."""
    v = np.asarray(A_or_importance, dtype=np.float64)
    if v.ndim == 2 and v.shape[0] == v.shape[1]:
        v = np.abs(v[np.triu_indices(v.shape[0], 1)])
    return _sparsity(v)


def rewired_null(A, n: int = 20, seed: int = 0):
    """Degree-preserving rewiring null for Q (V2): mean and sd of Leiden Q over ``n`` rewired
    graphs (binary structure, weights re-attached in random order)."""
    import networkx as nx
    A = np.asarray(A); G = nx.from_numpy_array((A != 0).astype(float))
    w = A[np.triu_indices(A.shape[0], 1)]; w = w[w != 0]
    rng = np.random.default_rng(seed); qs = []
    for i in range(n):
        H = G.copy()
        try:
            nx.double_edge_swap(H, nswap=max(1, 2 * H.number_of_edges()), max_tries=50 * max(1, H.number_of_edges()), seed=int(rng.integers(1 << 31)))
        except nx.NetworkXError:
            pass
        B = nx.to_numpy_array(H); iu = np.triu_indices(B.shape[0], 1); m = B[iu] != 0
        vals = np.zeros(m.sum()); vals[:] = rng.permutation(w)[:m.sum()] if len(w) >= m.sum() else np.resize(rng.permutation(w), m.sum())
        W = np.zeros_like(B); W[iu[0][m], iu[1][m]] = vals; W = W + W.T
        qs.append(modularity_q(W, seed=i)[0])
    return float(np.mean(qs)), float(np.std(qs))


__all__ = ["attribution_graph", "modularity", "treewidth_ub", "sparsity", "rewired_null", "symmetrise",
           "threshold_density", "modularity_sweep", "shuffled_q_null", "weight_sparsity", "DENSITIES"]
