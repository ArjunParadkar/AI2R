"""Degree-preserving rewiring nulls (Maslov-Sneppen swaps; weights travel with edges) and an ER size/density null."""
import numpy as np, networkx as nx

def rewire_directed(E, seed=0, n_swaps=None):
    """Directed swap (a->b, c->d) -> (a->d, c->b); exact in/out-degree preservation; no self loops / multi-edges."""
    E = np.asarray(E, float); rng = np.random.default_rng(seed); N = len(E)
    src, dst = np.nonzero(E); w = E[src, dst].copy(); m = len(src)
    if n_swaps is None: n_swaps = 5 * m
    present = set(zip(src.tolist(), dst.tolist())); done = 0; tries = 0
    while done < n_swaps and tries < 20 * n_swaps:
        tries += 1; i, j = rng.integers(0, m, 2)
        a, b, c, d = src[i], dst[i], src[j], dst[j]
        if a == d or c == b or (a, d) in present or (c, b) in present: continue
        present.remove((a, b)); present.remove((c, d)); present.add((a, d)); present.add((c, b))
        dst[i], dst[j] = d, b; done += 1
    R = np.zeros_like(E); R[src, dst] = w; return R

def rewire_undirected(A, seed=0, n_swaps=None):
    """Undirected double-edge swap on a symmetric weighted adjacency; degree sequence preserved."""
    A = np.asarray(A, float); rng = np.random.default_rng(seed)
    iu = np.triu_indices_from(A, 1); m0 = A[iu] > 0; src, dst = iu[0][m0].copy(), iu[1][m0].copy(); w = A[iu][m0]; m = len(src)
    if n_swaps is None: n_swaps = 5 * m
    present = set(); [present.add((min(a, b), max(a, b))) for a, b in zip(src, dst)]
    done = 0; tries = 0
    while done < n_swaps and tries < 20 * n_swaps:
        tries += 1; i, j = rng.integers(0, m, 2)
        a, b, c, d = src[i], dst[i], src[j], dst[j]
        if rng.random() < 0.5: c, d = d, c
        if a == d or c == b: continue
        e1, e2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if e1 in present or e2 in present or e1 == e2: continue
        present.discard((min(a, b), max(a, b))); present.discard((min(c, d), max(c, d))); present.add(e1); present.add(e2)
        src[i], dst[i], src[j], dst[j] = a, d, c, b; done += 1
    R = np.zeros_like(A); R[src, dst] = w; R[dst, src] = w; return R

def er_null(N, n_edges, seed=0):
    G = nx.gnm_random_graph(N, n_edges, seed=seed); return nx.to_numpy_array(G)
