"""Thin adapter: E4's legibility profile computed from a plain signed weighted adjacency E (u->v).
Calls e4lib.graph directly so biology and artificial nets share one implementation of every measure."""
import numpy as np
from e4lib.graph import symmetrise, threshold_density, modularity_q, treewidth_bound, modularity_sweep, sparsity, shuffled_q_null

DENSITIES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)   # identical to e4lib.legibility.DENSITIES

def weight_importance(E):
    """[JC] node importance for weight graphs: total weighted degree |in| + |out| (no activations exist for a connectome)."""
    E = np.abs(np.asarray(E, float)); return E.sum(0) + E.sum(1)

def threshold_count(A, n_edges):
    """Keep exactly the top-n_edges undirected edges (matched-edge-count secondary threshold)."""
    iu = np.triu_indices_from(A, 1); w = A[iu]; nz = w > 0
    if n_edges >= nz.sum(): return A.copy()
    thr = np.sort(w[nz])[::-1][n_edges - 1]
    B = np.zeros_like(A); m = (w >= thr) & nz; B[iu[0][m], iu[1][m]] = w[m]; return B + B.T

def profile_adjacency(E, importance=None, null_n=20, seed=0, with_sweep=True, densities=DENSITIES, match_edges=None):
    """Same keys as e4lib.legibility.legibility_profile's structural part."""
    E = np.asarray(E, float); N = len(E); A = symmetrise(E); out = dict(N=N)
    out["density_nonzero"] = float((E != 0).mean())
    B10 = threshold_density(A, 0.1); out["n_edges10"] = int((B10 > 0).sum() // 2)
    out["Q10"], memb = modularity_q(B10, seed=seed); out["membership10"] = memb
    out["n_comm10"] = int(len(set(memb.tolist())))
    out["tw10"] = treewidth_bound(B10); out["tw10_norm"] = out["tw10"] / N
    q_null, q_sd = shuffled_q_null(E, 0.1, n=null_n, seed=seed)
    out["Q10_null"], out["Q10_null_sd"], out["Q10_excess"] = q_null, q_sd, out["Q10"] - q_null
    out["sparsity"] = sparsity(weight_importance(E) if importance is None else importance)
    out["edge_sparsity"] = sparsity(A[np.triu_indices(N, 1)])
    if with_sweep:
        sw = modularity_sweep(E, densities, seed=seed); out["sweep"] = sw
        out["Q_auc"] = float(np.mean([r["Q"] for r in sw]))
        for r in sw:
            d = int(round(r["density"] * 100)); out[f"Q{d}"] = r["Q"]; out[f"tw{d}"] = r["tw"]
    if match_edges is not None:
        Bm = threshold_count(A, int(match_edges)); out["Q_matched"], _ = modularity_q(Bm, seed=seed)
        out["tw_matched"] = treewidth_bound(Bm); out["tw_matched_norm"] = out["tw_matched"] / N
        out["n_edges_matched"] = int((Bm > 0).sum() // 2)
    return out
