"""legibility_profile(model, task_batch) -> dict of structural measures + (optionally) discovery outcome.
Stage-2 whitebox.legibility reuse point. task_batch = (tokens, targets). Scope: small node-instrumented
models (see models.py); extrapolation beyond the E4 ensemble is flagged in FINDINGS."""
import numpy as np, torch
from .graph import symmetrise, threshold_density, modularity_q, treewidth_bound, modularity_sweep, sparsity, weight_sparsity, shuffled_q_null
from .discovery import discover_circuit, jaccard

DENSITIES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

@torch.no_grad()
def node_importance(model, tokens):
    model.forward(tokens, record=True)
    return np.array([o.detach().reshape(o.shape[0] if model.arch == "mlp" else o.shape[0], -1).norm(dim=-1).mean().item()
                     for o in model._node_out])

def legibility_profile(model, task_batch, discovery_batches=None, null_n=20, seed=0, with_sweep=True):
    tokens, targets = task_batch
    model.eval()
    means = model.node_means(tokens)
    E, node_sc = model.edge_scores(tokens, targets, means)
    A = symmetrise(E.numpy()); N = model.n_nodes
    out = dict(N=N)
    B10 = threshold_density(A, 0.1)
    out["Q10"], _ = modularity_q(B10, seed=seed)
    out["tw10"] = treewidth_bound(B10); out["tw10_norm"] = out["tw10"] / N
    q_null, q_sd = shuffled_q_null(E.numpy(), 0.1, n=null_n, seed=seed)
    out["Q10_null"], out["Q10_null_sd"], out["Q10_excess"] = q_null, q_sd, out["Q10"] - q_null
    out["sparsity"] = sparsity(node_importance(model, tokens))
    out["edge_sparsity"] = sparsity(A[np.triu_indices(N, 1)])
    out["weight_sparsity"] = weight_sparsity(model)
    if with_sweep:
        sw = modularity_sweep(E.numpy(), DENSITIES, seed=seed); out["sweep"] = sw
        out["Q_auc"] = float(np.mean([r["Q"] for r in sw]))
    if discovery_batches is not None:
        runs = []
        for s, (tk, tg) in enumerate(discovery_batches):
            mu = model.node_means(tk)
            runs.append(discover_circuit(model, tk, tg, mu, seed=s))
        out["S_raw"] = float(np.mean([1 - r["frac"] for r in runs]))
        out["F"] = float(np.nanmean([r["F"] for r in runs])); out["cost"] = float(np.mean([r["cost"] for r in runs]))
        out["wall"] = float(np.mean([r["wall"] for r in runs]))
        out["size"] = float(np.mean([r["size"] for r in runs]))
        js = [jaccard(runs[i]["keep"], runs[j]["keep"]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
        out["jaccard"] = float(np.mean(js)) if js else float("nan")
        out["runs"] = runs
    return out
