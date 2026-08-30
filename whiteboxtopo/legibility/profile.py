"""legibility_profile — structural measures of a model's attribution graph, and the
discovery outcome of the fixed pipeline. Instrument only: E4 (whitebox-v2,
2026-08-30, 87 small models) found NO structural measure that predicts
circuit-discovery success within an architecture, so there is no
``legibility_score``; calling it raises ``NotCalibratedError`` with the numbers.
"""
from __future__ import annotations
from typing import Dict, Optional, Sequence
import numpy as np

from .graph import (symmetrise, threshold_density, modularity_q, treewidth_bound, modularity_sweep, sparsity,
                    weight_sparsity, shuffled_q_null, DENSITIES)
from .discovery import discover_circuit, jaccard, logprob_metric
from .models import node_importance


def legibility_profile(model, task_batch, discovery_batches: Optional[Sequence] = None, density: float = 0.1,
                       null_n: int = 20, seed: int = 0, with_sweep: bool = True, densities=DENSITIES,
                       metric=logprob_metric, min_accuracy: float = 0.5) -> Dict[str, object]:
    """Structural profile of ``model`` (node-instrumented, see ``models.py``) on ``task_batch = (x, targets)``.

    Returns a dict with
      N              number of nodes
      Q10            Leiden modularity of the attribution graph thresholded at ``density`` (10 % of edges)
      Q10_null, Q10_null_sd, Q10_excess   shuffled-weight null (same edge slots) and Q10 - null
      tw10, tw10_norm   min-fill treewidth UPPER bound of that graph, and / N
      sparsity       effective sparsity 1 - PR/N of node importances; edge_sparsity, weight_sparsity likewise
      sweep, Q_auc   (if ``with_sweep``) Q and tw at every density, and mean Q over the sweep
      accuracy, learned   accuracy on ``task_batch`` and the guard ``accuracy >= min_accuracy``
    and, if ``discovery_batches`` (a list of (x, targets)) is given, the fixed pipeline's outcome per batch:
      S_raw (mean 1 - |C|/N), S (S_raw if learned else 0), F, cost, size, jaccard (pairwise across batches), runs.

    Scope: small models (E4 ensemble: 2-hidden-layer MLPs, 2-4 layer transformers, N <= 300 nodes).
    The discovery pipeline's compression S is only meaningful when the model computes something
    (E4 kill 3: untrained models "succeed" trivially) — hence the accuracy guard.
    """
    import torch
    x, targets = task_batch
    model.eval()
    means = model.node_means(x)
    E, _ = model.edge_scores(x, targets, means, metric=metric)
    A = symmetrise(E.numpy()); N = model.n_nodes
    out: Dict[str, object] = dict(N=N, density=density)
    B = threshold_density(A, density)
    out["Q10"], out["membership"] = modularity_q(B, seed=seed)
    out["tw10"] = treewidth_bound(B); out["tw10_norm"] = out["tw10"] / N
    q_null, q_sd = shuffled_q_null(E.numpy(), density, n=null_n, seed=seed)
    out["Q10_null"], out["Q10_null_sd"], out["Q10_excess"] = q_null, q_sd, out["Q10"] - q_null
    out["sparsity"] = sparsity(node_importance(model, x))
    out["edge_sparsity"] = sparsity(A[np.triu_indices(N, 1)])
    out["weight_sparsity"] = weight_sparsity(model)
    with torch.no_grad():
        out["accuracy"] = float((model(x).argmax(-1) == targets).float().mean().item())
    out["learned"] = bool(out["accuracy"] >= min_accuracy)
    if with_sweep:
        sw = modularity_sweep(E.numpy(), densities, seed=seed); out["sweep"] = sw
        out["Q_auc"] = float(np.mean([r["Q"] for r in sw]))
    if discovery_batches is not None:
        runs = []
        for s, (xk, tk) in enumerate(discovery_batches):
            runs.append(discover_circuit(model, xk, tk, model.node_means(xk), seed=s, metric=metric))
        out["S_raw"] = float(np.mean([1 - r["frac"] for r in runs]))
        out["S"] = out["S_raw"] if out["learned"] else 0.0
        out["F"] = float(np.nanmean([r["F"] for r in runs])); out["cost"] = float(np.mean([r["cost"] for r in runs]))
        out["size"] = float(np.mean([r["size"] for r in runs]))
        js = [jaccard(runs[i]["keep"], runs[j]["keep"]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
        out["jaccard"] = float(np.mean(js)) if js else float("nan")
        out["runs"] = runs
    return out


E4_CALIBRATION_ATTEMPT = dict(
    date="2026-08-30", n_models=87, ensemble="whitebox-v2 E4: mlp / tf2 / tf4 x modadd / sort / copy x 8 regularisation arms, "
    "+ random-label, untrained and block-masked controls",
    rho_Q10_S_pooled=(-0.37, (-0.55, -0.17)), rho_Q10_S_within_arch={"mlp": (0.32, (-0.12, 0.68)), "tf2": (-0.38, (-0.70, 0.04)),
                                                                     "tf4": (0.21, (-0.20, 0.55))},
    partial_rho_Q10_S_given_arch=(0.03, (-0.26, 0.33)), rho_tw_norm_S=(-0.11, (-0.35, 0.13)),
    knee_delta_bic=-2.6, binary_success_rate="1/87",
    verdict="no structural measure predicts discovery success within an architecture; the pooled Q-S relation is an "
            "architecture offset with the WRONG sign. A score would predict the architecture label.")


class NotCalibratedError(NotImplementedError):
    """Raised by ``legibility_score``: there is no validated mapping from structure to tractability."""


def legibility_score(model=None, task_batch=None, **kw):
    """NOT CALIBRATED — raises ``NotCalibratedError``. Use ``legibility_profile`` for the measures.

    E4 (whitebox-v2, 87 models) tried to calibrate "predicted circuit-discovery tractability" on
    modularity Q, treewidth bound and sparsity of the attribution graph: rho(Q10, S) = -0.37
    [-0.55, -0.17] pooled — the wrong sign, and an architecture confound (within-architecture
    rho CIs all cross 0: mlp +0.32 [-0.12, 0.68], tf2 -0.38 [-0.70, 0.04], tf4 +0.21 [-0.20, 0.55];
    partial rho | architecture = +0.03 [-0.26, 0.33]); treewidth/N: rho = -0.11 [-0.35, 0.13];
    no knee (Delta BIC = -2.6). See VALIDATION.md and benchmarks/PHASE6.md. Until an ensemble
    with within-architecture variation in outcome exists, this function refuses to return a number.
    """
    a = E4_CALIBRATION_ATTEMPT
    raise NotCalibratedError(
        "legibility_score is not calibrated — see VALIDATION.md. E4 (whitebox-v2, n=87): rho(Q10, S) = "
        f"{a['rho_Q10_S_pooled'][0]:+.2f} {list(a['rho_Q10_S_pooled'][1])} pooled (architecture-confounded, wrong sign); "
        "within-architecture " + ", ".join(f"{k} {v[0]:+.2f} {list(v[1])}" for k, v in a["rho_Q10_S_within_arch"].items())
        + f"; partial rho | arch = {a['partial_rho_Q10_S_given_arch'][0]:+.2f} {list(a['partial_rho_Q10_S_given_arch'][1])}. "
        "Use legibility_profile() for the measures; they carry no prediction.")
