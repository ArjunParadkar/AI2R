"""whiteboxtopo.discovery — the fixed circuit-discovery pipeline (handbook S4).

    r = find_circuit(model, (x, targets), faithfulness=0.9, seed=0)
    r["circuit"] (bool mask over nodes), r["size"], r["faithfulness"], r["wall"], r["cost"]

Attribution patching (mean corruption) + greedy prune, two passes, deterministic
given ``seed`` (E4's ``discover_circuit``). Frozen: no per-model tuning.
"""
from __future__ import annotations
from ..legibility.discovery import discover_circuit, jaccard, faithfulness as _faithfulness, logprob_metric


def find_circuit(model, task, faithfulness: float = 0.9, seed: int = 0, metric=logprob_metric, **kw):
    x, targets = task
    means = model.node_means(x)
    r = discover_circuit(model, x, targets, means, seed=seed, F_target=faithfulness, metric=metric, **kw)
    return {"circuit": r["keep"].numpy(), "size": r["size"], "frac": r["frac"], "faithfulness": r["F"],
            "cost": r["cost"], "wall": r["wall"], "gap": r["gap"], "success": bool(r["F"] >= faithfulness)}


__all__ = ["find_circuit", "discover_circuit", "jaccard", "logprob_metric"]
