"""The fixed circuit-discovery pipeline of E4 (``e4lib.discovery``): node attribution
patching with mean corruption + greedy pruning to a minimal faithful circuit at
F >= ``F_target``, two passes, re-scoring every ``rescore_every`` accepted removals.
``cost`` = number of ablated forward passes (not wall-clock).

Known property (E4 finding 5): a greedy prune touches each node ~once per pass,
so cost/N is 1-2 for every model by construction — cost from this pipeline
cannot carry a parameterized-complexity statement. It is a fixed instrument.
"""
from __future__ import annotations
import time
import numpy as np


def logprob_metric(logits, targets):
    """Mean log-prob of the correct class over batch (and any output slots). logits (..., C), targets (...)."""
    import torch
    lp = torch.log_softmax(logits, -1)
    return lp.gather(-1, targets.unsqueeze(-1)).mean()


def _m(model, x, targets, keep, means, metric):
    import torch
    with torch.no_grad():
        return metric(model(x, keep=keep, means=means), targets).item()


def faithfulness(model, x, targets, keep, means, m_clean, m_corr, metric=logprob_metric):
    """Gap-normalised metric: (m(keep) - m_corr) / (m_clean - m_corr); NaN if the gap is degenerate."""
    if abs(m_clean - m_corr) < 1e-8: return float("nan")
    return (_m(model, x, targets, keep, means, metric) - m_corr) / (m_clean - m_corr)


def _scores_under(model, x, targets, means, keep, metric):
    import torch
    model.zero_grad(set_to_none=True)
    logits = model(x, keep=keep, means=means, record=True)
    m = metric(logits, targets); outs = model._node_out
    grads = torch.autograd.grad(m, outs, allow_unused=True)
    return torch.tensor([0.0 if g is None else ((mu - o) * g).sum().item() for o, g, mu in zip(outs, grads, means)])


def discover_circuit(model, x, targets, means, seed: int = 0, F_target: float = 0.9, rescore_every: int = 16,
                     n_passes: int = 2, metric=logprob_metric):
    """Returns dict(keep, size, frac, F, cost, wall, gap); ``keep`` is a bool tensor over nodes.
    ``model`` must follow the node-instrumented protocol (see ``whiteboxtopo.legibility.models``)."""
    import torch
    model.eval(); t0 = time.time(); N = model.n_nodes
    rng = np.random.default_rng(seed)
    full = torch.ones(N, dtype=torch.bool); none = torch.zeros(N, dtype=torch.bool)
    m_clean = _m(model, x, targets, full, means, metric); m_corr = _m(model, x, targets, none, means, metric)
    gap = m_clean - m_corr; cost = 2
    keep = full.clone()

    def F(k):
        nonlocal cost; cost += 1
        return (_m(model, x, targets, k, means, metric) - m_corr) / gap if abs(gap) > 1e-8 else float("nan")

    if not np.isfinite(F(full)):
        return dict(keep=keep, size=N, frac=1.0, F=float("nan"), cost=cost, wall=time.time() - t0, gap=gap)
    for _ in range(n_passes):
        sc = model.node_scores(x, targets, means, metric=metric).abs().numpy()
        sc = sc + rng.normal(0, 1e-9 * (sc.max() + 1e-12), N)
        order = [i for i in np.argsort(sc) if keep[i]]; accepted = 0; pos = 0
        while pos < len(order):
            i = order[pos]; pos += 1
            if not keep[i]: continue
            keep[i] = False
            if F(keep) >= F_target:
                accepted += 1
                if accepted % rescore_every == 0:
                    sc = _scores_under(model, x, targets, means, keep, metric).abs().numpy()
                    sc = sc + rng.normal(0, 1e-9 * (sc.max() + 1e-12), N)
                    seen = set(order[:pos])
                    order = [j for j in np.argsort(sc) if keep[j] and j not in seen]; pos = 0
            else:
                keep[i] = True
    Ff = F(keep)
    return dict(keep=keep, size=int(keep.sum()), frac=float(keep.sum() / N), F=float(Ff), cost=cost,
                wall=time.time() - t0, gap=gap)


def jaccard(a, b) -> float:
    a, b = set(np.flatnonzero(np.asarray(a))), set(np.flatnonzero(np.asarray(b)))
    return len(a & b) / max(1, len(a | b))
