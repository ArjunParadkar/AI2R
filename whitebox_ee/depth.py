"""Depth of understanding and structural memorization (theory §3).

Phase 1 amendment (reports/phase1.md §2): on input x a node contributes its
generality only if its value C(x) is in its training range Rg(C) (taken on at
least n_min distinct training inputs, theory Def. 4.1); otherwise it is a leaf
on x and contributes 0. Inputs and outputs have weight 0.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .filter import Thresholds, tv
from .generality import Activity
from .graph import ConceptGraph
from .probe import Probe
from .systems import softmax


def value_support(G: ConceptGraph, probe: Probe, train_rows: np.ndarray) -> Dict[str, Dict[int, int]]:
    """Number of distinct training inputs per value of each concept."""
    X = probe.X
    if isinstance(X, np.ndarray):
        _, inv = np.unique(X[train_rows], axis=0, return_inverse=True)
        uid = inv.reshape(-1)
    else:
        uid = np.arange(len(train_rows))
    sup = {}
    for c in G.concepts():
        v = G.nodes[c].values[train_rows]
        d = {}
        for val in np.unique(v):
            d[int(val)] = len(np.unique(uid[v == val]))
        sup[c] = d
    return sup


def node_weights(G, A: Activity, g: Dict[str, float], support, n_min: int):
    """g_x(C) per row (len rows): g(C) if C(x) in range else 0."""
    w = {}
    for c in G.concepts():
        vals = G.nodes[c].values[A.rows]
        inr = np.array([support[c].get(int(v), 0) >= n_min for v in vals])
        w[c] = np.where(inr, g[c], 0.0)
    return w


def bottleneck(G: ConceptGraph, A: Activity, w: Dict[str, np.ndarray], b: str) -> np.ndarray:
    """lambda(x) = min over active input->out_b paths of max node weight on the path
    (Prop. 3.5 recursion: val(v) = max(w(v), min_{u->v active} val(u)))."""
    n = len(A.rows)
    INF = np.inf
    ob = [o for o in G.outputs() if G.nodes[o].meta["behaviour"] == b][0]
    val = {}
    for c in G.topo():
        nd = G.nodes[c]
        if nd.kind == "input":
            val[c] = np.zeros(n)
            continue
        if nd.kind == "output" and c != ob:
            continue
        alive = np.ones(n, bool) if nd.kind == "output" else A.node[c][b]
        best = np.full(n, INF)
        for p in G.parents.get(c, []):
            if p not in val:
                continue
            e = A.edge.get((p, c), np.zeros(n, bool))
            best = np.where(e, np.minimum(best, val[p]), best)
        own = w[c] if nd.kind == "concept" else np.zeros(n)
        v = np.maximum(own, best)
        val[c] = np.where(alive, v, INF)
    lam = val[ob]
    return np.where(np.isfinite(lam), lam, 0.0)        # no active path -> 0 (not grounded)


def memorization_level(G: ConceptGraph, A: Activity, w: Dict[str, np.ndarray], b: str) -> np.ndarray:
    n = len(A.rows)
    mu = np.zeros(n)
    for c in G.concepts():
        mu = np.maximum(mu, np.where(A.on_path[c][b], w[c], 0.0))
    return mu


def depth_scores(probe: Probe, G: ConceptGraph, A: Activity, g: Dict[str, float], th: Thresholds,
                 train_rows: Optional[np.ndarray] = None):
    train_rows = np.arange(probe.N) if train_rows is None else train_rows
    sup = value_support(G, probe, train_rows)
    w = node_weights(G, A, g, sup, th.n_min)
    out = {}
    for b in A.behaviours:
        lam = bottleneck(G, A, w, b)
        mu = memorization_level(G, A, w, b)
        out[b] = {"lambda": lam, "mu": mu}
    return out


def behaviour_verdicts(scores, th: Thresholds, tau: float):
    res = {}
    for b, d in scores.items():
        lam, mu = d["lambda"], d["mu"]
        Lam = float((lam >= tau).mean())
        memo = float((mu < tau).mean())
        if Lam >= 1 - th.eps_M:
            v = "grounded"
        elif memo >= 1 - th.eps_M:
            v = "memorized"
        else:
            v = "partly grounded"
        res[b] = {"Lambda": Lam, "lambda_bar": float(lam.mean()), "frac_mu_below": memo, "verdict": v}
    return res


def gamma_share(probe: Probe, G: ConceptGraph, principles: Sequence[str], rows, th: Thresholds, rng,
                m: int = 4, pool=None):
    """gamma_tau(x) = 1 - E d(f(x), f_{Pi<-x}(x')) / E d(f(x), f(x')) per behaviour."""
    pool = np.arange(probe.N) if pool is None else np.asarray(pool)
    rows = np.asarray(rows)
    rep = np.repeat(rows, m)
    xp = rng.choice(pool, len(rep))
    nodes = [(G.nodes[c], rep) for c in principles]
    if nodes:
        # run on x' with principles patched from x
        lg, _ = probe.run_patched(xp, nodes)
    else:
        lg = {b: probe.logits[b][xp] for b in probe.sys.behaviours}
    out = {}
    for b in probe.sys.behaviours:
        px = softmax(probe.logits[b][rep])
        num = tv(px, softmax(lg[b])).reshape(len(rows), m).mean(1)
        den = tv(px, softmax(probe.logits[b][xp])).reshape(len(rows), m).mean(1)
        out[b] = 1 - num / np.maximum(den, 1e-9)
    return out
