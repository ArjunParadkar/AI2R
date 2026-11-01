"""Activity, active derivations and generality (theory §3, with the Phase 1
edge-activity amendment, reports/phase1.md §2).

On input x and behaviour b:
  node C is active   if swapping C from partners with a different value changes
                     r_b on >= alpha of partners;
  edge P->C is active if swapping P from partners with a different P value,
                     holding every graph node strictly between P and C at its
                     clean value, changes C (or r_b if C is the output) on at least
                     alpha times the fraction of those partners whose own C value
                     differs from C(x) (normalised edge activity).
D_b(x) = active nodes + inputs + out_b, with active edges. A node lies on the
derivation of x if it is on an active-edge path from an input to out_b.
Generality g(C) = sum_b w_b E_x[ C on the derivation of x for b ].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .filter import Thresholds, random_member
from .graph import ConceptGraph
from .probe import Node, Probe


@dataclass
class Activity:
    rows: np.ndarray
    node: Dict[str, Dict[str, np.ndarray]]          # node -> behaviour -> bool (len rows)
    edge: Dict[tuple, np.ndarray]                   # (P, C) -> bool
    on_path: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)   # node -> b -> bool
    behaviours: List[str] = field(default_factory=list)


def _partner_rows(probe: Probe, values, rows, m, rng, pool):
    """m partners per row with a different value, drawn from `pool` indices."""
    out = np.full((len(rows), m), -1, np.int64)
    pv = values[pool]
    for i, r in enumerate(rows):
        cand = pool[pv != values[r]]
        if len(cand):
            out[i] = rng.choice(cand, m)
    return out


def compute_activity(probe: Probe, G: ConceptGraph, th: Thresholds, rng, rows=None, pool=None,
                     m: Optional[int] = None) -> Activity:
    m = m or th.n_partners
    rows = np.arange(probe.N) if rows is None else np.asarray(rows)
    pool = np.arange(probe.N) if pool is None else np.asarray(pool)
    behaviours = list(probe.sys.behaviours)
    concepts = [G.nodes[n] for n in G.concepts()]
    outs = {G.nodes[o].meta["behaviour"]: G.nodes[o] for o in G.outputs()}
    node_act = {}
    rep = np.repeat(rows, m)
    for C in concepts:
        src = _partner_rows(probe, C.values, rows, m, rng, pool).reshape(-1)
        ok = src >= 0
        src_ = np.where(ok, src, rep)
        lg, _ = probe.run_patched(rep, [(C, src_)], sites=())
        node_act[C.name] = {}
        for b in behaviours:
            ch = (lg[b].argmax(-1) != probe.read[b][rep]) & ok
            node_act[C.name][b] = ch.reshape(len(rows), m).mean(1) >= th.alpha
    # edges: one run per (parent, target site)
    edge_act = {}
    by_parent: Dict[str, List[str]] = {}
    for c, ps in G.parents.items():
        for p in ps:
            by_parent.setdefault(p, []).append(c)
    pool_nodes = [G.nodes[n] for n in G.concepts()]
    for p, chs in by_parent.items():
        P = G.nodes[p]
        src = _partner_rows(probe, P.values, rows, m, rng, pool).reshape(-1)
        ok = src >= 0
        src_ = np.where(ok, src, rep)
        by_site: Dict[int, List[Node]] = {}
        for c in chs:
            by_site.setdefault(G.nodes[c].site_index, []).append(G.nodes[c])
        for t, targets in by_site.items():
            freeze = [Q for Q in pool_nodes if Q is not P and P.site_index < Q.site_index < t]
            lg, st = probe.run_patched(rep, [(P, src_)], freeze=freeze,
                                       sites=sorted({C.primary.site for C in targets if C.members}))
            for C in targets:
                v = probe.node_values_from_states(C, st, lg)
                ch = ((v != C.values[rep]) & ok).reshape(len(rows), m).mean(1)
                # normalised edge activity: the P-swap transmits >= alpha of the variation
                # in C that the same partners carry (phase1.md §2)
                avail = ((C.values[src_] != C.values[rep]) & ok).reshape(len(rows), m).mean(1)
                edge_act[(p, C.name)] = (ch > 0) & (ch >= th.alpha * avail)
    A = Activity(rows, node_act, edge_act, behaviours=behaviours)
    A.on_path = paths(G, A)
    return A


def paths(G: ConceptGraph, A: Activity) -> Dict[str, Dict[str, np.ndarray]]:
    """on_path[node][b][i]: node lies on an active input->out_b path on row i."""
    n = len(A.rows)
    topo = G.topo()
    outs = {G.nodes[o].meta["behaviour"]: o for o in G.outputs()}
    res = {c: {} for c in G.nodes}
    for b in A.behaviours:
        ob = outs[b]
        alive = {}
        for c in topo:
            nd = G.nodes[c]
            if nd.kind == "input":
                alive[c] = np.ones(n, bool)
            elif nd.kind == "output":
                alive[c] = np.ones(n, bool) if c == ob else np.zeros(n, bool)
            else:
                alive[c] = A.node[c][b]
        fwd = {}
        for c in topo:
            nd = G.nodes[c]
            if nd.kind == "input":
                fwd[c] = alive[c].copy()
                continue
            r = np.zeros(n, bool)
            for p in G.parents.get(c, []):
                r |= fwd[p] & A.edge.get((p, c), np.zeros(n, bool))
            fwd[c] = r & alive[c]
        bwd = {c: np.zeros(n, bool) for c in topo}
        bwd[ob] = fwd[ob].copy()
        for c in reversed(topo):
            if c == ob:
                continue
            r = np.zeros(n, bool)
            for ch in G.children(c):
                r |= bwd[ch] & A.edge.get((c, ch), np.zeros(n, bool))
            bwd[c] = r & fwd[c]
        for c in topo:
            res[c][b] = bwd[c]
    return res


def generality(G: ConceptGraph, A: Activity, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    bs = A.behaviours
    w = weights or {b: 1 / len(bs) for b in bs}
    return {c: float(sum(w[b] * A.on_path[c][b].mean() for b in bs)) for c in G.concepts()}


def count_generality(G: ConceptGraph, A: Activity, alpha: float = 0.5) -> Dict[str, int]:
    return {c: int(sum(A.on_path[c][b].mean() >= alpha for b in A.behaviours)) for c in G.concepts()}


def random_realiser_generality(probe: Probe, th: Thresholds, rng, k: int = 4, n_per_site: int = 5,
                               rows=None, m: int = 4) -> np.ndarray:
    """Null for generality when the untrained-copy graph is empty: node-activity mass
    of random realisers (no path requirement, so this null is conservative)."""
    rows = np.arange(min(probe.N, 512)) if rows is None else rows
    gs = []
    for s in probe.sites:
        d = probe.sys.site_dim[s]
        for _ in range(n_per_site):
            mem = random_member(s, d, min(k, d), rng)
            R = Node("rand", "concept", [mem], None, probe.site_idx(s))
            rep = np.repeat(rows, m)
            src = rng.integers(0, probe.N, len(rep))
            lg, _ = probe.run_patched(rep, [(R, src)], sites=())
            act = []
            for b in probe.sys.behaviours:
                ch = lg[b].argmax(-1) != probe.read[b][rep]
                act.append((ch.reshape(len(rows), m).mean(1) >= th.alpha).mean())
            gs.append(float(np.mean(act)))
    return np.array(gs)
