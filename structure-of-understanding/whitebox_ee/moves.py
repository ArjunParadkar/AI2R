"""The three moves on a new input (theory §4, Defs. 4.1-4.2)."""
from __future__ import annotations

import copy
from typing import Callable, Dict, Optional

import numpy as np

from .filter import Thresholds
from .generality import compute_activity
from .graph import ConceptGraph
from .probe import Node, Probe


def extend(G: ConceptGraph, probe: Probe, X_new):
    """Probe and graph on train + new inputs. Node values on new rows come from the
    nodes' value maps (concepts), the task (inputs) and the readout (outputs)."""
    if isinstance(probe.X, np.ndarray):
        Xc = np.concatenate([probe.X, np.asarray(X_new)])
    else:
        Xc = list(probe.X) + list(X_new)
    P2 = Probe(probe.sys, Xc)
    nodes = {}
    for n, nd in G.nodes.items():
        nn = copy.copy(nd)
        if nd.kind == "input":
            nn.values = P2.inputs[n]
        elif nd.kind == "output":
            nn.values = P2.read[nd.meta["behaviour"]]
        else:
            m = nd.primary
            nn.values = m.values(P2.states[m.site])
        nodes[n] = nn
    G2 = ConceptGraph(nodes, G.parents, G.psi, certificates=G.certificates)
    return G2, P2, np.arange(probe.N), np.arange(probe.N, P2.N)


def classify_moves(G: ConceptGraph, probe: Probe, X_new, th: Thresholds, rng,
                   reference: Optional[Dict[str, Callable]] = None, m: int = 8):
    """For each new input: 'lower' | 'sideways' | 'diagonal' (per behaviour and overall),
    with the reasons, and for sideways inputs the interchange-assembled values at the
    unseen parent tuples (and, if a reference derivation is given for a node,
    whether they match it -> predicted correctness)."""
    G2, P2, tr, new = extend(G, probe, X_new)
    A = compute_activity(P2, G2, th, rng, rows=new, pool=tr, m=m)
    # ranges
    if isinstance(P2.X, np.ndarray):
        _, inv = np.unique(P2.X[tr], axis=0, return_inverse=True)
        uid = inv.reshape(-1)
    else:
        uid = np.arange(len(tr))
    rng_val, rng_tup, dist99 = {}, {}, {}
    for c in G2.concepts() + G2.outputs():
        nd = G2.nodes[c]
        v = nd.values[tr]
        rng_val[c] = {int(x) for x in np.unique(v) if len(np.unique(uid[v == x])) >= th.n_min}
        ps = G2.parents.get(c, [])
        tup = list(zip(*[G2.nodes[p].values[tr].tolist() for p in ps])) if ps else [()] * len(tr)
        cnt = {}
        for t, u in zip(tup, uid):
            cnt.setdefault(t, set()).add(int(u))
        rng_tup[c] = {t for t, s in cnt.items() if len(s) >= th.n_min}
        if nd.kind == "output":
            continue
        mm = nd.primary
        cd = np.sqrt(((mm.centroids[:, None] - mm.centroids[None]) ** 2).sum(-1))
        cd = cd[cd > 0].min() if (cd > 0).any() else 1.0
        # floor: a quarter of the closest centroid separation (noiseless systems have ~0 spread)
        dist99[c] = max(np.quantile(mm.dist_to_fibre(P2.states[mm.site][tr]), 0.99), 0.25 * cd)
    out = []
    behaviours = A.behaviours
    for i, r in enumerate(new):
        rec = {"per_behaviour": {}}
        worst = 0
        for b in behaviours:
            onp = [c for c in G2.concepts() if A.on_path[c][b][i]]
            reasons = []
            kind = 0
            unseen = []
            for c in onp:
                nd = G2.nodes[c]
                mm = nd.primary
                d = mm.dist_to_fibre(P2.states[mm.site][[r]])[0]
                if d > dist99[c]:
                    kind = 2
                    reasons.append(f"{c}: realiser state has no value")
                    continue
                if int(nd.values[r]) not in rng_val[c]:
                    kind = max(kind, 2)
                    reasons.append(f"{c}: value out of range")
                    continue
                ps = G2.parents.get(c, [])
                t = tuple(int(G2.nodes[p].values[r]) for p in ps)
                if t not in rng_tup[c]:
                    kind = max(kind, 1)
                    unseen.append((c, t))
                    reasons.append(f"{c}: parent tuple {t} unseen")
            # the output's own parent tuple (the final derivation step, M7)
            ob = [o for o in G2.outputs() if G2.nodes[o].meta["behaviour"] == b][0]
            ps = G2.parents.get(ob, [])
            if ps and kind < 2:
                t = tuple(int(G2.nodes[p].values[r]) for p in ps)
                if t not in rng_tup[ob]:
                    kind = max(kind, 1)
                    unseen.append((ob, t))
                    reasons.append(f"{ob}: parent tuple {t} unseen")
            if not onp and not ps:
                kind = 2
                reasons.append("no node on the derivation (output not mediated by the graph)")
            rec["per_behaviour"][b] = {"move": ["lower", "sideways", "diagonal"][kind], "reasons": reasons,
                                       "unseen": unseen}
            worst = max(worst, kind)
        rec["move"] = ["lower", "sideways", "diagonal"][worst]
        out.append(rec)
    # sideways: interchange-assembled values at unseen tuples
    for i, r in enumerate(new):
        for b, pb in out[i]["per_behaviour"].items():
            checks = []
            for c, t in pb["unseen"]:
                ps = G2.parents[c]
                # base: a training input; patch each parent from a training input with that value
                base = rng.choice(tr)
                srcs = []
                ok = True
                for p, v in zip(ps, t):
                    pool = tr[G2.nodes[p].values[tr] == v]
                    if len(pool) == 0:
                        ok = False
                        break
                    srcs.append((G2.nodes[p], np.array([rng.choice(pool)])))
                if not ok:
                    checks.append({"node": c, "tuple": t, "assembled": None})
                    continue
                lg, st = P2.run_patched(np.array([base]), srcs,
                                        sites=[G2.nodes[c].primary.site] if G2.nodes[c].members else ())
                val = int(P2.node_values_from_states(G2.nodes[c], st, lg)[0])
                ch = {"node": c, "tuple": t, "assembled": val, "native": int(G2.nodes[c].values[r])}
                if reference and c in reference:
                    ch["reference"] = int(reference[c](*t))
                    ch["matches_reference"] = ch["assembled"] == ch["reference"]
                checks.append(ch)
            pb["interchange"] = checks
            if pb["move"] == "lower":
                pb["predicted_correct"] = True
            elif pb["move"] == "diagonal":
                pb["predicted_correct"] = False
            else:
                ms = [ch.get("matches_reference") for ch in checks]
                pb["predicted_correct"] = (all(ms) if ms and all(x is not None for x in ms) else None)
    return out
