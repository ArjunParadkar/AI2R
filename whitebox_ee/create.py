"""Creation protocol (theory Defs. 4.5-4.6): valid new nodes and false creations."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .filter import Thresholds
from .graph import ConceptGraph
from .info import normalised_vi


def _equiv(vals, others, th, n_eff):
    return [n for n, v in others.items() if normalised_vi(vals, v, n_eff) <= th.eps_merge]


def creation(G0: ConceptGraph, G1: ConceptGraph, rec1: dict, th: Thresholds, n_eff: int,
             G_ctrl: Optional[ConceptGraph] = None, probe_values_G0: Optional[Dict[str, np.ndarray]] = None,
             probe_values_ctrl: Optional[Dict[str, np.ndarray]] = None, gen1: Optional[Dict[str, float]] = None,
             subthreshold_old: Optional[Dict[str, np.ndarray]] = None):
    """G0: graph before exposure; G1: after (same probe inputs). Node values of G0 must be
    evaluated on the same inputs as G1 (pass probe_values_G0 if G0 was built on a
    different probe). Returns valid new nodes and the false-creation table.

    subthreshold_old (amendment 3): partitions of before-exposure candidates that cleared the
    uncorrected per-behaviour screen (A_b > eta_b for some b) but not the family-wise criterion.
    A G1 node equivalent to one of them is not clearly absent before exposure, so it is reported
    as "relabelling (sub-threshold before)" and never counts as created (threshold hysteresis)."""
    old = probe_values_G0 or {n: G0.nodes[n].values for n in G0.nodes if G0.nodes[n].kind != "output"}
    old_concepts = {n for n in old if n in G0.nodes and G0.nodes[n].kind == "concept"} if probe_values_G0 is None else \
        {n for n in old if not n.startswith("in:")}
    inputs = {n for n, nd in G1.nodes.items() if nd.kind == "input"}
    ctrl = None
    if G_ctrl is not None:
        ctrl = probe_values_ctrl or {n: G_ctrl.nodes[n].values for n in G_ctrl.concepts()}
    new_status = {}
    # pass 1: relabellings
    for c in G1.concepts():
        eq = _equiv(G1.nodes[c].values, old, th, n_eff)
        if eq:
            new_status[c] = ("relabelling", eq)
        elif subthreshold_old:
            eq2 = _equiv(G1.nodes[c].values, subthreshold_old, th, n_eff)
            if eq2:
                new_status[c] = ("relabelling (sub-threshold before)", eq2)
    candidates = [c for c in G1.concepts() if c not in new_status]
    # map G1 nodes that are equivalent to old nodes or inputs -> "existing"
    existing = set(inputs) | {c for c, s in new_status.items() if s[0].startswith("relabelling")}
    valid = set()
    changed = True
    while changed:
        changed = False
        for c in candidates:
            if c in valid:
                continue
            ps = G1.parents.get(c, [])
            if ps and all(p in existing or p in valid for p in ps) and any(p not in inputs for p in ps):
                valid.add(c)
                changed = True
    for c in candidates:
        ps = G1.parents.get(c, [])
        if c in valid:
            if ctrl is not None and _equiv(G1.nodes[c].values, ctrl, th, n_eff):
                new_status[c] = ("exposure-generic", ps)
            else:
                new_status[c] = ("valid", ps)
        elif ps and all(p in inputs for p in ps):
            new_status[c] = ("new leaf (used, not derived)", ps)
        else:
            new_status[c] = ("not derived from existing nodes", ps)
    # decodable-but-unused new candidates (from the after-exposure discovery record)
    unused = [k for k, v in rec1["candidates"].items() if v["verdict"] == "represented, not used"]
    res = {"valid": [c for c, s in new_status.items() if s[0] == "valid"],
           "status": new_status,
           "represented_not_used_candidates": unused}
    if gen1 is not None:
        res["reach"] = {c: gen1.get(c) for c in res["valid"]}
    return res
