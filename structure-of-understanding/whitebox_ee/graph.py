"""Concept graph: merging, derivation maps psi, the high-level model F_G,
causal-use (IIA) and the discovery fixed point."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .filter import Thresholds, random_member, quotient_test
from .info import entropy, normalised_vi
from .parents import direct_effects, select_parents, sufficiency, minimal_sufficient_sets, parent_stability
from .probe import Member, Node, Probe
from .stats import bootstrap_ci


def fit_psi(C: Node, pa_nodes: Sequence[Node], extra=None):
    """Derivation map as a lookup table: majority value of C per parent tuple, on the
    observational tuples plus (M8) the interventionally produced ones in `extra`."""
    if not pa_nodes:
        return {(): Counter(C.values.tolist()).most_common(1)[0][0]}
    keys = list(zip(*[p.values.tolist() for p in pa_nodes]))
    vals = C.values.tolist()
    if extra:
        keys = keys + list(extra[0])
        vals = vals + list(extra[1])
    tab: Dict[tuple, Counter] = {}
    for k, v in zip(keys, vals):
        tab.setdefault(k, Counter())[v] += 1
    return {k: c.most_common(1)[0][0] for k, c in tab.items()}


@dataclass
class ConceptGraph:
    nodes: Dict[str, Node]
    parents: Dict[str, List[str]] = field(default_factory=dict)
    psi: Dict[str, dict] = field(default_factory=dict)
    log: List[dict] = field(default_factory=list)
    certificates: Dict[str, dict] = field(default_factory=dict)
    alternatives: Dict[str, list] = field(default_factory=dict)

    def topo(self) -> List[str]:
        return sorted(self.nodes, key=lambda n: (self.nodes[n].site_index, n))

    def children(self, name):
        return [c for c, ps in self.parents.items() if name in ps]

    def descendants(self, name):
        out, stack = set(), [name]
        while stack:
            u = stack.pop()
            for c in self.children(u):
                if c not in out:
                    out.add(c)
                    stack.append(c)
        return out

    def ancestors(self, name):
        out, stack = set(), [name]
        while stack:
            u = stack.pop()
            for p in self.parents.get(u, []):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out

    def rename(self, mapping: Dict[str, str]):
        f = lambda n: mapping.get(n, n)
        self.nodes = {f(n): v for n, v in self.nodes.items()}
        for n, v in self.nodes.items():
            v.name = n
        self.parents = {f(n): [f(p) for p in ps] for n, ps in self.parents.items()}
        self.psi = {f(n): t for n, t in self.psi.items()}
        self.certificates = {f(n): c for n, c in self.certificates.items()}
        self.alternatives = {f(n): a for n, a in self.alternatives.items()}

    def concepts(self):
        return [n for n, v in self.nodes.items() if v.kind == "concept"]

    def outputs(self):
        return [n for n, v in self.nodes.items() if v.kind == "output"]

    def counterfactual(self, rows: np.ndarray, do: Dict[str, np.ndarray]):
        """F_G(x; do(C=v)): recompute descendants of the intervened nodes with psi.
        Returns dict node -> predicted values (int, -1 where unidentified)."""
        vals = {n: v.values[rows].copy() for n, v in self.nodes.items()}
        undefined = np.zeros(len(rows), bool)
        for n, v in do.items():
            vals[n] = np.asarray(v).copy()
        affected = set()
        for n in do:
            affected |= self.descendants(n)
        for n in self.topo():
            if n not in affected or n in do:
                continue
            ps = self.parents.get(n, [])
            tab = self.psi[n]
            keys = zip(*[vals[p].tolist() for p in ps]) if ps else [()] * len(rows)
            out = np.empty(len(rows), np.int64)
            for i, k in enumerate(keys):
                if k in tab:
                    out[i] = tab[k]
                else:
                    out[i] = -1
                    undefined[i] = True
            vals[n] = out
        return vals, undefined


def iia(probe: Probe, G: ConceptGraph, name: str, th: Thresholds, rng, random_null: bool = True):
    """Interchange intervention accuracy of node `name` against F_G (theory Def. 2.9)."""
    C = G.nodes[name]
    rows = rng.integers(0, probe.N, th.n_pairs)
    src = probe.partners(C.values, 1, differ=True, rows=rows, rng=rng)[:, 0]
    ok = src >= 0
    rows, src = rows[ok], src[ok]
    pred, undef = G.counterfactual(rows, {name: C.values[src]})
    lg, _ = probe.run_patched(rows, [(C, src)], sites=())
    outs = G.outputs()
    agree_all, agree_eff, n_eff = [], [], 0
    rand_eff = []
    if random_null and C.members:
        R = Node("rand", "concept", [random_member(C.primary.site, probe.sys.site_dim[C.primary.site],
                                                   C.primary.U.shape[1], rng, C.primary.U)])
        lgr, _ = probe.run_patched(rows, [(R, src)], sites=())
    for o in outs:
        b = G.nodes[o].meta["behaviour"]
        sysv = lg[b].argmax(-1)
        pv = pred[o]
        d = ~undef & (pv >= 0)
        clean = G.nodes[o].values[rows]
        eff = d & (pv != clean)
        agree_all.append((sysv[d] == pv[d]).astype(float))
        agree_eff.append((sysv[eff] == pv[eff]).astype(float))
        n_eff += int(eff.sum())
        if random_null and C.members:
            rv = lgr[b].argmax(-1)
            rand_eff.append((rv[eff] == pv[eff]).astype(float))
    a_all = np.concatenate(agree_all) if agree_all else np.array([])
    a_eff = np.concatenate(agree_eff) if agree_eff else np.array([])
    r_eff = np.concatenate(rand_eff) if rand_eff else np.array([0.0])
    m_all = bootstrap_ci(a_all, n_boot=500) if len(a_all) else (np.nan,) * 3
    m_eff = bootstrap_ci(a_eff, n_boot=500) if len(a_eff) else (np.nan,) * 3
    rnd = float(r_eff.mean()) if len(r_eff) else 0.0
    frac_undef = float(undef.mean())
    pass_eff = (len(a_eff) >= th.min_effective_pairs and m_eff[0] >= th.tau_IIA and (m_eff[1] - rnd) >= th.delta_IIA)
    passed = pass_eff and m_all[0] >= th.tau_IIA
    return dict(passed=bool(passed), pass_eff=bool(pass_eff), iia=m_all, iia_eff=m_eff, n_eff=n_eff, random_eff=rnd, frac_undefined=frac_undef)


# ---------------------------------------------------------------------------
# merging candidates into nodes (theory Def. 2.12)
# ---------------------------------------------------------------------------

def merge_equivalent(nodes: List[Node], inputs: List[Node], th: Thresholds, n_eff: int) -> List[Node]:
    """Merge candidates with the same partition (normalised VI <= eps_merge).
    Candidates equivalent to an input node become realisers of that input node.
    Within a class the earliest site is the node's site; members keep all realisers."""
    out: List[Node] = []
    for c in sorted(nodes, key=lambda n: n.site_index):
        merged = False
        for inp in inputs:
            if normalised_vi(c.values, inp.values, n_eff) <= th.eps_merge:
                inp.members.extend(c.members)
                inp.meta.setdefault("merged", []).append(c.name)
                merged = True
                break
        if merged:
            continue
        for o in out:
            if normalised_vi(c.values, o.values, n_eff) <= th.eps_merge:
                o.members.extend(c.members)
                o.meta.setdefault("merged", []).append(c.name)
                merged = True
                break
        if not merged:
            out.append(c)
    return out


def label_nodes(G: ConceptGraph, labels: Dict[str, np.ndarray], n_eff: int, max_vi: float = 0.1,
                rows: Optional[np.ndarray] = None) -> Dict[str, Optional[str]]:
    """Match each concept to the known variable with the smallest normalised VI
    (<= max_vi), for evaluation against planted / known algorithms only."""
    out = {}
    for c in G.concepts():
        v = G.nodes[c].values if rows is None else G.nodes[c].values[rows]
        best, bv = None, np.inf
        for name, lab in labels.items():
            lab = lab if rows is None else lab[rows]
            d = normalised_vi(v, lab, n_eff)
            if d < bv:
                best, bv = name, d
        out[c] = best if bv <= max_vi else None
        G.nodes[c].meta["label"] = out[c]
        G.nodes[c].meta["label_vi"] = float(bv)
    return out


def align_values(G: ConceptGraph, labels: Dict[str, np.ndarray]):
    """Evaluation only: recode each labelled concept's values (and its members' value
    ids) to the matched known variable's coding by majority map. Codings are
    arbitrary up to invertible relabelling (Prop. 2.13), so this changes no verdict."""
    for c in G.concepts():
        lab = G.nodes[c].meta.get("label")
        if not lab:
            continue
        v, t = G.nodes[c].values, labels[lab]
        mp = {}
        for val in np.unique(v):
            u, cnt = np.unique(t[v == val], return_counts=True)
            mp[int(val)] = int(u[cnt.argmax()])
        if len(set(mp.values())) < len(mp):
            continue                      # not injective: leave as is
        G.nodes[c].values = np.array([mp[int(x)] for x in v])
        for m in G.nodes[c].members:
            m.value_ids = np.array([mp.get(int(x), int(x)) for x in m.value_ids])
