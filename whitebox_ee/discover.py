"""Discovery: propose -> quotient -> merge -> (parents, psi, IIA) fixed point."""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .filter import Thresholds, quotient_test, decodability
from .graph import ConceptGraph, fit_psi, iia, merge_equivalent
from .info import entropy, normalised_vi
from .parents import direct_effects, interventional_tuples, select_parents, sufficiency, minimal_sufficient_sets, parent_stability
from .probe import Member, Node, Probe


def make_candidate(probe: Probe, m: Member, name: str) -> Node:
    vals = m.values(probe.states[m.site])
    return Node(name, "concept", [m], vals, probe.site_layer(m.site), meta={"source": m.source})


def containment(U_small: np.ndarray, U_big: np.ndarray) -> float:
    """Fraction of span(U_small) inside span(U_big) (mean squared cosine of principal angles)."""
    return float((np.linalg.norm(U_big.T @ U_small) ** 2) / U_small.shape[1])


def separable(a: Node, b: Node, states: np.ndarray, rng, n: int = 400, min_frac: float = 0.95) -> bool:
    """State-level modularity: swapping a's realiser (from random partners) leaves b's
    decoded value unchanged on >= min_frac of rows, and vice versa."""
    N = len(states)
    r = rng.integers(0, N, n)
    s = rng.integers(0, N, n)
    for x, y in ((a, b), (b, a)):
        U = x.primary.U
        h = states[r] + ((states[s] - states[r]) @ U) @ U.T
        if (y.primary.values(h) == y.primary.values(states[r])).mean() < min_frac:
            return False
    return True


def find_composites(cands, inner=None, contain_min: float = 0.9, states=None, seed: int = 0,
                    n_eff: int = 0, eps_merge: float = 0.05):
    """A candidate is a composite if >= 2 smaller candidates lie inside its realiser,
    are pairwise separable (modular), and together span it."""
    out = {}
    inner = cands if inner is None else inner
    rng = np.random.default_rng(seed)
    for c in cands:
        inside = [o for o in inner if o is not c and o.primary.site == c.primary.site
                  and o.primary.U.shape[1] < c.primary.U.shape[1]
                  and containment(o.primary.U, c.primary.U) >= contain_min]
        if states is not None and len(inside) >= 2:
            st = states[c.primary.site]
            # keep a maximal pairwise-separable subset (greedy, largest realisers first)
            inside.sort(key=lambda o: -o.primary.U.shape[1])
            sel = []
            for o in inside:
                if all(separable(o, q, st, rng) for q in sel):
                    sel.append(o)
            inside = sel
        if len(inside) >= 2:
            B = np.linalg.qr(np.concatenate([o.primary.U for o in inside], 1))[0]
            if containment(c.primary.U, B) >= contain_min:
                out[c.name] = [o.name for o in inside]
    return out


def drop_coarsenings(cands, inputs, record, th, contain_min: float = 0.9):
    """Phase 1 canonicalisation at each site (reports/phase1.md §2):
    (1) composite: a candidate whose realiser is covered (>= contain_min) by the
        span of >= 2 smaller candidates it contains is a product of them -> dropped;
    (2) coarsening: a candidate whose realiser lies inside a larger non-composite
        candidate (or input realiser) at the same site that refines its partition
        is a coarsening -> dropped."""
    from .info import cond_entropy_plugin
    keep = []
    others = [(c.name, c.primary, c.values) for c in cands]
    for inp in inputs:
        # an input realiser is a coarsening container only at the input's primary site; at later
        # sites every derived variable is linearly readable from the persisting input blocks
        if inp.members:
            m0 = inp.members[0]
            others.append((inp.name, m0, inp.values))
    for c in cands:
        m = c.primary
        hit = None
        for name, om, ov in others:
            if name == c.name or om.site != m.site or om.U.shape[1] <= m.U.shape[1]:
                continue
            if containment(m.U, om.U) >= contain_min and cond_entropy_plugin(c.values, [ov]) <= th.eps_merge * max(entropy(c.values), 1e-9):
                hit = name
                break
        if hit is None:
            keep.append(c)
        else:
            record["candidates"][c.name]["verdict"] = f"coarsening of {hit}"
    return keep


def _one_pass(probe, inputs, nodes, outs, th, rng, strict, suspended=()):
    pool = inputs + nodes
    eff, nul, smp = direct_effects(probe, pool, list(outs) + list(suspended), th, rng, return_samples=True)
    allnodes = {n.name: n for n in pool + outs}
    parents, psi, drop, cert = {}, {}, {}, {}
    react = []
    for C in suspended:
        pa = select_parents(C, pool, eff, nul, th)
        suf = sufficiency(C, [allnodes[p] for p in pa])
        if 0 < len(pa) <= th.k_max and suf <= th.eps_D:
            react.append(C.name)
    for C in nodes + outs:
        pa = select_parents(C, pool, eff, nul, th)
        pa_nodes = [allnodes[p] for p in pa]
        suf = sufficiency(C, pa_nodes)
        parents[C.name] = pa
        psi[C.name] = fit_psi(C, pa_nodes, interventional_tuples(C, pa_nodes, smp))
        cert[C.name] = {"parents": pa, "suff": suf,
                        "edge_effects": {p: eff.get((p, C.name), 0.0) for p in pa},
                        "edge_null": {p: nul.get((p, C.name), 0.0) for p in pa}}
        if C.kind == "concept":
            if len(pa) == 0:
                drop[C.name] = "no parents (derivation fails)"
            elif len(pa) > th.k_max:
                drop[C.name] = f"{len(pa)} parents > k_max"
            elif suf > th.eps_D:
                drop[C.name] = f"parents not eps_D-sufficient ({suf:.3f}; pa={pa})"
    for inp in inputs:
        parents[inp.name] = []
        psi[inp.name] = {}
    G = ConceptGraph(allnodes, parents, psi)
    for C in nodes:
        r = iia(probe, G, C.name, th, rng)
        cert[C.name]["iia"] = r
        if C.name in drop:
            continue
        # non-strict passes: too few effective pairs means the provisional graph gives the
        # node no predicted downstream change yet -> pending, not failed
        ok = (r["pass_eff"] or r["n_eff"] < th.min_effective_pairs) if not strict else r["passed"]
        if not ok:
            drop[C.name] = (f"causal use fails ({'full' if strict else 'effective-pair'} test: IIA {r['iia'][0]:.3f}, "
                            f"eff {r['iia_eff'][0]:.3f}, n_eff {r['n_eff']}, random {r['random_eff']:.3f})")
    G.certificates = cert
    # drop schedule (phase1.md §2): derivation failures first, earliest failing site only
    # (parent sets depend on upstream correctness); otherwise only the
    # causal-use failures at the latest failing site (downstream verdicts depend on less
    # of the provisional graph, so they are trusted first).
    deriv = {k: v for k, v in drop.items() if not v.startswith("causal")}
    if deriv:
        first = min(allnodes[k].site_index for k in deriv)
        return G, cert, {k: v for k, v in deriv.items() if allnodes[k].site_index == first}, react
    if drop:
        last = max(allnodes[k].site_index for k in drop)
        drop = {k: v for k, v in drop.items() if allnodes[k].site_index == last}
    return G, cert, drop, react


def _peel(probe, nodes, drop, record):
    kept, gone = [], []
    for n in nodes:
        if n.name not in drop:
            kept.append(n)
            continue
        # peel the members at the failing site; re-test the node at its next site
        cur = (probe.site_layer(n.primary.site), probe.site_idx(n.primary.site))
        later = [m for m in n.members if (probe.site_layer(m.site), probe.site_idx(m.site)) > cur
                 and m.site != n.primary.site]
        record["dropped"][f"{n.name}[{n.primary.site}]"] = drop[n.name]
        if later:
            later.sort(key=lambda m: (probe.site_layer(m.site), probe.site_idx(m.site), getattr(m, "quality", 1.0), m.U.shape[1]))
            n.members = later
            n.site_index = probe.site_layer(later[0].site)
            n.values = later[0].values(probe.states[later[0].site])
            kept.append(n)
        else:
            gone.append(n)
    return kept, gone


def base_nodes(probe: Probe):
    inputs = [Node(n, "input", [], probe.inputs[n], -1) for n in probe.sys.input_names]
    top = max(probe.site_layer(s) for s in probe.sites) + 1
    outs = [Node(f"out:{b}", "output", [], probe.read[b], top, meta={"behaviour": b})
            for b in probe.sys.behaviours]
    return inputs, outs


def discover(system, X, members: Sequence[Member], th: Optional[Thresholds] = None, seed: int = 0,
             max_iter: int = 6, stability: bool = True, verbose: bool = False, probe: Optional[Probe] = None,
             minimal_sets: bool = True, coarsening_rule: bool = False):
    th = th or Thresholds()
    rng = np.random.default_rng(seed)
    t0 = time.time()
    probe = probe or Probe(system, X, seed=seed)
    inputs, outs = base_nodes(probe)
    n_eff = len(np.unique(np.asarray(X), axis=0)) if isinstance(X, np.ndarray) else len(X)
    H_X = sum(entropy(i.values) for i in inputs)
    cands = [make_candidate(probe, m, f"cand{i}@{m.site}:{m.source}") for i, m in enumerate(members)]
    record = {"candidates": {}, "dropped": {}, "iterations": []}
    # input-equivalent candidates (IE) become realisers of the input nodes (inputs are exempt
    # from Cond. 1-2) unless they are composites of quotient-passing candidates (checked below)
    rest, ie = [], []
    for c in cands:
        hit = None
        for inp in inputs:
            if normalised_vi(c.values, inp.values, n_eff) <= th.eps_merge:
                hit = inp
                break
        if hit is not None:
            ie.append((c, hit))
        else:
            rest.append(c)
    # Condition 1
    eta_cache = {}
    passed = []
    for c in rest:
        q = quotient_test(probe, c, H_X, th, eta_cache, rng)
        record["candidates"][c.name] = {"verdict": "quotient pass" if q.passed else "quotient fail",
                                       "delta_Q": q.delta_Q, "delta_ci": q.delta_ci, "A": q.A, "eta": q.eta,
                                       "H": q.H, "kappa": q.kappa, "p_perm": q.p_perm, "reasons": q.reasons,
                                       "site": c.primary.site, "k": c.primary.U.shape[1], "source": c.meta["source"],
                                       "A_b": None if q.A_b is None else np.round(q.A_b, 5).tolist(),
                                       "eta_b": None if q.eta_b is None else np.round(q.eta_b, 5).tolist(),
                                       "null_effects": None if q.null_effects is None else np.round(q.null_effects, 5).tolist()}
        c.primary.quality = q.delta_Q
        if q.passed:
            passed.append(c)
        else:
            # represented-not-used bookkeeping
            if q.reasons and q.reasons[0].startswith("realiser does not matter"):
                acc = decodability(probe.states[c.primary.site] @ c.primary.U, c.values)
                record["candidates"][c.name]["decode_acc"] = acc
                if acc >= 0.9:
                    record["candidates"][c.name]["verdict"] = "represented, not used"
    comp = find_composites(passed + [c for c, _ in ie], inner=passed, states=probe.states, seed=seed,
                           n_eff=n_eff, eps_merge=th.eps_merge)
    for c in passed + [c for c, _ in ie]:
        if c.name in comp:
            record["candidates"].setdefault(c.name, {"site": c.primary.site, "source": c.meta["source"]})
            record["candidates"][c.name]["verdict"] = "composite of " + ",".join(comp[c.name])
    passed = [c for c in passed if c.name not in comp]
    for c, inp in ie:
        if c.name in comp:
            continue
        inp.members.extend(c.members)
        record["candidates"][c.name] = {"verdict": f"realiser of input {inp.name}", "site": c.primary.site,
                                       "source": c.meta["source"]}
    for inp in inputs:
        inp.members.sort(key=lambda m: (probe.site_layer(m.site), probe.site_idx(m.site)))
    if coarsening_rule:
        passed = drop_coarsenings(passed, inputs, record, th)
    # M7: a candidate equivalent to a behaviour's output partition is a realiser of that
    # output, not a derivation step (otherwise the answer representation lies on every
    # derivation with generality equal to its behaviour's mass)
    keep = []
    for c in passed:
        hit = next((o for o in outs if normalised_vi(c.values, o.values, n_eff) <= th.eps_merge), None)
        if hit is None:
            keep.append(c)
        else:
            hit.meta.setdefault("realisers", []).append(c.name)
            record["candidates"][c.name]["verdict"] = f"realiser of output {hit.name}"
    passed = keep
    nodes = merge_equivalent(passed, inputs, th, n_eff)
    for nd in nodes + inputs:
        nd.members.sort(key=lambda m: (probe.site_layer(m.site), probe.site_idx(m.site), getattr(m, "quality", 1.0), m.U.shape[1]))
    # every member's value map is refitted to the node's own partition (values of the
    # primary member; for inputs, the input values), so that all realisers of a node
    # read the same variable with the same coding
    for nd in nodes + inputs:
        if not nd.members:
            continue
        base = nd.values if nd.kind == "input" else nd.primary.values(probe.states[nd.primary.site])
        nd.values = base
        ids = np.unique(base)
        for m in nd.members:
            z = probe.states[m.site] @ m.U
            m.centroids = np.stack([z[base == v].mean(0) for v in ids])
            m.value_ids = ids
    record["merges"] = {}
    for i, nd in enumerate(nodes):
        record["merges"][f"C{i}@{nd.primary.site}"] = [nd.name] + nd.meta.get("merged", [])
        nd.meta["candidate"] = nd.name
        nd.name = f"C{i}@{nd.primary.site}"
    G = None
    it_total = 0
    suspended: List[Node] = []
    strict = False
    while True:
        it_total += 1
        if it_total > max_iter * 4:
            record["not_converged"] = True
            break
        G, cert, drop, react = _one_pass(probe, inputs, nodes, outs, th, rng, strict, suspended)
        record["iterations"].append({"n_nodes": len(nodes), "strict": strict, "dropped": dict(drop),
                                     "reactivated": list(react), "n_suspended": len(suspended)})
        if verbose:
            print(f"[discover] pass {it_total} strict={strict}: {len(nodes)} nodes, dropping {len(drop)}, "
                  f"reactivating {len(react)}, suspended {len(suspended)}", flush=True)
        if react:
            nodes = nodes + [n for n in suspended if n.name in react]
            suspended = [n for n in suspended if n.name not in react]
        if not drop and not react:
            if strict:
                break
            strict = True
            continue
        if drop:
            nodes, gone = _peel(probe, nodes, drop, record)
            # derivation failures with no later realiser are suspended, not deleted
            for n in gone:
                if not drop[n.name].startswith("causal"):
                    suspended.append(n)
        strict = False
    for n in suspended:
        record["dropped"].setdefault(f"{n.name}[{n.primary.site}]", "derivation fails (suspended at convergence)")
    if G is not None:
        # final names reflect the node's (possibly peeled) site
        G.rename({c: f"C{i}@{G.nodes[c].primary.site}" for i, c in enumerate(sorted(G.concepts(), key=lambda c: (G.nodes[c].site_index, c)))})
        nodes = [G.nodes[c] for c in G.concepts()]
    if stability and G is not None:
        st = parent_stability(probe, inputs + nodes, outs, th, rng)
        for k, (j, ss) in st.items():
            G.certificates.setdefault(k, {})["stability_jaccard"] = j
    if minimal_sets and G is not None:
        pool = inputs + nodes
        for C in nodes + outs:
            G.alternatives[C.name] = minimal_sufficient_sets(C, pool, th)
    record["runtime_s"] = time.time() - t0
    record["n_forward_rows"] = probe.n_runs
    return G, probe, record
