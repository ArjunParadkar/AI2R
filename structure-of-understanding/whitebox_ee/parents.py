"""Condition 2 (derivation): parent sets.

Phase 1 canonical rule (reports/phase1.md §2): the parent set of C is the set
of pool nodes P at earlier sites with a *direct* interventional effect on C
(P patched from a partner with a different value, every other pool node at a
site strictly between P and C held at its clean value), above threshold and
above random realisers of matched dimension; the set must then be
eps_D-sufficient for C (held-out conditional entropy) and stable across
resamples. Observational minimal sufficient sets (theory Def. 2.3) are also
enumerated and reported as the range of alternatives (Remark 2.7).
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .filter import Thresholds, random_member
from .info import entropy, heldout_cond_xent
from .probe import Node, Probe


def _read_targets(probe: Probe, targets: Sequence[Node], logits, states):
    return {t.name: probe.node_values_from_states(t, states, logits) for t in targets}


def direct_effects(probe: Probe, pool: Sequence[Node], outputs: Sequence[Node], th: Thresholds, rng,
                   rows=None, n_pairs=None, with_null=True, return_samples=False):
    """effects[(P, C)] = fraction of pairs where patching P (others between held) changes C."""
    n_pairs = n_pairs or th.n_pairs
    nodes = list(pool) + list(outputs)
    by_site: Dict[int, List[Node]] = {}
    for c in nodes:
        if c.kind == "input":
            continue
        by_site.setdefault(c.site_index, []).append(c)
    effects, nulls = {}, {}
    samples = {}                       # (P, C) -> (rows, src rows, C value under the P patch)
    rows_all = np.arange(probe.N) if rows is None else np.asarray(rows)
    for P in pool:
        r = rng.choice(rows_all, n_pairs)
        src = probe.partners(P.values, 1, differ=True, rows=r, rng=rng)[:, 0]
        ok = src >= 0
        r, src = r[ok], src[ok]
        for t, targets in sorted(by_site.items()):
            if t <= P.site_index:
                continue
            freeze = [Q for Q in pool if Q is not P and Q.kind != "input" and P.site_index < Q.site_index < t]
            need = sorted({C.primary.site for C in targets if C.kind != "output" and C.members})
            lg, st = probe.run_patched(r, [(P, src)], freeze=freeze, sites=need)
            vals = _read_targets(probe, targets, lg, st)
            for C in targets:
                effects[(P.name, C.name)] = float((vals[C.name] != C.values[r]).mean())
                samples[(P.name, C.name)] = (r, src, vals[C.name])
            if with_null:
                # null: within-fibre swap of P (same P value, same perturbation class)
                src_eq = probe.partners(P.values, 1, differ=False, rows=r, rng=rng)[:, 0]
                lg2, st2 = probe.run_patched(r, [(P, src_eq)], freeze=freeze, sites=need)
                v2 = _read_targets(probe, targets, lg2, st2)
                for C in targets:
                    nulls[(P.name, C.name)] = float((v2[C.name] != C.values[r]).mean())
    if return_samples:
        return effects, nulls, samples
    return effects, nulls


def interventional_tuples(C: Node, pa_nodes, samples):
    """Parent tuples and C values produced by the direct-effect patches of C's parents
    (M8, Prop. 2.8b): the patched parent takes its source value, the other parents keep
    their clean values (they are upstream, parallel, or held fixed)."""
    keys, vals = [], []
    for j, P in enumerate(pa_nodes):
        smp = samples.get((P.name, C.name))
        if smp is None:
            continue
        r, src, cv = smp
        T = np.stack([p.values[r] for p in pa_nodes], 1)
        T[:, j] = P.values[src]
        keys += [tuple(t) for t in T.tolist()]
        vals += list(np.asarray(cv).tolist())
    return keys, vals


def select_parents(C: Node, pool: Sequence[Node], effects, nulls, th: Thresholds):
    pa = []
    for P in pool:
        if P.site_index >= C.site_index:
            continue
        e = effects.get((P.name, C.name), 0.0)
        nl = nulls.get((P.name, C.name), 0.0)
        if e - nl >= th.theta_dir:
            pa.append(P.name)
    return pa


def sufficiency(C: Node, pa_nodes: Sequence[Node]) -> float:
    """H(C | Pa) / H(C), held out."""
    H = entropy(C.values)
    if H <= 1e-12:
        return 0.0
    return heldout_cond_xent(C.values, [p.values for p in pa_nodes]) / H


def minimal_sufficient_sets(C: Node, pool: Sequence[Node], th: Thresholds, max_size: int = 3,
                            max_pool: int = 14) -> List[Tuple[str, ...]]:
    """Observational inclusion-minimal eps_D-sufficient sets (theory Def. 2.3), by
    exhaustive search up to max_size over the max_pool candidates with highest MI."""
    from .info import mutual_info
    cands = [P for P in pool if P.site_index < C.site_index]
    cands = sorted(cands, key=lambda P: -mutual_info(P.values, C.values))[:max_pool]
    H = entropy(C.values)
    found = []
    for k in range(1, max_size + 1):
        for S in itertools.combinations(cands, k):
            names = tuple(sorted(p.name for p in S))
            if any(set(f) <= set(names) for f in found):
                continue
            r = heldout_cond_xent(C.values, [p.values for p in S]) / max(H, 1e-12)
            if r <= th.eps_D:
                # minimal: no proper subset 2eps-sufficient
                minimal = True
                for j in range(1, k):
                    for T in itertools.combinations(S, j):
                        if heldout_cond_xent(C.values, [p.values for p in T]) / max(H, 1e-12) <= 2 * th.eps_D:
                            minimal = False
                            break
                    if not minimal:
                        break
                if minimal:
                    found.append(names)
    return found


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def parent_stability(probe: Probe, pool, outputs, th: Thresholds, rng, n_resamples: int = 5):
    """Parent sets re-estimated on n_resamples random halves of the probe set."""
    sets = {c.name: [] for c in list(pool) + list(outputs) if c.kind != "input"}
    for k in range(n_resamples):
        half = rng.permutation(probe.N)[: probe.N // 2]
        eff, nul = direct_effects(probe, pool, outputs, th, rng, rows=half, n_pairs=th.n_pairs // 2)
        for c in list(pool) + list(outputs):
            if c.kind == "input":
                continue
            sets[c.name].append(select_parents(c, pool, eff, nul, th))
    stab = {}
    for name, ss in sets.items():
        js = [jaccard(a, b) for a, b in itertools.combinations(ss, 2)]
        stab[name] = (float(np.mean(js)) if js else 1.0, ss)
    return stab
