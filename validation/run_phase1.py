"""Phase 1 planted-answer validation harness (prereg/PHASE_1.md).

Usage: python -m validation.run_phase1 --seeds 100 101 102 103 104 --out runs/phase1/val
Each cell writes one JSON file (idempotent: finished cells are skipped).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from validation import planted as PL
from whitebox_ee.create import creation
from whitebox_ee.depth import depth_scores
from whitebox_ee.discover import discover
from whitebox_ee.filter import Thresholds
from whitebox_ee.formation import formation
from whitebox_ee.generality import compute_activity, generality
from whitebox_ee.graph import label_nodes
from whitebox_ee.moves import classify_moves
from whitebox_ee.probe import Member, Probe
from whitebox_ee.propose import propose_probes
from whitebox_ee.stats import auroc, bootstrap_ci

TH = Thresholds()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def wrong_hypotheses(vals, rng, n=5):
    names = sorted(vals)
    out = {}
    for j in range(n):
        a, b = rng.choice(names, 2, replace=False)
        m = int(max(2, min(vals[a].max() + 1, vals[b].max() + 1)))
        out[f"wrong{j}:{a}+{b}"] = (vals[a] + vals[b]) % m
    return out


def oracle_members(P: PL.Planted, probe: Probe, vals):
    mem = []
    for n, v in P.vars.items():
        s = P.site_of[n]
        U = P.U[n]
        z = probe.states[s] @ U
        ids = np.unique(vals[n])
        mem.append(Member(s, U, np.stack([z[vals[n] == i].mean(0) for i in ids]), ids, f"oracle:{n}"))
    return mem


def graph_summary(G, lab):
    nodes = {c: lab.get(c) for c in G.concepts()}
    edges = []
    for c, ps in G.parents.items():
        cl = lab.get(c, c) if G.nodes[c].kind == "concept" else c
        for p in ps:
            pl = lab.get(p, p) if G.nodes[p].kind == "concept" else p
            edges.append((pl, cl))
    return nodes, edges


def run_discovery(P, X, vals, labels, seed, system=None):
    sys_ = system or P.system
    probe = Probe(sys_, X, seed=seed)
    mem = propose_probes(probe, labels)
    G, probe, rec = discover(sys_, X, mem, th=TH, probe=probe, seed=seed, stability=True, minimal_sets=False)
    n_eff = len(np.unique(np.asarray(X), axis=0))
    lab = label_nodes(G, vals, n_eff)
    return G, probe, rec, lab, len(mem)


def truth_edges(P: PL.Planted):
    concepts = {n for n, v in P.vars.items() if v.concept and v.layer > 0}
    E = set()
    for n in concepts:
        for p in P.vars[n].parents:
            E.add((p, n))
    for b, v in P.behaviours.items():
        E.add((v, f"out:{b}"))
    return concepts, E


def prf(found, truth):
    found, truth = set(found), set(truth)
    tp = len(found & truth)
    p = tp / len(found) if found else (1.0 if not truth else 0.0)
    r = tp / len(truth) if truth else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ---------------------------------------------------------------------------

def cell_v1(seed, variant, N=2000):
    kw = {"onehot": {}, "noise05": {"noise": 0.5}, "noise10": {"noise": 1.0}, "fourier": {"enc_sum": "fourier"}}[variant]
    P = PL.v1_addition(3, seed=seed, **kw)
    rng = np.random.default_rng(seed)
    X = P.sample(N, rng)
    vals = P.values(X)
    wrong = wrong_hypotheses({k: v for k, v in vals.items() if k != "decoy"}, rng)
    labels = {**vals, **wrong}
    t0 = time.time()
    G, probe, rec, lab, n_cand = run_discovery(P, X, vals, labels, seed)
    rt = time.time() - t0
    concepts, E = truth_edges(P)
    nodes, edges = graph_summary(G, lab)
    found_nodes = [l for l in nodes.values() if l]
    unlabelled = [c for c, l in nodes.items() if l is None]
    prec_n = len(set(found_nodes) & concepts) / max(len(nodes), 1)
    rec_n = len(set(found_nodes) & concepts) / len(concepts)
    ep, er, ef = prf(edges, E)
    # decoy and wrong hypotheses verdicts
    decoy_v = [v["verdict"] for k, v in rec["candidates"].items() if v.get("source", "").endswith(":decoy")
               and v.get("site") == "h1"]
    wrong_accepted = sorted({l for l in found_nodes if l.startswith("wrong")})
    # wrong hypotheses as labels of accepted nodes (label_nodes only matches vals, so check VI to wrong labels)
    from whitebox_ee.info import normalised_vi
    n_eff = len(np.unique(X, axis=0))
    wrong_hits = []
    for c in G.concepts():
        for wn, wv in wrong.items():
            if normalised_vi(G.nodes[c].values, wv, n_eff) <= 0.1 and nodes[c] is None:
                wrong_hits.append(wn)
    # generality vs oracle
    A = compute_activity(probe, G, TH, rng, rows=np.arange(min(600, len(X))))
    g = generality(G, A)
    Go, probe_o, reco, labo, _ = (None, None, None, None, None)
    po = Probe(P.system, X, seed=seed)
    Go, po, reco = discover(P.system, X, oracle_members(P, po, vals), th=TH, probe=po, seed=seed, stability=False,
                            minimal_sets=False)
    labo = label_nodes(Go, vals, n_eff)
    Ao = compute_activity(po, Go, TH, rng, rows=np.arange(min(600, len(X))))
    go = generality(Go, Ao)
    gl = {lab[c]: g[c] for c in g if lab.get(c)}
    gol = {labo[c]: go[c] for c in go if labo.get(c)}
    common = sorted(set(gl) & set(gol))
    from scipy.stats import spearmanr
    rho = float(spearmanr([gl[k] for k in common], [gol[k] for k in common]).correlation) if len(common) > 2 else np.nan
    onodes, oedges = graph_summary(Go, labo)
    return dict(cell="V1", variant=variant, seed=seed, N=N, node_precision=prec_n, node_recall=rec_n,
                edge_precision=ep, edge_recall=er, edge_f1=ef, decoy_verdicts=decoy_v, wrong_hits=wrong_hits,
                n_wrong=len(wrong), unlabelled_nodes=unlabelled, generality=gl, generality_oracle=gol,
                spearman_vs_oracle=rho, oracle_nodes=sorted(l for l in onodes.values() if l),
                oracle_edge_f1_vs_spec=prf(oedges, E)[2], n_candidates=n_cand, runtime_s=rt,
                n_forward_rows=probe.n_runs, stability={c: G.certificates[c].get("stability_jaccard")
                                                         for c in G.concepts()},
                edges=sorted(edges), not_converged=rec.get("not_converged", False))


def cell_null(seed, kind, family="V1", N=2000):
    P = PL.v1_addition(3, seed=seed) if family == "V1" else PL.v4_scaffold(seed=seed)
    rng = np.random.default_rng(seed + 7)
    X = P.sample(N, rng)
    vals = P.values(X)
    labels = {**vals, **wrong_hypotheses(vals, rng)}
    system = P.system
    if kind == "untrained":
        system = P.system.untrained_copy(seed)
    elif kind == "shuffled":
        labels = {k: rng.permutation(v) for k, v in labels.items()}
    G, probe, rec, lab, n_cand = run_discovery(P, X, vals, labels, seed, system=system)
    n_acc = len(G.concepts())
    return dict(cell="NULL", kind=kind, family=family, seed=seed, n_candidates=n_cand, n_accepted=n_acc,
                fp_rate=n_acc / max(n_cand, 1), accepted_labels=[lab.get(c) for c in G.concepts()])


def cell_v2(seed, N=2000):
    P = PL.v2_memo(seed=seed)
    rng = np.random.default_rng(seed)
    X = P.sample(N, rng)
    vals = P.values(X)
    labels = {**vals, **wrong_hypotheses(vals, rng)}
    G, probe, rec, lab, _ = run_discovery(P, X, vals, labels, seed)
    rows = np.arange(800)
    A = compute_activity(probe, G, TH, rng, rows=rows)
    g = generality(G, A)
    sc = depth_scores(probe, G, A, g, TH)
    memo = P.meta["is_memo"](X[rows])
    lam, mu = sc["out"]["lambda"], sc["out"]["mu"]
    from whitebox_ee.stats import auroc_ci
    # unit of analysis for the CI: distinct input pairs
    grp = np.array([a * 100 + b for a, b in X[rows]])
    return dict(cell="V2", seed=seed, frac_memo=float(memo.mean()),
                auroc_neg_mu=auroc_ci(-mu, memo, groups=grp), auroc_neg_lambda=auroc_ci(-lam, memo, groups=grp),
                memorized_at_tau=float((mu[memo] < 0.25).mean()), grounded_rule=float((lam[~memo] >= 0.25).mean()),
                generality={lab.get(c) or c: g[c] for c in g})


def cell_v3(seed, compositional, N=1500, n_test=80):
    P = PL.v3_sideways(seed=seed, compositional=compositional)
    rng = np.random.default_rng(seed)
    X = P.sample(N, rng, train=True)
    vals = P.values(X)
    labels = {**vals, **wrong_hypotheses(vals, rng)}
    G, probe, rec, lab, _ = run_discovery(P, X, vals, labels, seed)
    from whitebox_ee.graph import align_values
    align_values(G, vals)
    q = P.meta["q"]
    ref = {c: (lambda a, b: (a + b) % q) for c, l in lab.items() if l == "C"}
    out = {}
    for nm, Xn in [("withheld", P.sample(n_test, rng, withheld_only=True)), ("seen", P.sample(n_test, rng, train=True))]:
        res = classify_moves(G, probe, Xn, TH, rng, reference=ref)
        mv = [r["move"] for r in res]
        pc = [r["per_behaviour"]["out"].get("predicted_correct") for r in res]
        correct = P.system.run(Xn)[0]["out"].argmax(-1) == (P.meta["a_of"](Xn) + P.meta["b_of"](Xn)) % q
        want = "sideways" if nm == "withheld" else "lower"
        out[nm] = dict(move_acc=float(np.mean([m == want for m in mv])),
                       pred_correct_acc=float(np.mean([p == c for p, c in zip(pc, correct)])),
                       moves=dict(zip(*np.unique(mv, return_counts=True))) if mv else {},
                       actual_correct=float(correct.mean()))
    out["moves"] = {k: {kk: int(vv) for kk, vv in v["moves"].items()} for k, v in out.items() if "moves" in v}
    for k in ("withheld", "seen"):
        out[k].pop("moves", None)
    return dict(cell="V3", seed=seed, compositional=compositional, node_labels=sorted(l for l in lab.values() if l), **out)


def cell_v4(seed, N=2000):
    P = PL.v4_scaffold(seed=seed)
    rng = np.random.default_rng(seed)
    X = P.sample(N, rng)
    vals = P.values(X)
    labels = {**vals, **wrong_hypotheses(vals, rng)}
    G, probe, rec, lab, _ = run_discovery(P, X, vals, labels, seed)
    A = compute_activity(probe, G, TH, rng, rows=np.arange(min(600, len(X))))
    g = generality(G, A)
    gl = {lab[c]: g[c] for c in g if lab.get(c)}
    scaf = P.meta["scaffold"]
    other = P.meta["content"] + P.meta["conjunctive"]
    s_vals = [gl[k] for k in scaf if k in gl]
    o_vals = [gl[k] for k in other if k in gl]
    au = auroc(np.array(s_vals + o_vals), np.array([1] * len(s_vals) + [0] * len(o_vals))) if s_vals and o_vals else np.nan
    return dict(cell="V4", seed=seed, generality=gl, auroc_scaffold_vs_content=au,
                scaffold_recall=len([k for k in scaf if k in gl]) / len(scaf),
                content_recall=len([k for k in other if k in gl]) / len(other),
                principles_025=sorted(k for k, v in gl.items() if v >= 0.25))


def cell_v5(seed, variant, N=1500):
    rng = np.random.default_rng(seed)
    if variant == "generic":
        before, after = PL.v5_creation(seed=seed, variant="valid")
        ctrl = after
    else:
        before, after = PL.v5_creation(seed=seed, variant=variant)
        ctrl = before
    X = before.sample(N, rng)
    va = after.values(X)
    labels = {**va, "AplusB": (va["A"] + va["B"]) % 5, **wrong_hypotheses(va, rng)}
    G0, p0, r0, l0, _ = run_discovery(before, X, va, labels, seed)
    G1, p1, r1, l1, _ = run_discovery(after, X, va, labels, seed)
    Gc, pc, rc, lc, _ = run_discovery(ctrl, X, va, labels, seed)
    n_eff = len(np.unique(X, axis=0))
    res = creation(G0, G1, r1, TH, n_eff, G_ctrl=Gc,
                   probe_values_G0={n: G0.nodes[n].values for n in G0.concepts()},
                   probe_values_ctrl={n: Gc.nodes[n].values for n in Gc.concepts()})
    valid = [(l1.get(c), sorted(l1.get(p, p) for p in G1.parents[c])) for c in res["valid"]]
    status = {l1.get(c) or c: s[0] for c, s in res["status"].items()}
    return dict(cell="V5", seed=seed, variant=variant, valid=valid, status=status,
                n_represented_not_used=len(res["represented_not_used_candidates"]))


def cell_formation(seed, N=1500):
    seq = PL.formation_sequence(seed=seed)
    per = []
    for i, P in enumerate(seq):
        rng = np.random.default_rng(seed)
        X = P.sample(N, rng)
        vals = P.values(X)
        G, probe, rec, lab, _ = run_discovery(P, X, vals, {**vals}, seed)
        A = compute_activity(probe, G, TH, rng, rows=np.arange(min(600, len(X))))
        g = generality(G, A)
        sc = depth_scores(probe, G, A, g, TH)
        per.append(dict(ckpt=i, G=G, g=g, scores=sc))
    n_eff = len(np.unique(X, axis=0))
    f = formation(per, 0.25, TH.eps_merge, n_eff)
    births = [b for b in f["births"] if b["label"] == "sum"]
    return dict(cell="FORMATION", seed=seed, events=f["events"],
                Lambda=[r["Lambda"]["out"] for r in f["per_checkpoint"]],
                sum_principle_from=births[0]["principle_from"] if births else None,
                sum_first_seen=births[0]["first_seen"] if births else None)


def cell_sensitivity(seed, N):
    r = cell_v1(seed, "onehot", N=N)
    r["cell"] = "SENS"
    return r


CELLS = {
    "V1": lambda s: [("V1", s, v) for v in ("onehot", "noise05", "noise10", "fourier")],
    "NULL": lambda s: [("NULL", s, k, f) for k in ("untrained", "shuffled") for f in ("V1", "V4")],
    "V2": lambda s: [("V2", s)],
    "V3": lambda s: [("V3", s, c) for c in (True, False)],
    "V4": lambda s: [("V4", s)],
    "V5": lambda s: [("V5", s, v) for v in ("valid", "decoy", "leaf", "relabel", "generic")],
    "FORMATION": lambda s: [("FORMATION", s)],
    "SENS": lambda s: [("SENS", s, n) for n in (250, 500, 1000)],
}


def run_cell(spec):
    kind = spec[0]
    if kind == "V1":
        return cell_v1(spec[1], spec[2])
    if kind == "NULL":
        return cell_null(spec[1], spec[2], spec[3])
    if kind == "V2":
        return cell_v2(spec[1])
    if kind == "V3":
        return cell_v3(spec[1], spec[2])
    if kind == "V4":
        return cell_v4(spec[1])
    if kind == "V5":
        return cell_v5(spec[1], spec[2])
    if kind == "FORMATION":
        return cell_formation(spec[1])
    if kind == "SENS":
        return cell_sensitivity(spec[1], spec[2])
    raise ValueError(kind)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    for s in a.seeds:
        for c in a.cells:
            for spec in CELLS[c](s):
                name = "_".join(str(x) for x in spec) + ".json"
                f = out / name
                if f.exists():
                    continue
                t0 = time.time()
                try:
                    r = run_cell(spec)
                    r["status"] = r.get("status", "ok") if spec[0] != "V5" else r["status"]
                except Exception as e:  # every failure is recorded
                    import traceback
                    r = {"cell": spec[0], "spec": list(map(str, spec)), "error": repr(e), "traceback": traceback.format_exc()}
                r.update(commit=commit, spec=[str(x) for x in spec], wall_s=time.time() - t0, node=platform.node(),
                         start=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)))
                f.write_text(json.dumps(_jsonable(r), indent=1, default=str))
                print(f"[phase1] {name} {time.time() - t0:.0f}s {'ERROR ' + r['error'] if 'error' in r else ''}", flush=True)


if __name__ == "__main__":
    main()
