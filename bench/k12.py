"""K1 (ground-truth derivation) and K2 (depth predicts per-input error) for one
(model, task, seed) cell, as pre-registered in prereg/PHASE_2.md.

Resumable: each stage writes its artefact into the cell directory and is
skipped if present. Every artefact carries commit, config hash, model revision,
node, GPU and timestamps.

  python -m bench.k12 --model Qwen/Qwen2.5-7B --task add3 --seed 0 --out /scrfs/.../k12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from bench.tasks import AdditionTask, AgreementTask, MultiplicationTask
from whitebox_ee import baselines as BL
from whitebox_ee.depth import depth_scores, gamma_share, value_support, node_weights, bottleneck, memorization_level
from whitebox_ee.discover import discover
from whitebox_ee.filter import Thresholds
from whitebox_ee.generality import compute_activity, generality
from whitebox_ee.graph import label_nodes
from whitebox_ee.moves import extend
from whitebox_ee.probe import Probe
from whitebox_ee.propose import propose_probes

# ----------------------------------------------------------------------------- config

TASKS = {
    # name: (constructor, sampler kwargs for train/test, behaviours kept, known-algorithm nodes and edges)
    "add3": dict(make=lambda: AdditionTask(n_digits=3, n_shots=3), n_digits=3),
    "add2": dict(make=lambda: AdditionTask(n_digits=2, n_shots=3), n_digits=2),
    "mul21": dict(make=lambda: MultiplicationTask(n_digits_a=2, n_shots=3)),
    "agree": dict(make=lambda: AgreementTask(n_shots=0)),
}


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return os.environ.get("WB_COMMIT", "unknown")


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def sample_sets(task_name, task, n_train, n_test, seed):
    """Training-distribution probe set and a disjoint held-out set."""
    rng = np.random.default_rng(seed)
    if task_name.startswith("add"):
        nd = task.n
        # carry-out fixed to 1 (answers have n+1 digits, leading digit 1): the out{n}
        # behaviour is constant and dropped; all other carry patterns equally often
        pats = [tuple(list(p) + [1]) for p in np.ndindex(*([2] * (nd - 1)))]
        def draw(n, r):
            X = []
            for p in pats:
                X += task.sample(n // len(pats), r, carries=p)
            return X
        Xtr = draw(n_train, rng)
        seen = {(e["a"], e["b"]) for e in Xtr}
        Xte = [e for e in draw(int(n_test * 1.3), rng) if (e["a"], e["b"]) not in seen][:n_test]
        behaviours = [f"out{j}" for j in range(nd)]
    elif task_name == "mul21":
        Xtr = task.sample(n_train, rng)
        seen = {(e["a"], e["b"]) for e in Xtr}
        Xte = [e for e in task.sample(int(n_test * 2), rng) if (e["a"], e["b"]) not in seen][:n_test]
        behaviours = list(task.behaviours)
    else:
        Xtr = task.sample(n_train, rng)
        key = lambda e: tuple(sorted((k, v) for k, v in e.items() if isinstance(v, (int, np.integer))))
        seen = {key(e) for e in Xtr}
        Xte = [e for e in task.sample(int(n_test * 2), rng) if key(e) not in seen][:n_test]
        behaviours = list(task.behaviours)
    return Xtr, Xte, behaviours


def write(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".json":
        tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    else:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--layer_step", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--untrained", action="store_true", help="run on the untrained copy (null)")
    ap.add_argument("--sharpness_n", type=int, default=300)
    ap.add_argument("--sharpness_hutch", type=int, default=32)
    ap.add_argument("--activity_rows", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_random_realisers", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from whitebox_ee.llm import HFSystem

    cfg = vars(a).copy()
    cfg.pop("out")
    ch = config_hash(cfg)
    cell = Path(a.out) / f"{a.model.replace('/', '__')}{'@' + a.revision if a.revision else ''}_{a.task}_s{a.seed}" \
                         f"{'_untrained' if a.untrained else ''}{'_8bit' if a.load_in_8bit else ''}"
    cell.mkdir(parents=True, exist_ok=True)
    meta = dict(config=cfg, config_hash=ch, commit=git_commit(), node=platform.node(),
                gpu=torch.cuda.get_device_name(0) if (torch.cuda.is_available() and a.device.startswith("cuda")) else "cpu",
                start=time.strftime("%Y-%m-%dT%H:%M:%S"))
    log = open(cell / "log.txt", "a")

    def say(*x):
        msg = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(v) for v in x)
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    th = Thresholds(n_random_realisers=a.n_random_realisers)
    task = TASKS[a.task]["make"]()
    Xtr, Xte, behaviours = sample_sets(a.task, task, a.n_train, a.n_test, a.seed)
    say("cell", cell.name, "config", ch, "n_train", len(Xtr), "n_test", len(Xte))

    sysm = HFSystem(a.model, revision=a.revision, task=task, layers=[0], dtype=a.dtype, load_in_8bit=a.load_in_8bit,
                    device=a.device)
    L = sysm.n_layers
    # every layer_step-th layer in the first two thirds, every layer in the last third (the
    # development model computed addition in a narrow late band; prereg §2)
    cut = int(np.ceil(2 * L / 3))
    layers = sorted(set(list(range(0, cut, a.layer_step)) + list(range(cut, L + 1))))
    sysm = HFSystem(a.model, revision=a.revision, task=task, layers=layers, dtype=a.dtype,
                    load_in_8bit=a.load_in_8bit, _model=sysm.model, _tokenizer=sysm.tok, device=a.device)
    if a.untrained:
        sysm = sysm.untrained_copy(a.seed)
    sysm.behaviours = behaviours
    meta["model_meta"] = sysm.meta

    # ------------------------------------------------------------------ stage 1: probe + specimen gate
    probe = Probe(sysm, Xtr, seed=a.seed)
    tg = task.targets(Xtr)
    acc = {b: float((probe.read[b] == tg[b]).mean()) for b in behaviours}
    full = float(np.mean(np.all([probe.read[b] == tg[b] for b in behaviours], 0)))
    meta["train_accuracy"] = acc
    meta["train_full_accuracy"] = full
    say("train accuracy", acc, "full", full)

    # ------------------------------------------------------------------ stage 2: discovery (K1)
    gpath = cell / "graph.pkl"
    labels = {k: v for k, v in task.labels(Xtr).items() if len(np.unique(v)) > 1}
    if gpath.exists():
        G, rec = pickle.load(open(gpath, "rb"))
        say("graph loaded")
    else:
        mem = propose_probes(probe, labels)
        say(len(mem), "candidates")
        G, probe, rec = discover(sysm, Xtr, mem, th=th, probe=probe, seed=a.seed, verbose=True,
                                 stability=True, minimal_sets=False)
        for c in G.nodes.values():
            for m in c.members:
                m.U = m.U.astype(np.float32)
        write(gpath, (G, rec))
    lab = label_nodes(G, labels, len(Xtr))
    say("nodes", {c: lab[c] for c in G.concepts()})
    say("parents", {c: G.parents[c] for c in G.nodes if G.nodes[c].kind != "input"})

    rng = np.random.default_rng(a.seed + 1)
    act_rows = np.arange(min(a.activity_rows, len(Xtr)))
    A = compute_activity(probe, G, th, rng, rows=act_rows)
    g = generality(G, A)
    gin = {c: float(np.mean([A.on_path[c][b].mean() for b in behaviours])) for c in G.nodes
           if G.nodes[c].kind == "input"}
    k1 = dict(nodes={c: dict(label=lab[c], label_vi=G.nodes[c].meta.get("label_vi"), site=G.nodes[c].primary.site,
                             parents=G.parents[c], g=g[c], cert=G.certificates.get(c)) for c in G.concepts()},
              outputs={o: G.parents[o] for o in G.outputs()}, input_generality=gin,
              dropped=rec["dropped"], verdicts={k: v["verdict"] for k, v in rec["candidates"].items()},
              n_candidates=len(rec["candidates"]), runtime_s=rec["runtime_s"], forward_rows=probe.n_runs)
    write(cell / "k1.json", dict(meta=meta, **k1))
    say("K1 written; generality", {lab[c] or c: round(g[c], 3) for c in g})

    if a.untrained:
        say("untrained null: done")
        return

    # ------------------------------------------------------------------ stage 3: K2 on held-out inputs
    G2, P2, tr, te = extend(G, probe, Xte)
    rng = np.random.default_rng(a.seed + 2)
    A2 = compute_activity(P2, G2, th, rng, rows=te, pool=tr)
    sup = value_support(G2, P2, tr)
    w = node_weights(G2, A2, g, sup, th.n_min)
    lam = {b: bottleneck(G2, A2, w, b) for b in behaviours}
    mu = {b: memorization_level(G2, A2, w, b) for b in behaviours}
    tgt = task.targets(Xte)
    correct_b = {b: (P2.read[b][te] == tgt[b]) for b in behaviours}
    correct = np.all([correct_b[b] for b in behaviours], 0)
    prin = [c for c in G.concepts() if g[c] >= 0.25]
    gam = gamma_share(P2, G2, prin, te, th, rng, m=4, pool=tr) if prin else {b: np.zeros(len(te)) for b in behaviours}
    logits_te = {b: P2.logits[b][te] for b in behaviours}
    table = dict(correct=correct, **{f"correct_{b}": correct_b[b] for b in behaviours},
                 depth_lambda=np.min([lam[b] for b in behaviours], 0),
                 depth_mu=np.min([mu[b] for b in behaviours], 0),
                 depth_gamma=np.min([gam[b] for b in behaviours], 0),
                 max_softmax=BL.max_softmax(logits_te, reduce="min"),
                 neg_entropy=BL.neg_entropy(logits_te, reduce="min"),
                 margin=BL.margin(logits_te, reduce="min"))
    for b in behaviours:
        table[f"lambda_{b}"] = lam[b]
    # answer-funnel diagnostic (prereg §4.3): is the bottleneck node of the minimising
    # behaviour an answer-like direct parent of that behaviour's output?
    from whitebox_ee.info import cond_entropy_plugin, entropy as _H
    answer_like = {}
    for b in behaviours:
        ob = [o for o in G2.outputs() if G2.nodes[o].meta["behaviour"] == b][0]
        yv = G2.nodes[ob].values[tr]
        for c in G2.parents.get(ob, []):
            if G2.nodes[c].kind == "concept":
                answer_like[(b, c)] = cond_entropy_plugin(yv, [G2.nodes[c].values[tr]]) <= 0.1 * max(_H(yv), 1e-9)
    lam_stack = np.stack([lam[b] for b in behaviours], 0)
    bmin = lam_stack.argmin(0)
    funnel = np.zeros(len(te), bool)
    for i in range(len(te)):
        b = behaviours[bmin[i]]
        for c in G2.concepts():
            if A2.on_path[c][b][i] and abs(w[c][i] - lam[b][i]) < 1e-12 and lam[b][i] > 0:
                funnel[i] = bool(answer_like.get((b, c), False))
                break
    table["funnel"] = funnel
    # CCGP per input (prereg §4): known-algorithm variables; conditions per variable are the
    # task-variable combinations of its own column, (a_i, b_i, c_{i-1}) for arithmetic and
    # (subj_num, attr_num) for agreement; states reduced to the top 64 PCs (training rows);
    # site search over every 4th exposed layer.
    lab_all = {k: v for k, v in task.labels(list(Xtr) + list(Xte)).items() if len(np.unique(v)) > 1}
    allX = list(Xtr) + list(Xte)
    iv = task.input_values(allX)

    def joint(cols):
        return np.unique(np.stack(cols, 1), axis=0, return_inverse=True)[1].reshape(-1)

    if a.task.startswith("add") or a.task == "mul21":
        known = [k for k in lab_all if k[0] in "scrpk" and k[1:].isdigit()]
        def cond_for(k):
            i = int(k[1:])
            cols = [iv[n] for n in (f"a{i}", f"b{i}", "b") if n in iv]
            if i > 0 and f"c{i-1}" in lab_all:
                cols.append(lab_all[f"c{i-1}"])
            return joint(cols)
    else:
        known = [k for k in ("subj_num", "attr_num", "match") if k in lab_all]
        def cond_for(k):
            return joint([lab_all["subj_num"], lab_all["attr_num"]])
    ex_layers = sorted({P2.sys.site_layer(s_) for s_ in P2.sites})
    ccgp_layers = set(ex_layers[1::2])            # every 4th layer (exposed layers are every 2nd)
    ccgp_sites = [s_ for s_ in P2.sites if P2.sys.site_layer(s_) in ccgp_layers]
    red = {}
    for s_ in ccgp_sites:
        Hs = P2.states[s_]
        mu = Hs[tr].mean(0)
        _, _, Vt = np.linalg.svd(Hs[tr] - mu, full_matrices=False)
        red[s_] = (Hs - mu) @ Vt[:64].T
    per_var = []
    for k in known:
        try:
            per_var.append(BL.ccgp_per_input(red, {k: lab_all[k]}, cond_for(k), tr, te, seed=a.seed,
                                             site_select_splits=3))
        except Exception as e:
            say("ccgp failed for", k, repr(e))
    table["ccgp"] = np.nanmean(np.stack(per_var, 0), 0) if per_var else np.full(len(te), np.nan)
    write(cell / "k2_partial.pkl", table)
    # sharpness (subset; expensive)
    try:
        sub = np.arange(min(a.sharpness_n, len(te)))
        sh = BL.sharpness_per_input(sysm.model, sysm.tok, task, [Xte[i] for i in sub], k_blocks=2,
                                    n_hutch=a.sharpness_hutch, seed=a.seed)
        s_full = np.full(len(te), np.nan)
        s_full[sub] = -np.asarray(sh)
        table["neg_sharpness"] = s_full
    except Exception as e:
        say("sharpness failed", repr(e))
        table["neg_sharpness"] = np.full(len(te), np.nan)
    meta["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["test_accuracy"] = {b: float(correct_b[b].mean()) for b in behaviours}
    meta["test_full_accuracy"] = float(correct.mean())
    meta["principles_025"] = [lab.get(c) or c for c in prin]
    write(cell / "k2.pkl", dict(meta=meta, table=table))
    say("K2 written; test full accuracy", meta["test_full_accuracy"])


if __name__ == "__main__":
    main()
