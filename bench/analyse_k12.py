"""Phase 2 analysis, exactly as pre-registered in prereg/PHASE_2.md (K1 §3, K2 §4).

  python -m bench.analyse_k12 RESULTS_DIR [--out reports/phase2_tables.md]
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from whitebox_ee.stats import auroc, holm, paired_auroc_diff_ci

BASELINES = ["ccgp", "neg_sharpness", "max_softmax"]
SECONDARY = ["depth_gamma", "neg_entropy", "margin"]

KNOWN = {  # known intermediate variables (not read directly by a behaviour, M7) and edges
    "add3": dict(nodes=["s0", "s1", "s2", "c0", "c1"],
                 edges=[("a0", "s0"), ("b0", "s0"), ("a1", "s1"), ("b1", "s1"), ("a2", "s2"), ("b2", "s2"),
                        ("s0", "c0"), ("s1", "c1"), ("c0", "c1"),
                        ("s0", "out:out0"), ("s1", "out:out1"), ("c0", "out:out1"), ("s2", "out:out2"), ("c1", "out:out2")],
                 operand=["a0", "a1", "a2", "b0", "b1", "b2"], shared=["s0", "s1", "s2", "c0", "c1"]),
    "add2": dict(nodes=["s0", "s1", "c0"],
                 edges=[("a0", "s0"), ("b0", "s0"), ("a1", "s1"), ("b1", "s1"), ("s0", "c0"),
                        ("s0", "out:out0"), ("s1", "out:out1"), ("c0", "out:out1")],
                 operand=["a0", "a1", "b0", "b1"], shared=["s0", "s1", "c0"]),
}


def k1_cell(cell: Path, task: str):
    k1 = json.loads((cell / "k1.json").read_text())
    meta = k1["meta"]
    kn = KNOWN.get(task)
    labels = {c: v["label"] for c, v in k1["nodes"].items()}
    found = {l for l in labels.values() if l}
    out = dict(cell=cell.name, train_acc=meta["train_accuracy"], gate=min(meta["train_accuracy"].values()) >= 0.6)
    if kn is None:
        return out
    out["K1a_node_recall"] = len(found & set(kn["nodes"])) / len(kn["nodes"])
    # graph edges relabelled; an edge P->C counts as present if P is an ancestor of C through
    # at most one unlabelled intermediate node
    par = {c: v["parents"] for c, v in k1["nodes"].items()}
    par.update(k1["outputs"])
    name = lambda n: labels.get(n) or n

    def present(p, c):
        cn = [n for n in par if name(n) == c or n == c]
        for n in cn:
            for q in par.get(n, []):
                if name(q) == p:
                    return True
                if labels.get(q) is None and q in par:
                    if any(name(r) == p for r in par[q]):
                        return True
        return False
    cand = [(p, c) for p, c in kn["edges"] if (p in found or p in kn["operand"]) and (c in found or c.startswith("out:"))]
    out["K1b_edge_recovery"] = (sum(present(p, c) for p, c in cand) / len(cand)) if cand else float("nan")
    g = {labels[c]: v["g"] for c, v in k1["nodes"].items() if labels.get(c)}
    gin = k1["input_generality"]
    sh = [g[k] for k in kn["shared"] if k in g]
    op = [gin[k] for k in kn["operand"] if k in gin]
    out["K1c_generality_gap"] = (np.mean(sh) - np.mean(op)) if sh and op else float("nan")
    import re
    wrong_labels = {m.group(1) for k in k1["verdicts"] for m in [re.search(r":probe:([wq]\d+)$", k)] if m}
    acc_wrong = {l for l in found if l in wrong_labels}
    out["K1d_wrong_accepted"] = len(acc_wrong) / len(wrong_labels) if wrong_labels else 0.0
    out["K1_pass"] = (out["K1a_node_recall"] >= 0.6 and out["K1b_edge_recovery"] >= 0.7
                      and out["K1c_generality_gap"] > 0 and out["K1d_wrong_accepted"] <= 0.2)
    return out


def k2_cell(cell: Path, n_boot=2000, seed=0):
    d = pickle.load(open(cell / "k2.pkl", "rb"))
    t = {k: np.asarray(v, float) for k, v in d["table"].items()}
    y = t["correct"].astype(int)
    res = dict(cell=cell.name, n=len(y), n_err=int((1 - y).sum()), n_ok=int(y.sum()),
               gate=min(d["meta"]["train_accuracy"].values()) >= 0.6 and (1 - y).sum() >= 30 and y.sum() >= 30)
    for s in ["depth_lambda"] + BASELINES + SECONDARY:
        v = t.get(s)
        if v is None:
            continue
        ok = ~np.isnan(v)
        res[f"auroc_{s}"] = auroc(v[ok], y[ok]) if ok.sum() > 10 else float("nan")
    for b in BASELINES + SECONDARY:
        v = t.get(b)
        if v is None:
            continue
        ok = ~np.isnan(v)
        if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
            res[f"d_{b}"] = (float("nan"),) * 3
            continue
        res[f"d_{b}"] = paired_auroc_diff_ci(t["depth_lambda"][ok], v[ok], y[ok], n_boot=n_boot, seed=seed)
    f = t.get("funnel")
    if f is not None:
        res["funnel_frac"] = float(f.mean())
        nf = f < 0.5
        res["auroc_lambda_nonfunnel"] = auroc(t["depth_lambda"][nf], y[nf]) if nf.sum() > 20 and len(np.unique(y[nf])) == 2 else float("nan")
    # disagreement analysis (Prop. P7)
    if "ccgp" in t and not np.isnan(t["ccgp"]).all():
        lam_hi = t["depth_lambda"] > np.nanmedian(t["depth_lambda"])
        cc_hi = t["ccgp"] > np.nanmedian(t["ccgp"])
        a = cc_hi & ~lam_hi
        b = ~cc_hi & lam_hi
        res["disagree_highCCGP_lowDepth"] = (float(a.mean()), float(1 - y[a].mean()) if a.any() else float("nan"))
        res["disagree_lowCCGP_highDepth"] = (float(b.mean()), float(1 - y[b].mean()) if b.any() else float("nan"))
    return res


def k2_decision(cells, n_boot=4000, seed=0):
    """Prereg §4 decision rule over gate-passing cells (unit = cell)."""
    cells = [c for c in cells if c["gate"]]
    out = dict(n_cells=len(cells))
    if len(cells) < 4:
        out["verdict"] = "NOT DECIDABLE (fewer than 4 cells pass the specimen gate)"
        return out
    rng = np.random.default_rng(seed)
    pvals, cis = [], {}
    for b in BASELINES:
        d = np.array([c[f"d_{b}"][0] for c in cells if not np.isnan(c[f"d_{b}"][0])])
        if len(d) == 0:
            pvals.append(1.0)
            cis[b] = (np.nan,) * 3
            continue
        bs = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)])
        p = float((1 + (bs <= 0).sum()) / (1 + n_boot))       # one-sided bootstrap p for mean > 0
        pvals.append(p)
        cis[b] = (float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), len(d))
    adj = holm(pvals)
    out["mean_delta"] = cis
    out["holm_p"] = dict(zip(BASELINES, map(float, adj)))
    ccgp_wins = sum(1 for c in cells if not np.isnan(c["d_ccgp"][1]) and c["d_ccgp"][1] > 0)
    out["ccgp_cell_wins"] = f"{ccgp_wins}/{len(cells)}"
    all_pos = all(cis[b][0] > 0 and adj[i] < 0.05 for i, b in enumerate(BASELINES))
    out["verdict"] = "PASS" if (all_pos and ccgp_wins >= len(cells) / 2) else "FAIL"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.results)
    k1s, k2s = [], []
    for cell in sorted(p for p in root.iterdir() if p.is_dir()):
        task = cell.name.split("_")[-2] if "_s" in cell.name else ""
        for t in KNOWN:
            if f"_{t}_s" in cell.name:
                task = t
        if (cell / "k1.json").exists():
            k1s.append(k1_cell(cell, task))
        if (cell / "k2.pkl").exists():
            k2s.append(k2_cell(cell))
    dec = k2_decision([c for c in k2s if "_untrained" not in c["cell"]])
    report = dict(K1=k1s, K2=k2s, K2_decision=dec)
    txt = json.dumps(report, indent=1, default=str)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)


if __name__ == "__main__":
    main()
