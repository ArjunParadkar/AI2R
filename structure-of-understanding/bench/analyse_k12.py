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
    out["K1_stats_pass"] = bool(out["K1a_node_recall"] >= 0.6 and out["K1b_edge_recovery"] >= 0.7
                                and out["K1c_generality_gap"] > 0 and out["K1d_wrong_accepted"] <= 0.2)
    out["n_candidates"] = k1.get("n_candidates", 0)
    out["n_accepted"] = len(k1["nodes"])
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


DEV_MODELS = ("pythia-70m", "pythia-160m", "Qwen2.5-0.5B", "Qwen2.5-1.5B")   # prereg §1: never confirmatory
FAMILY = (("Qwen", "Qwen2.5"), ("OLMo", "OLMo-2"), ("pythia", "Pythia"), ("Mistral", "Mistral"))
# K1 untrained-copy null per model (prereg §3; amendment: a model > 7B uses its family's <= 7B null)
NULL_OF = {"Qwen__Qwen2.5-14B": "Qwen__Qwen2.5-7B", "Qwen__Qwen2.5-32B": "Qwen__Qwen2.5-7B",
           "allenai__OLMo-2-1124-13B": "allenai__OLMo-2-1124-7B", "EleutherAI__pythia-12b": "EleutherAI__pythia-6.9b"}


def parse_cell(name):
    """model, task, seed, untrained, 8bit from a k12 cell directory name."""
    import re
    m = re.match(r"(.+?)_(add3|add2|mul21|agree)_s(\d+)(_untrained)?(_8bit)?$", name)
    if not m:
        return None
    return dict(model=m.group(1), task=m.group(2), seed=int(m.group(3)), untrained=bool(m.group(4)),
                q8=bool(m.group(5)))


def family(model):
    return next((f for k, f in FAMILY if k in model), model)


QUANT_CHECK_CELL = "Qwen__Qwen2.5-7B_add3_s0_8bit"     # prereg §4.4; a check, not a unit
EIGHT_BIT_MODELS = ("Qwen__Qwen2.5-32B",)                # run only in 8-bit (prereg §1), units only if §4.4 passes


def is_unit(name, quant_ok=True):
    """A confirmatory model x task cell: seed 0, the pre-registered precision, trained, not a development model."""
    c = parse_cell(name)
    if not c or c["seed"] != 0 or c["untrained"] or any(d in c["model"] for d in DEV_MODELS) or name == QUANT_CHECK_CELL:
        return False
    if c["model"] in EIGHT_BIT_MODELS:
        return c["q8"] and quant_ok
    return not c["q8"]


def k1_decision(k1s, quant_ok=True):
    by = {c["cell"]: c for c in k1s}
    rows = []
    for c in k1s:
        if not is_unit(c["cell"], quant_ok) or parse_cell(c["cell"])["task"] != "add3":
            continue
        model = parse_cell(c["cell"])["model"]
        nm = NULL_OF.get(model, model) + "_add3_s0_untrained"
        nl = by.get(nm)
        null_fp = (nl["n_accepted"] / max(nl["n_candidates"], 1)) if nl else float("nan")
        ok = bool(c.get("gate") and c.get("K1_stats_pass") and nl is not None and null_fp <= 0.02)
        rows.append(dict(cell=c["cell"], family=family(model), gate=c.get("gate"), stats_pass=c.get("K1_stats_pass"),
                         null_cell=nm if nl else None, null_fp=null_fp, K1_pass=ok))
    fams = sorted({r["family"] for r in rows})
    anchor = {f: any(r["K1_pass"] for r in rows if r["family"] == f) for f in fams}
    return dict(cells=rows, family_pass=anchor, verdict="PASS" if any(anchor.values()) else "FAIL")


def quant_check(root):
    """prereg §4.4: Qwen2.5-7B add3 s0 bf16 vs 8-bit; node-label Jaccard >= 0.8 and Spearman(lambda) >= 0.8."""
    from scipy.stats import spearmanr
    a, b = root / "Qwen__Qwen2.5-7B_add3_s0", root / "Qwen__Qwen2.5-7B_add3_s0_8bit"
    if not ((a / "k2.pkl").exists() and (b / "k2.pkl").exists()):
        return dict(verdict="NOT AVAILABLE")
    la = {v["label"] for v in json.loads((a / "k1.json").read_text())["nodes"].values() if v["label"]}
    lb = {v["label"] for v in json.loads((b / "k1.json").read_text())["nodes"].values() if v["label"]}
    jac = len(la & lb) / max(len(la | lb), 1)
    xa = pickle.load(open(a / "k2.pkl", "rb"))["table"]["depth_lambda"]
    xb = pickle.load(open(b / "k2.pkl", "rb"))["table"]["depth_lambda"]
    rho = float(spearmanr(xa, xb).correlation)
    return dict(jaccard=jac, spearman=rho, verdict="PASS" if jac >= 0.8 and rho >= 0.8 else "FAIL")


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
    qc = quant_check(root)
    qok = qc["verdict"] == "PASS"
    dec = k2_decision([c for c in k2s if is_unit(c["cell"], qok)])
    dec["units"] = [c["cell"] for c in k2s if is_unit(c["cell"], qok) and c["gate"]]
    dec["families"] = {c: family(parse_cell(c)["model"]) for c in dec["units"]}
    report = dict(K1=k1s, K1_decision=k1_decision(k1s, qok), K2=k2s, K2_decision=dec, quantization_check=qc,
                  not_units=[c["cell"] for c in k2s if not is_unit(c["cell"], qok)])
    txt = json.dumps(report, indent=1, default=str)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)


if __name__ == "__main__":
    main()
