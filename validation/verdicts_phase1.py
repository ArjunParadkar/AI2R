"""Mechanical Phase 1 verdicts from the validation JSONs, by the letter of prereg/PHASE_1.md."""
import json
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/phase1/val")
SEEDS = [100, 101, 102, 103, 104]
R = {f.stem: json.loads(f.read_text()) for f in d.glob("*.json")}


def get(name):
    r = R.get(name)
    if r is None or "error" in r:
        return None
    return r


rows = []


def add(cell, crit, values, passed, note=""):
    rows.append((cell, crit, values, "PASS" if passed else "FAIL", note))


# P1a, P1b, P1c, P1e on gated variants
for var in ("onehot", "noise05"):
    v = [get(f"V1_{s}_{var}") for s in SEEDS]
    ok = all(x is not None for x in v)
    pr = [x["node_precision"] for x in v if x]
    rc = [x["node_recall"] for x in v if x]
    ef = [x["edge_f1"] for x in v if x]
    dc = [bool(x["decoy_verdicts"]) and all(t == "represented, not used" for t in x["decoy_verdicts"]) for x in v if x]
    mae = [x["generality_mae_vs_oracle"] for x in v if x]
    add(f"P1a V1-{var}", "node precision & recall >= 0.9 on 5/5", f"P {min(pr):.2f}-{max(pr):.2f}, R {min(rc):.2f}-{max(rc):.2f}",
        ok and min(pr) >= 0.9 and min(rc) >= 0.9)
    add(f"P1b V1-{var}", "edge F1 >= 0.9 on 5/5", f"{min(ef):.2f}-{max(ef):.2f}", ok and min(ef) >= 0.9)
    add(f"P1c V1-{var}", "decoy represented-not-used 5/5", f"{sum(dc)}/5", ok and sum(dc) == 5)
    add(f"P1e V1-{var}", "generality MAE vs oracle <= 0.05 on 5/5", f"{max(mae):.3f} (max)", ok and max(mae) <= 0.05)
# P1d pooled over gated variants
hits, tot = 0, 0
for var in ("onehot", "noise05"):
    for s in SEEDS:
        x = get(f"V1_{s}_{var}")
        if x:
            hits += len(set(x["wrong_hits"]))
            tot += x["n_wrong"]
add("P1d V1", "wrong hypotheses accepted <= 10% (pooled)", f"{hits}/{tot}", tot > 0 and hits / tot <= 0.10)
# P1f
fps, accs = [], []
for k in ("untrained", "shuffled"):
    for f in ("V1", "V4"):
        for s in SEEDS:
            x = get(f"NULL_{s}_{k}_{f}")
            if x:
                fps.append(x["fp_rate"])
                accs.append(x["n_accepted"])
add("P1f NULL", "FP <= 0.02 every seed; mean accepted <= 1", f"max FP {max(fps):.3f}, mean accepted {np.mean(accs):.2f} ({len(fps)} cells)",
    len(fps) == 20 and max(fps) <= 0.02 and np.mean(accs) <= 1)
# P2
v = [get(f"V2_{s}") for s in SEEDS]
pts = [x["auroc_neg_mu"][0] for x in v if x]
los = [x["auroc_neg_mu"][1] for x in v if x]
add("P2 V2", "AUROC(-mu) >= 0.9 on 5/5 and lower CI > 0.8 on 4/5", f"pt {min(pts):.3f}-{max(pts):.3f}, lo {min(los):.3f}-{max(los):.3f}",
    len(pts) == 5 and min(pts) >= 0.9 and sum(l > 0.8 for l in los) >= 4)
# P3
for comp in (True, False):
    v = [get(f"V3_{s}_{comp}") for s in SEEDS]
    m = [min(x["withheld"]["move_acc"], x["seen"]["move_acc"]) for x in v if x]
    p = [x["withheld"]["pred_correct_acc"] for x in v if x]
    add(f"P3a V3-{'comp' if comp else 'lookup'}", "move accuracy >= 0.9 (withheld and seen) 5/5", f"{min(m):.2f}", len(m) == 5 and min(m) >= 0.9)
    add(f"P3b V3-{'comp' if comp else 'lookup'}", "predicted-correctness accuracy >= 0.9 5/5", f"{min(p):.2f}", len(p) == 5 and min(p) >= 0.9)
# P4
v = [get(f"V4_{s}") for s in SEEDS]
au = [x["auroc_scaffold_vs_content"] for x in v if x]
rc = [x["scaffold_recall"] for x in v if x]
add("P4 V4", "AUROC >= 0.95 and scaffold recall >= 0.9, 5/5", f"AUROC {min(au):.3f}, recall {min(rc):.2f}",
    len(au) == 5 and min(au) >= 0.95 and min(rc) >= 0.9)
# P5
v = [get(f"V5_{s}_valid") for s in SEEDS]
ok = sum(1 for x in v if x and len(x["valid"]) == 1 and x["valid"][0][0] == "N" and sorted(x["valid"][0][1]) == ["A", "B"])
add("P5a V5-valid", "exactly one valid node N <- {A,B}, 4/5", f"{ok}/5", ok >= 4)
for var, want in (("decoy", None), ("leaf", "new leaf (used, not derived)"), ("relabel", None), ("generic", "exposure-generic")):
    v = [get(f"V5_{s}_{var}") for s in SEEDS]
    good = 0
    for x in v:
        if not x:
            continue
        if len(x["valid"]) != 0:
            continue
        if want and x["status"].get("N") != want:
            continue
        good += 1
    add(f"P5b V5-{var}", "zero valid nodes" + (f" and N '{want}'" if want else "") + ", 4/5", f"{good}/5", good >= 4)
# P6
v = [get(f"FORMATION_{s}") for s in SEEDS]
target = 1 - np.array([1.0, 1.0, 1.0, 0.8, 0.1, 0.1, 0.1])
a = sum(1 for x in v if x and abs(x["events"]["out"]["grounding_event"] - 4) <= 1
        and x["sum_principle_from"] is not None and abs(x["sum_principle_from"] - 3) <= 1)
b = sum(1 for x in v if x and np.max(np.abs(np.array(x["Lambda"]) - target)) <= 0.2)
devs = [float(np.max(np.abs(np.array(x["Lambda"]) - target))) for x in v if x]
add("P6a FORM", "grounding event 4+-1 and sum principle 3+-1, 4/5", f"{a}/5", a >= 4)
add("P6b FORM", "|Lambda_t - (1 - memo_t)| <= 0.2 every checkpoint, 4/5", f"{b}/5 (max deviations {', '.join(f'{x:.2f}' for x in devs)})", b >= 4)

print("| Cell | Criterion | Result | Verdict |")
print("|---|---|---|---|")
for c, k, val, p, n in rows:
    print(f"| {c} | {k} | {val} | {p} |")
