"""PHASE 3 — strengthen H1 (n=10 seeds), H4 (full noise curve), H2 (100 epochs).

Reuses exp_vision.train_one_seed (ResNet-18 / CIFAR-10) writing into the CLONE results
dir. Computes the headline statistics:
  H1: cross-seed reproducibility of the SV(t) phase-transition (mean pairwise corr).
  H4: critical label-noise threshold where topological complexity (TC) collapses < 50%.
  H2: Spearman rho between concept-crystallization order and class distinctness.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import pearsonr, spearmanr

import exp_vision as V

OUT = "/home/god/whitebox_experiments_clone/results/whitebox_breakthrough"
LAYERS = ["layer1", "layer2", "layer3", "fc"]


def _load(p):
    return json.load(open(p))


def run_h1(tsdir, seeds=range(10), epochs=30):
    d = Path(tsdir) / "raw" / "vision_h1"; d.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in seeds:
        paths.append(V.train_one_seed(s, epochs=epochs, ckpt_every=2, layers=LAYERS,
                                      label_noise=0.0, tag="h1", out_dir=str(d)))
    # cross-seed reproducibility: mean pairwise Pearson of SV(t) per layer
    runs = [_load(p) for p in paths]
    rho_by_layer = {}
    for k in LAYERS:
        svs = [np.array(r["metrics"][k]["SV"]) for r in runs]
        L = min(len(s) for s in svs); svs = [s[:L] for s in svs]
        rs = [pearsonr(a, b)[0] for a, b in combinations(svs, 2)]
        rho_by_layer[k] = float(np.nanmean(rs))
    peak_epochs = {k: [int(np.argmax(r["metrics"][k]["SV"])) for r in runs] for k in LAYERS}
    res = {"n_seeds": len(runs), "epochs": epochs, "rho_by_layer": rho_by_layer,
           "mean_rho": float(np.nanmean(list(rho_by_layer.values()))),
           "peak_epoch_marks": {k: peak_epochs[k] for k in LAYERS},
           "epoch_marks": runs[0]["epoch_marks"]}
    json.dump(res, open(Path(tsdir)/"tables"/"H1_seed10.json", "w"), indent=2)
    print(f"[H1] mean cross-seed rho (n={len(runs)}) = {res['mean_rho']:.3f}  "
          f"per-layer={rho_by_layer}", flush=True)
    return res


def run_h4(tsdir, noises=range(0, 101, 10), epochs=12):
    d = Path(tsdir) / "raw" / "vision_h4"; d.mkdir(parents=True, exist_ok=True)
    rows = []
    for npct in noises:
        p = V.train_one_seed(0, epochs=epochs, ckpt_every=2, layers=LAYERS,
                             label_noise=npct/100.0, tag=f"h4n{npct}", out_dir=str(d))
        r = _load(p)
        tc_final = r["metrics"]["layer3"]["TC"][-1]
        sv_peak = float(np.max(r["metrics"]["layer3"]["SV"]))
        rows.append({"noise_pct": npct, "TC_final_layer3": tc_final, "SV_peak_layer3": sv_peak,
                     "train_acc": r["train_acc"][-1], "test_acc": r["test_acc"][-1]})
    tc0 = rows[0]["TC_final_layer3"]
    thr = 0.5 * tc0
    crit = None
    for r in rows:
        if r["TC_final_layer3"] < thr:
            crit = r["noise_pct"]; break
    res = {"epochs": epochs, "clean_TC": tc0, "threshold_50pct": thr,
           "critical_noise_pct": crit, "curve": rows}
    json.dump(res, open(Path(tsdir)/"tables"/"H4_noise_curve.json", "w"), indent=2)
    print(f"[H4] clean TC={tc0:.3f} 50%thr={thr:.3f} critical_noise={crit}%", flush=True)
    return res


def run_h2(tsdir, seeds=(0, 1, 2), epochs=100):
    d = Path(tsdir) / "raw" / "vision_h2"; d.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in seeds:
        paths.append(V.train_one_seed(s, epochs=epochs, ckpt_every=1, layers=LAYERS,
                                      label_noise=0.0, tag="h2", out_dir=str(d)))
    runs = [_load(p) for p in paths]
    marks = runs[0]["epoch_marks"]
    # crystallization epoch per class = epoch at which class-conditional SV last exceeds
    # 20% of its own max (i.e. settles); distinctness = final fc residual SV magnitude.
    cryst_rank, distinct_rank = {}, {}
    cryst_ep = {c: [] for c in range(10)}; distinct = {c: [] for c in range(10)}
    for r in runs:
        for c in range(10):
            sv = np.array(r["classcond_sv"][str(c)])
            mx = sv.max() if sv.size else 0.0
            tc_star = int(np.where(sv > 0.2*mx)[0].max()) if mx > 0 else 0
            cryst_ep[c].append(marks[min(tc_star, len(marks)-1)])
            distinct[c].append(float(sv[-1]) if sv.size else 0.0)
    cryst_mean = {c: float(np.mean(cryst_ep[c])) for c in range(10)}
    distinct_mean = {c: float(np.mean(distinct[c])) for c in range(10)}
    classes = list(range(10))
    cvals = [cryst_mean[c] for c in classes]; dvals = [distinct_mean[c] for c in classes]
    rho, p = spearmanr(cvals, dvals)
    res = {"n_seeds": len(runs), "epochs": epochs,
           "crystallization_epoch_mean": cryst_mean, "distinctness_mean": distinct_mean,
           "spearman_rho": float(rho), "spearman_p": float(p)}
    json.dump(res, open(Path(tsdir)/"tables"/"H2_crystallization_100ep.json", "w"), indent=2)
    print(f"[H2] Spearman rho(crystallization, distinctness) = {rho:.3f} (p={p:.4f})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsdir", required=True)
    ap.add_argument("--which", default="h4,h1,h2", help="comma list of h1,h2,h4")
    args = ap.parse_args()
    which = args.which.split(",")
    if "h4" in which: run_h4(args.tsdir)
    if "h1" in which: run_h1(args.tsdir)
    if "h2" in which: run_h2(args.tsdir)


if __name__ == "__main__":
    main()
