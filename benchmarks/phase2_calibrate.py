"""Phase 2: calibrate memorization_signal on the battery and report ROC.

Ground truth per model:
  memorizing   := label-noise >= 50 % (the model is fitting noise; we also
                  record whether it actually reproduced the flipped labels)
  generalizing := label-noise 0 % (subset and full+aug)
  graded       := 20-40 % noise, used only to check monotonicity.

Evaluation that matters: (1) leave-one-seed-out AUC within architecture,
(2) CROSS-ARCHITECTURE AUC — calibrate on ResNet, score GPT-2 and vice versa.
The shipped calibration is fitted on everything; the reported numbers are
held-out. Also reports the size-dependence shift of each feature (n=500 vs
m=250 subsamples) that the CI construction relies on.
"""
from __future__ import annotations
import csv, glob, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from whitebox.signals.memorization import memorization_features, Calibration, calib_path
from whitebox.core.bootstrap import subsample_indices
from common import RESULTS

rows = []
for p in sorted(glob.glob(f"{RESULTS}/battery_cifar/*.json")):
    tag = os.path.basename(p)[:-5]; gt = json.load(open(p)); z = np.load(p[:-5] + ".npz")
    ep = max(int(k.split("_")[0][2:]) for k in z.files); noise = 0 if "full" in tag else int(tag.split("_n")[1][:3])
    for layer in ("pool", "layer3"):
        rows.append(dict(arch="resnet18", tag=tag, layer=layer, noise=noise, seed=int(tag[-1]), acts=z[f"ep{ep}_{layer}"],
                         memorized=gt["memorized_flipped_frac"], gap=gt["generalization_gap"], test_acc=gt["test_acc"]))
for p in sorted(glob.glob(f"{RESULTS}/battery_gpt2/*.json")):
    tag = os.path.basename(p)[:-5]; gt = json.load(open(p)); z = np.load(p[:-5] + ".npz")
    for layer in ("L12", "L6"):
        rows.append(dict(arch="gpt2", tag=tag, layer=layer, noise=gt["noise"], seed=gt["seed"], acts=z[f"ep10_{layer}"],
                         memorized=gt["memorized_flipped_frac"], gap=gt["train_nll_noisy"] - gt["heldout_nll_clean"], test_acc=gt["heldout_ppl"]))
print(f"{len(rows)} (model, layer) rows", flush=True)
for r in rows:
    r["feat"] = memorization_features(r["acts"], distance_device="cuda")
    idx = subsample_indices(len(r["acts"]), 250, 10, 0)
    subs = [memorization_features(r["acts"][i], distance_device="cuda") for i in idx]
    r["feat_sub"] = {k: float(np.mean([s[k] for s in subs])) for k in subs[0]}
    r["y"] = 1 if r["noise"] >= 50 else (0 if r["noise"] == 0 else -1)
    print(r["tag"], r["layer"], {k: round(v, 3) for k, v in r["feat"].items() if k in ("tc_ratio", "pe1", "tc_norm", "b1", "tc")}, "y=", r["y"], flush=True)

with open(f"{RESULTS}/phase2_features.csv", "w", newline="") as f:
    keys = ["arch", "tag", "layer", "noise", "seed", "y", "memorized", "gap", "test_acc"] + list(rows[0]["feat"]) + [f"sub250_{k}" for k in rows[0]["feat"]]
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in rows:
        w.writerow({**{k: r[k] for k in keys[:9]}, **r["feat"], **{f"sub250_{k}": v for k, v in r["feat_sub"].items()}})

LOG = ["tc_ratio", "tc_norm", "tc0_norm", "b1", "n_bars"]
FEATSETS = {"tc_ratio": ["tc_ratio"], "pe1": ["pe1"], "tc_norm": ["tc_norm"], "b1": ["b1"], "tc0_norm": ["tc0_norm"],
            "tc_ratio+pe1": ["tc_ratio", "pe1"], "tc_ratio+pe1+tc_norm": ["tc_ratio", "pe1", "tc_norm"], "all5": ["tc_ratio", "pe1", "tc_norm", "b1", "tc0_norm"]}


def X_of(rs, feats):
    return np.array([[np.log(max(r["feat"][k], 1e-12)) if k in LOG else r["feat"][k] for k in feats] for r in rs])


def fit(rs, feats):
    X = X_of(rs, feats); y = np.array([r["y"] for r in rs]); mu, sd = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(C=1.0).fit((X - mu) / sd, y)
    return Calibration(feats, mu, sd, lr.coef_[0], float(lr.intercept_[0]), [k for k in feats if k in LOG])


def auc(cal, rs):
    y = np.array([r["y"] for r in rs]); s = np.array([cal.score(r["feat"]) for r in rs])
    return roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")


table = []
for layer_set, name in ((("pool", "L12"), "penultimate"), (("layer3", "L6"), "mid")):
    lab = [r for r in rows if r["layer"] in layer_set and r["y"] >= 0]
    res = [r for r in lab if r["arch"] == "resnet18"]; gpt = [r for r in lab if r["arch"] == "gpt2"]
    for fname, feats in FEATSETS.items():
        row = dict(layers=name, features=fname)
        row["auc_all_insample"] = auc(fit(lab, feats), lab)
        # leave-one-seed-out within each architecture
        loso = []
        for arch_rows in (res, gpt):
            for s in sorted({r["seed"] for r in arch_rows}):
                tr = [r for r in arch_rows if r["seed"] != s]; te = [r for r in arch_rows if r["seed"] == s]
                if len({r["y"] for r in tr}) == 2 and len({r["y"] for r in te}) == 2: loso.append(auc(fit(tr, feats), te))
        row["auc_leave_one_seed_out"] = float(np.mean(loso)) if loso else float("nan")
        row["auc_resnet_to_gpt2"] = auc(fit(res, feats), gpt) if res and gpt else float("nan")
        row["auc_gpt2_to_resnet"] = auc(fit(gpt, feats), res) if res and gpt else float("nan")
        # monotonicity with noise on graded models (Spearman of score vs noise, all models incl graded)
        from scipy.stats import spearmanr
        cal = fit(lab, feats); allr = [r for r in rows if r["layer"] in layer_set]
        row["spearman_score_vs_noise"] = spearmanr([r["noise"] for r in allr], [cal.score(r["feat"]) for r in allr])[0]
        table.append(row); print(row, flush=True)
with open(f"{RESULTS}/phase2_calibration_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)

# choose PER FAMILY: penultimate layer, best leave-one-seed-out AUC within the family (cross-family transfer failed, see table)
FAM = {"cnn_classifier": ("resnet18", "pool"), "lm_transformer": ("gpt2", "L12")}
chosen = {}
for fam, (arch, layer) in FAM.items():
    fr = [r for r in rows if r["arch"] == arch and r["layer"] == layer and r["y"] >= 0]
    best, bestv = None, -1
    for fname, feats in FEATSETS.items():
        loso = []
        for s_ in sorted({r["seed"] for r in fr}):
            tr = [r for r in fr if r["seed"] != s_]; te = [r for r in fr if r["seed"] == s_]
            if len({r["y"] for r in tr}) == 2 and len({r["y"] for r in te}) == 2: loso.append(auc(fit(tr, feats), te))
        v = float(np.mean(loso)) if loso else float("nan")
        if v > bestv: best, bestv = fname, v
    feats = FEATSETS[best]; cal = fit(fr, feats)
    other = [r for r in rows if r["layer"] in ("pool", "L12") and r["arch"] != arch and r["y"] >= 0]
    chosen[fam] = dict(features=best, auc_loso=bestv, auc_insample=auc(cal, fr), auc_on_other_family=auc(cal, other), n_models=len(fr))
    gsc = sorted(cal.score(r["feat"]) for r in fr if r["y"] == 0); msc = sorted(cal.score(r["feat"]) for r in fr if r["y"] == 1)
    grd = sorted(cal.score(r["feat"]) for r in rows if r["arch"] == arch and r["layer"] == layer and r["y"] == -1)
    thr = 0.5 * (gsc[-1] + msc[0]) if gsc[-1] < msc[0] else 0.5
    cal.meta = dict(family=fam, arch=arch, layer=layer, **chosen[fam], probe_n=500, pca=30, thresh_percentile=95, threshold=thr,
                    battery_scores=dict(generalizing=[round(x, 3) for x in gsc], memorizing=[round(x, 3) for x in msc], graded_noise=[round(x, 3) for x in grd]),
                    note="memorizing := label noise >= 50%; generalizing := 0%. Cross-family transfer FAILS (sign flip); never apply to another family.")
    cal.save(calib_path(fam)); print("saved", fam, chosen[fam], flush=True)
best = chosen; lab = [r for r in rows if r["layer"] in ("pool", "L12") and r["y"] >= 0]
# size shift of features
shift = {k: float(np.median([r["feat_sub"][k] / max(r["feat"][k], 1e-12) for r in rows])) for k in rows[0]["feat"]}
json.dump(dict(chosen=chosen, table=table, sub250_over_full_median_ratio=shift), open(f"{RESULTS}/phase2_report.json", "w"), indent=1)
print("sub250/full feature ratio:", shift)

# ROC figure: per family, shipped calibration in-sample + LOSO curve; plus score vs noise / vs memorisation
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
cals = {fam: Calibration.load(fam) for fam in FAM}
for (fam, (arch, layer)), c in zip(FAM.items(), ("C0", "C1")):
    fr = [r for r in rows if r["arch"] == arch and r["layer"] == layer and r["y"] >= 0]; y = [r["y"] for r in fr]
    s_in = [cals[fam].score(r["feat"]) for r in fr]; fpr, tpr, _ = roc_curve(y, s_in)
    ax[0].plot(fpr, tpr, c, label=f"{fam} ({arch}) in-sample AUC={roc_auc_score(y, s_in):.2f}")
    s_lo = np.zeros(len(fr)); feats = FEATSETS[chosen[fam]["features"]]
    for s_ in sorted({r["seed"] for r in fr}):
        tr = [r for r in fr if r["seed"] != s_]; cal_ = fit(tr, feats)
        for i, r in enumerate(fr):
            if r["seed"] == s_: s_lo[i] = cal_.score(r["feat"])
    fpr, tpr, _ = roc_curve(y, s_lo); ax[0].plot(fpr, tpr, c + "--", label=f"{fam} leave-one-seed-out AUC={roc_auc_score(y, s_lo):.2f}")
    other = [r for r in rows if r["layer"] in ("pool", "L12") and r["arch"] != arch and r["y"] >= 0]
    s_o = [cals[fam].score(r["feat"]) for r in other]; fpr, tpr, _ = roc_curve([r["y"] for r in other], s_o)
    ax[0].plot(fpr, tpr, c + ":", lw=1, label=f"{fam} calib. applied to OTHER family AUC={roc_auc_score([r['y'] for r in other], s_o):.2f}")
    allf = [r for r in rows if r["arch"] == arch and r["layer"] == layer]; mk = "o" if arch == "resnet18" else "s"
    ax[1].scatter([r["noise"] + np.random.uniform(-2, 2) for r in allf], [cals[fam].score(r["feat"]) for r in allf], marker=mk, label=fam, alpha=0.8)
    ax[2].scatter([r["memorized"] if r["memorized"] == r["memorized"] else 0 for r in allf], [cals[fam].score(r["feat"]) for r in allf], marker=mk, label=fam, alpha=0.8)
ax[0].plot([0, 1], [0, 1], "k:", lw=0.7); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate"); ax[0].legend(fontsize=6); ax[0].set_title("memorization_signal ROC, per-family calibration")
ax[1].axhline(0.5, color="k", ls=":", lw=0.7); ax[1].set_xlabel("label noise %"); ax[1].set_ylabel("calibrated score"); ax[1].set_title("score vs injected noise (penultimate layer)"); ax[1].legend()
ax[2].axhline(0.5, color="k", ls=":", lw=0.7); ax[2].set_xlabel("fraction of flipped labels reproduced"); ax[2].set_ylabel("calibrated score"); ax[2].set_title("score vs measured memorisation"); ax[2].legend()
fig.tight_layout(); fig.savefig(f"{RESULTS}/phase2_roc.png", dpi=300); print("figure saved")
