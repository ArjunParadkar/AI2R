"""Phase 6 (v2): does per-unit standardisation — the E2 gauge repair — cost the v1
memorization_signal its discriminative power?

Re-runs the Phase 2 feature computation on the CACHED battery activations
(results/battery_cifar/*.npz: 21 ResNet-18 / CIFAR-10 models, pool + layer3 at
the final epoch; results/battery_gpt2/*.npz: 10 GPT-2 124M fine-tunes, L12 + L6
at epoch 10) twice: raw (v1, native unit coordinates) and standardize=True
(each unit z-scored over the 500-point probe, dead units dropped, then
PCA -> 30 -> VR persistence as before). Same labels as Phase 2 (memorising :=
noise >= 50 %, generalising := 0 %), same leave-one-seed-out protocol, every
Phase 2 feature set. Adds a bootstrap-over-models 95 % CI on the LOSO AUC
(2000 resamples of the held-out (score, label) pairs) and a paired bootstrap
on the raw - standardised AUC difference.

Nothing is retrained. What cannot be tested here: other layers than the cached
ones, other probe sizes, and — for GPT-2 — whether per-unit standardisation
is even the right gauge repair (a residual stream has no per-unit scaling
symmetry; LayerNorm's group is different). Runs on CPU in a few minutes.
"""
from __future__ import annotations
import csv, glob, json, os, sys, time
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
from whiteboxtopo.signals.memorization import memorization_features, Calibration, calib_path, FAMILIES

OUT = os.path.join(RESULTS, "phase6"); os.makedirs(OUT, exist_ok=True)
LOG = ["tc_ratio", "tc_norm", "tc0_norm", "b1", "n_bars"]
FEATSETS = {"tc_ratio": ["tc_ratio"], "pe1": ["pe1"], "tc_norm": ["tc_norm"], "b1": ["b1"], "tc0_norm": ["tc0_norm"],
            "tc_ratio+pe1": ["tc_ratio", "pe1"], "all5": ["tc_ratio", "pe1", "tc_norm", "b1", "tc0_norm"]}
SHIPPED = {"cnn_classifier": ("resnet18", "pool", "tc_ratio"), "lm_transformer": ("gpt2", "L12", "pe1")}
N_BOOT = 2000

rows = []
for p in sorted(glob.glob(f"{RESULTS}/battery_cifar/*.json")):
    tag = os.path.basename(p)[:-5]; gt = json.load(open(p)); z = np.load(p[:-5] + ".npz")
    ep = max(int(k.split("_")[0][2:]) for k in z.files); noise = 0 if "full" in tag else int(tag.split("_n")[1][:3])
    for layer in ("pool", "layer3"):
        rows.append(dict(arch="resnet18", tag=tag, layer=layer, noise=noise, seed=int(tag[-1]), acts=z[f"ep{ep}_{layer}"]))
for p in sorted(glob.glob(f"{RESULTS}/battery_gpt2/*.json")):
    tag = os.path.basename(p)[:-5]; gt = json.load(open(p)); z = np.load(p[:-5] + ".npz")
    for layer in ("L12", "L6"):
        rows.append(dict(arch="gpt2", tag=tag, layer=layer, noise=gt["noise"], seed=gt["seed"], acts=z[f"ep10_{layer}"]))
for r in rows:
    r["y"] = 1 if r["noise"] >= 50 else (0 if r["noise"] == 0 else -1)
print(f"{len(rows)} (model, layer) rows from cached activations", flush=True)
t0 = time.time()
for i, r in enumerate(rows):
    r["raw"] = memorization_features(r["acts"], standardize=False)
    r["std"] = memorization_features(r["acts"], standardize=True)
    print(f"[{i+1}/{len(rows)} {time.time()-t0:.0f}s] {r['tag']} {r['layer']} y={r['y']}  raw tc_ratio={r['raw']['tc_ratio']:.3f} pe1={r['raw']['pe1']:.3f} | "
          f"std tc_ratio={r['std']['tc_ratio']:.3f} pe1={r['std']['pe1']:.3f}", flush=True)

with open(f"{OUT}/phase6_standardize_features.csv", "w", newline="") as f:
    keys = ["arch", "tag", "layer", "noise", "seed", "y"] + [f"raw_{k}" for k in rows[0]["raw"]] + [f"std_{k}" for k in rows[0]["std"]]
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in rows:
        w.writerow({**{k: r[k] for k in keys[:6]}, **{f"raw_{k}": v for k, v in r["raw"].items()}, **{f"std_{k}": v for k, v in r["std"].items()}})


def X_of(rs, feats, var):
    return np.array([[np.log(max(r[var][k], 1e-12)) if k in LOG else r[var][k] for k in feats] for r in rs])


def fit(rs, feats, var):
    X = X_of(rs, feats, var); y = np.array([r["y"] for r in rs]); mu, sd = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(C=1.0).fit((X - mu) / sd, y)
    return Calibration(feats, mu, sd, lr.coef_[0], float(lr.intercept_[0]), [k for k in feats if k in LOG])


def loso_scores(rs, feats, var):
    """Held-out score for every labelled model under leave-one-seed-out."""
    s = np.full(len(rs), np.nan)
    for sd_ in sorted({r["seed"] for r in rs}):
        tr = [r for r in rs if r["seed"] != sd_]
        if len({r["y"] for r in tr}) < 2: continue
        cal = fit(tr, feats, var)
        for i, r in enumerate(rs):
            if r["seed"] == sd_: s[i] = cal.score(r[var])
    return s


def boot_auc(y, s, s2=None, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(s); a = []; d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2: continue
        a.append(roc_auc_score(y[i], s[i]))
        if s2 is not None: d.append(roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], s2[i]))
    lo, hi = np.percentile(a, [2.5, 97.5])
    out = dict(ci=(float(lo), float(hi)))
    if s2 is not None: out["diff_ci"] = tuple(float(v) for v in np.percentile(d, [2.5, 97.5]))
    return out


table = []
for fam, (arch, layer, shipped_feat) in SHIPPED.items():
    for lay in ([layer] + (["layer3"] if arch == "resnet18" else ["L6"])):
        rs = [r for r in rows if r["arch"] == arch and r["layer"] == lay and r["y"] >= 0]
        allr = [r for r in rows if r["arch"] == arch and r["layer"] == lay]
        y = np.array([r["y"] for r in rs])
        for fname, feats in FEATSETS.items():
            s_raw, s_std = loso_scores(rs, feats, "raw"), loso_scores(rs, feats, "std")
            ok = ~np.isnan(s_raw) & ~np.isnan(s_std)
            row = dict(family=fam, arch=arch, layer=lay, features=fname, shipped=(lay == layer and fname == shipped_feat), n=int(ok.sum()))
            row["auc_loso_raw"] = roc_auc_score(y[ok], s_raw[ok]); row["auc_loso_std"] = roc_auc_score(y[ok], s_std[ok])
            b = boot_auc(y[ok], s_raw[ok], s_std[ok]); row["auc_loso_raw_lo"], row["auc_loso_raw_hi"] = b["ci"]; row["diff_raw_minus_std_lo"], row["diff_raw_minus_std_hi"] = b["diff_ci"]
            b2 = boot_auc(y[ok], s_std[ok]); row["auc_loso_std_lo"], row["auc_loso_std_hi"] = b2["ci"]
            row["auc_insample_raw"] = roc_auc_score(y, [fit(rs, feats, "raw").score(r["raw"]) for r in rs])
            row["auc_insample_std"] = roc_auc_score(y, [fit(rs, feats, "std").score(r["std"]) for r in rs])
            cr, cs = fit(rs, feats, "raw"), fit(rs, feats, "std")
            row["spearman_noise_raw"] = spearmanr([r["noise"] for r in allr], [cr.score(r["raw"]) for r in allr])[0]
            row["spearman_noise_std"] = spearmanr([r["noise"] for r in allr], [cs.score(r["std"]) for r in allr])[0]
            table.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
with open(f"{OUT}/phase6_standardize_auc.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)

# decision per family on the SHIPPED feature at the shipped layer
decision = {}
for fam, (arch, layer, feat) in SHIPPED.items():
    row = next(r for r in table if r["shipped"])if False else next(r for r in table if r["family"] == fam and r["layer"] == layer and r["features"] == feat)
    best_std = max((r for r in table if r["family"] == fam and r["layer"] == layer), key=lambda r: r["auc_loso_std"])
    keep = row["auc_loso_std"] >= row["auc_loso_raw"] - 1e-9
    decision[fam] = dict(shipped_feature=feat, auc_loso_raw=row["auc_loso_raw"], auc_loso_std=row["auc_loso_std"],
                         raw_ci=(row["auc_loso_raw_lo"], row["auc_loso_raw_hi"]), std_ci=(row["auc_loso_std_lo"], row["auc_loso_std_hi"]),
                         diff_ci=(row["diff_raw_minus_std_lo"], row["diff_raw_minus_std_hi"]),
                         best_std_featureset=best_std["features"], best_std_auc=best_std["auc_loso_std"],
                         standardized_keeps_auc=bool(keep), flip_default=bool(keep))
    if keep:   # ship a standardised calibration for this family (same recipe as phase2_calibrate)
        rs = [r for r in rows if r["arch"] == arch and r["layer"] == layer and r["y"] >= 0]
        cal = fit(rs, FEATSETS[feat], "std")
        gsc = sorted(cal.score(r["std"]) for r in rs if r["y"] == 0); msc = sorted(cal.score(r["std"]) for r in rs if r["y"] == 1)
        grd = sorted(cal.score(r["std"]) for r in rows if r["arch"] == arch and r["layer"] == layer and r["y"] == -1)
        thr = 0.5 * (gsc[-1] + msc[0]) if gsc[-1] < msc[0] else 0.5
        cal.meta = dict(family=fam, arch=arch, layer=layer, features=feat, standardize=True, auc_loso=row["auc_loso_std"],
                        auc_insample=row["auc_insample_std"], n_models=len(rs), probe_n=500, pca=30, thresh_percentile=95, threshold=thr,
                        battery_scores=dict(generalizing=[round(x, 3) for x in gsc], memorizing=[round(x, 3) for x in msc], graded_noise=[round(x, 3) for x in grd]),
                        note="Phase 6: per-unit standardised (gauge-invariant) features; same battery and labels as Phase 2.")
        cal.save(os.path.join(os.path.dirname(calib_path(fam)), f"memorization_{fam}_std.json"))
        print("saved standardised calibration for", fam, flush=True)
    print(fam, decision[fam], flush=True)
json.dump(dict(decision=decision, table=table, n_boot=N_BOOT, seconds=time.time() - t0), open(f"{OUT}/phase6_standardize_report.json", "w"), indent=1)
print("done", flush=True)
