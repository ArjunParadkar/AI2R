"""M4 — cross-family scope of the memorisation signal (HPCC): recalibrate ``memorization_signal`` per family
(CNN = the cached ResNet-18/CIFAR battery, LM = the cached GPT-2 battery, ViT = a new ViT-S/8 CIFAR battery trained
here with the same label-noise grid) and report leave-one-seed-out AUC *within* each family and the full cross-family
AUC matrix honestly (v1: CNN -> LM 0.00 / 0.67; handbook M4). NOVELTY_BOARD.md #9: the detector claim is TAKEN
(Topo^2 2608.30487, MOIT, CrossSplit); this is scope-setting, not a detection claim.

Usage:  python admire_vega.py --cnn-dir ../benchmarks/results/battery_cifar --lm-dir ../benchmarks/results/battery_gpt2 \
            --out results/m4 --train-vit --resume
Phase-2 protocol reused verbatim (penultimate probe layer, 500 probe points, features = memorization_features,
logistic calibration on log-features, y = 1 iff noise >= 50 %, y = 0 iff noise = 0). Checkpoint: per ViT model (json +
npz in out/battery_vit, same layout as the CNN battery); the AUC table is recomputed at the end."""
import os, sys, json, glob, time, argparse, itertools
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
from whiteboxtopo.signals.memorization import memorization_features, Calibration
LOG = ["tc_ratio", "tc_norm", "tc0_norm", "b1", "n_bars"]; FEATS = ["tc_ratio", "pe1", "tc_norm", "b1", "tc0_norm"]
NOISES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]; SEEDS = [0, 1, 2]


def load_family(fam, d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        tag = os.path.basename(p)[:-5]; gt = json.load(open(p)); z = np.load(p[:-5] + ".npz")
        if fam == "lm":
            rows.append(dict(family=fam, tag=tag, noise=gt["noise"], seed=gt["seed"], acts=z["ep10_L12"], memorized=gt["memorized_flipped_frac"]))
        else:
            ep = max(int(k.split("_")[0][2:]) for k in z.files); noise = 0 if "full" in tag else int(tag.split("_n")[1][:3])
            rows.append(dict(family=fam, tag=tag, noise=noise, seed=int(tag[-1]), acts=z[f"ep{ep}_pool"], memorized=gt["memorized_flipped_frac"]))
    for r in rows: r["y"] = 1 if r["noise"] >= 50 else (0 if r["noise"] == 0 else -1)
    return rows


def train_vit_battery(out, device="cuda", epochs=40, n_train=5000):
    """ViT-S/8 (6 blocks, d=192, 3 heads, patch 8 on 32x32) on the Phase-2 5k CIFAR-10 subset with symmetric label noise."""
    import torch, torch.nn as nn, torch.nn.functional as F
    from models import cifar_tensors, noisy_subset, probe_from
    os.makedirs(out, exist_ok=True)
    Xtr, ytr = cifar_tensors(True); Xte, yte = cifar_tensors(False); Xp, yp = probe_from(Xtr, ytr, 50, seed=0)

    class ViT(nn.Module):
        def __init__(s, d=192, depth=6, heads=3, patch=8, n_cls=10):
            super().__init__(); s.patch = nn.Conv2d(3, d, patch, patch); n = (32 // patch) ** 2
            s.cls = nn.Parameter(torch.zeros(1, 1, d)); s.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
            s.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True), depth)
            s.norm = nn.LayerNorm(d); s.head = nn.Linear(d, n_cls)
        def features(s, x):
            t = s.patch(x).flatten(2).transpose(1, 2); t = torch.cat([s.cls.expand(len(t), -1, -1), t], 1) + s.pos
            return s.norm(s.blocks(t))[:, 0]
        def forward(s, x): return s.head(s.features(x))
    for noise, seed in itertools.product(NOISES, SEEDS):
        tag = f"vit_n{int(noise*100):03d}_s{seed}"; jp = os.path.join(out, tag + ".json")
        if os.path.exists(jp): continue
        torch.manual_seed(seed); Xs, yn, yc, flipped = noisy_subset(Xtr, ytr, n_train, noise, seed)
        m = ViT().to(device); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.05); t0 = time.time()
        for ep in range(epochs):
            m.train(); perm = torch.randperm(n_train)
            for i in range(0, n_train, 128):
                idx = perm[i:i + 128]; loss = F.cross_entropy(m(Xs[idx].to(device)), yn[idx].to(device)); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pred = torch.cat([m(Xs[i:i + 500].to(device)).argmax(1).cpu() for i in range(0, n_train, 500)])
            te = torch.cat([m(Xte[i:i + 1000].to(device)).argmax(1).cpu() for i in range(0, len(Xte), 1000)])
            feats = torch.cat([m.features(Xp[i:i + 250].to(device)).cpu() for i in range(0, 500, 250)]).numpy()
        gt = dict(train_acc_noisy=float((pred == yn).float().mean()), train_acc_clean=float((pred == yc).float().mean()), test_acc=float((te == yte).float().mean()),
                  memorized_flipped_frac=float((pred[flipped] == yn[flipped]).float().mean()) if flipped.any() else float("nan"), epochs=epochs, n_train=n_train, seconds=time.time() - t0)
        np.savez(os.path.join(out, tag + ".npz"), **{f"ep{epochs}_pool": feats}); json.dump(gt, open(jp, "w"), indent=1); print(tag, gt, flush=True)


def X_of(rs): return np.array([[np.log(max(r["feat"][k], 1e-12)) if k in LOG else r["feat"][k] for k in FEATS] for r in rs])


def fit(rs):
    from sklearn.linear_model import LogisticRegression
    X = X_of(rs); y = np.array([r["y"] for r in rs]); mu, sd = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(C=1.0).fit((X - mu) / sd, y)
    return Calibration(FEATS, mu, sd, lr.coef_[0], float(lr.intercept_[0]), [k for k in FEATS if k in LOG])


def auc(cal, rs):
    from sklearn.metrics import roc_auc_score
    y = np.array([r["y"] for r in rs]); s = np.array([cal.score(r["feat"]) for r in rs])
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cnn-dir", default=os.path.join(ROOT, "benchmarks", "results", "battery_cifar"))
    ap.add_argument("--lm-dir", default=os.path.join(ROOT, "benchmarks", "results", "battery_gpt2")); ap.add_argument("--out", default="results/m4")
    ap.add_argument("--train-vit", action="store_true"); ap.add_argument("--resume", action="store_true"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); vit_dir = os.path.join(a.out, "battery_vit")
    if a.train_vit: train_vit_battery(vit_dir)
    fams = {"cnn": load_family("cnn", a.cnn_dir), "lm": load_family("lm", a.lm_dir)}
    if os.path.isdir(vit_dir) and glob.glob(os.path.join(vit_dir, "*.json")): fams["vit"] = load_family("vit", vit_dir)
    fp = os.path.join(a.out, "features.json"); cache = json.load(open(fp)) if (a.resume and os.path.exists(fp)) else {}
    for fam, rs in fams.items():
        for r in rs:
            k = f"{fam}/{r['tag']}"
            if k not in cache: cache[k] = memorization_features(r["acts"], distance_device="cuda"); json.dump(cache, open(fp, "w"))
            r["feat"] = cache[k]
    lab = {f: [r for r in rs if r["y"] >= 0] for f, rs in fams.items()}
    table = []
    for src in lab:
        loso = []
        for s in sorted({r["seed"] for r in lab[src]}):
            tr = [r for r in lab[src] if r["seed"] != s]; te = [r for r in lab[src] if r["seed"] == s]
            if len({r["y"] for r in tr}) == 2 and len({r["y"] for r in te}) == 2: loso.append(auc(fit(tr), te))
        row = dict(calibrated_on=src, auc_loso_within=float(np.mean(loso)) if loso else float("nan"))
        cal = fit(lab[src]); cal.meta = dict(family=src, protocol="M4 recalibration, Phase-2 features, penultimate probe layer")
        cal.save(os.path.join(a.out, f"calibration_{src}.json"))
        for dst in lab: row[f"auc_on_{dst}"] = auc(cal, lab[dst])
        table.append(row); print(row, flush=True)
    pd.DataFrame(table).to_csv(os.path.join(a.out, "M4_cross_family_auc.csv"), index=False)
    print("cross-family AUC matrix written; off-diagonal entries are the honest scope statement (handbook M4)")


if __name__ == "__main__":
    main()
