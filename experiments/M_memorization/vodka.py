"""M2 (benchmark against baselines on the retrained v1 battery) and M3 (per-class scoping study).
SPEC §M2–M3. GPU. Writes results/M2_models.csv, M2_roc.csv, M2_delong.json, M3_perclass.csv, fig/."""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", ".."); sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
from models import ResNet18, cifar_tensors, noisy_subset, probe_from
from whiteboxtopo.activations import ActivationExtractor
from whiteboxtopo.signals.memorization import memorization_features, Calibration
R = os.path.join(HERE, "results"); FIG = os.path.join(R, "fig"); os.makedirs(FIG, exist_ok=True); DEV = "cuda"
NOISES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]; SEEDS = [0, 1, 2]; N_TRAIN = 5000; EPOCHS = 40


def delong(y, s1, s2):
    """DeLong test for two correlated AUCs (Sun & Xu 2014 fast version). Returns (auc1, auc2, z, p)."""
    y = np.asarray(y); pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); m, n = len(pos), len(neg)
    def comps(s):
        s = np.asarray(s, dtype=float); P, N = s[pos], s[neg]
        V10 = np.array([(np.sum(p > N) + 0.5 * np.sum(p == N)) / n for p in P]); V01 = np.array([(np.sum(P > q) + 0.5 * np.sum(P == q)) / m for q in N])
        return V10.mean(), V10, V01
    a1, v10a, v01a = comps(s1); a2, v10b, v01b = comps(s2)
    S10 = np.cov(np.stack([v10a, v10b])); S01 = np.cov(np.stack([v01a, v01b])); S = S10 / m + S01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return float(a1), float(a2), float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


@torch.no_grad()
def acc(model, X, y, bs=1000):
    model.eval(); c = 0
    for i in range(0, len(X), bs): c += (model(X[i:i + bs].to(DEV)).argmax(1).cpu() == y[i:i + bs]).sum().item()
    return c / len(X)


@torch.no_grad()
def predict(model, X, bs=1000):
    model.eval(); return torch.cat([model(X[i:i + bs].to(DEV)).argmax(1).cpu() for i in range(0, len(X), bs)])


@torch.no_grad()
def per_sample_loss(model, X, y, bs=1000):
    model.eval(); return torch.cat([F.cross_entropy(model(X[i:i + bs].to(DEV)), y[i:i + bs].to(DEV), reduction="none").cpu() for i in range(0, len(X), bs)])


def train_resnet(Xs, yn, epochs, augment, lr, seed, probe_batches=None, record_early=True):
    torch.manual_seed(seed); model = ResNet18().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); scaler = torch.amp.GradScaler("cuda")
    dl = DataLoader(TensorDataset(Xs, yn), 128, shuffle=True, drop_last=True); early = []; wnorm = []
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            if augment:
                if torch.rand(1) < 0.5: x = x.flip(3)
                pad = F.pad(x, (4, 4, 4, 4), mode="reflect"); i, j = np.random.randint(0, 9, 2); x = pad[:, :, i:i + 32, j:j + 32]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16): loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        if record_early and ep <= 10: early.append(per_sample_loss(model, Xs[:2000], yn[:2000]).numpy())
        wnorm.append(float(sum(p.norm() ** 2 for p in model.parameters()).sqrt()))
    return model, np.array(early), np.array(wnorm)


def probe_feats(model, Xp):
    ex = ActivationExtractor(model, ["layer3", "pool"]); a = ex([(Xp[i:i + 250], None) for i in range(0, len(Xp), 250)]); ex.remove()
    return {k: v.astype(np.float32) for k, v in a.items()}


def susceptibility(model, Xh, yh, Xte, yte, seed):
    """Forouzesh-style: fine-tune one epoch on a held-out set with 20 % of labels flipped;
    susceptibility = fraction of clean test predictions that change (and test-acc drop)."""
    import copy
    vodka = copy.deepcopy(model); rng = np.random.default_rng(seed); yf = yh.clone(); flip = torch.as_tensor(rng.random(len(yh)) < 0.2)
    yf[flip] = torch.as_tensor(rng.integers(0, 10, int(flip.sum())))
    p0 = predict(vodka, Xte); opt = torch.optim.SGD(vodka.parameters(), lr=0.01, momentum=0.9); vodka.train()
    for x, y in DataLoader(TensorDataset(Xh, yf), 128, shuffle=True):
        opt.zero_grad(); F.cross_entropy(vodka(x.to(DEV)), y.to(DEV)).backward(); opt.step()
    p1 = predict(vodka, Xte); return float((p0 != p1).float().mean()), float(acc(model, Xte, yte) - acc(vodka, Xte, yte))


def eff_rank(A):
    s = np.linalg.svd(A - A.mean(0), compute_uv=False); p = s ** 2 / (s ** 2).sum(); return float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))


def vodka(args):
    Xtr, ytr = cifar_tensors(True); Xte, yte = cifar_tensors(False); Xp, yp = probe_from(Xtr, ytr, 50, seed=0)
    Xh, yh = Xtr[45000:46000], ytr[45000:46000]
    cal = Calibration.load("cnn_classifier"); out = os.path.join(R, "M2_models.csv")
    done = set(pd.read_csv(out).tag) if os.path.exists(out) else set(); rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    cached = os.path.join(ROOT, "benchmarks", "results", "battery_cifar")
    plan = [(f"resnet18_n{int(n * 100):03d}_s{s}", n, s, False) for s in SEEDS for n in NOISES] + [(f"resnet18_full_aug_s{s}", 0.0, s, True) for s in SEEDS]
    for tag, noise, seed, full in plan:
        if tag in done: continue
        t0 = time.time()
        if full: Xs, yn, yc, flip = Xtr, ytr, ytr, torch.zeros(len(ytr), dtype=torch.bool); epochs, lr = 12, 0.1
        else: Xs, yn, yc, flip = noisy_subset(Xtr, ytr, N_TRAIN, noise, seed); epochs, lr = EPOCHS, 0.05
        model, early, wnorm = train_resnet(Xs, yn, epochs, full, lr, seed)
        pred = predict(model, Xs); gt = dict(tag=tag, noise=noise, seed=seed, full_aug=int(full), train_acc_noisy=acc(model, Xs, yn), test_acc=acc(model, Xte, yte),
                                            memorized_flipped_frac=float((pred[flip] == yn[flip]).float().mean()) if flip.any() else np.nan)
        # topological signal (shipped calibration) on the final 'pool' probe cloud; agreement with the v1 cache
        fa = probe_feats(model, Xp); feat = memorization_features(fa["pool"]); gt["topo_score"] = float(cal.score(feat)); gt["topo_tc_ratio"] = feat.get("tc_ratio", np.nan)
        cz = os.path.join(cached, f"{tag}.npz")
        if os.path.exists(cz):
            z = np.load(cz); key = [k for k in z.files if k.endswith("_pool")][-1]; f_old = memorization_features(z[key]); gt["topo_score_v1cache"] = float(cal.score(f_old))
        # baselines
        e = early[:5] if len(early) else np.zeros((1, 1))
        gt["elr_early_loss_mean"] = float(e.mean()); gt["elr_early_consistency"] = float((early[:5] < np.log(10)).mean()) if len(early) else np.nan
        gt["elr_late_minus_early_loss"] = float(early[-1].mean() - early[0].mean()) if len(early) > 1 else np.nan
        gt["susceptibility_flip_frac"], gt["susceptibility_acc_drop"] = susceptibility(model, Xh, yh, Xte, yte, seed)
        gt["grok_eff_rank_pool"] = eff_rank(fa["pool"]); gt["grok_wnorm_ratio"] = float(wnorm[-1] / wnorm[0]); gt["seconds"] = time.time() - t0
        rows.append(gt); pd.DataFrame(rows).to_csv(out, index=False); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in gt.items()}, flush=True)
        del model; torch.cuda.empty_cache()
    # ---- ROC + DeLong
    df = pd.DataFrame(rows); y = (df.noise > 0).astype(int).values
    methods = {"topological_signal": df.topo_score.values, "susceptibility_flip_frac": df.susceptibility_flip_frac.values,
               "elr_early_loss_mean": df.elr_early_loss_mean.values, "elr_early_consistency": -df.elr_early_consistency.values,
               "grok_eff_rank_pool": -df.grok_eff_rank_pool.values, "grok_wnorm_ratio": df.grok_wnorm_ratio.values, "generalization_gap": (df.train_acc_noisy - df.test_acc).values}
    roc = []; dl = {}
    for k, s in methods.items():
        s = np.nan_to_num(s.astype(float)); a = roc_auc_score(y, s); a2 = roc_auc_score(y, -s)
        rho = stats.spearmanr(s[df.noise > 0], df.memorized_flipped_frac.values[df.noise > 0], nan_policy="omit")[0]
        roc.append(dict(method=k, auc=a, auc_best_orientation=max(a, a2), spearman_vs_memorized_frac=rho))
        if k != "topological_signal":
            a1, a2_, z, p = delong(y, np.nan_to_num(methods["topological_signal"].astype(float)), s); dl[k] = dict(auc_topo=a1, auc_baseline=a2_, z=z, p=p)
    pd.DataFrame(roc).to_csv(os.path.join(R, "M2_roc.csv"), index=False); json.dump(dl, open(os.path.join(R, "M2_delong.json"), "w"), indent=1)
    print(pd.DataFrame(roc).to_string()); print(json.dumps(dl, indent=1))


def daiwa_scarlet(args):
    Xtr, ytr = cifar_tensors(True); Xte, yte = cifar_tensors(False); cal = Calibration.load("cnn_classifier"); rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed); idx = rng.permutation(len(Xtr))[:N_TRAIN]; Xs, yc = Xtr[idx], ytr[idx]; yn = yc.clone()
        corrupt = torch.as_tensor(np.isin(yc.numpy(), [0, 1, 2]) & (rng.random(N_TRAIN) < 0.4))
        yn[corrupt] = torch.as_tensor(rng.integers(3, 10, int(corrupt.sum())))       # within-class 40 % flips to other classes
        model, early, wnorm = train_resnet(Xs, yn, EPOCHS, False, 0.05, seed)
        fa_all = probe_feats(model, Xs[:3000]); pool = fa_all["pool"]; ycl = yc[:3000].numpy()
        for c in range(10):
            sel = np.flatnonzero(ycl == c)[:150]; feat = memorization_features(pool[sel])
            e = early[:5][:, sel] if len(early) else np.zeros((1, 1))
            rows.append(dict(seed=seed, cls=c, corrupted=int(c in (0, 1, 2)), topo_score=float(cal.score(feat)), tc_ratio=feat.get("tc_ratio", np.nan),
                             elr_early_loss=float(e.mean()), elr_flip_rate=float((np.argmin(early[:5][:, sel], 0) != 4).mean()) if len(early) >= 5 else np.nan,
                             class_train_acc_noisy=float((predict(model, Xs[sel]) == yn[sel]).float().mean())))
        print(rows[-10:], flush=True); del model; torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df.to_csv(os.path.join(R, "M3_perclass.csv"), index=False)
    summ = []
    for k in ("topo_score", "elr_early_loss", "class_train_acc_noisy"):
        aucs = [max(roc_auc_score(d.corrupted, d[k]), roc_auc_score(d.corrupted, -d[k])) for _, d in df.groupby("seed")]
        summ.append(dict(method=k, auc_mean=np.mean(aucs), auc_per_seed=aucs))
    pd.DataFrame(summ).to_csv(os.path.join(R, "M3_summary.csv"), index=False); print(pd.DataFrame(summ).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--daiwa-scarlet", "--m3", dest="daiwa_scarlet", action="store_true"); a = ap.parse_args()
    (daiwa_scarlet if a.daiwa_scarlet else vodka)(a)
