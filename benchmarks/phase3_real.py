"""Phase 3 (real data): correspondence between the input space and the
penultimate representation for every model of the Phase 2 CIFAR battery
(21 ResNet-18s with known label noise, test accuracy and memorisation
fraction), plus an MNIST MLP mini-battery (label noise {0,25,50,100}% x 3
seeds, 3k-image subset, fitted to completion).

Data manifold: pixel space (flattened normalised images). This is the honest
weak point — pixel-space Euclidean distances are a poor estimate of the image
manifold; it is what RTD uses, and PHASE3.md says so.
Hypotheses: (a) high for 0% noise, (b) low for >=50%, (c) score predicts
test accuracy across models without touching test data.
"""
from __future__ import annotations
import csv, glob, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from whitebox.signals.correspondence import correspondence
from models import cifar_tensors, probe_from
from common import RESULTS
DEV = "cuda"; rows = []

# ---- CIFAR battery (reuse) ------------------------------------------------
Xtr, ytr = cifar_tensors(True); Xp, yp = probe_from(Xtr, ytr, 50, seed=0); P = Xp.flatten(1).numpy().astype(np.float64)
for p in sorted(glob.glob(f"{RESULTS}/battery_cifar/*.json")):
    tag = os.path.basename(p)[:-5]; gt = json.load(open(p)); z = np.load(p[:-5] + ".npz")
    ep = max(int(k.split("_")[0][2:]) for k in z.files); noise = 0 if "full" in tag else int(tag.split("_n")[1][:3])
    for layer in ("pool",):   # layer3 cross-barcodes cost ~35 s each (vs 2 s): dropped, penultimate is the hypothesis layer
        c = correspondence(P, z[f"ep{ep}_{layer}"], n_trials=5, batch=300, seed=0, device=DEV)
        rows.append(dict(dataset="cifar10", arch="resnet18", tag=tag, layer=layer, noise=noise, score=c.score, ci_low=c.ci_low, ci_high=c.ci_high,
                         rtd=c.rtd.point, rtd_null=c.rtd_null.point, w2_h1=c.diagram_w2, test_acc=gt["test_acc"], train_acc=gt["train_acc_noisy"], memorized=gt["memorized_flipped_frac"]))
        print(tag, layer, f"score={c.score:.3f} rtd={c.rtd.point:.3f} null={c.rtd_null.point:.3f} test={gt['test_acc']:.3f}", flush=True)

# ---- MNIST MLP mini-battery -----------------------------------------------
import torchvision, torchvision.transforms as T
root = "/home/god/whitebox-lib/benchmarks/data"; os.makedirs(root, exist_ok=True)
tr = torchvision.datasets.MNIST(root, train=True, download=True); te = torchvision.datasets.MNIST(root, train=False, download=True)
Xm = (tr.data.float() / 255).flatten(1); ym = tr.targets; Xmt = (te.data.float() / 255).flatten(1); ymt = te.targets
rng = np.random.default_rng(0); pidx = np.concatenate([rng.choice(np.where(ym.numpy() == c)[0], 50, replace=False) for c in range(10)])
Pm = Xm[pidx].numpy().astype(np.float64)


class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.f1 = nn.Linear(784, 256); self.f2 = nn.Linear(256, 128); self.out = nn.Linear(128, 10)
    def forward(self, x):
        return self.out(F.relu(self.f2(F.relu(self.f1(x)))))


for seed in range(3):
    for noise in (0, 25, 50, 100):
        tag = f"mlp_n{noise:03d}_s{seed}"; r = np.random.default_rng(seed); idx = r.choice(len(Xm), 3000, replace=False)
        X, yc = Xm[idx].to(DEV), ym[idx].clone(); yn = yc.clone(); flip = torch.as_tensor(r.random(3000) < noise / 100)
        yn[flip] = (yc[flip] + torch.as_tensor(r.integers(1, 10, int(flip.sum())))) % 10; yn = yn.to(DEV)
        torch.manual_seed(seed); m = MLP().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
        for ep in range(300):
            perm = torch.randperm(3000, device=DEV)
            for i in range(0, 3000, 128):
                b = perm[i:i + 128]; opt.zero_grad(); F.cross_entropy(m(X[b]), yn[b]).backward(); opt.step()
        with torch.no_grad():
            m.eval(); tr_acc = (m(X).argmax(1) == yn).float().mean().item(); te_acc = (m(Xmt.to(DEV)).argmax(1).cpu() == ymt).float().mean().item()
            H = F.relu(m.f2(F.relu(m.f1(Xm[pidx].to(DEV))))).cpu().numpy()
        c = correspondence(Pm, H, n_trials=5, batch=300, seed=0, device=DEV)
        rows.append(dict(dataset="mnist", arch="mlp", tag=tag, layer="f2", noise=noise, score=c.score, ci_low=c.ci_low, ci_high=c.ci_high, rtd=c.rtd.point, rtd_null=c.rtd_null.point,
                         w2_h1=c.diagram_w2, test_acc=te_acc, train_acc=tr_acc, memorized=float("nan")))
        print(tag, f"score={c.score:.3f} rtd={c.rtd.point:.3f} null={c.rtd_null.point:.3f} train={tr_acc:.3f} test={te_acc:.3f}", flush=True)

with open(f"{RESULTS}/phase3_real.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
stats = {}
for ds, layer in (("cifar10", "pool"), ("mnist", "f2")):
    rs = [r for r in rows if r["dataset"] == ds and r["layer"] == layer]
    gen = [r["score"] for r in rs if r["noise"] == 0]; mem = [r["score"] for r in rs if r["noise"] >= 50]
    from sklearn.metrics import roc_auc_score
    stats[f"{ds}/{layer}"] = dict(n=len(rs), mean_score_generalizing=float(np.mean(gen)), mean_score_memorizing=float(np.mean(mem)),
                                  auc_gen_vs_mem=float(roc_auc_score([0] * len(gen) + [1] * len(mem), gen + mem)) if gen and mem else None,
                                  spearman_score_vs_test_acc=spearmanr([r["score"] for r in rs], [r["test_acc"] for r in rs])[0],
                                  spearman_p=spearmanr([r["score"] for r in rs], [r["test_acc"] for r in rs])[1])
json.dump(stats, open(f"{RESULTS}/phase3_real_stats.json", "w"), indent=1); print(json.dumps(stats, indent=1))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for ds, layer, mk in (("cifar10", "pool", "o"), ("mnist", "f2", "s")):
    rs = [r for r in rows if r["dataset"] == ds and r["layer"] == layer]
    ax[0].errorbar([r["noise"] + np.random.uniform(-2, 2) for r in rs], [r["score"] for r in rs], yerr=[[r["score"] - r["ci_low"] for r in rs], [r["ci_high"] - r["score"] for r in rs]], fmt=mk, capsize=2, label=f"{ds}/{layer}", alpha=0.8)
    ax[1].scatter([r["test_acc"] for r in rs], [r["score"] for r in rs], marker=mk, label=f"{ds}/{layer} ρ={stats[f'{ds}/{layer}']['spearman_score_vs_test_acc']:.2f}", alpha=0.8)
ax[0].set_xlabel("label noise %"); ax[0].set_ylabel("correspondence"); ax[0].legend(); ax[0].set_title("(a)/(b): correspondence vs injected noise")
ax[1].set_xlabel("test accuracy (never seen by the signal)"); ax[1].set_ylabel("correspondence"); ax[1].legend(); ax[1].set_title("(c): correspondence vs held-out accuracy")
fig.tight_layout(); fig.savefig(f"{RESULTS}/phase3_real.png", dpi=300); print("done")
