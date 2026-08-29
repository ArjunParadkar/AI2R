"""Phase 3 (synthetic): data manifolds of KNOWN topology -> is the data-manifold
persistence recovered, and does correspondence separate generalising from
memorising models?

Manifolds: circle (b1=1), torus (b1=2, b2=1), sphere (b2=1), 600 points each,
isometrically embedded in R^20 with Gaussian noise. Per manifold x 3 seeds:
  iso        the data itself, rotated+scaled (control: score -> 1)
  untrained  random-init MLP hidden layer
  general    MLP regressing a SMOOTH function of the manifold coordinates
  memorize   the same MLP regressing i.i.d. random targets (must fit them)
Ground truth for (c): held-out R^2 of each trained model on fresh points.
"""
from __future__ import annotations
import csv, json, os, time
import numpy as np, torch, torch.nn as nn
from whitebox.core.persistence import compute_diagrams
from whitebox.signals.correspondence import correspondence
from common import SYNTHETIC, embed, RESULTS
torch.set_num_threads(4)
MAN = {"circle": (SYNTHETIC["circle"], [1, 1, 0]), "torus": (SYNTHETIC["torus"], [1, 2, 1]), "sphere": (SYNTHETIC["sphere"], [1, 0, 1])}
N, D = 600, 20


def smooth_target(X3):  # smooth scalar field on the intrinsic coordinates
    return np.sin(2 * X3[:, 0]) + np.cos(1.5 * X3[:, 1]) + (X3[:, 2] if X3.shape[1] > 2 else 0)


def betti_from_diagram(dg, k, frac=0.5):
    """Number of 'dominant' bars: lifetime > frac * max lifetime (k>=1); for H0 count bars > frac*max among finite bars + 1 infinite."""
    if dg[k].size == 0: return 0
    lt = dg[k][:, 1] - dg[k][:, 0]
    return int(np.sum(lt > frac * lt.max()))


class MLP(nn.Module):
    def __init__(self, d_in, h=64):
        super().__init__(); self.f1 = nn.Linear(d_in, h); self.f2 = nn.Linear(h, h); self.act = nn.Tanh(); self.out = nn.Linear(h, 1)
    def forward(self, x):
        return self.out(self.act(self.f2(self.act(self.f1(x)))))


def train(X, y, seed, epochs=3000, lr=3e-3):
    torch.manual_seed(seed); m = MLP(X.shape[1]); opt = torch.optim.Adam(m.parameters(), lr)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)[:, None]
    for _ in range(epochs):
        opt.zero_grad(); loss = ((m(Xt) - yt) ** 2).mean(); loss.backward(); opt.step()
    return m, float(loss)


@torch.no_grad()
def hidden(m, X):
    return m.act(m.f2(m.act(m.f1(torch.tensor(X, dtype=torch.float32))))).numpy()


@torch.no_grad()
def r2(m, X, y):
    p = m(torch.tensor(X, dtype=torch.float32)).numpy()[:, 0]; return float(1 - ((p - y) ** 2).mean() / y.var())


rows, checks = [], []
for name, (gen, betti) in MAN.items():
    for seed in range(3):
        X3 = gen(N, seed=seed); X = embed(X3, D, 0.03, seed); Xte3 = gen(300, seed=seed + 50); Xte = embed(Xte3, D, 0.03, seed)
        # (1) data-manifold persistence check, H0..H2 on a 300-landmark subsample (H2 cost)
        dg = compute_diagrams(X, maxdim=2, thresh_percentile=100, backend="landmark", n_landmarks=250)
        rec = [betti_from_diagram(dg, k) for k in range(3)]
        checks.append(dict(manifold=name, seed=seed, expected_b1=betti[1], expected_b2=betti[2], recovered_b1=rec[1], recovered_b2=rec[2],
                           top_lifetimes_h1=np.sort(dg.lifetimes(1))[::-1][:4].round(3).tolist(), top_lifetimes_h2=np.sort(dg.lifetimes(2))[::-1][:3].round(3).tolist()))
        print(checks[-1], flush=True)
        # (2) models
        y_s, y_r = smooth_target(X3), np.random.default_rng(seed).standard_normal(N)
        yte_s = smooth_target(Xte3)
        rng = np.random.default_rng(seed); Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        reps = {"iso": (X @ Q.T * 3.0, None, None)}
        m0 = MLP(D); torch.manual_seed(seed); reps["untrained"] = (hidden(m0, X), None, None)
        t0 = time.time(); mg, lg = train(X, y_s, seed); reps["general"] = (hidden(mg, X), r2(mg, X, y_s), r2(mg, Xte, yte_s))
        mm, lm = train(X, y_r, seed, epochs=6000); reps["memorize"] = (hidden(mm, X), r2(mm, X, y_r), r2(mm, Xte, np.random.default_rng(seed + 9).standard_normal(300)))
        for cond, (H, tr_r2, te_r2) in reps.items():
            c = correspondence(X, H, n_trials=8, batch=400, seed=seed)
            rows.append(dict(manifold=name, seed=seed, condition=cond, score=c.score, ci_low=c.ci_low, ci_high=c.ci_high, rtd=c.rtd.point, rtd_null=c.rtd_null.point,
                             w2_h1=c.diagram_w2, train_r2=tr_r2, heldout_r2=te_r2))
            print(f"  {name} s{seed} {cond:10s} score={c.score:.3f} [{c.ci_low:.2f},{c.ci_high:.2f}] rtd={c.rtd.point:.3f} null={c.rtd_null.point:.3f} trainR2={tr_r2} testR2={te_r2}", flush=True)
os.makedirs(RESULTS, exist_ok=True)
for fn, rs in (("phase3_synthetic_manifold_check.csv", checks), ("phase3_synthetic.csv", rows)):
    with open(f"{RESULTS}/{fn}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); conds = ["iso", "untrained", "general", "memorize"]
for i, name in enumerate(MAN):
    for j, cond in enumerate(conds):
        rs = [r for r in rows if r["manifold"] == name and r["condition"] == cond]
        ax[0].errorbar([i + (j - 1.5) * 0.18] * len(rs), [r["score"] for r in rs], yerr=[[r["score"] - r["ci_low"] for r in rs], [r["ci_high"] - r["score"] for r in rs]],
                       fmt="o", color=f"C{j}", label=cond if i == 0 else None, capsize=2)
ax[0].set_xticks(range(len(MAN))); ax[0].set_xticklabels(list(MAN)); ax[0].set_ylabel("correspondence (1 - RTD/RTD_null)"); ax[0].legend(); ax[0].set_title("synthetic manifolds: hidden-layer correspondence")
tr = [r for r in rows if r["heldout_r2"] is not None]
ax[1].scatter([r["heldout_r2"] for r in tr], [r["score"] for r in tr], c=["C2" if r["condition"] == "general" else "C3" for r in tr])
ax[1].set_xlabel("held-out R^2 (ground truth, not used by the signal)"); ax[1].set_ylabel("correspondence"); ax[1].set_title("(c): correspondence vs held-out generalisation")
fig.tight_layout(); fig.savefig(f"{RESULTS}/phase3_synthetic.png", dpi=300); print("done")
