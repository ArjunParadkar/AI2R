"""Phase 4: validation battery for the remaining candidate signals, against
ground truth, with proper nulls. Two ResNet-18/CIFAR-10 models (full data,
augmentation, 12 epochs, seeds 0/1).

A. P4 failure neighbourhoods. Original claim: misclassified test inputs sit
   in topologically sparser layer-3 regions (bootstrap TC, Mann-Whitney).
   Here: (i) the same statistic but with a LABEL-PERMUTATION null (the
   original's bootstrap draws were pseudo-replicates); (ii) the deployable
   per-sample version — kNN radius in layer-3 space as a misclassification
   score, AUC — against the softmax-confidence baseline. Ships only if it
   adds to confidence.
B. BiasTop permutation test: false-positive rate under a true null (random
   splits of one class distribution, 30 repeats, alpha=0.05) and power vs
   subgroup contrast (subgroup B = mixture of two classes at fraction f).
C. P1 replication: per-class top-3 FPL (pool layer) vs per-class test acc.
D. P5 replication: per-class residual TC / FPL vs accuracy drop under
   Gaussian input noise (expected null).
"""
from __future__ import annotations
import csv, json, os, time
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
import whiteboxtopo as wb
from whiteboxtopo.activations import ActivationExtractor
from models import ResNet18, cifar_tensors, probe_from
from common import RESULTS
DEV = "cuda"; OUT = f"{RESULTS}/phase4"; os.makedirs(OUT, exist_ok=True)
Xtr, ytr = cifar_tensors(True); Xte, yte = cifar_tensors(False)


def train_full(seed, epochs=12):
    torch.manual_seed(seed); m = ResNet18().to(DEV); opt = torch.optim.SGD(m.parameters(), 0.1, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); sc = torch.amp.GradScaler("cuda"); dl = DataLoader(TensorDataset(Xtr, ytr), 128, shuffle=True, drop_last=True)
    for ep in range(epochs):
        m.train()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            if torch.rand(1) < 0.5: x = x.flip(3)
            pad = F.pad(x, (4, 4, 4, 4), mode="reflect"); i, j = np.random.randint(0, 9, 2); x = pad[:, :, i:i + 32, j:j + 32]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16): loss = F.cross_entropy(m(x), y)
            sc.scale(loss).backward(); sc.step(opt); sc.update()
        sch.step()
    return m.eval()


@torch.no_grad()
def acts_and_preds(m, X, layers=("layer3", "pool"), sigma=0.0, bs=500):
    ex = ActivationExtractor(m, list(layers)); A = {l: [] for l in layers}; logits = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(DEV)
        if sigma > 0: x = x + sigma * torch.randn_like(x)
        ex._store = {}; ex._mask = None; logits.append(m(x).float().cpu())
        for l in layers: A[l].append(ex._store[l][0])
    ex.remove(); return {l: np.concatenate(v) for l, v in A.items()}, torch.cat(logits)


report = {}; rows_fpl = []
for seed in (0, 1):
    t0 = time.time(); m = train_full(seed); A, logits = acts_and_preds(m, Xte); pred = logits.argmax(1); corr = (pred == yte).numpy(); conf = logits.softmax(1).max(1).values.numpy()
    test_acc = float(corr.mean()); print(f"seed {seed}: test acc {test_acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    R = {"test_acc": test_acc}
    # ---- A. failure neighbourhoods -------------------------------------
    L3 = A["layer3"]; red = wb.fit_reducer(L3[corr], 30); Z = red.transform(L3); rng = np.random.default_rng(seed)
    def tc_sub(idx_pool, n=20, size=150):
        return np.array([wb.topological_complexity(wb.compute_diagrams(Z[rng.choice(idx_pool, size, replace=False)])[1]) for _ in range(n)])
    ci, mi = np.where(corr)[0], np.where(~corr)[0]
    tcC, tcM = tc_sub(ci), tc_sub(mi); obs = tcM.mean() - tcC.mean(); u, p_mw = mannwhitneyu(tcC, tcM)
    null = []
    for _ in range(100):
        lab = rng.permutation(corr); null.append(tc_sub(np.where(~lab)[0], n=5).mean() - tc_sub(np.where(lab)[0], n=5).mean())
    null = np.array(null); p_perm = float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null)))
    nn_ = NearestNeighbors(n_neighbors=11).fit(Z); d, _ = nn_.kneighbors(Z); knn_r = d[:, 1:].mean(1)
    R["A_failure"] = dict(TC_correct=float(tcC.mean()), TC_mis=float(tcM.mean()), rel_diff=float(obs / tcC.mean()), mannwhitney_p_pseudo=float(p_mw), permutation_p=p_perm,
                          auc_knn_radius=float(roc_auc_score(~corr, knn_r)), auc_softmax_conf=float(roc_auc_score(~corr, -conf)),
                          auc_combined=float(roc_auc_score(~corr, (knn_r - knn_r.mean()) / knn_r.std() - (conf - conf.mean()) / conf.std())), n_mis=int((~corr).sum()))
    print("A", R["A_failure"], flush=True)
    # ---- B. BiasTop: FPR under null, power vs contrast -------------------
    P = A["pool"]; redp = wb.fit_reducer(P, 30); cls = [np.where(yte.numpy() == c)[0] for c in range(10)]
    fpr_hits = 0; NREP = 30
    for r_ in range(NREP):
        idx = rng.choice(cls[0], 200, replace=False); a, b = idx[:100], idx[100:]
        _, p, _ = wb.bias_top_permutation_test(P[a], P[b], redp, n_permutations=99, rng=np.random.default_rng(r_))
        fpr_hits += p < 0.05
    power = {}
    for f in (0.1, 0.25, 0.5, 1.0):
        hits = 0
        for r_ in range(10):
            a = rng.choice(cls[0], 100, replace=False); nb = int(100 * f); b = np.concatenate([rng.choice(cls[1], nb, replace=False), rng.choice(np.setdiff1d(cls[0], a), 100 - nb, replace=False)])
            _, p, _ = wb.bias_top_permutation_test(P[a], P[b], redp, n_permutations=99, rng=np.random.default_rng(100 + r_)); hits += p < 0.05
        power[str(f)] = hits / 10
    R["B_biastop"] = dict(false_positive_rate_alpha05=fpr_hits / NREP, n_null_repeats=NREP, power_vs_mixing_fraction=power)
    print("B", R["B_biastop"], flush=True)
    # ---- C/D. per-class FPL vs accuracy; vs fragility -------------------
    fpl, tcc, acc, accn = [], [], [], []
    An, logn = acts_and_preds(m, Xte, sigma=0.1); predn = logn.argmax(1)
    for c in range(10):
        idx = rng.choice(cls[c], 300, replace=False); D = wb.compute_diagrams(P[idx], redp)
        fpl.append(wb.top_k_lifetime(D[1], 3)); tcc.append(wb.topological_complexity(D[1]))
        acc.append(float((pred[cls[c]] == c).float().mean())); accn.append(float((predn[cls[c]] == c).float().mean()))
    drop = np.array(acc) - np.array(accn)
    R["C_fpl_vs_acc"] = dict(spearman=spearmanr(fpl, acc)[0], p=spearmanr(fpl, acc)[1], fpl=fpl, acc=acc)
    R["D_fragility"] = dict(spearman_fpl_vs_drop=spearmanr(fpl, drop)[0], p_fpl=spearmanr(fpl, drop)[1], spearman_tc_vs_drop=spearmanr(tcc, drop)[0], p_tc=spearmanr(tcc, drop)[1], drop=drop.tolist())
    print("C", {k: v for k, v in R["C_fpl_vs_acc"].items() if k in ("spearman", "p")}, "D", {k: v for k, v in R["D_fragility"].items() if k.startswith(("spearman", "p_"))}, flush=True)
    report[f"seed{seed}"] = R; del m; torch.cuda.empty_cache()
json.dump(report, open(f"{OUT}/phase4_report.json", "w"), indent=1, default=float)
# CSV summary
with open(f"{OUT}/phase4_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["seed", "test_acc", "A_TC_rel_diff", "A_perm_p", "A_auc_knn", "A_auc_softmax", "A_auc_combined", "B_FPR", "B_power_f0.1", "B_power_f0.25", "B_power_f0.5", "B_power_f1", "C_rho_fpl_acc", "C_p", "D_rho_fpl_drop", "D_rho_tc_drop"])
    for s, R in report.items():
        w.writerow([s, R["test_acc"], R["A_failure"]["rel_diff"], R["A_failure"]["permutation_p"], R["A_failure"]["auc_knn_radius"], R["A_failure"]["auc_softmax_conf"], R["A_failure"]["auc_combined"],
                    R["B_biastop"]["false_positive_rate_alpha05"], *[R["B_biastop"]["power_vs_mixing_fraction"][k] for k in ("0.1", "0.25", "0.5", "1.0")],
                    R["C_fpl_vs_acc"]["spearman"], R["C_fpl_vs_acc"]["p"], R["D_fragility"]["spearman_fpl_vs_drop"], R["D_fragility"]["spearman_tc_vs_drop"]])
print("phase4 done")
