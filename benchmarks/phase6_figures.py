"""Phase 6 (v2) figures, 300 dpi, under results/phase6/:
  phase6_invariance_cv.png     orbit CV per quantity (E2 table: 687 trained nets, perm / scale / combined)
  phase6_legibility_phase.png  legibility phase diagram (E4: Q10 vs treewidth/N and vs sparsity, colour = S, marker = arch)
  phase6_standardize_auc.png   memorization_signal LOSO AUC raw vs standardised (if the benchmark has run)
Inputs are the CSVs copied from whitebox-v2/E2/results and E4/results into results/phase6/ (data, not code).
"""
import csv, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
P6 = os.path.join(RESULTS, "phase6")


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- 1. invariance CV bar chart
rows = read(f"{P6}/e2_invariance_table.csv")
order = {"sanity": 0, "topology_repair": 1, "indictment_sorted": 2, "spectrum": 3, "topology": 4, "indictment": 5, "indictment_discrete": 6}
rows = [r for r in rows if r["quantity"] != "gates_index_agreement"]
rows.sort(key=lambda r: (order.get(r["family"], 9), float(r["cv_combined"])))
names = [r["quantity"] for r in rows]; x = np.arange(len(rows)); floor = 1e-16
fig, ax = plt.subplots(figsize=(15, 5.2))
for i, (g, c) in enumerate((("perm", "C0"), ("scale", "C1"), ("combined", "C3"))):
    v = np.array([max(float(r[f"cv_{g}"]), floor) for r in rows]); lo = np.array([max(float(r[f"cv_{g}_lo"]), floor) for r in rows]); hi = np.array([max(float(r[f"cv_{g}_hi"]), floor) for r in rows])
    ax.bar(x + (i - 1) * 0.27, v, 0.27, color=c, label=f"{g}", yerr=[v - lo, hi - v], error_kw=dict(lw=0.5, capsize=1))
bn = [float(r["between_network_cv"]) for r in rows]
ax.plot(x, bn, "k_", ms=9, mew=1.5, label="between-network CV (vacuity check)")
ax.axhline(1e-6, color="g", ls="--", lw=0.8); ax.text(0.5, 1.6e-6, "exact (< 1e-6)", color="g", ha="left", fontsize=8)
ax.axhline(0.05, color="r", ls="--", lw=0.8); ax.text(7.6, 0.07, "not invariant (> 0.05)", color="r", ha="left", fontsize=8)
ax.set_yscale("log"); ax.set_ylim(5e-17, 30); ax.set_xticks(x); ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
ax.set_ylabel("orbit CV (median over 687 trained ReLU MLPs, 100 reps)"); ax.legend(fontsize=8, loc="center left")
ax.set_title("Gauge invariance under hidden-unit permutations x per-unit scalings (E2). Shipped in whitebox.invariants: I1-I3, I8, sorted N2/N3/N6")
shipped = {"logits_tt", "input_grad_tt", "W_product", "tc0_box_std", "tc1_box_std", "gradxact_sorted", "ablation_sorted", "gates_sorted"}
for i, n in enumerate(names):
    if n in shipped: ax.get_xticklabels()[i].set_color("darkgreen"); ax.get_xticklabels()[i].set_fontweight("bold")
fig.tight_layout(); fig.savefig(f"{P6}/phase6_invariance_cv.png", dpi=300); plt.close(fig); print("phase6_invariance_cv.png")

# ---------------------------------------------------------------- 2. legibility phase diagram
m = read(f"{P6}/e4_models.csv")
mk = {"mlp": "o", "tf2": "s", "tf4": "^"}; kinds = {"trained": 1.0, "randlabel": 0.5, "untrained": 0.3, "blockmask": 0.3}
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for j, (xcol, xl) in enumerate((("tw10_norm", "treewidth bound / N (10 % edge density)"), ("sparsity", "effective node sparsity 1 - PR/N"))):
    for arch in mk:
        rs = [r for r in m if r["arch"] == arch]
        sc = ax[j].scatter([float(r[xcol]) for r in rs], [float(r["Q10"]) for r in rs], c=[float(r["S"]) for r in rs], marker=mk[arch], s=45,
                           cmap="viridis", vmin=0, vmax=1, edgecolors=["k" if r["kind"] == "trained" else "r" for r in rs], linewidths=0.6, label=arch)
    ax[j].set_xlabel(xl); ax[j].set_ylabel("modularity Q10 (Leiden)"); ax[j].legend(fontsize=8)
cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02); cb.set_label("discovery success S = 1 - |C|/N at F >= 0.9 (guarded)")
t = read(f"{P6}/e4_tests.csv"); h = next(r for r in t if r["test"].startswith("H-E4(i)")); tw = next(r for r in t if r["test"].startswith("H-E4b rho(tw/N,S)"))
fig.suptitle(f"E4 legibility phase diagram, 87 models (red edge = control). rho(Q10, S) = {float(h['rho']):+.2f} [{float(h['lo']):.2f}, {float(h['hi']):.2f}] pooled — "
             f"an architecture offset (partial rho | arch = +0.03 [-0.26, 0.33]); rho(tw/N, S) = {float(tw['rho']):+.2f} [{float(tw['lo']):.2f}, {float(tw['hi']):.2f}]. No calibrated score.", fontsize=9)
fig.savefig(f"{P6}/phase6_legibility_phase.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("phase6_legibility_phase.png")

# ---------------------------------------------------------------- 3. standardisation vs memorisation AUC
rp = f"{P6}/phase6_standardize_report.json"
if os.path.exists(rp):
    R = json.load(open(rp)); T = R["table"]
    fams = [("cnn_classifier", "pool"), ("cnn_classifier", "layer3"), ("lm_transformer", "L12"), ("lm_transformer", "L6")]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True)
    for ax, (fam, lay) in zip(axs, fams):
        rs = [r for r in T if r["family"] == fam and r["layer"] == lay]; xx = np.arange(len(rs))
        for i, (var, c) in enumerate((("raw", "C0"), ("std", "C2"))):
            v = np.array([r[f"auc_loso_{var}"] for r in rs]); lo = np.array([r[f"auc_loso_{var}_lo"] for r in rs]); hi = np.array([r[f"auc_loso_{var}_hi"] for r in rs])
            ax.bar(xx + (i - 0.5) * 0.38, v, 0.38, color=c, label="raw (native coordinates)" if var == "raw" else "per-unit standardised (gauge-invariant)",
                   yerr=[np.maximum(v - lo, 0), np.maximum(hi - v, 0)], error_kw=dict(lw=0.6, capsize=2))
        for i, r in enumerate(rs):
            if r["shipped"]: ax.text(i, 1.03, "shipped", ha="center", fontsize=7, color="k")
        ax.axhline(0.5, color="k", ls=":", lw=0.7); ax.set_xticks(xx); ax.set_xticklabels([r["features"] for r in rs], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{fam} / {lay} (n={rs[0]['n']})", fontsize=9); ax.set_ylim(0, 1.1)
    axs[0].set_ylabel("leave-one-seed-out AUC (95 % bootstrap over models)"); axs[0].legend(fontsize=7, loc="lower left")
    d = R["decision"]
    fig.suptitle("Does the E2 gauge repair cost memorization_signal its power? " + "; ".join(
        f"{f}: raw {v['auc_loso_raw']:.2f} vs std {v['auc_loso_std']:.2f} -> {'standardised keeps AUC' if v['standardized_keeps_auc'] else 'standardised LOSES AUC'}" for f, v in d.items()), fontsize=9)
    fig.tight_layout(); fig.savefig(f"{P6}/phase6_standardize_auc.png", dpi=300); plt.close(fig); print("phase6_standardize_auc.png")
else:
    print("standardisation report not found; run phase6_standardize_memorization.py first")
