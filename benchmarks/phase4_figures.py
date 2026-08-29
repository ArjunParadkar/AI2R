import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
from common import RESULTS
R = json.load(open(f"{RESULTS}/phase4/phase4_report.json"))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for i, (s, r) in enumerate(R.items()):
    a = r["A_failure"]; ax[0].bar([i - 0.2, i + 0.2], [a["auc_knn_radius"], a["auc_softmax_conf"]], 0.4, color=["C0", "C3"], label=["topology (kNN radius, layer3)", "softmax confidence"] if i == 0 else [None, None])
    ax[0].text(i, 0.95, f"perm p={a['permutation_p']:.2f}\nΔTC={100*a['rel_diff']:.0f}%", ha="center", fontsize=8)
    b = r["B_biastop"]["power_vs_mixing_fraction"]; ax[1].plot([float(k) for k in b], list(b.values()), "o-", label=f"{s} (FPR@0.05 = {r['B_biastop']['false_positive_rate_alpha05']:.2f})")
    c = r["C_fpl_vs_acc"]; ax[2].scatter(c["fpl"], c["acc"], label=f"{s}: ρ={c['spearman']:.2f} p={c['p']:.2f}")
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(list(R)); ax[0].set_ylim(0.4, 1.02); ax[0].set_ylabel("AUC for predicting a misclassification"); ax[0].legend(fontsize=7); ax[0].set_title("P4 failure neighbourhoods: topology vs confidence")
ax[1].set_xlabel("fraction of subgroup B drawn from another class"); ax[1].set_ylabel("power (p<0.05) at n=100/group"); ax[1].set_ylim(-0.02, 1.02); ax[1].legend(fontsize=7); ax[1].set_title("BiasTop permutation test: power")
ax[2].set_xlabel("per-class top-3 FPL (pool layer)"); ax[2].set_ylabel("per-class test accuracy"); ax[2].legend(fontsize=7); ax[2].set_title("P1: FPL vs per-class accuracy (full-data ResNets)")
fig.tight_layout(); fig.savefig(f"{RESULTS}/phase4/phase4_validation.png", dpi=300); print("ok")
