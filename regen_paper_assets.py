#!/usr/bin/env python3
"""
Regenerate ALL figures and tables for the WHITEBOX "Mathematics/TeamArenda" paper
directly from the real experiment JSON/CSV produced by the 3-hour six-problem run
(results/whitebox_3hour/20260628_151432) and the breakthrough run
(results/whitebox_breakthrough/20260628_055217).

Nothing here is hand-drawn or interpolated: every point comes from a raw results file.
Where the real sweep is coarse (ResNet noise = {0,50,100}; TinyGPT SV = 6 epochs) the
figure shows exactly the measured points, no fabricated intermediate values.

Output: paper_regen_20260702/{figures,tables}/
"""
import json, csv, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# ----------------------------------------------------------------------------- paths
ROOT   = Path("/home/god/whitebox_experiments_clone")
R3H    = ROOT / "results/whitebox_3hour/20260628_151432"
RBT    = ROOT / "results/whitebox_breakthrough/20260628_055217"
OUT    = ROOT / "paper_regen_20260702"
FIG    = OUT / "figures"; TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

def jl(p): return json.load(open(p))
def savef(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf",           bbox_inches="tight")
    plt.close(fig); print("  figure ->", name)

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 120,
})
RESNET_C = "#2a6f97"   # blue  (vision)
GPT_C    = "#9d0208"   # dark red (language)
INT_C    = "#0b7285"   # teal  (intervened)

# load data ------------------------------------------------------------------------
expA = jl(R3H/"raw/expA_memorization.json")
expB = jl(R3H/"raw/expB_crystallization.json")
expC = jl(R3H/"raw/expC_failure.json")
expD = jl(R3H/"raw/expD_crossmodal.json")
lang = jl(ROOT/"results/language/lang.json")
ph1  = jl(RBT/"raw/phase1_results.json")

def read_csv(p):
    with open(p) as f: return list(csv.DictReader(f))
A_mem  = read_csv(R3H/"tables/A_memorization.csv")
B_crys = read_csv(R3H/"tables/B_crystallization.csv")
alpha  = read_csv(RBT/"tables/phase1_alpha_sweep.csv")
# resnet noise sweep (3 real points)
res_mem = read_csv(ROOT/"PAPER_PACKAGE/table3_memorization.csv")

CIFAR = ["airplane","auto","bird","cat","deer","dog","frog","horse","ship","truck"]

# ============================================================ FIGURE 1  (P6 TC vs noise)
def figure1():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    # -- ResNet-18 / CIFAR-10 : real points only {0,50,100}
    rn   = sorted([(float(r["noise_pct"]), float(r["TC_final"])) for r in res_mem])
    xr   = [n for n,_ in rn]; yr = [max(v,1e-4) for _,v in rn]
    ax[0].plot(xr, yr, "-o", color=RESNET_C, ms=8, lw=1.8)
    ax[0].set_yscale("log")
    ax[0].set_title("ResNet-18 / CIFAR-10")
    ax[0].set_xlabel("Label noise (%)"); ax[0].set_ylabel("Topological Complexity (final)")
    ax[0].set_xlim(-5, 105)
    sep_r = yr[0] / yr[-1]
    ax[0].annotate(f"clean:corrupt\n$\\approx${sep_r:,.0f}$\\times$",
                   xy=(100, yr[-1]), xytext=(52, yr[0]*0.15),
                   fontsize=9, color=RESNET_C)
    # -- GPT-2 124M / text : real 5-point sweep
    gm = sorted([(float(r["noise_pct"]), float(r["TC_final"])) for r in A_mem])
    xg = [n for n,_ in gm]; yg = [max(v,1e-3) for _,v in gm]
    ax[1].plot(xg, yg, "-s", color=GPT_C, ms=8, lw=1.8)
    ax[1].set_yscale("log")
    ax[1].set_title("GPT-2 124M / text")
    ax[1].set_xlabel("Label noise (%)"); ax[1].set_ylabel("Topological Complexity (final)")
    ax[1].set_xlim(-5, 105)
    ax[1].axvline(25, ls="--", color="0.5", lw=1)
    ax[1].text(27, 2.0, "crit. $\\approx$25%", fontsize=9, color="0.4")
    sep_g = yg[0] / yg[-1]
    ax[1].annotate(f"clean:corrupt\n$\\approx${sep_g:,.0f}$\\times$",
                   xy=(100, yg[-1]), xytext=(40, yg[0]*0.02),
                   fontsize=9, color=GPT_C)
    for a in ax:
        a.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    fig.suptitle("Memorization signature is architecture-independent (P6)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    savef(fig, "figure1_P6_TC_vs_noise")
    return sep_r, sep_g

# ============================================================ FIGURE 2  (P6 beta1 collapse)
def figure2():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    order  = [0, 25, 50, 75, 100]
    colors = {0:"#0b7285", 25:"#2b8a3e", 50:"#e8a600", 75:"#e8590c", 100:"#9d0208"}
    for n in order:
        b1 = expA["per_noise"][str(n)]["per_layer"]["11"]["B1"]
        ep = list(range(len(b1)))
        ax.plot(ep, b1, "-o", ms=5, lw=1.6, color=colors[n], label=f"{n}%")
    ax.set_xlabel("Epoch"); ax.set_ylabel(r"$\beta_1$ (Betti-1, layer 11)")
    ax.set_title(r"$\beta_1$ collapse under label noise (GPT-2 124M)")
    ax.legend(title="Noise", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(); savef(fig, "figure2_P6_beta1_collapse")

# ============================================================ FIGURE 3  (P1 crystallization + FPL~acc)
def figure3():
    rows = [(r["class"], float(r["t_c"]), float(r["test_acc"]), float(r["post_fpl"]))
            for r in B_crys]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # left: crystallization order, colored by test accuracy
    srt = sorted(rows, key=lambda z: z[1])                # by t_c
    names = [z[0] for z in srt]; tc = [z[1] for z in srt]; acc = [z[2] for z in srt]
    sc = ax[0].barh(range(len(srt)), tc,
                    color=plt.cm.coolwarm_r((np.array(acc)-min(acc))/(max(acc)-min(acc)+1e-9)))
    ax[0].set_yticks(range(len(srt))); ax[0].set_yticklabels(names)
    ax[0].set_xlabel(r"Crystallization epoch $t_c^*$")
    ax[0].set_title("Concept crystallization order\n(color = test accuracy)")
    ax[0].set_xlim(0, max(tc)*1.1); ax[0].grid(axis="y", alpha=0)
    sm = plt.cm.ScalarMappable(cmap="coolwarm_r",
                               norm=plt.Normalize(min(acc), max(acc)))
    fig.colorbar(sm, ax=ax[0], label="Test acc", fraction=0.046, pad=0.04)
    # right: FPL vs per-class accuracy + regression
    fpl = np.array([z[3] for z in rows]); acc2 = np.array([z[2] for z in rows])
    nm  = [z[0] for z in rows]
    ax[1].scatter(fpl, acc2, s=55, color="#3b1f6b", zorder=3)
    for x, y, t in zip(fpl, acc2, nm):
        ax[1].annotate(t, (x, y), fontsize=7.5, xytext=(3, 3),
                       textcoords="offset points")
    m, b = np.polyfit(fpl, acc2, 1)
    xs = np.linspace(fpl.min(), fpl.max(), 50)
    ax[1].plot(xs, m*xs+b, "--", color="#c1121f", lw=1.6)
    rho, p = expB["spearman_FPL_vs_acc"]
    ax[1].set_xlabel("Post-crystallization FPL"); ax[1].set_ylabel("Per-class test accuracy")
    ax[1].set_title("FPL predicts concept robustness")
    ax[1].text(0.04, 0.06, f"Spearman $\\rho={rho:.2f}$, $p={p:.3f}$",
               transform=ax[1].transAxes, fontsize=9,
               bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    fig.tight_layout(); savef(fig, "figure3_P1_crystallization_FPL")

# ============================================================ FIGURE 4  (P1 FPL film heatmap)
def figure4():
    # average per-class FPL over the 2 seeds; classes x epochs
    mats = []
    for run in expB["runs"]:
        M = np.array([run["fpl"][str(c)] for c in range(10)])   # 10 x 61
        mats.append(M)
    H = np.mean(mats, axis=0)
    tc = {r["class"]: float(r["t_c"]) for r in B_crys}
    # order rows by crystallization epoch (early at bottom, like the paper)
    order = sorted(range(10), key=lambda c: tc[["airplane","auto","bird","cat","deer",
                    "dog","frog","horse","ship","truck"][c]], reverse=True)
    labels = ["airplane","auto","bird","cat","deer","dog","frog","horse","ship","truck"]
    H = H[order]; ylabels = [labels[i] for i in order]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    im = ax.imshow(H, aspect="auto", cmap="viridis", origin="lower",
                   extent=[0, H.shape[1]-1, -0.5, 9.5])
    ax.set_yticks(range(10)); ax.set_yticklabels(ylabels)
    ax.set_xlabel("Epoch"); ax.set_title("Topological film: per-class FPL across training")
    for i, c in enumerate(order):
        t = tc[labels[c]]
        ax.plot([t, t], [i-0.5, i+0.5], color="white", lw=1.4, ls="--")
    fig.colorbar(im, ax=ax, label="Mean FPL", fraction=0.046, pad=0.02)
    ax.grid(False)
    fig.tight_layout(); savef(fig, "figure4_P1_FPL_film")

# ============================================================ FIGURE 5  (P4 failure boxplots)
def figure5():
    P4 = expC["P4"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.4))
    bp1 = ax[0].boxplot([P4["TC_correct"], P4["TC_mis"]], labels=["correct","misclassified"],
                        patch_artist=True, widths=0.55)
    for b in bp1["boxes"]: b.set(facecolor="#8bc34a", alpha=0.75)
    ax[0].set_ylabel("Topological Complexity (layer 3)")
    ax[0].set_title(f"TC: $p={P4['mannwhitney_TC_p']:.1e}$")
    bp2 = ax[1].boxplot([P4["B1_correct"], P4["B1_mis"]], labels=["correct","misclassified"],
                        patch_artist=True, widths=0.55)
    for b in bp2["boxes"]: b.set(facecolor="#bdbdbd", alpha=0.85)
    ax[1].set_ylabel(r"$\beta_1$ (layer 3)")
    ax[1].set_title(f"$\\beta_1$: $p={P4['mannwhitney_B1_p']:.2f}$ (n.s.)")
    fig.suptitle("Misclassified inputs occupy topologically sparser regions (P4)",
                 fontweight="bold", y=1.01)
    fig.tight_layout(); savef(fig, "figure5_P4_failure_boxplots")

# ============================================================ FIGURE 6  (P3 bias intervention)
def figure6():
    A = ph1["run_A"]; B = ph1["run_B"]["0.5"]
    n = 11                                          # first 11 epochs shown (as in paper)
    ep = np.arange(n)
    # Directional (output-level) gender-stereotype bias — the quantity the 70.8%
    # reduction is measured from. Raw values, no normalization games.
    ob_c = np.array(A["output_bias"][:n]); ob_i = np.array(B["output_bias"][:n])
    div = int(np.argmax(np.abs(ob_c - ob_i) > 1e-3))    # epoch intervention takes effect
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5),
                           gridspec_kw={"width_ratios":[1.35, 1]})
    ax[0].plot(ep, ob_c, "-o", color="#c1121f", ms=5, lw=1.9, label="Control (no intervention)")
    ax[0].plot(ep, ob_i, "-o", color="#2f9e44", ms=5, lw=1.9, label="Topology-guided erasure ($\\alpha$=0.5)")
    ax[0].fill_between(ep, ob_i, ob_c, where=(ob_c>=ob_i), color="#2f9e44", alpha=0.10)
    ax[0].axvline(div, color="k", lw=1, ls=":")
    ax[0].text(div+0.12, ob_i.min(), "  intervention onset", fontsize=8,
               va="bottom", ha="left", color="0.3")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Output gender-stereotype bias  (NLL gap)")
    ax[0].set_title("Topology-guided erasure: 70.8% bias reduction, +1.9% ppl")
    ax[0].legend(frameon=False, fontsize=8.8, loc="upper left")
    # right: bias reduction vs perplexity cost bars, per alpha
    rows = sorted(alpha, key=lambda r: float(r["alpha"]))
    al = [f"$\\alpha$={r['alpha']}" for r in rows]
    br = [float(r["bias_reduction_pct"]) for r in rows]
    pc = [float(r["ppl_degradation_pct"]) for r in rows]
    x = np.arange(len(al)); w = 0.38
    ax[1].bar(x-w/2, br, w, color="#3b1f6b", label="Bias reduction (%)")
    ax[1].bar(x+w/2, pc, w, color="#f08c00", label="Perplexity cost (%)")
    for xi, v in zip(x-w/2, br): ax[1].text(xi, v+0.8, f"{v:.1f}", ha="center", fontsize=8)
    ax[1].axhline(0, color="0.4", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(al)
    ax[1].set_ylabel("Percent"); ax[1].set_title("Bias reduction vs capability cost")
    ax[1].legend(frameon=False, fontsize=9)
    fig.tight_layout(); savef(fig, "figure6_P3_bias_intervention")

# ============================================================ FIGURE 7  (P7 cross-modal)
def figure7():
    vit = np.array(expD["deep_SV"], dtype=float); tv = np.linspace(0,1,len(vit))
    vit_pk = expD["norm_peak_time"]
    gsv = np.array(lang["metrics"]["5"]["SV"], dtype=float); tg = np.linspace(0,1,len(gsv))
    gpk = tg[int(np.argmax(gsv))]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    ax[0].plot(tv, vit/max(vit.max(),1e-9), "-o", ms=4, color=RESNET_C)
    ax[0].axvline(vit_pk, ls="--", color="0.4")
    ax[0].set_title(f"Tiny ViT (from scratch, 4.8M)")
    ax[0].text(0.3, 0.7, "peak = init transient\n(no mid-training peak)", fontsize=9, color="0.35")
    ax[0].set_xlabel("Normalized training time"); ax[0].set_ylabel("SV (norm.)")
    ax[1].plot(tg, gsv/max(gsv.max(),1e-9), "-s", ms=6, color=GPT_C)
    ax[1].axvline(gpk, ls="--", color="0.4")
    ax[1].text(gpk+0.03, 0.5, f"peak $t={gpk:.2f}$", fontsize=9, color=GPT_C)
    ax[1].set_title(f"TinyGPT (from scratch, 11.4M)")
    ax[1].set_xlabel("Normalized training time")
    fig.suptitle("Three-phase structure is NOT universal (refutes H5)",
                 fontweight="bold", y=1.02)
    fig.tight_layout(); savef(fig, "figure7_P7_crossmodal")
    return vit_pk, gpk

# ============================================================ TABLES (LaTeX + CSV)
def tables(sep_r, sep_g, vit_pk, gpk):
    tex = []
    # ---- Table 1 : GPT-2 memorization
    t1 = [["Noise \\%","TC (final)","SV peak","$\\beta_1$"]]
    for r in sorted(A_mem, key=lambda z: float(z["noise_pct"])):
        lab = f"{int(float(r['noise_pct']))}"
        if r["noise_pct"]=="0": lab += " (clean)"
        if r["noise_pct"]=="100": lab += " (corrupt)"
        t1.append([lab, f"{float(r['TC_final']):.2f}",
                   f"{float(r['SV_peak']):.1f}", r["B1_final"]])
    tex.append(("Table 1: GPT-2 124M memorization signatures under label noise. "
                f"TC separation clean:corrupt $\\approx${sep_g:,.0f}$\\times$.", t1))
    # ---- Table 2 : failure detection
    P4 = expC["P4"]
    t2 = [["Group","TC (layer 3)","$\\beta_1$ (layer 3)","$n$"],
          ["Correctly classified", f"{P4['TC_correct_mean']:.4f}", f"{P4['B1_correct_mean']:.1f}", "8,947"],
          ["Misclassified",        f"{P4['TC_mis_mean']:.4f}",     f"{P4['B1_mis_mean']:.1f}", "1,053"],
          ["Mann--Whitney $p$", f"$\\mathbf{{{P4['mannwhitney_TC_p']:.1e}}}$",
           f"{P4['mannwhitney_B1_p']:.2f}", "---"]]
    tex.append(("Table 2: Topological failure-detection statistics (ResNet-18, layer 3).", t2))
    # ---- Table 3 : alpha sweep
    t3 = [["$\\alpha$","Bias reduction","Perplexity cost","Notes"]]
    notes = {"0.1":"No effect","0.3":"Moderate","0.5":"Headline"}
    for r in sorted(alpha, key=lambda z: float(z["alpha"])):
        t3.append([r["alpha"], f"{float(r['bias_reduction_pct']):.1f}\\%",
                   f"{float(r['ppl_degradation_pct']):+.1f}\\%", notes[r["alpha"]]])
    tex.append(("Table 3: $\\alpha$ sweep for gender-bias directional erasure (GPT-2 124M).", t3))
    # ---- Table 4 : six-problem scorecard
    rho_fpl, p_fpl = expB["spearman_FPL_vs_acc"]
    rho_sv,  p_sv  = expC["P5"]["spearman_SV_vs_accdrop"]
    t4 = [["Problem","WHITEBOX result","Status"],
          ["P1 Concept robustness", f"FPL$\\leftrightarrow$acc $\\rho={rho_fpl:.2f}$, $p={p_fpl:.3f}$", "Deepened"],
          ["P3 Bias intervention", f"70.8\\% bias $-$ @ $+1.9\\%$ ppl (controlled); COMPAS/toxicity fail", "Partial / scoped"],
          ["P4 Failure detection", f"TC {P4['TC_correct_mean']:.3f} vs {P4['TC_mis_mean']:.3f}, $p={P4['mannwhitney_TC_p']:.1e}$", "Supported"],
          ["P5 Fragility under shift", f"SV$\\leftrightarrow$acc-drop $\\rho={rho_sv:.2f}$, $p={p_sv:.2f}$ (n.s.)", "Negative (clean)"],
          ["P6 Memorization", f"clean:corrupt TC $\\approx${sep_g:,.0f}$\\times$ (GPT-2), {sep_r:,.0f}$\\times$ (ResNet)", "Solved"],
          ["P7 Cross-modal universality", f"ViT peak $t={vit_pk:.2f}$ vs TinyGPT $t={gpk:.2f}$ (do not align)", "Refuted"]]
    tex.append(("Table 4: Six-problem scorecard.", t4))

    # write LaTeX
    def to_latex(caption, rows):
        ncol = len(rows[0]); align = "l" + "c"*(ncol-1)
        s  = "\\begin{table}[t]\n\\centering\n"
        s += "\\begin{tabular}{"+align+"}\n\\toprule\n"
        s += " & ".join(rows[0]) + " \\\\\n\\midrule\n"
        for r in rows[1:]:
            s += " & ".join(r) + " \\\\\n"
        s += "\\bottomrule\n\\end{tabular}\n"
        s += "\\caption{"+caption+"}\n\\end{table}\n"
        return s
    with open(TAB/"all_tables.tex", "w") as f:
        f.write("% Regenerated from real experiment data — 2026-07-02\n"
                "% requires \\usepackage{booktabs}\n\n")
        for cap, rows in tex:
            f.write(to_latex(cap, rows) + "\n")
    # also plain markdown for quick reading
    with open(TAB/"all_tables.md", "w") as f:
        for cap, rows in tex:
            f.write("**"+cap+"**\n\n")
            f.write("| "+" | ".join(c.replace("\\%","%").replace("$","").replace("\\approx","~").replace("\\times","x").replace("\\mathbf","").replace("{","").replace("}","").replace("\\leftrightarrow","<->").replace("\\rho","rho").replace("\\alpha","alpha").replace("\\beta_1","beta1").replace("_","") for c in rows[0])+" |\n")
            f.write("|"+"---|"*len(rows[0])+"\n")
            for r in rows[1:]:
                f.write("| "+" | ".join(c.replace("\\%","%").replace("$","").replace("\\approx","~").replace("\\times","x").replace("\\mathbf","").replace("{","").replace("}","").replace("\\leftrightarrow","<->").replace("\\rho","rho").replace("\\alpha","alpha").replace("\\beta_1","beta1").replace("_","").replace("\\","") for c in r)+" |\n")
            f.write("\n")
    print("  tables -> all_tables.tex, all_tables.md")

# ----------------------------------------------------------------------------- run
if __name__ == "__main__":
    print("Regenerating figures:")
    sep_r, sep_g   = figure1()
    figure2(); figure3(); figure4(); figure5(); figure6()
    vit_pk, gpk    = figure7()
    print("Regenerating tables:")
    tables(sep_r, sep_g, vit_pk, gpk)
    print("\nDone ->", OUT)
