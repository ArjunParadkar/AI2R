"""E5 analysis: H-E5a/b/c, kill attempts, figures. Reads results/e5_points_raw.csv, e5_nulls_raw.csv, e5_sweep.csv and
E4/results/e4_models.csv + e4_sweep.csv. NOT FOR EXTERNAL CLAIMS until cleared with the Fiete lab."""
import os, sys, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results"); E4R = os.path.join(os.path.dirname(HERE), "E4", "results")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C = dict(fly="#c8102e", rnn="#2a78d6", func="#7b3fe4", gru="#1baf7a", ring="#e0a800", ctrl="#888888", size="#eb6834")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300, "savefig.dpi": 300})
COORDS = ["Q10", "tw10_norm", "sparsity"]

def mahal(X, mu, Sinv):
    D = np.atleast_2d(X) - mu; return np.sqrt(np.einsum("ij,jk,ik->i", D, Sinv, D))

def main():
    e4 = pd.read_csv(os.path.join(E4R, "e4_models.csv")); e4sw = pd.read_csv(os.path.join(E4R, "e4_sweep.csv"))
    e5 = pd.read_csv(os.path.join(R, "e5_points_raw.csv")); nulls = pd.read_csv(os.path.join(R, "e5_nulls_raw.csv")); sw5 = pd.read_csv(os.path.join(R, "e5_sweep.csv"))
    for d in (5, 20):   # E4 tw at other densities for kill 5
        s = e4sw[np.isclose(e4sw.density, d / 100)].set_index("id"); e4[f"Q{d}"] = e4.id.map(s.Q); e4[f"tw{d}_norm"] = e4.id.map(s.tw) / e4.N
    e5["tw5_norm"] = e5.tw5 / e5.N; e5["tw20_norm"] = e5.tw20 / e5.N
    out, lines = [], []
    def rec(**k): out.append(k); lines.append(" | ".join(f"{a}={b:.4g}" if isinstance(b, (float, np.floating)) else f"{a}={b}" for a, b in k.items())); print(lines[-1])
    learned = e4[e4.learned.astype(bool)]
    for tag, coords in (("d10", COORDS), ("d5", ["Q5", "tw5_norm", "sparsity"]), ("d20", ["Q20", "tw20_norm", "sparsity"])):
        X4 = e4[coords].values.astype(float); mu4, sd4 = X4.mean(0), X4.std(0); Z4 = (X4 - mu4) / sd4
        Sigma = np.cov(Z4.T); Sinv = np.linalg.pinv(Sigma)
        z = lambda df: (df[coords].values.astype(float) - mu4) / sd4
        L = learned; q_hi, q_lo = np.percentile(L.S, [200 / 3, 100 / 3])
        Sset, Fset = L[L.S >= q_hi], L[L.S <= q_lo]; ZS, ZF = z(Sset), z(Fset)
        loo = np.array([mahal(ZS[i], np.delete(ZS, i, 0).mean(0), Sinv)[0] for i in range(len(ZS))]); loo95 = np.percentile(loo, 95)
        fly = e5[e5.id == "fly_core148"].iloc[0]; zf = z(e5[e5.id == "fly_core148"])
        dS, dF = mahal(zf, ZS.mean(0), Sinv)[0], mahal(zf, ZF.mean(0), Sinv)[0]
        dall = mahal(z(e4), zf[0], Sinv); nn5 = np.argsort(dall)[:5]; nn5_S = e4.S.values[nn5].mean(); nn5_ids = ";".join(e4.id.values[nn5])
        rec(test=f"H-E5a[{tag}]", d_fly_success=dS, loo95_success=loo95, d_fly_failure=dF, pass_i=bool(dS <= loo95), pass_ii=bool(dS < dF),
            PASS=bool(dS <= loo95 and dS < dF), n_success=len(Sset), n_failure=len(Fset), S_cut_hi=q_hi, S_cut_lo=q_lo, nn5_meanS=nn5_S, learned_meanS=L.S.mean(), nn5=nn5_ids)
        # within-architecture scatter threshold
        meds = []
        for a in ("mlp", "tf2", "tf4"):
            Za = z(L[L.arch == a]); D = np.array([mahal(Za[i], Za[j], Sinv)[0] for i in range(len(Za)) for j in range(i + 1, len(Za))]); meds.append(np.median(D))
        thr = float(np.median(meds))
        rnn = e5[(e5.kind == "trained") & (e5.cell == "rnn") & (e5.N == 148) & (e5.graph == "weight")]
        for gname, sub in (("weight", rnn), ("functional", e5[(e5.kind == "trained") & (e5.cell == "rnn") & (e5.N == 148) & (e5.graph == "functional")])):
            cen = z(sub).mean(0); d_cf = mahal(cen[None], zf[0], Sinv)[0]
            pts = np.vstack([z(e4), zf]); dc = mahal(pts, cen, Sinv); rank_fly = int(np.argsort(dc).tolist().index(len(pts) - 1)) + 1
            d_seeds = mahal(z(sub), zf[0], Sinv)
            dS_c, dF_c = mahal(cen[None], ZS.mean(0), Sinv)[0], mahal(cen[None], ZF.mean(0), Sinv)[0]
            rec(test=f"H-E5b[{tag},{gname}]", d_centroid_fly=d_cf, threshold_within_arch_median=thr, PASS=bool(d_cf < thr), fly_rank_among_88=rank_fly, nn_pass=bool(rank_fly <= 5),
                d_seed_fly_mean=d_seeds.mean(), d_seed_fly_min=d_seeds.min(), d_seed_fly_max=d_seeds.max(), n_seeds=len(sub), n_learned=int(sub.learned.sum()),
                d_centroid_success=dS_c, d_centroid_failure=dF_c, centroid_in_success_region=bool(dS_c <= loo95 and dS_c < dF_c))
        # H-E5c weight vs functional per seed
        w = e5[(e5.graph == "weight") & (e5.kind != "connectome") & (e5.kind != "planted_ring")].set_index("id"); f = e5[e5.graph == "functional"].set_index("id")
        ids = [i for i in w.index if i in f.index]; Zw, Zf = z(w.loc[ids]), z(f.loc[ids]); dwf = np.sqrt(np.einsum("ij,jk,ik->i", Zw - Zf, Sinv, Zw - Zf))
        main_ids = [i for i in ids if i.startswith("trained|rnn|N148")]; sel = [ids.index(i) for i in main_ids]
        diffs = (f.loc[main_ids][coords].values - w.loc[main_ids][coords].values).astype(float)
        rec(test=f"H-E5c[{tag}]", frac_agree_main=float(np.mean(dwf[sel] < thr)), d_wf_mean_main=float(dwf[sel].mean()), threshold=thr,
            frac_agree_all=float(np.mean(dwf < thr)), dQ10_func_minus_weight=float(diffs[:, 0].mean()), dtw_norm=float(diffs[:, 1].mean()), dsparsity=float(diffs[:, 2].mean()),
            edge_rank_rho=float(w.loc[main_ids].edge_rank_rho_w_vs_f.mean()))
        if tag == "d10":
            ctx = dict(Sinv=Sinv, mu4=mu4, sd4=sd4, thr=thr, loo95=loo95, ZS=ZS, ZF=ZF, dwf=dict(zip(ids, dwf)))
    # ---- kills ----
    fly = e5[e5.id == "fly_core148"].iloc[0]
    for nm in ("rewire_directed", "rewire_undirected_top10", "ER_matched"):
        nn = nulls[(nulls.target == "fly_core148") & (nulls.null == nm)]
        for col in ("Q10", "tw10_norm"):
            mu, sd = nn[col].mean(), nn[col].std(); zsc = (fly[col] - mu) / (sd + 1e-12)
            p = (np.sum(nn[col] >= fly[col]) + 1) / (len(nn) + 1) if col == "Q10" else (np.sum(nn[col] <= fly[col]) + 1) / (len(nn) + 1)
            rec(test=f"kill_fly_{nm}", coord=col, observed=float(fly[col]), null_mean=mu, null_sd=sd, z=zsc, p_onesided=p, n=len(nn))
    rw = e5[(e5.graph == "weight") & (e5.kind == "trained") & (e5.N == 148) & (e5.cell == "rnn")]
    rec(test="kill_rnn_rewire", zQ_mean=rw.rewire_Q_z.mean(), zQ_min=rw.rewire_Q_z.min(), zQ_max=rw.rewire_Q_z.max(), ztw_mean=rw.rewire_tw_z.mean(), frac_seeds_zQ_gt2=float((rw.rewire_Q_z > 2).mean()),
        Q10_mean=rw.Q10.mean(), Q10_rewire_mean=rw.rewire_Q_mean.mean())
    rec(test="kill_fly_label_shuffle_NMI", nmi_type=float(fly.nmi_type), nmi_type_null=float(fly.nmi_type_null_mean), p_type=float(fly.nmi_type_p),
        nmi_wedge=float(fly.nmi_wedge), nmi_wedge_null=float(fly.nmi_wedge_null_mean), p_wedge=float(fly.nmi_wedge_p), n_comm=int(fly.n_comm10))
    van = e5[(e5.graph == "weight") & (e5.kind == "trained") & (e5.cell == "rnn")]
    Sinv, mu4, sd4 = ctx["Sinv"], ctx["mu4"], ctx["sd4"]; zf = ((fly[COORDS].values.astype(float) - mu4) / sd4)
    for col in COORDS + ["Q10_excess", "n_edges10"]:
        rec(test="kill_size_arms", coord=col, spearman_vs_N=float(stats.spearmanr(van.N, van[col]).statistic), mean_N64=float(van[van.N == 64][col].mean()), mean_N148=float(van[van.N == 148][col].mean()), mean_N256=float(van[van.N == 256][col].mean()))
    for n in (64, 148, 256):
        Zc = ((van[van.N == n][COORDS].values.astype(float) - mu4) / sd4).mean(0); rec(test="kill_size_arms_distance", N=n, d_centroid_fly=float(mahal(Zc[None], zf, Sinv)[0]))
    fm = e5[e5.graph == "weight"].set_index("id")
    rec(test="matched_edge_count", fly_n_edges10=int(fly.n_edges10), rnn_n_edges10=float(rw.n_edges10.mean()), rnn_Q_matched=float(rw.Q_matched.mean()), rnn_Q10=float(rw.Q10.mean()),
        rnn_tw_matched_norm=float(rw.tw_matched_norm.mean()), rnn_tw10_norm=float(rw.tw10_norm.mean()), fly_Q10=float(fly.Q10), fly_tw10_norm=float(fly.tw10_norm))
    full = e5[e5.id == "fly_full439"].iloc[0]; zfull = ((full[COORDS].values.astype(float) - mu4) / sd4)
    rec(test="sensitivity_full439", Q10=float(full.Q10), tw10_norm=float(full.tw10_norm), sparsity=float(full.sparsity), d_full_core=float(mahal(zfull[None], zf, Sinv)[0]), d_full_success=float(mahal(zfull[None], ctx["ZS"].mean(0), Sinv)[0]), d_full_failure=float(mahal(zfull[None], ctx["ZF"].mean(0), Sinv)[0]))
    for kind, cell, n in (("untrained", "rnn", 148), ("shuffled_target", "rnn", 148), ("trained", "gru", 148), ("planted_ring", "ring", 148)):
        sub = e5[(e5.kind == kind) & (e5.cell == cell) & (e5.graph == "weight")]
        if len(sub) == 0: continue
        Zc = ((sub[COORDS].values.astype(float) - mu4) / sd4).mean(0)
        rec(test="arm_placement", arm=f"{kind}/{cell}", n=len(sub), Q10=float(sub.Q10.mean()), tw10_norm=float(sub.tw10_norm.mean()), sparsity=float(sub.sparsity.mean()), d_centroid_fly=float(mahal(Zc[None], zf, Sinv)[0]),
            d_success=float(mahal(Zc[None], ctx["ZS"].mean(0), Sinv)[0]), d_failure=float(mahal(Zc[None], ctx["ZF"].mean(0), Sinv)[0]), S_raw=float(sub.S_raw.mean()) if "S_raw" in sub and sub.S_raw.notna().any() else np.nan)
    rec(test="rnn_discovery", n=len(rw), S_raw_mean=float(rw.S_raw.mean()), S_raw_sd=float(rw.S_raw.std()), F_mean=float(rw.F.mean()), jaccard_mean=float(rw.jaccard.mean()), cost_over_N=float((rw.cost / rw.N).mean()), acc_tail_mean=float(rw.acc_tail.mean()),
        e4_learned_S_mean=float(learned.S.mean()), e4_S_cut_hi=float(np.percentile(learned.S, 200 / 3)))
    pd.DataFrame(out).to_csv(os.path.join(R, "e5_tests.csv"), index=False); open(os.path.join(R, "analyze_stdout.txt"), "w").write("\n".join(lines))
    # combined points table
    e4c = e4[["id", "arch", "task", "arm", "kind", "N", "Q10", "tw10", "tw10_norm", "Q10_null", "Q10_excess", "sparsity", "edge_sparsity", "Q_auc", "S", "S_raw", "F", "jaccard", "learned"]].copy(); e4c["substrate"] = "E4"; e4c["graph"] = "attribution"
    e5c = e5.copy(); e5c["substrate"] = np.where(e5c.kind == "connectome", "fly", np.where(e5c.kind == "planted_ring", "planted", "rnn"))
    pd.concat([e5c, e4c], ignore_index=True).to_csv(os.path.join(R, "e5_points.csv"), index=False)
    nulls.groupby(["target", "null"]).agg(Q10_mean=("Q10", "mean"), Q10_sd=("Q10", "std"), tw10_mean=("tw10", "mean"), tw10_sd=("tw10", "std"), n=("k", "count")).reset_index().to_csv(os.path.join(R, "e5_nulls.csv"), index=False)
    figures(e4, e5, nulls, sw5, e4sw, ctx)

def figures(e4, e5, nulls, sw5, e4sw, ctx):
    fly = e5[e5.id == "fly_core148"].iloc[0]; full = e5[e5.id == "fly_full439"].iloc[0]; ring = e5[e5.id == "planted_ring148"].iloc[0]
    W = e5[e5.graph == "weight"]; Fn = e5[e5.graph == "functional"]
    main = W[(W.kind == "trained") & (W.cell == "rnn") & (W.N == 148)]; mainf = Fn[(Fn.kind == "trained") & (Fn.cell == "rnn") & (Fn.N == 148)]
    gru = W[(W.cell == "gru")]; sz = W[(W.kind == "trained") & (W.cell == "rnn") & (W.N != 148)]; unt = W[W.kind == "untrained"]; shf = W[W.kind == "shuffled_target"]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, yv, yl in ((axs[0], "tw10_norm", "treewidth bound / N (top-10% graph)"), (axs[1], "sparsity", "effective sparsity (1 − PR/N)")):
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        L = e4[e4.learned.astype(bool)]; ctr = e4[~e4.learned.astype(bool)]
        sc = ax.scatter(L.Q10, L[yv], c=L.S, cmap="Greys", vmin=-0.2, vmax=1, s=26, edgecolor=INK2, lw=0.3, zorder=2, label="E4 ensemble (grey = S)")
        ax.scatter(ctr.Q10, ctr[yv], marker="x", c="#b0b0b0", s=18, lw=0.7, zorder=2, label="E4 controls")
        ax.scatter(sz.Q10, sz[yv], marker="o", facecolor="none", edgecolor=C["size"], s=30, lw=0.9, zorder=3, label="HD RNN N=64/256 (weight)")
        ax.scatter(unt.Q10, unt[yv], marker="P", c=C["ctrl"], s=34, zorder=3, label="untrained RNN"); ax.scatter(shf.Q10, shf[yv], marker="X", c=C["ctrl"], s=34, zorder=3, label="shuffled-target RNN")
        ax.scatter(gru.Q10, gru[yv], marker="D", c=C["gru"], s=30, zorder=3, label="HD GRU N=148 (weight)")
        for _, a in main.iterrows():
            b = mainf[mainf.id == a.id]
            if len(b): ax.plot([a.Q10, b.Q10.iloc[0]], [a[yv], b[yv].iloc[0]], color=C["func"], lw=0.5, alpha=0.5, zorder=2)
        ax.scatter(main.Q10, main[yv], marker="o", c=C["rnn"], s=42, edgecolor="white", lw=0.5, zorder=4, label="HD RNN N=148, weight graph (10 seeds)")
        ax.scatter(mainf.Q10, mainf[yv], marker="^", c=C["func"], s=42, edgecolor="white", lw=0.5, zorder=4, label="HD RNN N=148, functional (EAP) graph")
        ax.scatter([ring.Q10], [ring[yv]], marker="h", c=C["ring"], s=80, edgecolor=INK, lw=0.6, zorder=5, label="planted ring attractor")
        ax.scatter([full.Q10], [full[yv]], marker="*", facecolor="none", edgecolor=C["fly"], s=160, lw=1.0, zorder=5, label="fly HD full 439")
        ax.scatter([fly.Q10], [fly[yv]], marker="*", c=C["fly"], s=220, edgecolor=INK, lw=0.7, zorder=6, label="fly HD core (EPG/PEG/PEN/Δ7, N=148)")
        ax.set_xlabel("modularity Q₁₀ (Leiden, top-10% edges)"); ax.set_ylabel(yl)
    cb = fig.colorbar(sc, ax=axs, pad=0.02, fraction=0.03); cb.set_label("E4 discovery success S"); cb.outline.set_visible(False)
    axs[0].legend(frameon=False, fontsize=6.2, loc="upper left"); axs[0].set_title("E5: biology and task-matched RNNs on E4's legibility phase diagram", loc="left", fontsize=10, color=INK)
    fig.text(0.01, 0.005, "PRELIMINARY — NOT FOR EXTERNAL CLAIMS until cleared with the Fiete lab", fontsize=7, color=C["fly"])
    fig.savefig(os.path.join(R, "fig_e5_phase.png"), bbox_inches="tight"); plt.close(fig)
    # support
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
    ax = axs[0, 0]; ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for a, col in (("mlp", "#9e9e9e"), ("tf2", "#bdbdbd"), ("tf4", "#d6d6d6")):
        ids = e4[(e4.arch == a) & e4.learned.astype(bool)].id; s = e4sw[e4sw.id.isin(ids)].groupby("density").Q.mean(); ax.plot(s.index * 100, s.values, color=col, lw=1.2, label=f"E4 {a} (mean)")
    for lab, ids, col, ls in (("fly core", ["fly_core148"], C["fly"], "-"), ("fly full 439", ["fly_full439"], C["fly"], "--"), ("planted ring", ["planted_ring148"], C["ring"], "-"),
                              ("RNN weight", main.id, C["rnn"], "-"), ("RNN functional", main.id, C["func"], "-"), ("untrained RNN", unt.id, C["ctrl"], ":")):
        g = "functional" if lab == "RNN functional" else "weight"; s = sw5[(sw5.id.isin(ids)) & (sw5.graph == g)].groupby("density").Q.mean(); ax.plot(s.index * 100, s.values, color=col, ls=ls, lw=1.6, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("edge density kept (%)"); ax.set_ylabel("modularity Q_d"); ax.legend(frameon=False, fontsize=6.5); ax.set_title("Q sweep (E4 protocol)", loc="left", fontsize=9)
    ax = axs[0, 1]; ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for nm, col in (("rewire_directed", C["fly"]), ("rewire_undirected_top10", "#f08a9b"), ("ER_matched", C["ctrl"])):
        nn = nulls[(nulls.target == "fly_core148") & (nulls.null == nm)]; ax.hist(nn.Q10, bins=20, alpha=0.6, color=col, label=f"{nm} (n={len(nn)})")
    ax.axvline(fly.Q10, color=C["fly"], lw=2, label=f"fly core Q₁₀ = {fly.Q10:.2f}"); ax.axvline(main.rewire_Q_mean.mean(), color=C["rnn"], ls="--", lw=1.2, label="RNN rewired mean")
    ax.axvline(main.Q10.mean(), color=C["rnn"], lw=2, label=f"RNN weight Q₁₀ = {main.Q10.mean():.2f}"); ax.set_xlabel("Q₁₀"); ax.set_ylabel("count"); ax.legend(frameon=False, fontsize=6.5); ax.set_title("degree-preserving rewiring / ER nulls", loc="left", fontsize=9)
    ax = axs[1, 0]; ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    van = W[(W.kind == "trained") & (W.cell == "rnn")]
    for col, lab, mk in (("Q10", "Q₁₀", "o"), ("tw10_norm", "tw₁₀/N", "s"), ("sparsity", "sparsity", "^")):
        g = van.groupby("N")[col]; ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values, marker=mk, color=C["size"] if col == "Q10" else (C["rnn"] if col == "tw10_norm" else C["gru"]), capsize=2, label=f"RNN {lab}")
        ax.axhline(fly[col], color=C["fly"], ls=":" if col != "Q10" else "-", lw=1); ax.text(262, fly[col], f"fly {lab}", color=C["fly"], fontsize=6.5, va="center")
    ax.set_xscale("log"); ax.set_xticks([64, 148, 256]); ax.set_xticklabels([64, 148, 256]); ax.set_xlabel("RNN hidden size N"); ax.set_ylabel("coordinate"); ax.legend(frameon=False, fontsize=6.5); ax.set_title("size arms (kill: N confound)", loc="left", fontsize=9)
    ax = axs[1, 1]; ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    d = [ctx["dwf"].get(i, np.nan) for i in main.id]; ax.bar(range(len(d)), d, color=C["func"]); ax.axhline(ctx["thr"], color=INK2, ls="--", lw=1, label="H-E5b threshold (E4 within-arch median d_M)")
    ax.set_xticks(range(len(d))); ax.set_xticklabels([f"s{i}" for i in main.seed]); ax.set_ylabel("d_M(weight, functional)"); ax.legend(frameon=False, fontsize=6.5); ax.set_title("weight vs functional placement per seed (H-E5c)", loc="left", fontsize=9)
    fig.text(0.01, 0.005, "PRELIMINARY — NOT FOR EXTERNAL CLAIMS until cleared with the Fiete lab", fontsize=7, color=C["fly"])
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_e5_support.png"), bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__": main()
