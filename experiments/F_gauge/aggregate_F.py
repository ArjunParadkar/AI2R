"""F4 (pooled orbits), F5 (hypotheses + figures), F6 (autopsy) from results/explanations.csv.
Both explanation definitions (handbook bit-at-0 'hb'; Méloux thresholds 'ml') are analysed."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(__file__), "results"); FIG = os.path.join(R, "fig")
os.makedirs(FIG, exist_ok=True)


def load():
    E = pd.read_csv(os.path.join(R, "explanations.csv"), dtype={"key6": str, "key4": str})
    sp = E.specimen.str.extract(r"^(?P<function>[A-Z0-9_]+)_d(?P<depth>\d)_w(?P<width>\d+)_s(?P<seed>\d+)$")
    E = pd.concat([E, sp], axis=1); E["depth"] = E.depth.astype(int); E["width"] = E.width.astype(int); E["seed"] = E.seed.astype(int)
    E["cell"] = E.function + "_d" + E.depth.astype(str) + "_w" + E.width.astype(str)
    st = pd.read_csv(os.path.join(R, "enumeration_status.csv"))
    return E, st


def orbits(df, definition, keycol="key6"):
    """N and K for a set of explanation rows. hb: interpretable circuits, weight 1 each.
    ml: every circuit, weight n_meloux; gauge-equivalent circuits share n_meloux (invariant)."""
    if definition == "hb":
        d = df[df.interpretable == 1]; w = np.ones(len(d))
    else:
        d = df[df.n_meloux > 0]; w = d.n_meloux.values.astype(float)
    N = float(w.sum())
    if N == 0:
        return 0.0, 0.0, 0
    g = pd.DataFrame({"k": d[keycol].values, "w": w}).groupby("k").w.max()
    return N, float(g.sum()), len(d)


def main():
    E, st = load()
    rows_w, rows_p = [], []
    for (spec, crit), d in E.groupby(["specimen", "criterion"]):
        r = dict(specimen=spec, criterion=crit, cell=d.cell.iloc[0], function=d.function.iloc[0], depth=d.depth.iloc[0],
                 width=d.width.iloc[0], seed=d.seed.iloc[0], n_minimal=len(d))
        for defn in ("hb", "ml"):
            N, K, nc = orbits(d, defn); r[f"N_{defn}"] = N; r[f"K_within_{defn}"] = K
            r[f"ratio_within_{defn}"] = K / N if N > 0 else np.nan
            N4, K4, _ = orbits(d, defn, "key4"); r[f"K_within_{defn}_eps4"] = K4
        rows_w.append(r)
    W = pd.DataFrame(rows_w); W.to_csv(os.path.join(R, "orbits.csv"), index=False)
    for (cell, crit), d in E.groupby(["cell", "criterion"]):
        r = dict(cell=cell, criterion=crit, function=d.function.iloc[0], depth=d.depth.iloc[0], width=d.width.iloc[0],
                 n_specimens=d.specimen.nunique())
        for defn in ("hb", "ml"):
            N, K, _ = orbits(d, defn); N4, K4, _ = orbits(d, defn, "key4")
            r[f"N_{defn}"] = N; r[f"K_pooled_{defn}"] = K; r[f"ratio_{defn}"] = K / N if N > 0 else np.nan
            r[f"K_pooled_{defn}_eps4"] = K4; r[f"ratio_{defn}_eps4"] = K4 / N if N > 0 else np.nan
            r[f"n_specimens_N0_{defn}"] = int(sum(orbits(dd, defn)[0] == 0 for _, dd in d.groupby("specimen")))
        rows_p.append(r)
    P = pd.DataFrame(rows_p); P.to_csv(os.path.join(R, "orbits_pooled.csv"), index=False)
    # ---- F5
    out = {}
    rng = np.random.default_rng(0)
    for defn in ("hb", "ml"):
        for crit in ("zero", "mean"):
            p = P[P.criterion == crit]
            # H-F5a: median over specimens of the pooled-cell ratio (each specimen carries its cell's ratio)
            w = W[W.criterion == crit].merge(p[["cell", f"ratio_{defn}"]], on="cell")
            med = float(np.nanmedian(w[f"ratio_{defn}"])); med_within = float(np.nanmedian(w[f"ratio_within_{defn}"]))
            # H-F5b: per (function, depth) slopes of log K and log N on width, seed bootstrap
            diffs = []
            for (fn, dep), g in E[E.criterion == crit].groupby(["function", "depth"]):
                widths = sorted(g.width.unique())
                if len(widths) < 3: continue
                def slopes(seed_sets):
                    lk, ln, lw = [], [], []
                    for wd in widths:
                        gg = g[(g.width == wd) & g.seed.isin(seed_sets[wd])]
                        N, K, _ = orbits(gg, defn)
                        if N > 0: lk.append(np.log(K)); ln.append(np.log(N)); lw.append(np.log(wd))
                    if len(lw) < 3: return np.nan
                    lw = np.array(lw); return np.polyfit(lw, ln, 1)[0] - np.polyfit(lw, lk, 1)[0]
                full = {wd: set(g[g.width == wd].seed.unique()) for wd in widths}
                point = slopes(full)
                boots = []
                for b in range(1000):
                    ss = {wd: set(rng.choice(sorted(full[wd]), len(full[wd]), replace=True)) for wd in widths}   # de-duplicated (E1 amendment 5)
                    boots.append(slopes(ss))
                boots = np.array([x for x in boots if np.isfinite(x)])
                lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan))
                diffs.append(dict(function=fn, depth=dep, slope_diff=point, ci_lo=lo, ci_hi=hi, passes=bool(point > 0 and lo > 0)))
            D = pd.DataFrame(diffs); D.to_csv(os.path.join(R, f"F5b_slopes_{defn}_{crit}.csv"), index=False)
            out[f"{defn}_{crit}"] = dict(H_F5a_median_ratio_pooled=med, H_F5a_pass=bool(med <= 0.5), median_ratio_within=med_within,
                                         H_F5b_pairs=int(len(D)), H_F5b_pass_pairs=int(D.passes.sum()) if len(D) else 0,
                                         H_F5b_pass=bool(len(D) and D.passes.all()),
                                         n_cells=int(len(p)), n_specimens_N0=int(p[f"n_specimens_N0_{defn}"].sum()))
    json.dump(out, open(os.path.join(R, "F5_verdicts.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    # ---- figures
    for defn in ("hb", "ml"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
        for ax, crit in zip(axes, ("zero", "mean")):
            p = P[P.criterion == crit]
            for fn, g in p.groupby("function"):
                g = g.sort_values("width")
                ax.plot(g.width, g[f"N_{defn}"], "o-", ms=3, lw=1, alpha=.7)
                ax.plot(g.width, g[f"K_pooled_{defn}"], "x--", ms=3, lw=1, alpha=.7, color=ax.lines[-1].get_color())
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("width"); ax.set_ylabel("N (o) and K (x), pooled per cell")
            ax.set_title(f"{crit}-ablation, {defn}")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"F5_KN_vs_width_{defn}.png")); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=300)
        for ax, crit in zip(axes, ("zero", "mean")):
            p = P[P.criterion == crit]
            ax.hist(p[f"ratio_{defn}"].dropna(), bins=np.linspace(0, 1.02, 30), color="#4477AA")
            ax.axvline(0.5, color="#EE6677", ls="--", lw=1); ax.set_xlabel("K/N (pooled per cell)"); ax.set_title(f"{crit}, {defn}")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"F5_KN_hist_{defn}.png")); plt.close(fig)
    # ---- F6 autopsy: distinct algorithms per function (handbook gate strings of interpretable circuits)
    rows = []
    for (fn, crit), d in E[E.interpretable == 1].groupby(["function", "criterion"]):
        algs = d.algorithm.value_counts()
        rows.append(dict(function=fn, criterion=crit, n_explanations=len(d), n_orbits=d.key6.nunique(),
                         n_distinct_gate_strings=len(algs), top_algorithms=" | ".join(f"{a} ({c})" for a, c in algs.head(5).items())))
    pd.DataFrame(rows).to_csv(os.path.join(R, "F6_algorithms.csv"), index=False)
    # specimens with K_within < N (gauge-mergeable within one network)
    m = W[(W.K_within_hb < W.N_hb) | (W.K_within_ml < W.N_ml)]
    m.to_csv(os.path.join(R, "F6_within_network_merges.csv"), index=False)
    print("specimens with within-network orbit merges:", len(m), "of", len(W))


if __name__ == "__main__":
    main()
