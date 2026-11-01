"""E4 analysis: H-E4 knee test, H-E4b/c, kill attempts, phase-diagram figures. Reads results/e4_models.csv."""
import os, sys, json, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RNG = np.random.default_rng(0); NB = 2000
C = {"mlp": "#2a78d6", "tf2": "#eb6834", "tf4": "#1baf7a"}; INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300, "savefig.dpi": 300})

def _rho_rows(X, Y):
    """Spearman per row of X vs Y (vectorised)."""
    rx = stats.rankdata(X, axis=1); ry = stats.rankdata(Y, axis=1)
    rx = rx - rx.mean(1, keepdims=True); ry = ry - ry.mean(1, keepdims=True)
    return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1) + 1e-12)

def spearman_ci(x, y, nb=NB):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    n = len(x)
    if n < 4: return dict(rho=np.nan, lo=np.nan, hi=np.nan, p_perm=np.nan, n=n)
    r = stats.spearmanr(x, y).statistic
    I = RNG.integers(0, n, (nb, n)); bs = _rho_rows(x[I], y[I]); lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    P = np.array([RNG.permutation(n) for _ in range(nb)]); perm = _rho_rows(x[P], np.tile(y, (nb, 1)))
    p = (np.sum(np.abs(perm) >= abs(r)) + 1) / (nb + 1)
    return dict(rho=r, lo=lo, hi=hi, p_perm=p, n=n)

def partial_spearman_ci(x, y, Z, nb=500):
    x, y = np.asarray(x, float), np.asarray(y, float); Z = np.asarray(Z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1); x, y, Z = x[m], y[m], Z[m]
    def pr(x, y, Z):
        rx, ry = stats.rankdata(x), stats.rankdata(y); A = np.column_stack([np.ones(len(x)), stats.rankdata(Z, axis=0)])
        ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]; ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        return stats.pearsonr(ex, ey).statistic
    r = pr(x, y, Z); n = len(x)
    bs = [pr(x[i], y[i], Z[i]) for i in (RNG.integers(0, n, n) for _ in range(nb))]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(rho=r, lo=lo, hi=hi, n=n)

def bic(rss, n, k): return n * np.log(max(rss, 1e-12) / n) + k * np.log(n)

def fit_lin(x, y):
    A = np.column_stack([np.ones_like(x), x]); b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b, np.sum((y - A @ b) ** 2)

def fit_seg(x, y):
    best = None
    if len(x) < 8 or np.ptp(x) == 0: return (np.zeros(3), np.nan, np.inf)
    for k in np.linspace(*np.percentile(x, [10, 90]), 41):
        A = np.column_stack([np.ones_like(x), x, np.maximum(x - k, 0)]); b, *_ = np.linalg.lstsq(A, y, rcond=None)
        rss = np.sum((y - A @ b) ** 2)
        if best is None or rss < best[2]: best = (b, k, rss)
    return best

def knee_test(x, y, nb=1000):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = len(x)
    def dbic(x, y):
        _, rl = fit_lin(x, y); bs, k, rs = fit_seg(x, y); return bic(rl, len(x), 2) - bic(rs, len(x), 4), k, bs
    d, k, bs = dbic(x, y)
    boot = [dbic(x[i], y[i]) for i in (RNG.integers(0, n, n) for _ in range(nb))]
    db = np.array([b[0] for b in boot]); kb = np.array([b[1] for b in boot]); b2 = np.array([b[2][2] for b in boot])
    null = np.array([dbic(RNG.permutation(x), y)[0] for _ in range(nb)])
    return dict(dBIC=d, knee=k, knee_lo=np.percentile(kb, 2.5), knee_hi=np.percentile(kb, 97.5), b1=bs[1], b2=bs[2],
                b2_lo=np.percentile(b2, 2.5), b2_hi=np.percentile(b2, 97.5), frac_boot_pos=np.mean(db > 0),
                p_perm=(np.sum(null >= d) + 1) / (nb + 1), n=n, lin=fit_lin(x, y)[0], seg=bs)

def main():
    df = pd.read_csv(os.path.join(R, "e4_models.csv")); sw = pd.read_csv(os.path.join(R, "e4_sweep.csv"))
    df["cost_norm"] = df["cost"] / df["N"]; tests = []
    def rec(name, subset, **kw): tests.append(dict(test=name, subset=subset, **{k: (float(v) if np.isscalar(v) else str(v)) for k, v in kw.items()}))
    for name, sub in [("all", df), ("learned", df[df.learned]), ("trained_only", df[df.kind == "trained"])]:
        s = spearman_ci(sub.Q10, sub.S); rec("H-E4(i) rho(Q10,S)", name, **s)
        s = spearman_ci(sub.Q10, sub.S_raw); rec("rho(Q10,S_raw)", name, **s)
        k = knee_test(sub.Q10, sub.S); rec("H-E4(ii) knee Q10->S", name, **{a: b for a, b in k.items() if a not in ("lin", "seg")})
        s = spearman_ci(sub.tw10_norm, sub.S); rec("H-E4b rho(tw/N,S)", name, **s)
        s = spearman_ci(sub.tw10_norm, sub.cost_norm); rec("H-E4b rho(tw/N,cost/N)", name, **s)
        s = spearman_ci(sub.tw10, sub.cost); rec("rho(tw,cost raw)", name, **s)
        p = partial_spearman_ci(sub.Q10, sub.S, sub[["sparsity", "N"]].values); rec("H-E4c partial rho(Q10,S|sparsity,N)", name, **p)
        p = partial_spearman_ci(sub.Q10_excess, sub.S, sub[["sparsity", "N"]].values); rec("partial rho(Q10_excess,S|sparsity,N)", name, **p)
        p = partial_spearman_ci(sub.sparsity, sub.S, sub[["Q10", "N"]].values); rec("partial rho(sparsity,S|Q10,N)", name, **p)
        for pred in ["Q10", "Q10_excess", "Q_auc", "tw10_norm", "tw10", "sparsity", "edge_sparsity", "weight_sparsity", "N"]:
            for tgt in ["S", "cost_norm", "jaccard_guarded", "F"]:
                s = spearman_ci(sub[pred], sub[tgt]); rec(f"horse_race rho({pred},{tgt})", name, **s)
        for arch in ["mlp", "tf2", "tf4"]:
            ss = sub[sub.arch == arch]
            if len(ss) > 5: s = spearman_ci(ss.Q10, ss.S); rec("kill6 within-arch rho(Q10,S)", f"{name}/{arch}", **s)
        for d in [0.05, 0.2]:
            q = sw[sw.density == d].set_index("id")["Q"].reindex(sub.id).values
            k = knee_test(q, sub.S); rec(f"kill5 knee at d={d}", name, **{a: b for a, b in k.items() if a not in ("lin", "seg")})
            s = spearman_ci(q, sub.S); rec(f"kill5 rho(Q_{d},S)", name, **s)
    ub = df[df.kind == "untrained_block"]; ut = df[df.kind == "untrained"]; tr = df[df.kind == "trained"]
    rec("kill3 untrained_block S_raw mean", "controls", value=ub.S_raw.mean(), Q10=ub.Q10.mean(), n=len(ub))
    rec("kill3 untrained S_raw mean", "controls", value=ut.S_raw.mean(), Q10=ut.Q10.mean(), n=len(ut))
    rec("kill3 trained S mean", "controls", value=tr.S.mean(), Q10=tr.Q10.mean(), n=len(tr))
    rec("kill3 randlabel S mean", "controls", value=df[df.kind == "randlabel"].S.mean(), Q10=df[df.kind == "randlabel"].Q10.mean(), n=(df.kind == "randlabel").sum())
    pd.DataFrame(tests).to_csv(os.path.join(R, "e4_tests.csv"), index=False)
    # per-arm summary
    g = df.groupby(["kind", "arch", "arm"]).agg(n=("S", "size"), acc=("acc", "mean"), Q10=("Q10", "mean"), Q10_excess=("Q10_excess", "mean"),
        tw10n=("tw10_norm", "mean"), sparsity=("sparsity", "mean"), S=("S", "mean"), S_raw=("S_raw", "mean"), cost=("cost_norm", "mean"),
        J=("jaccard", "mean")).reset_index(); g.to_csv(os.path.join(R, "e4_by_arm.csv"), index=False)
    figures(df, sw, tests)
    return df, tests

def figures(df, sw, tests):
    learned = df[df.learned]; ctrl = df[~df.learned]
    # phase diagrams
    for xv, xl, fn in [("tw10_norm", "treewidth bound / N (top-10% graph)", "fig_phase_Q_tw.png"), ("sparsity", "effective sparsity (1 - PR/N)", "fig_phase_Q_sparsity.png")]:
        fig, ax = plt.subplots(figsize=(5.2, 4)); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        sc = ax.scatter(learned.Q10, learned[xv], c=learned.S, cmap="Blues", vmin=0, vmax=1, s=34, edgecolor=INK2, lw=0.4, zorder=3)
        ax.scatter(ctrl.Q10, ctrl[xv], marker="x", c=INK2, s=26, lw=0.8, label="controls (not learned; S := 0)", zorder=3)
        for a, mk in [("mlp", "o"), ("tf2", "s"), ("tf4", "^")]:
            ss = learned[learned.arch == a]; ax.scatter(ss.Q10, ss[xv], facecolor="none", edgecolor=C[a], marker=mk, s=70, lw=0.9, label=a, zorder=4)
        cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("discovery success S = 1 − |C|/N at F ≥ 0.9"); cb.outline.set_visible(False)
        ax.set_xlabel("modularity Q₁₀ (Leiden, top-10% edges)"); ax.set_ylabel(xl); ax.legend(frameon=False, fontsize=7, loc="best")
        ax.set_title("E4 legibility phase diagram", loc="left", fontsize=10, color=INK); fig.tight_layout(); fig.savefig(os.path.join(R, fn)); plt.close(fig)
    # knee figure
    k = knee_test(df.Q10, df.S, nb=300); x = np.linspace(df.Q10.min(), df.Q10.max(), 200)
    fig, ax = plt.subplots(figsize=(5.2, 3.8)); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for a, mk in [("mlp", "o"), ("tf2", "s"), ("tf4", "^")]:
        ss = df[(df.arch == a) & df.learned]; ax.scatter(ss.Q10, ss.S, c=C[a], marker=mk, s=28, label=a, edgecolor="white", lw=0.5, zorder=3)
    ss = df[~df.learned]; ax.scatter(ss.Q10, ss.S, marker="x", c=INK2, s=22, lw=0.8, label="controls (S := 0)", zorder=3)
    ax.plot(x, k["lin"][0] + k["lin"][1] * x, color=INK2, lw=1.6, ls="--", label="linear")
    ax.plot(x, k["seg"][0] + k["seg"][1] * x + k["seg"][2] * np.maximum(x - k["knee"], 0), color=INK, lw=2, label=f"segmented (knee {k['knee']:.2f})")
    ax.axvspan(k["knee_lo"], k["knee_hi"], color=GRID, alpha=0.7, zorder=0, label="knee 95% CI (bootstrap)")
    ax.set_xlabel("modularity Q₁₀"); ax.set_ylabel("discovery success S"); ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"H-E4 knee test: ΔBIC(lin−seg) = {k['dBIC']:.1f}, perm p = {k['p_perm']:.3f}", loc="left", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_knee.png")); plt.close(fig)
    # cost
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
    for ax, xv, xl in [(axs[0], "tw10_norm", "treewidth bound / N"), (axs[1], "Q10", "modularity Q₁₀")]:
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for a, mk in [("mlp", "o"), ("tf2", "s"), ("tf4", "^")]:
            ss = df[(df.arch == a)]; ax.scatter(ss[xv], ss.cost / ss.N, c=C[a], marker=mk, s=26, label=a, edgecolor="white", lw=0.5)
        ax.set_xlabel(xl)
    axs[0].set_ylabel("cost / N (ablated passes per node)"); axs[0].legend(frameon=False, fontsize=7)
    axs[0].set_title("Discovery cost vs structure", loc="left", fontsize=10, color=INK); fig.tight_layout(); fig.savefig(os.path.join(R, "fig_cost.png")); plt.close(fig)
    # sweep
    m = sw.merge(df[["id", "arch", "kind", "arm"]], on="id")
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2), sharey=True)
    for ax, a in zip(axs, ["mlp", "tf2", "tf4"]):
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for kind, col, ls in [("trained", C[a], "-"), ("untrained", INK2, ":"), ("untrained_block", INK, "--"), ("randlabel", "#e34948", "-.")]:
            g = m[(m.arch == a) & (m.kind == kind)].groupby("density")["Q"]
            if len(g): mu, sd = g.mean(), g.std().fillna(0); ax.plot(mu.index, mu.values, color=col, ls=ls, lw=1.6, label=kind); ax.fill_between(mu.index, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
        ax.set_xscale("log"); ax.set_xlabel("edge density kept"); ax.set_title(a, loc="left", fontsize=9)
    axs[0].set_ylabel("Leiden Q"); axs[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(R, "fig_sweep.png")); plt.close(fig)

if __name__ == "__main__":
    df, tests = main()
    t = pd.DataFrame(tests); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(t[~t.test.str.startswith("horse")].to_string(index=False))
    h = t[t.test.str.startswith("horse") & (t.subset == "all")].copy(); h["abs"] = h.rho.abs()
    print(h.sort_values("abs", ascending=False)[["test", "rho", "lo", "hi", "p_perm"]].to_string(index=False))
