"""V4 (phase diagram + changepoint test) and V5 (drift) analysis. Writes results/V4_fits.json, fig/V4_*.png, V5_summary.json, fig/V5_drift.png."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); FIG = os.path.join(R, "fig"); os.makedirs(FIG, exist_ok=True)


def logistic_fit(X, y, changepoint=False):
    """Returns (negative log-lik, k params). With changepoint: piecewise-linear hinge on the first column."""
    n, d = X.shape
    def nll(theta, cp=None):
        Xa = X if cp is None else np.hstack([X, np.maximum(X[:, :1] - cp, 0)])
        z = Xa @ theta[1:] + theta[0]; z = np.clip(z, -30, 30)
        return -np.sum(y * z - np.log1p(np.exp(z))) + 1e-3 * np.sum(theta[1:] ** 2)
    if not changepoint:
        r = minimize(nll, np.zeros(d + 1), method="L-BFGS-B"); return r.fun, d + 1, None
    best = None
    for cp in np.quantile(X[:, 0], np.linspace(0.1, 0.9, 17)):
        r = minimize(lambda t: nll(t, cp), np.zeros(d + 2), method="L-BFGS-B")
        if best is None or r.fun < best[0]: best = (r.fun, d + 3, float(cp))
    return best


def bic(nll, k, n): return 2 * nll + k * np.log(n)


def main():
    s = pd.read_csv(os.path.join(R, "V2_structure.csv")); d = pd.read_csv(os.path.join(R, "V3_discovery.csv")); m = pd.read_csv(os.path.join(R, "V1_models.csv"))
    df = s.merge(d, on=["name", "task", "arch"]).merge(m[["name", "untrained", "random_labels", "eval_acc"]], on="name")
    df["learned"] = df.eval_acc >= 0.5
    df["success_v4"] = ((df.F >= 0.9) & (df.frac <= 0.5) & df.learned).astype(int); df["cost_v4"] = 1 - df.frac
    df["family"] = df.arch.str.replace(r"\d+", "", regex=True)
    out = {}; n = len(df)
    for feat in ("Qexcess_0.1", "tw_0.1", "sparsity_nodes"):
        X = np.stack([df[feat].values, (df["tw_0.1"] / df.N).values, df.sparsity_nodes.values], 1) if feat == "Qexcess_0.1" else df[[feat]].values
        X = (X - X.mean(0)) / (X.std(0) + 1e-9); y = df.success_v4.values
        nl, k, _ = logistic_fit(X, y); nc, kc, cp = logistic_fit(X, y, changepoint=True)
        out[f"success~{feat}"] = dict(bic_linear=bic(nl, k, n), bic_changepoint=bic(nc, kc, n), delta_bic=bic(nl, k, n) - bic(nc, kc, n), changepoint_std=cp, pass_pre_registered=bool(bic(nl, k, n) - bic(nc, kc, n) > 10))
    # continuous cost with piecewise-linear knee
    x = df["Qexcess_0.1"].values; yc = df.cost_v4.values
    lin = np.polyfit(x, yc, 1); rss_l = np.sum((yc - np.polyval(lin, x)) ** 2); best = None
    for cp in np.quantile(x, np.linspace(0.1, 0.9, 17)):
        A = np.stack([np.ones_like(x), x, np.maximum(x - cp, 0)], 1); coef, *_ = np.linalg.lstsq(A, yc, rcond=None); rss = np.sum((yc - A @ coef) ** 2)
        if best is None or rss < best[0]: best = (rss, cp)
    bic_l = n * np.log(rss_l / n) + 2 * np.log(n); bic_k = n * np.log(best[0] / n) + 4 * np.log(n)
    out["cost~Qexcess_knee"] = dict(bic_linear=bic_l, bic_knee=bic_k, delta_bic=bic_l - bic_k, knee=best[1], pass_pre_registered=bool(bic_l - bic_k > 10))
    # KILL (V AMENDMENTS #12): the Q_excess changepoint is the only pre-registered pass — (a) label-permutation null of its
    # delta-BIC (the 17-candidate search inflates it), (b) family (mlp/tf) one-hot in BOTH models (the E4 confound), (c) per family
    X = np.stack([df["Qexcess_0.1"].values, (df["tw_0.1"] / df.N).values, df.sparsity_nodes.values], 1); X = (X - X.mean(0)) / (X.std(0) + 1e-9); y = df.success_v4.values
    def dbic(X, y):
        nl, k, _ = logistic_fit(X, y); nc, kc, cp = logistic_fit(X, y, changepoint=True); return bic(nl, k, len(y)) - bic(nc, kc, len(y)), cp
    obs, _ = dbic(X, y); rng = np.random.default_rng(0); null = np.array([dbic(X, rng.permutation(y))[0] for _ in range(200)])
    fam = (df.family.values == "tf").astype(float)[:, None]; obs_fam, cp_fam = dbic(np.hstack([X, fam]), y)
    per_family = {}
    for f, g in df.groupby("family"):
        Xg = X[(df.family == f).values]; yg = y[(df.family == f).values]; dg, cpg = dbic(Xg, yg); per_family[f] = dict(n=int(len(yg)), n_success=int(yg.sum()), delta_bic=dg, changepoint_std=cpg)
    out["kill_Qexcess_changepoint"] = dict(observed_delta_bic=obs, perm_null_p=float(np.mean(null >= obs)), perm_null_q95=float(np.percentile(null, 95)), perm_null_max=float(null.max()),
                                           delta_bic_with_family_covariate=obs_fam, changepoint_std_with_family=cp_fam, per_family=per_family,
                                           success_rate_by_family={f: float(g.success_v4.mean()) for f, g in df.groupby("family")},
                                           survives=bool(np.mean(null >= obs) < 0.05 and obs_fam > 10))
    # correlations pooled and within family
    out["spearman"] = {}
    for feat in ("Qexcess_0.1", "Q_0.1", "tw_0.1", "sparsity_nodes"):
        out["spearman"][feat] = dict(pooled=stats.spearmanr(df[feat], df.cost_v4, nan_policy="omit")[0],
                                     **{f: stats.spearmanr(g[feat], g.cost_v4, nan_policy="omit")[0] for f, g in df.groupby("family")})
    out["n_models"] = n; out["n_success"] = int(df.success_v4.sum())
    json.dump(out, open(os.path.join(R, "V4_fits.json"), "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=300)
    for ax, feat in zip(axes[:2], ("Qexcess_0.1", "tw_0.1")):
        for fam, g in df.groupby("family"): ax.scatter(g[feat], g.cost_v4, s=14, alpha=.7, label=fam)
        ax.set_xlabel(feat); ax.set_ylabel("discovery compression 1-|C|/N"); ax.legend(fontsize=7)
    h = axes[2].hexbin(df["Qexcess_0.1"], df["tw_0.1"] / df.N, C=df.cost_v4, gridsize=12, cmap="viridis"); plt.colorbar(h, ax=axes[2], label="compression")
    axes[2].set_xlabel("Q excess"); axes[2].set_ylabel("tw / N"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "V4_phase.png")); plt.close(fig)
    for feat, fn in (("Qexcess_0.1", "V4_phase_Q.png"), ("tw_0.1", "V4_phase_tw.png")):
        fig, ax = plt.subplots(figsize=(5, 3.6), dpi=300)
        for fam, g in df.groupby("family"): ax.scatter(g[feat], g.success_v4 + np.random.default_rng(0).normal(0, 0.02, len(g)), s=14, alpha=.7, label=fam)
        ax.set_xlabel(feat); ax.set_ylabel("success (F>=0.9, |C|/N<=0.5)"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, fn)); plt.close(fig)
    # ---- V5 drift
    p5 = os.path.join(R, "V5_drift.csv")
    if os.path.exists(p5):
        v = pd.read_csv(p5); v["family"] = v.arch.str.replace(r"\d+", "", regex=True); summ = {}
        slopes = {}
        for name, g in v.groupby("name"):
            g = g.sort_values("frac")
            if len(g) >= 5: slopes[name] = dict(slope=np.polyfit(g.frac, g["Qexcess_0.1"], 1)[0], random_labels=int(g.random_labels.iloc[0]),
                                                knee=float(g.frac.iloc[int(np.argmax(np.abs(np.diff(g["Qexcess_0.1"]))))]) if len(g) > 1 else np.nan,
                                                loss_half=float(g.frac.iloc[int(np.argmax(g.loss <= g.loss.iloc[0] - 0.5 * (g.loss.iloc[0] - g.loss.iloc[-1])))]))
        sl = pd.DataFrame(slopes).T
        for lab, gg in sl.groupby("random_labels"):
            bs = [np.mean(np.random.default_rng(i).choice(gg.slope.values, len(gg))) for i in range(1000)]
            summ[f"slope_random_labels={int(lab)}"] = dict(mean=float(gg.slope.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=int(len(gg)),
                                                        knee_minus_loss_half_median=float((gg.knee - gg.loss_half).median()))
        json.dump(summ, open(os.path.join(R, "V5_summary.json"), "w"), indent=1); print(summ)
        fig, ax = plt.subplots(figsize=(6, 3.8), dpi=300)
        for name, g in v.groupby("name"):
            g = g.sort_values("frac"); ax.plot(g.frac, g["Qexcess_0.1"], lw=0.8, alpha=0.5, color="#EE6677" if g.random_labels.iloc[0] else "#4477AA")
        ax.set_xlabel("training fraction"); ax.set_ylabel("Q excess (density 0.1)"); ax.set_title("blue: real labels, red: random labels"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "V5_drift.png"))


if __name__ == "__main__":
    main()
