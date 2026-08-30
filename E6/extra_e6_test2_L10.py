"""Fill the reboot gap: Test 2 (topology, tc_ratio_bootstrap) for layer 10 only, which never
ran (the reboot hit during Test 1 L10's np.savez). Appends to e6_test2_topology.csv, regenerates
fig_tc_ratio.png, fig_probe.png, and the verdict table e6_verdict.csv from the now-complete data.
Mirrors run_e6.py's test2()/fig_tc()/fig_probe()/verdict logic exactly; no other numbers touched."""
import os, sys, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6lib import gaussian_matched_cov, tc_ratio_bootstrap

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache"); RES = os.path.join(HERE, "results")

t0 = time.time()
z = np.load(os.path.join(CACHE, "L10.npz")); X, E = z["X"].astype(np.float64), z["E"].astype(np.float64)
print(f"L10: N={len(E)} d={E.shape[1]}", flush=True)

rows = []
for name, C in (("E", E), ("X", X), ("gauss_cov(E)", gaussian_matched_cov(E, 0, n=len(E)))):
    t = tc_ratio_bootstrap(C, n_points=2000, n_boot=20, seed=0, n_jobs=6)
    rows.append(dict(layer=10, cloud=name, **{k: v for k, v in t.items() if k != "samples"},
                      tc_ratio_ci_low=t["tc_ratio_ci"][0], tc_ratio_ci_high=t["tc_ratio_ci"][1]))
    print(f"  L10 {name:14s} tc_ratio={t['tc_ratio']:.3f} CI={t['tc_ratio_ci']} b1={t['b1']:.0f}  ({time.time()-t0:.0f}s)", flush=True)

old = pd.read_csv(os.path.join(RES, "e6_test2_topology.csv"))
full = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
full.to_csv(os.path.join(RES, "e6_test2_topology.csv"), index=False)
print("wrote e6_test2_topology.csv with", len(full), "rows", flush=True)

# ---- regenerate fig_tc_ratio.png (identical to run_e6.fig_tc) ----
LAYERS = [2, 6, 10]
def fig_tc(df):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    clouds = ["E", "X", "gauss_cov(E)"]; cols = ["#4C72B0", "#55A868", "#8172B2"]
    for i, (c, col) in enumerate(zip(clouds, cols)):
        d = df[df.cloud == c].sort_values("layer")
        ax.errorbar(d.layer + (i - 1) * 0.25, d.tc_ratio, yerr=[d.tc_ratio - d.tc_ratio_ci_low, d.tc_ratio_ci_high - d.tc_ratio],
                    fmt="o", color=col, capsize=3, label=c)
    ax.axhline(1.2, color="#C44E52", ls="--", lw=1, label="H-E6 threshold 1.2"); ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xticks(LAYERS); ax.set_xlabel("layer (resid_pre)"); ax.set_ylabel("TC(H1) / TC(H1, feature-shuffled)")
    ax.set_title("Test 2 — topology vs v1 shuffle null (2 000-token clouds, PCA-30, 95 % CI over 20 subsamples)", fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_tc_ratio.png"), dpi=300); plt.close(fig)
fig_tc(full)
print("wrote fig_tc_ratio.png", flush=True)

# ---- regenerate fig_probe.png (identical to run_e6.fig_probe) ----
pr = pd.read_csv(os.path.join(RES, "e6_probe.csv"))
def fig_probe(pr):
    lp = os.path.join(RES, "e6_probe_logprob.csv")
    if os.path.exists(lp):
        l = pd.read_csv(lp).set_index("layer"); pr = pr.copy()
        for c in ("r2_E", "r2_Eshuf", "r2_X"): pr[c + "_lp"] = pr.layer.map(l[c])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    w = 0.25
    for ax, (cols, title, yl) in zip(axes, [(("r2_E", "r2_Eshuf", "r2_X"), "Test 3a — held-out R², probe → raw logit delta", "R²"),
                                            (("r2_E_lp", "r2_Eshuf_lp", "r2_X_lp"), "Test 3a' (amend. 3) — held-out R², probe → log-prob delta", "R²"),
                                            (("nt_top5_E", "nt_top5_Eshuf", "nt_top5_X"), "Test 3b — held-out top-5 next-token accuracy", "top-5 acc")]):
        for i, (c, lab, col) in enumerate(zip(cols, ("e", "shuffled e", "x"), ("#4C72B0", "#C44E52", "#55A868"))):
            ax.bar(np.arange(len(pr)) + (i - 1) * w, pr[c], w, color=col, label=lab)
        if "nt_majority" in pr and "top5" in cols[0]:
            ax.plot(np.arange(len(pr)), pr["nt_majority"], "k_", ms=20, label="majority (top-1)")
        ax.set_xticks(np.arange(len(pr))); ax.set_xticklabels([f"L{l}" for l in pr.layer]); ax.set_title(title, fontsize=9); ax.set_ylabel(yl); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_probe.png"), dpi=300); plt.close(fig)
fig_probe(pr)
print("wrote fig_probe.png", flush=True)

# ---- regenerate the verdict table ----
# NOTE (Amendment 2): the H-E6 CI criterion uses the without-replacement subsample CI
# (e6_test1_subsample_ci.csv) for cloud E's n_outliers, not the with-replacement block
# bootstrap CI stored alongside e6_test1_rmt.csv (that one is reported too, but is not the
# criterion — the original run_e6.py verdict block predates Amendment 2 and used the wrong
# (with-replacement) CI; fixed here).
t1 = pd.read_csv(os.path.join(RES, "e6_test1_rmt.csv"))
import ast
subci = pd.read_csv(os.path.join(RES, "e6_test1_subsample_ci.csv"))
t2 = full
ver = []
for L in sorted(t1.layer.unique()):
    e = t1[(t1.layer == L) & (t1.cloud == "E") & (t1.form == "correlation")].iloc[0]
    sh = t1[(t1.layer == L) & (t1.cloud == "shuffle(E)") & (t1.form == "correlation")].iloc[0]
    x = t1[(t1.layer == L) & (t1.cloud == "X") & (t1.form == "correlation")].iloc[0]
    tt = t2[(t2.layer == L) & (t2.cloud == "E")].iloc[0]; g = t2[(t2.layer == L) & (t2.cloud == "gauss_cov(E)")].iloc[0]
    e_ci_withrep = ast.literal_eval(e.n_outliers_ci) if isinstance(e.n_outliers_ci, str) else e.n_outliers_ci
    e_sub = subci[(subci.layer == L) & (subci.cloud == "E")].n_outliers
    e_ci_noRep = (float(np.percentile(e_sub, 2.5)), float(np.percentile(e_sub, 97.5)))
    valid = sh.n_outliers <= 1
    p1 = bool(valid and e.n_outliers > 10 and e_ci_noRep[0] > 10)
    p2 = bool(tt.tc_ratio > 1.2 and tt.tc_ratio_ci_low > 1.0)
    ver.append(dict(layer=L, n_out_E=e.n_outliers, n_out_E_ci_noreplace=e_ci_noRep, n_out_E_ci_withreplace=e_ci_withrep,
                    n_out_X=x.n_outliers, n_out_shuffle=sh.n_outliers,
                    test1_valid=valid, test1_pass=p1, structured_fraction_E=e.structured_fraction, excess_fraction_E=e.excess_fraction,
                    tc_ratio_E=tt.tc_ratio, tc_ratio_E_ci=(tt.tc_ratio_ci_low, tt.tc_ratio_ci_high), tc_ratio_gauss_cov=g.tc_ratio,
                    test2_pass=p2, H_E6=bool(p1 and p2)))
vdf = pd.DataFrame(ver)
vdf.to_csv(os.path.join(RES, "e6_verdict.csv"), index=False)
print(vdf.to_string())
print("done", time.time() - t0)
