"""Second half of extra_e6_test2_L10.py that crashed on fig_tc's yerr bug (L2's tc_ratio_E point
estimate, seed 0, falls outside its own 20-seed bootstrap CI -- clip yerr at 0 rather than pass
negative values to errorbar; this is a plotting fix only, not a numbers change, and the same bug
was latent in run_e6.py's fig_tc, which never ran to completion so never hit it). Regenerates
fig_tc_ratio.png, fig_probe.png, and e6_verdict.csv from the now-complete Test 1 + Test 2 CSVs."""
import os, sys, numpy as np, pandas as pd, ast
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
LAYERS = [2, 6, 10]

full = pd.read_csv(os.path.join(RES, "e6_test2_topology.csv"))

def fig_tc(df):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    clouds = ["E", "X", "gauss_cov(E)"]; cols = ["#4C72B0", "#55A868", "#8172B2"]
    for i, (c, col) in enumerate(zip(clouds, cols)):
        d = df[df.cloud == c].sort_values("layer")
        lo = np.clip(d.tc_ratio - d.tc_ratio_ci_low, 0, None)
        hi = np.clip(d.tc_ratio_ci_high - d.tc_ratio, 0, None)
        ax.errorbar(d.layer + (i - 1) * 0.25, d.tc_ratio, yerr=[lo, hi],
                    fmt="o", color=col, capsize=3, label=c)
    ax.axhline(1.2, color="#C44E52", ls="--", lw=1, label="H-E6 threshold 1.2"); ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xticks(LAYERS); ax.set_xlabel("layer (resid_pre)"); ax.set_ylabel("TC(H1) / TC(H1, feature-shuffled)")
    ax.set_title("Test 2 — topology vs v1 shuffle null (2 000-token clouds, PCA-30, 95% CI over 20 subsamples;\n"
                  "point estimate is seed-0, may fall outside its own CI band — see FINDINGS)", fontsize=8)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_tc_ratio.png"), dpi=300); plt.close(fig)
fig_tc(full)
print("wrote fig_tc_ratio.png")

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
print("wrote fig_probe.png")

t1 = pd.read_csv(os.path.join(RES, "e6_test1_rmt.csv"))
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
    valid = bool(sh.n_outliers <= 1)
    p1 = bool(valid and e.n_outliers > 10 and e_ci_noRep[0] > 10)
    # Amendment 2: point estimate can fall outside its own bootstrap CI (seed-0 vs seeds 1..20);
    # apply the criterion to both the point estimate and the bootstrap mean, report both.
    p2_point = bool(tt.tc_ratio > 1.2 and tt.tc_ratio_ci_low > 1.0)
    p2_mean = bool(tt.tc_ratio_boot_mean > 1.2 and tt.tc_ratio_ci_low > 1.0)
    ver.append(dict(layer=int(L), n_out_E=int(e.n_outliers), n_out_E_ci_noreplace=e_ci_noRep, n_out_E_ci_withreplace=e_ci_withrep,
                    n_out_X=int(x.n_outliers), n_out_shuffle=int(sh.n_outliers),
                    test1_valid=valid, test1_pass=p1, structured_fraction_E=e.structured_fraction, excess_fraction_E=e.excess_fraction,
                    tc_ratio_E_point=tt.tc_ratio, tc_ratio_E_boot_mean=tt.tc_ratio_boot_mean, tc_ratio_E_ci=(tt.tc_ratio_ci_low, tt.tc_ratio_ci_high),
                    tc_ratio_gauss_cov=g.tc_ratio, test2_pass_point=p2_point, test2_pass_bootmean=p2_mean,
                    H_E6_point=bool(p1 and p2_point), H_E6_bootmean=bool(p1 and p2_mean)))
vdf = pd.DataFrame(ver)
vdf.to_csv(os.path.join(RES, "e6_verdict.csv"), index=False)
print(vdf.to_string())
