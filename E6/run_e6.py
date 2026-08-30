"""E6 analysis (Tests 1 & 2 + figures) on the cached activations from extract_e6.py. See SPEC.md."""
import os, sys, time, json, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6lib import (mp_outliers, feature_shuffle, gaussian_matched_diag, gaussian_matched_cov, block_bootstrap_outliers,
                   project_out_top_pcs, mp_density, tc_ratio_bootstrap)

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache"); RES = os.path.join(HERE, "results")
LAYERS = [2, 6, 10]


def test1(L, X, E, seq, B):
    rows, spectra = [], {}
    clouds = {"E": E, "X": X, "gauss_diag(E)": gaussian_matched_diag(E, 0), "shuffle(E)": feature_shuffle(E, 0),
              "E_minus_top10PC(X)": project_out_top_pcs(E, X, 10), "E_minus_top50PC(X)": project_out_top_pcs(E, X, 50)}
    for name, C in clouds.items():
        for form in ("correlation", "covariance"):
            r = mp_outliers(C, form)
            row = dict(layer=L, cloud=name, **r.as_row())
            if name in ("E", "X") or (name == "shuffle(E)" and form == "correlation"):
                row.update(block_bootstrap_outliers(C, seq, B=B, seed=0, form=form))
            rows.append(row); spectra[(name, form)] = r
            print(f"  L{L} {name:20s} {form:11s} n_out={r.n_outliers:4d} sf={r.structured_fraction:.3f} ef={r.excess_fraction:.3f} thr={r.tw_threshold:.3f}")
    return rows, spectra


def test2(L, X, E, n_points, n_boot, n_jobs):
    rows = {}
    for name, C in (("E", E), ("X", X), ("gauss_cov(E)", gaussian_matched_cov(E, 0, n=len(E)))):
        t = tc_ratio_bootstrap(C, n_points=n_points, n_boot=n_boot, seed=0, n_jobs=n_jobs)
        rows[name] = dict(layer=L, cloud=name, **{k: v for k, v in t.items() if k != "samples"},
                          tc_ratio_ci_low=t["tc_ratio_ci"][0], tc_ratio_ci_high=t["tc_ratio_ci"][1])
        print(f"  L{L} {name:14s} tc_ratio={t['tc_ratio']:.3f} CI={t['tc_ratio_ci']} b1={t['b1']:.0f}")
    return list(rows.values())


def fig_spectrum(L, spectra):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    panels = [("E", "correlation"), ("X", "correlation"), ("shuffle(E)", "correlation"),
              ("gauss_diag(E)", "correlation"), ("E_minus_top50PC(X)", "correlation"), ("E", "covariance")]
    for ax, key in zip(axes.ravel(), panels):
        r = spectra[key]; ev = r.eigenvalues
        lo, hi = (1 - np.sqrt(r.gamma)) ** 2 * r.sigma2, r.mp_edge
        xs = np.linspace(max(lo * 0.8, 1e-6), max(hi * 1.5, ev.max() * 1.05), 600)
        bins = np.linspace(0, min(ev.max() * 1.05, max(hi * 4, ev.max() * 1.05)), 80)
        ax.hist(ev, bins=bins, density=True, color="#4C72B0", alpha=0.75, label="eigenvalues")
        ax.plot(xs, mp_density(xs, r.gamma, r.sigma2), "k-", lw=1.4, label="Marchenko–Pastur")
        ax.axvline(r.tw_threshold, color="#C44E52", ls="--", lw=1.2, label="TW edge (99%)")
        out = ev[ev > r.tw_threshold]
        if len(out): ax.plot(out, np.full(len(out), ax.get_ylim()[1] * 0.05), "v", color="#C44E52", ms=4, label=f"{len(out)} outliers")
        ax.set_title(f"{key[0]} — {key[1]} form\n n_out={r.n_outliers}, structured={r.structured_fraction:.2f}, excess={r.excess_fraction:.2f}", fontsize=9)
        ax.set_xlabel("eigenvalue"); ax.set_yscale("log"); ax.legend(fontsize=7)
    fig.suptitle(f"GPT-2 small, resid_pre layer {L}, gpt2-small-res-jb SAE error e = x − SAE(x)  (N={r.n}, d={r.p})", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(RES, f"fig_spectrum_L{L}.png"), dpi=300); plt.close(fig)


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


def fig_probe(pr):
    lp = os.path.join(RES, "e6_probe_logprob.csv")
    if os.path.exists(lp):   # SPEC amendment 3
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-points", type=int, default=2000); ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--B", type=int, default=50); ap.add_argument("--n-jobs", type=int, default=8); ap.add_argument("--layers", default="2,6,10")
    a = ap.parse_args(); t0 = time.time()
    t1_rows, t2_rows = [], []
    for L in [int(x) for x in a.layers.split(",")]:
        z = np.load(os.path.join(CACHE, f"L{L}.npz")); X, E, seq = z["X"].astype(np.float64), z["E"].astype(np.float64), z["seq"]
        print(f"L{L}: N={len(E)} d={E.shape[1]} FVU={float(z['fvu']):.4f}")
        rows, spectra = test1(L, X, E, seq, a.B); t1_rows += rows; fig_spectrum(L, spectra)
        pd.DataFrame(t1_rows).to_csv(os.path.join(RES, "e6_test1_rmt.csv"), index=False)
        np.savez(os.path.join(RES, f"e6_spectra_L{L}.npz"), **{f"{k[0]}__{k[1]}": v.eigenvalues for k, v in spectra.items()})
        t2_rows += test2(L, X, E, a.n_points, a.n_boot, a.n_jobs)
        pd.DataFrame(t2_rows).to_csv(os.path.join(RES, "e6_test2_topology.csv"), index=False)
        print(f"  elapsed {time.time()-t0:.0f}s")
    fig_tc(pd.DataFrame(t2_rows))
    pp = os.path.join(RES, "e6_probe.csv")
    if os.path.exists(pp): fig_probe(pd.read_csv(pp))
    # verdict table
    t1 = pd.DataFrame(t1_rows); t2 = pd.DataFrame(t2_rows); ver = []
    for L in sorted(t1.layer.unique()):
        e = t1[(t1.layer == L) & (t1.cloud == "E") & (t1.form == "correlation")].iloc[0]
        sh = t1[(t1.layer == L) & (t1.cloud == "shuffle(E)") & (t1.form == "correlation")].iloc[0]
        x = t1[(t1.layer == L) & (t1.cloud == "X") & (t1.form == "correlation")].iloc[0]
        tt = t2[(t2.layer == L) & (t2.cloud == "E")].iloc[0]; g = t2[(t2.layer == L) & (t2.cloud == "gauss_cov(E)")].iloc[0]
        valid = sh.n_outliers <= 1
        p1 = valid and e.n_outliers > 10 and e.n_outliers_ci[0] > 10
        p2 = tt.tc_ratio > 1.2 and tt.tc_ratio_ci_low > 1.0
        ver.append(dict(layer=L, n_out_E=e.n_outliers, n_out_E_ci=e.n_outliers_ci, n_out_X=x.n_outliers, n_out_shuffle=sh.n_outliers,
                        test1_valid=valid, test1_pass=p1, structured_fraction_E=e.structured_fraction, excess_fraction_E=e.excess_fraction,
                        tc_ratio_E=tt.tc_ratio, tc_ratio_E_ci=(tt.tc_ratio_ci_low, tt.tc_ratio_ci_high), tc_ratio_gauss_cov=g.tc_ratio,
                        test2_pass=p2, H_E6=bool(p1 and p2)))
    pd.DataFrame(ver).to_csv(os.path.join(RES, "e6_verdict.csv"), index=False); print(pd.DataFrame(ver).to_string())
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
