"""Seiun Sky (Z1) and Manhattan Cafe (Z2) analysis — SPEC §Z1/§Z2, Z AMENDMENTS #2.
Z1: mislabelled fraction (film events the zigzag joins) per model, real vs random-label; pre-registered: > 0.2 → the film is not a
valid tracker. Z2: Spearman of the two-parameter summary (Hilbert mass H1 / argmax) with V3 discovery cost vs the one-parameter total
persistence at the final checkpoint; pre-registered: |rho_2p| - |rho_1p| > 0.1 with bootstrap CI > 0.
Writes results/Z1_summary.json, results/Z2_summary.json."""
import os, json
import numpy as np, pandas as pd
from scipy import stats
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "V_legibility", "results")


def z1():
    p = os.path.join(R, "Z1_zigzag.csv")
    if not os.path.exists(p): print("Z1: no rows yet"); return None
    z = pd.read_csv(p); z["family"] = z.arch.str.replace(r"\d+", "", regex=True)
    out = dict(n=int(len(z)), n_random_labels=int(z.random_labels.sum()), budget_reduced=int(z.budget_reduced.sum()), n_sub_min=int(z.n_sub.min()))
    for lab, g in z.groupby("random_labels"):
        k = f"random_labels={int(lab)}"; mf = g.mislabelled_fraction.values
        bs = [np.median(np.random.default_rng(i).choice(mf, len(mf))) for i in range(2000)] if len(mf) > 1 else [float(mf[0])]
        out[k] = dict(n=int(len(g)), mislabelled_median=float(np.median(mf)), mislabelled_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                      mislabelled_min=float(mf.min()), mislabelled_max=float(mf.max()), n_above_0_2=int((mf > 0.2).sum()),
                      film_events_median=float(g.film_events.median()), zigzag_events_median=float(g.zigzag_events.median()),
                      zigzag_bars_median=float(g.n_zigzag_bars_H1.median()), long_bars_median=float(g.n_long_bars.median()),
                      long_bar_fraction_median=float((g.n_long_bars / g.n_zigzag_bars_H1.clip(lower=1)).median()),
                      scale_quantile_median=float(g.scale_quantile.median()), n_simplices_median=float(g.n_simplices.median()))
    out["by_family"] = {f: dict(n=int(len(g)), mislabelled_median=float(g.mislabelled_fraction.median()), long_bars_median=float(g.n_long_bars.median())) for f, g in z.groupby("family")}
    if z.random_labels.nunique() == 2:
        a, b = z[z.random_labels == 0].mislabelled_fraction, z[z.random_labels == 1].mislabelled_fraction
        out["real_vs_random_mannwhitney_p"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    out["pre_registered_film_invalid_gt_0_2"] = bool(np.median(z[z.random_labels == 0].mislabelled_fraction) > 0.2) if (z.random_labels == 0).any() else None
    # betti-1 curves: does beta1 at odd (union) positions differ from even (cloud) positions? (a zigzag-specific readout)
    curves = [json.loads(c) for c in z.betti1_curve]; ev = [np.mean(c[0::2]) for c in curves]; od = [np.mean(c[1::2]) for c in curves]
    out["beta1_cloud_positions_mean"] = float(np.mean(ev)); out["beta1_union_positions_mean"] = float(np.mean(od))
    json.dump(out, open(os.path.join(R, "Z1_summary.json"), "w"), indent=1); print(json.dumps(out, indent=1)); return out


def z2():
    p = os.path.join(R, "Z2_multiparam.csv")
    if not os.path.exists(p): print("Z2: no rows yet"); return None
    z = pd.read_csv(p); d = pd.read_csv(os.path.join(V, "V3_discovery.csv")); m = z.merge(d[["name", "cost", "frac", "F"]], on="name")
    out = dict(n=int(len(m)))
    if len(m) < 5: print("Z2: too few rows", len(m)); return out
    def rho(x, y): return float(stats.spearmanr(x, y)[0])
    for target in ("cost", "frac"):
        r2 = {k: rho(m[k], m[target]) for k in ("hilbert_mass_H1", "hilbert_mass_H0", "argmax_H1_param", "argmax_H1_scale", "frac_stable_by_half")}
        r1 = rho(m.total_persistence_H1_final, m[target]); best2 = max(r2, key=lambda k: abs(r2[k]))
        rng = np.random.default_rng(0); diffs = []
        for _ in range(2000):
            i = rng.integers(0, len(m), len(m)); mm = m.iloc[i]
            if mm[best2].nunique() < 2 or mm[target].nunique() < 2: continue
            diffs.append(abs(rho(mm[best2], mm[target])) - abs(rho(mm.total_persistence_H1_final, mm[target])))
        out[target] = dict(rho_2p=r2, rho_1p_total_persistence=r1, best_2p=best2, gain=abs(r2[best2]) - abs(r1),
                           gain_ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                           pass_pre_registered=bool(abs(r2[best2]) - abs(r1) > 0.1 and np.percentile(diffs, 2.5) > 0))
    json.dump(out, open(os.path.join(R, "Z2_summary.json"), "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float)); return out


if __name__ == "__main__":
    z1(); z2()
