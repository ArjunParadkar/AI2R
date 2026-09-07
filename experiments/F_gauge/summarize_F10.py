"""F8/F9/F10 summary tables + fig/F10_audit_heatmap.png from results/F8_identity.csv, F9_anchor.csv, audit_table.csv."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); FIG = os.path.join(R, "fig")
EXPECTED = {"weight_product": "invariant", "ph_raw_probe": "invariant(perm)/measured(scale)", "ph_raw_truthtable": "invariant(perm)/measured(scale)",
            "ph_std_probe": "invariant", "inv_poly_frobenius": "invariant", "inv_poly_gram_det": "invariant", "inv_poly_roundtrip": "invariant", "loop_ratios": "invariant",
            "singular_values": "covariant", "per_unit_activation": "artifact", "per_unit_gradxinput": "artifact", "topk_units": "artifact", "weight_norms": "artifact"}


def main():
    out = {}
    f8 = pd.read_csv(os.path.join(R, "F8_identity.csv")); f9 = pd.read_csv(os.path.join(R, "F9_anchor.csv"))
    out["F8"] = dict(n_specimens=int(len(f8)), identity_ok_frac=float(f8.identity_ok.mean()), max_dev=float(f8[["max_dev_perm", "max_dev_scale", "max_dev_combined"]].max().max()))
    out["F9"] = dict(n_specimens=int(len(f9)), anchor_ok_frac=float(f9.anchor_ok.mean()), max_cv=float(f9[["cv_perm", "cv_scale", "cv_combined"]].max().max()))
    A = pd.read_csv(os.path.join(R, "audit_table.csv")); A = A[A.error.isna() | (A.error == "")]
    A["base_metric"] = A.metric.str.replace(r"_L\d+", "", regex=True)
    rows = []
    for m, g in A.groupby("base_metric"):
        vc = g.verdict.value_counts(normalize=True); va = g.verdict_aligned.dropna()
        rows.append(dict(metric=m, n=len(g), majority=vc.index[0], agreement=float(vc.iloc[0]), frac_invariant=float((g.verdict == "invariant").mean()),
                         frac_covariant=float((g.verdict == "covariant").mean()), frac_artifact=float((g.verdict == "artifact").mean()),
                         cv_perm_med=float(g.cv_perm.median()), cv_scale_med=float(g.cv_scale.median()), cv_combined_med=float(g.cv_combined.median()),
                         exponent_med=float(g.scaling_exponent.median()) if g.scaling_exponent.notna().any() else np.nan, r2_med=float(g.scaling_r2.median()) if g.scaling_r2.notna().any() else np.nan,
                         aligned_majority=va.value_counts().index[0] if len(va) else "", aligned_frac_invariant=float((va == "invariant").mean()) if len(va) else np.nan,
                         perm_vs_combined_flip=float(((g.cv_perm < 0.01) != (g.cv_combined < 0.01)).mean()), expected=EXPECTED.get(m, "")))
    S = pd.DataFrame(rows).sort_values("metric"); S.to_csv(os.path.join(R, "F10_summary.csv"), index=False); print(S.to_string())
    json.dump(out, open(os.path.join(R, "F8_F9_summary.json"), "w"), indent=1); print(out)
    # heatmap: metric x kind, median log10 CV
    M = S.set_index("metric")[["cv_perm_med", "cv_scale_med", "cv_combined_med"]]
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(M) + 1.5), dpi=300)
    im = ax.imshow(np.log10(M.values + 1e-16), cmap="RdYlBu_r", vmin=-16, vmax=1, aspect="auto")
    ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=7); ax.set_xticks(range(3)); ax.set_xticklabels(["perm", "scale", "combined"])
    for i, m in enumerate(M.index):
        ax.text(3.1, i, S.set_index("metric").loc[m, "majority"], fontsize=7, va="center")
    plt.colorbar(im, ax=ax, label="log10 median CV"); ax.set_title("F10 audit: median per-component CV over specimens"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "F10_audit_heatmap.png")); plt.close(fig)


if __name__ == "__main__":
    main()
