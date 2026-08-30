"""Aggregate E2 per-network CVs into the invariance table, per-cell CIs, scaling verdicts,
the untrained-null comparison, and 300-dpi figures. Verdict thresholds from SPEC."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e2lib.quantities import QUANTITY_INFO  # noqa: E402
from e2lib.core import cv_of, EXACT_CV, BROKEN_CV  # noqa: E402

RES = os.path.join(HERE, "results")
GROUPS = ["perm", "scale", "combined"]
COL = {"perm": "#2a78d6", "scale": "#eb6834", "combined": "#1baf7a"}
rng = np.random.default_rng(0)


def boot_median(x, n=1000):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def verdict(med, hi):
    if np.isnan(med):
        return "n/a"
    if not np.isfinite(med):
        return "not_invariant"
    if med < EXACT_CV and (np.isnan(hi) or hi < EXACT_CV):
        return "exact"
    if med > BROKEN_CV:
        return "not_invariant"
    return "approximate"


def between_network_cv(bases):
    """Median over cells of the relative dispersion of base values across the cell's networks."""
    out = {}
    df = pd.DataFrame([{k: v for k, v in b.items() if not k.startswith("base::")} for b in bases])
    keys = [k for k in bases[0] if k.startswith("base::")]
    for k in keys:
        per_cell = []
        for (f, d, w), idx in df.groupby(["function", "depth", "width"]).groups.items():
            vals = [bases[i][k] for i in idx]
            if len(vals) < 2:
                continue
            per_cell.append(cv_of(np.stack(vals)))
        out[k[6:]] = float(np.nanmedian(per_cell)) if per_cell else np.nan
    return out


def build_table(pn, bases, suffix=""):
    pn = pn[pn.converged].copy()
    bcv = between_network_cv(bases)
    rows = []
    for q in sorted(pn.quantity.unique()):
        info = QUANTITY_INFO.get(q, ("indictment_discrete", "N6", False, "gate-of-neuron-i agreement (raw index)"))
        rec = {"quantity": q, "family": info[0], "spec_id": info[1], "expected_invariant": info[2],
               "description": info[3], "between_network_cv": bcv.get(q, np.nan)}
        for g in GROUPS:
            s = pn[(pn.quantity == q) & (pn.group == g)]
            med, lo, hi = boot_median(s.cv.values)
            rec[f"cv_{g}"] = med; rec[f"cv_{g}_lo"] = lo; rec[f"cv_{g}_hi"] = hi
            rec[f"agree_{g}"] = float(np.nanmedian(s.agreement.values)) if len(s) else np.nan
            rec[f"verdict_{g}"] = verdict(med, hi)
            rec["n_networks"] = int(s.shape[0])
        broken = [g for g in ("perm", "scale") if rec[f"verdict_{g}"] == "not_invariant"]
        rec["broken_by"] = "+".join(broken) if broken else ("none" if rec["verdict_combined"] == "exact" else "combined-only")
        rec["constant_across_networks"] = bool(rec["between_network_cv"] < EXACT_CV) if not np.isnan(rec["between_network_cv"]) else False
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, f"e2_invariance_table{suffix}.csv"), index=False)
    return t


def build_cells(pn):
    pn = pn[pn.converged]
    rows = []
    for (f, d, w, g, q), s in pn.groupby(["function", "depth", "width", "group", "quantity"]):
        med, lo, hi = boot_median(s.cv.values, n=300)
        rows.append({"function": f, "depth": d, "width": w, "group": g, "quantity": q, "n": len(s),
                     "cv_median": med, "cv_lo": lo, "cv_hi": hi, "agreement_median": float(np.nanmedian(s.agreement)),
                     "verdict": verdict(med, hi)})
    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(RES, "e2_cells.csv"), index=False)
    return c


def scaling_verdicts(sc, table):
    rows = []
    for q, s in sc.groupby("quantity"):
        pos = s[s.all_positive]
        n_pos = len(pos)
        const = s.constant.mean()
        if n_pos:
            k = pos.exponent.values
            r2 = pos.r2.values
            integer_ok = (np.abs(k - np.round(k)) < 0.01) & (r2 > 0.999)
            ints = np.round(k[integer_ok]).astype(int)
            same_int = len(set(ints.tolist())) == 1 if len(ints) else False
            k_med = float(np.median(k)); frac_ok = float(integer_ok.mean())
            k_int = int(ints[0]) if same_int else None
        else:
            k_med, frac_ok, same_int, k_int = np.nan, np.nan, False, None
        cv_scale = float(table.set_index("quantity").loc[q, "cv_scale"]) if q in table.quantity.values else np.nan
        if cv_scale < EXACT_CV:
            v = "scaling-invariant"
        elif (n_pos and frac_ok == 1.0 and same_int and k_int == 0) or const > 0.99:
            v = f"invariant to uniform scaling (exponent 0); NOT invariant under per-unit scaling (CV={cv_scale:.2f})"
        elif n_pos and frac_ok == 1.0 and same_int:
            v = f"scaling-covariant (degree {k_int}) under uniform scaling; NOT invariant under per-unit scaling (CV={cv_scale:.2f})"
        elif False:
            v = f"constant along uniform sweep (invariant to uniform scaling); NOT invariant under per-unit scaling (CV={cv_scale:.2f})"
        else:
            v = f"neither (uniform-sweep exponent median {k_med:.3f}, integer-fit fraction {frac_ok:.2f}); per-unit CV={cv_scale:.2f}"
        rows.append({"quantity": q, "n_fits": len(s), "n_all_positive": n_pos, "frac_constant_along_sweep": float(const),
                     "exponent_median": k_med, "frac_integer_fit": frac_ok, "integer": k_int, "cv_per_unit_scale": cv_scale,
                     "verdict": v})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "e2_scaling_verdicts.csv"), index=False)
    return out


def fig_cv(table, path):
    order = [q for q in QUANTITY_INFO if q in table.quantity.values] + ["gates_index_agreement"]
    order = [q for q in order if q in table.quantity.values and q != "gates_index_agreement"]
    t = table.set_index("quantity").loc[order]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(order) + 1.5))
    y = np.arange(len(order))
    floor = 1e-17
    for i, g in enumerate(GROUPS):
        v = np.clip(t[f"cv_{g}"].values.astype(float), floor, None)
        lo = np.clip(t[f"cv_{g}_lo"].values.astype(float), floor, None)
        hi = np.clip(t[f"cv_{g}_hi"].values.astype(float), floor, None)
        v = np.where(np.isfinite(v), v, 1e2); lo = np.where(np.isfinite(lo), lo, v); hi = np.where(np.isfinite(hi), hi, v)
        ax.errorbar(v, y + (i - 1) * 0.22, xerr=[v - lo, hi - v], fmt="o", ms=4, color=COL[g], label=g, lw=1)
    ax.axvline(EXACT_CV, color="#888", ls="--", lw=1); ax.text(EXACT_CV, -1.0, "exact (1e-6)", fontsize=7, color="#555", ha="center")
    ax.axvline(BROKEN_CV, color="#888", ls=":", lw=1); ax.text(BROKEN_CV, -1.0, "broken (0.05)", fontsize=7, color="#555", ha="center")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("median CV across orbit representatives (bootstrap 95% CI over networks)")
    ax.set_title("E2: gauge-orbit coefficient of variation per quantity", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_scaling(sweep_raw, sc, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    sub = sweep_raw[sweep_raw.layer == 0]
    keys = list(sub.groupby(["function", "depth", "width", "seed"]).groups)[:12]
    for i, k in enumerate(keys):
        s = sub[(sub.function == k[0]) & (sub.depth == k[1]) & (sub.width == k[2]) & (sub.seed == k[3])].sort_values("alpha")
        ax.plot(s.alpha, s.tc0_box, "-", color="#2a78d6", alpha=0.6, lw=1.2, label="TC(H0), box" if i == 0 else None)
        s1 = s[s.tc1_box > 0]
        ax.plot(s1.alpha, s1.tc1_box, "-", color="#eb6834", alpha=0.6, lw=1.2, label="TC(H1), box" if i == 0 else None)
    a = np.logspace(-2, 2, 9)
    ax.plot(a, a ** 2, "k--", lw=1, label="slope 2 reference")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("uniform scale α of hidden layer 0"); ax.set_ylabel("TC")
    ax.set_title("TC vs uniform layer scaling (12 example networks)", fontsize=9); ax.legend(fontsize=7)
    ax = axes[1]
    for q, c in (("tc0_box", "#2a78d6"), ("tc1_box", "#eb6834"), ("pe0_box", "#1baf7a"), ("b0_box", "#eda100")):
        s = sc[(sc.quantity == q) & sc.all_positive]
        if len(s):
            ax.hist(s.exponent, bins=np.linspace(-0.5, 2.5, 61), color=c, alpha=0.6, label=f"{q} (n={len(s)})")
    ax.set_xlabel("fitted exponent k in TC ∝ α^k (per network, per layer)"); ax.set_ylabel("count")
    ax.set_title("Uniform-scaling exponents", fontsize=9); ax.legend(fontsize=7)
    for a_ in axes:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_agreement(pn, path):
    pn = pn[pn.converged]
    qs = ["top_neuron_mag", "top_neuron_abl", "gates_index_agreement", "gates_sorted", "b1_tt"]
    fig, axes = plt.subplots(1, len(qs), figsize=(3 * len(qs), 3.2), sharey=True)
    for ax, q in zip(axes, qs):
        for g in GROUPS:
            s = pn[(pn.quantity == q) & (pn.group == g)].groupby("width").agreement.median()
            ax.plot(s.index, s.values, "o-", color=COL[g], label=g, ms=4)
        ax.set_title(q, fontsize=9); ax.set_xlabel("width"); ax.set_ylim(-0.02, 1.05)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("agreement rate with base network"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def markdown_table(t):
    lines = ["| id | quantity | expected | CV perm | CV scale | CV combined | between-net CV | verdict (combined) | broken by |",
             "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda x: ("inf" if not np.isfinite(x) else f"{x:.1e}") if not np.isnan(x) else "n/a"
    for _, r in t.iterrows():
        lines.append(f"| {r.spec_id} | `{r.quantity}` | {'inv' if r.expected_invariant else 'NOT'} | {fmt(r.cv_perm)} | {fmt(r.cv_scale)} | "
                     f"{fmt(r.cv_combined)} | {fmt(r.between_network_cv)} | {r.verdict_combined}"
                     f"{' (CONSTANT)' if r.constant_across_networks else ''} | {r.broken_by} |")
    return "\n".join(lines)


def main():
    pn = pd.read_csv(os.path.join(RES, "e2_per_network.csv"))
    bases = pd.read_pickle(os.path.join(RES, "e2_base_values.pkl"))
    sc = pd.read_csv(os.path.join(RES, "e2_scaling.csv"))
    sweep_raw = pd.read_csv(os.path.join(RES, "e2_sweep_raw.csv"))
    t = build_table(pn, bases)
    build_cells(pn)
    sv = scaling_verdicts(sc, t)
    fig_cv(t, os.path.join(RES, "fig_cv_per_quantity.png"))
    fig_scaling(sweep_raw, sc, os.path.join(RES, "fig_tc_scale_exponent.png"))
    fig_agreement(pn, os.path.join(RES, "fig_agreement.png"))
    print(f"networks: {pn[pn.converged].groupby(['function','depth','width','seed']).ngroups} converged, "
          f"{pn[~pn.converged].groupby(['function','depth','width','seed']).ngroups} dropped")
    print("param distance moved (median over nets of per-net median):")
    print(pn.groupby("group").param_dist_median.median().to_string())
    print(pn.groupby("group").param_dist_min.min().to_string())
    print(markdown_table(t))
    print(sv[["quantity", "n_all_positive", "exponent_median", "frac_integer_fit", "cv_per_unit_scale", "verdict"]].to_string())
    ut = os.path.join(RES, "e2_per_network_untrained.csv")
    if os.path.exists(ut):
        pu = pd.read_csv(ut); pu["converged"] = True
        bu = pd.read_pickle(os.path.join(RES, "e2_base_values_untrained.pkl"))
        tu = build_table(pu, bu, "_untrained")
        m = t.set_index("quantity"); mu = tu.set_index("quantity")
        diff = [(q, g, m.loc[q, f"verdict_{g}"], mu.loc[q, f"verdict_{g}"]) for q in m.index for g in GROUPS
                if q in mu.index and m.loc[q, f"verdict_{g}"] != mu.loc[q, f"verdict_{g}"]]
        print(f"untrained null: {len(diff)} verdict differences: {diff}")


if __name__ == "__main__":
    main()
