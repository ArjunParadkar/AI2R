"""Aggregate E1 results: per-cell pooled ratios with seed bootstrap, H-E1a/b verdicts, figures."""
import os, sys, glob, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1lib.gauge import orbit_counts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = [0.01, 0.02, 0.05, 0.1, 0.2]
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rng = np.random.default_rng(0)
B = 1000


def load_cell(fname, d, w):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, "networks", f"{fname}_d{d}_w{w}_s*.recs.pkl"))):
        seed = int(f.split("_s")[-1].split(".")[0])
        with open(f, "rb") as fh:
            out[seed] = pickle.load(fh)
    return out


def pooled_ratio(recs_by_seed, scheme, eps, seeds):
    recs = [r for s in seeds for r in recs_by_seed[s][scheme]]
    oc = orbit_counts(recs, EPS)
    return (oc[f"orbits_eps{eps}"] / oc["n_expl"] if oc["n_expl"] else np.nan), oc


def _load_parts(stem):
    fs = sorted(f for f in glob.glob(os.path.join(RES, f"{stem}*.csv")) if os.path.basename(f) != f"{stem}.csv")
    return pd.concat([pd.read_csv(os.path.join(RES, f"{stem}.csv"))] + [pd.read_csv(f) for f in fs], ignore_index=True)


per = _load_parts("e1_per_network")
cells = per[per.scheme.isin(["zero", "mean"])][["function", "depth", "width"]].drop_duplicates()
rows = []
for _, c in cells.iterrows():
    rb = load_cell(c.function, c.depth, c.width)
    seeds = sorted(rb)
    for scheme in ("zero", "mean"):
        r, oc = pooled_ratio(rb, scheme, 0.05, seeds)
        # amendment 5: resampled seeds are de-duplicated — a network duplicated by the resample is gauge-identical
        # to itself and would fake orbit collapse; this is a subsampling bootstrap over unique seed subsets.
        boots = [pooled_ratio(rb, scheme, 0.05, sorted(set(rng.choice(seeds, len(seeds)))))[0] for _ in range(B)]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        sub = per[(per.function == c.function) & (per.depth == c.depth) & (per.width == c.width) & (per.scheme == scheme)]
        row = {"function": c.function, "depth": c.depth, "width": c.width, "scheme": scheme,
               "n_seeds": len(seeds), "pooled_n_expl": oc["n_expl"], "pooled_n_circuits": oc["n_circuits"],
               "pooled_ratio_eps0.05": r, "ratio_ci_lo": lo, "ratio_ci_hi": hi,
               "H_E1a_pass": bool(r <= 0.5 and hi < 0.5),
               "within_ratio_median": sub["ratio_eps0.05"].median(),
               "median_n_minimal": sub["n_minimal"].median(), "median_n_expl": sub["n_expl"].median(),
               "median_interp_per_circuit": (sub["n_expl"] / sub["n_minimal"].replace(0, np.nan)).median(),
               "uninterpretable_rate": sub["n_uninterpretable"].sum() / max(sub["n_minimal"].sum(), 1),
               "truncated": int(sub["n_truncated"].sum())}
        for e in EPS:
            row[f"pooled_ratio_eps{e}"] = oc[f"orbits_eps{e}"] / oc["n_expl"] if oc["n_expl"] else np.nan
            row[f"pooled_circuit_orbits_eps{e}"] = oc[f"circuit_orbits_eps{e}"]
        row["pooled_ratio_functional"] = oc["orbits_functional"] / oc["n_expl"] if oc["n_expl"] else np.nan
        row["pooled_circuit_orbits_functional"] = oc["circuit_orbits_functional"]
        rows.append(row)
cells_df = pd.DataFrame(rows).sort_values(["function", "depth", "width", "scheme"])
cells_df.to_csv(os.path.join(RES, "e1_cells.csv"), index=False)

# H-E1b: slope of log(pooled count) vs width, orbits vs explanations, seed bootstrap
hrows = []
for (fname, d), g in cells_df[cells_df.scheme == "mean"].groupby(["function", "depth"]):
    if g.width.nunique() < 3:
        continue
    rbs = {w: load_cell(fname, d, w) for w in g.width}
    def slopes(seed_sets):
        we, wo = [], []
        for w in sorted(rbs):
            recs = [r for s in seed_sets[w] for r in rbs[w][s]["mean"]]
            oc = orbit_counts(recs, EPS)
            if oc["n_expl"] > 0:
                we.append((w, np.log(oc["n_expl"]))); wo.append((w, np.log(oc["orbits_eps0.05"])))
        if len(we) < 3:
            return np.nan, np.nan
        se = np.polyfit(*zip(*we), 1)[0]; so = np.polyfit(*zip(*wo), 1)[0]
        return se, so
    full = {w: sorted(rbs[w]) for w in rbs}
    se, so = slopes(full)
    diffs = []
    for _ in range(B):
        ss = {w: sorted(set(rng.choice(full[w], len(full[w])))) if full[w] else [] for w in rbs}
        a, b = slopes(ss); diffs.append(b - a)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    hrows.append({"function": fname, "depth": d, "slope_expl": se, "slope_orbits": so,
                  "diff": so - se, "diff_ci_lo": lo, "diff_ci_hi": hi, "H_E1b_pass": bool(hi < 0)})
pd.DataFrame(hrows).to_csv(os.path.join(RES, "e1_h1b.csv"), index=False)

# Figures
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
mk = {2: "o", 3: "^"}
funcs = sorted(cells_df.function.unique())
cols = plt.cm.tab10(np.linspace(0, 1, 10))
for i, f in enumerate(funcs):
    g = cells_df[(cells_df.function == f) & (cells_df.scheme == "mean")]
    ax.scatter(g.pooled_n_expl, g["pooled_ratio_eps0.05"] * g.pooled_n_expl, s=8 * g.width,
               marker="o", color=cols[i], label=f, alpha=0.8)
    g = cells_df[(cells_df.function == f) & (cells_df.scheme == "zero")]
    ax.scatter(g.pooled_n_expl, g["pooled_ratio_eps0.05"] * g.pooled_n_expl, s=8 * g.width,
               marker="x", color=cols[i], alpha=0.8)
lim = [1, max(cells_df.pooled_n_expl.max(), 10) * 2]
ax.plot(lim, lim, "k--", lw=0.8, label="orbits = explanations")
ax.plot(lim, [l / 2 for l in lim], "k:", lw=0.8, label="H-E1a boundary (½)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("pooled explanations per cell (24 seeds)"); ax.set_ylabel("pooled gauge orbits (ε = 0.05)")
ax.set_title("E1: orbits vs explanations\n(o mean-ablation, x zero-ablation; size = width)", fontsize=9)
ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_orbits_vs_expl.png"))

fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=300, sharey=True)
for ax, key, title in [(axes[0], "pooled_ratio_eps0.05", "parameter gauge, ε=0.05"), (axes[1], "pooled_ratio_functional", "functional key")]:
    for i, f in enumerate(funcs):
        for d in (2, 3):
            g = cells_df[(cells_df.function == f) & (cells_df.scheme == "mean") & (cells_df.depth == d)].sort_values("width")
            if len(g):
                ax.errorbar(g.width, g[key], yerr=[(g[key] - g.ratio_ci_lo).clip(lower=0), (g.ratio_ci_hi - g[key]).clip(lower=0)] if key.endswith("0.05") else None,
                            marker=mk[d], color=cols[i], label=f"{f} d{d}", lw=1, ms=4, capsize=2)
    ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.set_xlabel("width"); ax.set_title(title)
axes[0].set_ylabel("orbits / explanations (pooled)"); axes[1].legend(fontsize=5, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_ratio_vs_width.png"))

fd = _load_parts("e1_funcdim")
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
for i, f in enumerate(funcs):
    g = fd[fd.function == f]
    ax.scatter(g.width + 0.1 * i + 0.05 * rng.normal(size=len(g)), g.unexplained_deficit, s=8, color=cols[i], label=f)
ax.set_xlabel("width"); ax.set_ylabel("functional-dimension deficit beyond scaling + dead units")
ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_funcdim.png"))
print(cells_df[["function", "depth", "width", "scheme", "n_seeds", "pooled_n_expl", "pooled_n_circuits", "pooled_ratio_eps0.05", "ratio_ci_lo", "ratio_ci_hi", "H_E1a_pass", "pooled_ratio_functional", "median_interp_per_circuit", "truncated"]].to_string())
print(pd.DataFrame(hrows).to_string())
