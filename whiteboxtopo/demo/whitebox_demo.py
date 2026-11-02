#!/usr/bin/env python
"""WhiteBox / whiteboxtopo in 60 seconds: the four instruments on synthetic data, no model, no GPU.

    cd whiteboxtopo && pip install -e ".[legibility]" matplotlib && python demo/whitebox_demo.py

Prints one verdict line per instrument and writes demo/whitebox_demo.png (2x2 panel).  Everything is
computed in this run; nothing is hard-coded.  The data are the README quickstart's synthetic cases:
  1. invariants  - a "top neuron activation" reading audited under the ReLU-MLP gauge orbit
  2. gluing      - two planted explanation regimes; does the explanation glue (Omega vs shuffle null)?
  3. darkmatter  - a pure-noise residual cloud; the instrument must say "neither" (no structure)
  4. legibility  - a planted 4-block attribution graph; modularity + treewidth bound
"""
import time, pathlib, sys
import numpy as np

t0 = time.time()
rng = np.random.default_rng(0)
out = pathlib.Path(__file__).resolve().parent
print("whiteboxtopo demo -- four instruments, numpy-only (seed 0)")

# 1. invariants ---------------------------------------------------------------------------------
from whiteboxtopo.invariants import check_invariance_e2, ReluMLP
mlp = ReluMLP([rng.normal(size=(3, 6)), rng.normal(size=(6, 1))], [rng.normal(size=6), rng.normal(size=1)])
X = rng.uniform(-1, 1, size=(50, 3))
def raw_top_neuron(m, X):                      # a classic non-invariant reading
    h = np.maximum(X @ m.Ws[0] + m.bs[0], 0)
    return float(h[:, 0].mean())
def output_mean(m, X):                         # a function-level reading, for contrast
    h = np.maximum(X @ m.Ws[0] + m.bs[0], 0)
    return float((h @ m.Ws[1] + m.bs[1]).mean())
inv_bad = check_invariance_e2(lambda m: raw_top_neuron(m, X), mlp, n_reps=50)
inv_good = check_invariance_e2(lambda m: output_mean(m, X), mlp, n_reps=50)
print(f"[1 invariants]  'top neuron activation': verdict={inv_bad.verdict}, broken_by={inv_bad.broken_by}, CV={inv_bad.cv:.3g}"
      f"   |   'network output mean': verdict={inv_good.verdict}, CV={inv_good.cv:.2g}")

# 2. gluing -------------------------------------------------------------------------------------
from whiteboxtopo.gluing import explanation_obstruction
n = 400
scores = rng.normal(0, 0.3, size=(n, 20))
scores[: n // 2, :5] += 5.0
scores[n // 2 :, 10:15] += 5.0
emb = np.c_[np.r_[np.zeros(n // 2), np.ones(n - n // 2)], rng.normal(0, 0.1, n)]
g = explanation_obstruction(scores, embedding=emb, k=4, k_e=8, n_shuffle=40, n_boot=40)
print(f"[2 gluing]      Omega={g.omega:.3g} (CI {g.omega_ci[0]:.3g}..{g.omega_ci[1]:.3g}), shuffle null [{g.null_lo:.3g}, {g.null_hi:.3g}], "
      f"p_null={g.p_null:.3g}, significant_vs_null={g.significant_vs_null}, nerve edges={len(g.edges)}, frustrated={g.n_frustrated_edges}")

# 3. darkmatter ---------------------------------------------------------------------------------
from whiteboxtopo.darkmatter import residual_structure
errors = rng.standard_normal((1200, 60))         # e.g. e = x - SAE(x); here pure noise
dm = residual_structure(errors, n_points=400, n_boot=5, n_components=10)
print(f"[3 darkmatter]  verdict={dm.verdict!r}, MP outliers={dm.n_outliers}/{dm.p} (CI {dm.n_outliers_ci}), "
      f"lambda+={dm.mp_edge:.3g}, TC ratio={dm.tc_ratio:.3g}" if dm.topology_run else
      f"[3 darkmatter]  verdict={dm.verdict!r}, MP outliers={dm.n_outliers}/{dm.p}, lambda+={dm.mp_edge:.3g}")

# 4. legibility ---------------------------------------------------------------------------------
from whiteboxtopo.legibility import modularity_q, treewidth_bound, symmetrise
N = 40
lab = np.arange(N) // 10
A = symmetrise((lab[:, None] == lab[None, :]) * (rng.random((N, N)) < 0.6))
Q, membership = modularity_q(A)
tw = treewidth_bound(A)
print(f"[4 legibility]  modularity Q={Q:.3f} ({len(set(membership))} communities found, 4 planted), treewidth bound={tw}")
print(f"done in {time.time() - t0:.1f} s")

# figure ----------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("WhiteBox / whiteboxtopo -- four instruments, numpy-only, seed 0", fontsize=13)
a = ax[0, 0]
r_bad = np.abs(inv_bad.values.ravel() / inv_bad.base_value) + 1e-12
r_good = np.abs(inv_good.values.ravel() / inv_good.base_value) + 1e-12
bins = np.logspace(np.log10(min(r_bad.min(), 0.5)), np.log10(max(r_bad.max(), 2.0)), 30)
a.hist(r_bad, bins=bins, color="#c0392b", alpha=.75, label=f"top-neuron reading ({inv_bad.verdict}, CV={inv_bad.cv:.2f})")
a.axvline(1.0, color="#2980b9", lw=3, label=f"output mean: all 50 samples at 1.0 ({inv_good.verdict})")
a.set_xscale("log"); a.set_xlabel("metric / base value over 50 gauge-orbit samples"); a.set_title("1. invariants: is the metric about the function?"); a.legend(fontsize=8)
a = ax[0, 1]
a.bar(["observed Ω"], [g.omega], color="#8e44ad", yerr=[[g.omega - g.omega_ci[0]], [g.omega_ci[1] - g.omega]], capsize=6)
a.axhspan(g.null_lo, g.null_hi, color="grey", alpha=.3, label="shuffle null (2.5–97.5 %)")
a.set_title(f"2. gluing: obstruction Ω, p_null={g.p_null:.3g}"); a.legend(fontsize=8)
a = ax[1, 0]
ev = np.sort(dm.eigenvalues)[::-1]
a.plot(ev, "o-", ms=3, color="#16a085", label="residual covariance spectrum")
a.axhline(dm.mp_edge, color="k", ls="--", label=f"Marchenko–Pastur edge λ+={dm.mp_edge:.2f}")
a.set_title(f"3. darkmatter: verdict '{dm.verdict}', {dm.n_outliers} outliers"); a.set_xlabel("component"); a.legend(fontsize=8)
a = ax[1, 1]
order = np.argsort(membership)
a.imshow(A[np.ix_(order, order)], cmap="Greys", interpolation="nearest")
a.set_title(f"4. legibility: Q={Q:.2f}, treewidth ≤ {tw}"); a.set_xlabel("nodes sorted by found community")
fig.tight_layout()
fig.savefig(out / "whitebox_demo.png", dpi=110)
print(f"figure -> {out / 'whitebox_demo.png'}")
