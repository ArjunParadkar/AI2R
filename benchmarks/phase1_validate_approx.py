"""Phase 1(c): does the approximation preserve the signal?

Ground truth = exact VR PH (ripser, full cloud, NO threshold) on clouds of
n=500. Approximations compared: the original 95th-percentile threshold,
maxmin landmarks (k=100/200/300), random subsample mean (k=100/200/300, 10
draws). Clouds: six synthetic manifolds x 3 noise seeds, plus real GPT-2 124M
activations (layers 0/6/12, 500 sentences, PCA->30) and their label-noise-free
vs shuffled-feature 'collapsed' counterparts.

Reports, per approximation: Spearman rho and Pearson r of log TC vs exact
across clouds, median relative TC error, beta_1 agreement, W2 to exact H1
diagram, and time. CSV + 300 dpi figure.
"""
from __future__ import annotations
import csv, json, os, time
import numpy as np
from scipy.stats import spearmanr, pearsonr
import whitebox as wb
from whitebox.metrics.sv import diagram_distance
from common import SYNTHETIC, embed, probe_sentences, gpt2_activations, RESULTS

N = 500


def build_clouds():
    clouds = {}
    for name, f in SYNTHETIC.items():
        for s in range(3):
            clouds[f"{name}/s{s}"] = embed(f(N, seed=s), d=30, noise=0.05, seed=s)
    try:
        sents = probe_sentences(N)
        _, acts = gpt2_activations("gpt2", sents, [0, 6, 12])
        for l, A in acts.items():
            r = wb.fit_reducer(A, 30); Z = r.transform(A)
            clouds[f"gpt2/L{l}"] = Z
            rng = np.random.default_rng(l)
            clouds[f"gpt2/L{l}/shuffled"] = np.stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])], 1)
    except Exception as e:  # pragma: no cover
        print("GPT-2 clouds skipped:", e)
    return clouds


def measure(D, exact_D1=None):
    d1 = D[1]
    out = {"TC": wb.topological_complexity(d1), "B1": wb.betti_at_scale(d1, wb.median_scale(d1)),
           "FPL3": wb.top_k_lifetime(d1, 3), "n_bars": len(d1)}
    if exact_D1 is not None:
        out["W2_to_exact"] = diagram_distance(d1, exact_D1)
    return out


def main():
    clouds = build_clouds(); rows = []
    approx = [("thresh95", dict(thresh_percentile=95, backend="ripser")),
              ("gph_collapse", dict(thresh_percentile=100, backend="giotto-ph"))]
    approx += [(f"landmark{k}", dict(thresh_percentile=100, backend="landmark", n_landmarks=k)) for k in (100, 200, 300)]
    for cname, X in clouds.items():
        t0 = time.time(); ex = wb.compute_diagrams(X, thresh_percentile=100, backend="ripser"); te = time.time() - t0
        rows.append(dict(cloud=cname, method="exact", seconds=te, **measure(ex, ex[1])))
        for mname, kw in approx:
            t0 = time.time(); D = wb.compute_diagrams(X, **kw); rows.append(dict(cloud=cname, method=mname, seconds=time.time() - t0, **measure(D, ex[1])))
        for k in (100, 200, 300):
            t0 = time.time(); ms = []
            idx = wb.core.bootstrap.subsample_indices(len(X), k, 10, 0)
            for i in idx:
                ms.append(measure(wb.compute_diagrams(X[i], thresh_percentile=100, backend="giotto-ph"), ex[1]))
            agg = {key: float(np.mean([m[key] for m in ms])) for key in ms[0]}
            rows.append(dict(cloud=cname, method=f"subsample{k}x10", seconds=time.time() - t0, **agg))
        print(f"{cname:28s} exact TC={rows[-9]['TC']:.3g} B1={rows[-9]['B1']} ({te:.1f}s)", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(f"{RESULTS}/phase1_approx_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # summary per method
    exact = {r["cloud"]: r for r in rows if r["method"] == "exact"}
    summ = []
    for m in sorted({r["method"] for r in rows} - {"exact"}):
        rs = [r for r in rows if r["method"] == m]
        a = np.array([exact[r["cloud"]]["TC"] for r in rs]); b = np.array([r["TC"] for r in rs])
        la, lb = np.log(a + 1e-12), np.log(b + 1e-12)
        relerr = np.abs(b - a) / np.maximum(a, 1e-12)
        b1a = np.array([exact[r["cloud"]]["B1"] for r in rs]); b1b = np.array([r["B1"] for r in rs])
        summ.append(dict(method=m, n_clouds=len(rs), spearman_logTC=spearmanr(la, lb)[0], pearson_logTC=pearsonr(la, lb)[0],
                         median_rel_TC_err=float(np.median(relerr)), max_rel_TC_err=float(relerr.max()),
                         B1_spearman=spearmanr(b1a, b1b)[0], B1_exact_match_frac=float(np.mean(b1a == b1b)),
                         median_W2_to_exact=float(np.median([r["W2_to_exact"] for r in rs])),
                         median_seconds=float(np.median([r["seconds"] for r in rs])),
                         exact_median_seconds=float(np.median([exact[r["cloud"]]["seconds"] for r in rs]))))
    with open(f"{RESULTS}/phase1_approx_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    print(json.dumps(summ, indent=1))
    # figure: log TC approx vs exact, one panel per method
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ms = [s["method"] for s in summ]; fig, axes = plt.subplots(2, 4, figsize=(14, 7)); axes = axes.ravel()
    for ax, m in zip(axes, ms):
        rs = [r for r in rows if r["method"] == m]
        a = [exact[r["cloud"]]["TC"] for r in rs]; b = [r["TC"] for r in rs]
        col = ["C1" if r["cloud"].startswith("gpt2") else "C0" for r in rs]
        ax.scatter(a, b, s=14, c=col); lim = [min(min(a), min(b)) * 0.5 + 1e-6, max(max(a), max(b)) * 2]
        ax.plot(lim, lim, "k--", lw=0.7); ax.set_xscale("log"); ax.set_yscale("log")
        s = next(x for x in summ if x["method"] == m)
        ax.set_title(f"{m}\nρ={s['spearman_logTC']:.3f} relerr={s['median_rel_TC_err']:.2f} t={s['median_seconds']:.2f}s", fontsize=9)
        ax.set_xlabel("exact TC"); ax.set_ylabel("approx TC")
    for ax in axes[len(ms):]: ax.axis("off")
    fig.suptitle("Phase 1: approximate vs exact TC (blue synthetic, orange GPT-2 activations), n=500")
    fig.tight_layout(); fig.savefig(f"{RESULTS}/phase1_approx_vs_exact.png", dpi=300)


if __name__ == "__main__":
    main()
