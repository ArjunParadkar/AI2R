"""F4a — cross-seed orbit test (SPEC_F4_reformulated §F4a). Writes results/F4a_pairs.csv (subsampled),
F4a_cells.csv (K(tau)/N, K_alg, K_fkey per cell), F4a_kills.json, figures."""
import os, sys, csv, json, re, itertools, time, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flib.models import MLP
from flib.functions import truth_table
from flib.crossseed import effective_subimpl, residual, size_signature
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); FIG = os.path.join(R, "fig")
TAUS = (1e-6, 1e-3, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5)
_models = {}


def model_of(spec):
    if spec not in _models: _models[spec] = MLP.load(os.path.join(R, "networks", f"{spec}.npz"))
    return _models[spec]


def alg_key(s):
    """Gate-composition string with unit labels stripped: per-layer sorted gate multisets."""
    if not isinstance(s, str) or not s: return ""
    parts = s.split("->"); out = []
    for p in parts:
        if p.startswith("out:"): out.append(p); continue
        gates = sorted(re.sub(r"u\d+:", "", p.strip("[]")).split(",")); out.append("[" + ",".join(gates) + "]")
    return "->".join(out)


def cell_job(args):
    cell, crit, recs, fn, max_pairs = args
    X, y = truth_table(fn); rng = np.random.default_rng(abs(hash((cell, crit))) % (1 << 31))
    subs = []
    for spec, mid in recs:
        try: subs.append(effective_subimpl(model_of(spec), X, int(mid), crit))
        except Exception: subs.append(None)
    N = len(recs); idx_by_sig = {}
    for i, s in enumerate(subs):
        if s is not None: idx_by_sig.setdefault(size_signature(s), []).append(i)
    parent = list(range(N))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    rs = []; K = {t: None for t in TAUS}; edges = {t: [] for t in TAUS}
    pair_rows = []
    for sig, ids in idx_by_sig.items():
        pairs = [(i, j) for i, j in itertools.combinations(ids, 2) if recs[i][0] != recs[j][0]]
        if len(pairs) > max_pairs: pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
        for i, j in pairs:
            r, _ = residual(subs[i], subs[j]); rs.append(r)
            for t in TAUS:
                if r < t: edges[t].append((i, j))
            if len(pair_rows) < 2000: pair_rows.append(dict(cell=cell, criterion=crit, spec_a=recs[i][0], spec_b=recs[j][0], r=r, sig=str(sig)))
    out = dict(cell=cell, criterion=crit, function=fn, N=N, n_seeds=len({s for s, _ in recs}), n_pairs=len(rs),
               r_median=float(np.median(rs)) if rs else np.nan, r_q10=float(np.quantile(rs, .1)) if rs else np.nan, r_min=float(np.min(rs)) if rs else np.nan)
    for t in TAUS:
        parent = list(range(N))
        for i, j in edges[t]:
            a, b = find(i), find(j)
            if a != b: parent[a] = b
        K = len({find(i) for i in range(N)}); out[f"K_{t}"] = K; out[f"ratio_{t}"] = K / N if N else np.nan
    return out, pair_rows, rs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--max-pairs", type=int, default=20000)
    ap.add_argument("--kills", action="store_true"); a = ap.parse_args()
    E = pd.read_csv(os.path.join(R, "explanations.csv"), dtype={"key6": str, "key4": str})
    sp = E.specimen.str.extract(r"^(?P<function>[A-Z0-9_]+)_d(?P<depth>\d)_w(?P<width>\d+)_s(?P<seed>\d+)$"); E = pd.concat([E, sp], axis=1)
    E["cell"] = E.function + "_d" + E.depth + "_w" + E.width; E["alg_key"] = E.algorithm.map(alg_key)
    if a.kills: return kills(E)
    jobs = []; alg_rows = []
    for (cell, crit), g in E.groupby(["cell", "criterion"]):
        recs = list(zip(g.specimen, g.circuit_id)); jobs.append((cell, crit, recs, g.function.iloc[0], a.max_pairs))
        gi = g[g.interpretable == 1]
        alg_rows.append(dict(cell=cell, criterion=crit, N=len(g), N_interp=len(gi), K_alg=gi.alg_key.nunique(), K_fkey=g.fkey.nunique(),
                             K_alg_ratio=gi.alg_key.nunique() / len(gi) if len(gi) else np.nan, K_fkey_ratio=g.fkey.nunique() / len(g)))
    print(len(jobs), "cells", flush=True)
    cells, pairs, all_r = [], [], {}
    with Pool(a.procs) as pool:
        for i, (out, prow, rs) in enumerate(pool.imap_unordered(cell_job, jobs)):
            cells.append(out); pairs += prow; all_r[(out["cell"], out["criterion"])] = rs
            if i % 20 == 0: print(i, out["cell"], out["criterion"], "N", out["N"], "pairs", out["n_pairs"], "r_med", round(out["r_median"], 3) if out["r_median"] == out["r_median"] else None, "K0.1/N", round(out["ratio_0.1"], 3), flush=True)
    C = pd.DataFrame(cells).merge(pd.DataFrame(alg_rows), on=["cell", "criterion", "N"]); C.to_csv(os.path.join(R, "F4a_cells.csv"), index=False)
    pd.DataFrame(pairs).to_csv(os.path.join(R, "F4a_pairs.csv"), index=False)
    # different-function null: pairs of circuits from different functions, same depth/width and same signature
    null_r = []
    rng = np.random.default_rng(0)
    for (dw, crit), g in E.assign(dw=E.depth + "_" + E.width).groupby(["dw", "criterion"]):
        fns = g.function.unique()
        if len(fns) < 2: continue
        for t in range(300):
            f1, f2 = rng.choice(fns, 2, replace=False); a_ = g[g.function == f1].sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]; b_ = g[g.function == f2].sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
            X1, _ = truth_table(f1); X2, _ = truth_table(f2)
            if X1.shape[1] != X2.shape[1]: continue
            sa = effective_subimpl(model_of(a_.specimen), X1, int(a_.circuit_id), crit); sb = effective_subimpl(model_of(b_.specimen), X2, int(b_.circuit_id), crit)
            r, _ = residual(sa, sb)
            if np.isfinite(r): null_r.append(dict(dw=dw, criterion=crit, r=r))
    NR = pd.DataFrame(null_r); NR.to_csv(os.path.join(R, "F4a_null_pairs.csv"), index=False)
    same = np.array([r for rs in all_r.values() for r in rs]); diff = NR.r.values
    from scipy import stats
    ks = stats.ks_2samp(same, diff, alternative="greater") if len(same) and len(diff) else None
    verdict = {}
    for crit in ("zero", "mean"):
        c = C[C.criterion == crit]
        verdict[crit] = dict(H_F4a_1_median_ratio_tau01=float(c["ratio_0.1"].median()), H_F4a_1_pass=bool(c["ratio_0.1"].median() <= 0.5),
                             H_F4a_2_median_Kalg_ratio=float(c.K_alg_ratio.median()), H_F4a_2_pass=bool(c.K_alg_ratio.median() <= c["ratio_0.1"].median()),
                             K_fkey_ratio_median=float(c.K_fkey_ratio.median()), n_cells=int(len(c)))
    verdict["K4_null"] = dict(same_function_r_median=float(np.median(same)) if len(same) else None, different_function_r_median=float(np.median(diff)) if len(diff) else None,
                              ks_stat=float(ks.statistic) if ks else None, ks_p_same_smaller=float(ks.pvalue) if ks else None, n_same=int(len(same)), n_diff=int(len(diff)))
    json.dump(verdict, open(os.path.join(R, "F4a_verdicts.json"), "w"), indent=1); print(json.dumps(verdict, indent=1))
    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=300); bins = np.linspace(0, 2, 60)
    ax.hist(same, bins=bins, alpha=.6, density=True, label="same function, different seeds"); ax.hist(diff, bins=bins, alpha=.6, density=True, label="different functions (null)")
    kp = os.path.join(R, "F4a_kills.json")
    if os.path.exists(kp):
        k = json.load(open(kp))
        for name, col in (("K2_delta_0.001", "#228833"), ("K2_delta_0.01", "#CCBB44"), ("K2_delta_0.03", "#EE6677"), ("K3_cross_algorithm", "#AA3377")):
            if name in k and k[name].get("r_median") is not None: ax.axvline(k[name]["r_median"], color=col, ls="--", lw=1, label=name)
    ax.set_xlabel("gauge residual r"); ax.set_ylabel("density"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, "F4a_residual_hist.png")); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=300)
    for ax, crit in zip(axes, ("zero", "mean")):
        c = C[C.criterion == crit]
        for fn, g in c.groupby("function"): ax.plot(TAUS, [g[f"ratio_{t}"].median() for t in TAUS], "o-", ms=3, lw=1, alpha=.7, label=fn)
        ax.set_xscale("log"); ax.set_xlabel("tau"); ax.set_ylabel("median K(tau)/N"); ax.set_title(crit); ax.axhline(0.5, color="k", ls=":", lw=1)
    axes[0].legend(fontsize=5, ncol=2); fig.tight_layout(); fig.savefig(os.path.join(FIG, "F4a_KN_vs_tau.png")); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=300)
    for fn, g in C[C.criterion == "zero"].groupby("function"):
        g = g.assign(width=g.cell.str.extract(r"_w(\d+)$")[0].astype(int)).sort_values("width"); ax.plot(g.width, g.K_alg_ratio, "o-", ms=3, lw=1, alpha=.7)
    ax.set_xlabel("width"); ax.set_ylabel("K_alg / N_interpretable (zero)"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "F4a_Kalg_vs_width.png")); plt.close(fig)


def kills(E):
    from flib.gauge import apply_gauge, random_gauge, permute_mask_id
    from flib.planted import xor3_single, _xor3_alg_B, _xor3_alg_C, with_bias_unit
    out = {}; rng = np.random.default_rng(0)
    conv = E.drop_duplicates("specimen")[["specimen", "function", "criterion"]]
    sel = conv.sample(min(100, len(conv)), random_state=0)
    # K1 exact relabelling; K2 noise
    r1 = []; r2 = {d: [] for d in (1e-3, 1e-2, 3e-2)}
    for _, row in sel.iterrows():
        m = model_of(row.specimen); X, y = truth_table(row.function); crit = "zero"
        mids = E[(E.specimen == row.specimen) & (E.criterion == crit)].circuit_id.values[:5]
        mg, perms, scales = random_gauge(m, rng, log_scale_sd=1.0)
        for mid in mids:
            sa = effective_subimpl(m, X, int(mid), crit); sb = effective_subimpl(mg, X, permute_mask_id(int(mid), perms, m.widths), crit)
            r1.append(residual(sa, sb)[0])
            for d in r2:
                mn = MLP([W + d * np.abs(W).mean() * rng.normal(size=W.shape) for W in m.Ws], [b + d * (np.abs(b).mean() + 1e-3) * rng.normal(size=b.shape) for b in m.bs])
                r2[d].append(residual(effective_subimpl(m, X, int(mid), crit, collapse=False), effective_subimpl(mn, X, int(mid), crit, collapse=False))[0])
    out["K1_exact_relabel"] = dict(n=len(r1), r_max=float(np.nanmax(r1)), pass_=bool(np.nanmax(r1) < 1e-9), n_inf=int(np.sum(~np.isfinite(r1))))
    for d, rs in r2.items():
        rs = np.array(rs); fin = rs[np.isfinite(rs)]
        out[f"K2_delta_{d}"] = dict(delta=d, n=len(rs), r_median=float(np.median(fin)) if len(fin) else None, r_over_delta=float(np.median(fin) / d) if len(fin) else None,
                                   n_inf=int((~np.isfinite(rs)).sum()), pass_=bool(len(fin) and 1 / 3 <= np.median(fin) / d <= 3))
    # K3 planted algorithms A/B/C (with bias units so each is one circuit)
    X, y = truth_table("XOR3"); algs = {"A": with_bias_unit(xor3_single()), "B": with_bias_unit(_xor3_alg_B()), "C": with_bias_unit(_xor3_alg_C())}
    subs = {k: effective_subimpl(m, X, (1 << sum(m.widths)) - 1, "zero", collapse=False) for k, m in algs.items()}
    rr = {f"{a}-{b}": residual(subs[a], subs[b])[0] for a, b in itertools.combinations(algs, 2)}
    fin = [v for v in rr.values() if np.isfinite(v)]
    out["K3_cross_algorithm"] = dict(residuals=rr, r_median=float(np.median(fin)) if fin else None, pass_=bool(all((not np.isfinite(v)) or v > 0.3 for v in rr.values())))
    json.dump(out, open(os.path.join(R, "F4a_kills.json"), "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
