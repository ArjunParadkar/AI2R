"""Z4b (Z AMENDMENTS #1, 2026-09-09) — the two pieces PAUSE_5 §Z4 flagged as missing: (i) F4a's scale gauge-fix before
matching (every hidden unit's incoming vector (w_u, b_u) normalised to unit length, outgoing row rescaled inversely —
an exact symmetry), and (ii) a calibrated null (random unit assignment instead of Hungarian, in the same gauge-fixed
frame). Same 380 pairs as Z4 (same seed, same sampling loop). Writes results/Z4b_paths.csv, Z4b_summary.json."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_Z4_Z5 import F, R, MLP, truth_table, valid
from flib.gauge import apply_gauge
from scipy.optimize import linear_sum_assignment


def scale_fix(m):
    scales = [1.0 / np.sqrt((m.Ws[l] ** 2).sum(0) + m.bs[l] ** 2) for l in range(m.depth)]
    return apply_gauge(m, None, scales)


def match(m_a, m_b, rng=None):
    perms = []
    for l in range(m_a.depth):
        A = np.vstack([m_a.Ws[l], m_a.bs[l][None]]); B = np.vstack([m_b.Ws[l], m_b.bs[l][None]])
        if rng is None:
            cost = ((A.T[:, None, :] - B.T[None, :, :]) ** 2).sum(-1); ri, ci = linear_sum_assignment(cost); p = np.empty(A.shape[1], int); p[ri] = ci
        else:
            p = rng.permutation(A.shape[1])
        perms.append(p)
    return apply_gauge(m_b, perms, None)


def path_stats(ma, mb, X, y, mid, crit):
    ok = np.array([valid(MLP([(1 - s) * Wa + s * Wb for Wa, Wb in zip(ma.Ws, mb.Ws)], [(1 - s) * ba + s * bb for ba, bb in zip(ma.bs, mb.bs)]), X, y, mid, crit)
                   for s in np.linspace(0, 1, 50)])
    return dict(frac_valid_path=float(ok.mean()), valid_at_end=bool(ok[-1]), first_break=float(np.argmin(ok) / 49) if not ok.all() else 1.0)


def main():
    E = pd.read_csv(os.path.join(F, "results", "explanations.csv"), dtype={"key6": str, "key4": str})
    sp = E.specimen.str.extract(r"^(?P<function>[A-Z0-9_]+)_d(?P<depth>\d)_w(?P<width>\d+)_s(?P<seed>\d+)$"); E = pd.concat([E, sp], axis=1)
    E["cell"] = E.function + "_d" + E.depth + "_w" + E.width; rows = []; rng = np.random.default_rng(0); nrng = np.random.default_rng(1)
    for (cell, crit), g in E[E.interpretable == 1].groupby(["cell", "criterion"]):
        specs = g.specimen.unique()
        if len(specs) < 2: continue
        for t in range(10):
            a, b = rng.choice(specs, 2, replace=False); ra = g[g.specimen == a].sample(1, random_state=t).iloc[0]
            X, y = truth_table(ra.function); mid = int(ra.circuit_id)
            ma = MLP.load(os.path.join(F, "results", "networks", f"{a}.npz")); mb = MLP.load(os.path.join(F, "results", "networks", f"{b}.npz"))
            base = dict(cell=cell, criterion=crit, spec_a=a, spec_b=b, circuit_id=mid)
            rows.append(dict(base, condition="hungarian_raw", **path_stats(ma, match(ma, mb), X, y, mid, crit)))
            fa, fb = scale_fix(ma), scale_fix(mb)
            assert valid(fa, X, y, mid, crit) == valid(ma, X, y, mid, crit)          # the scale fix is an exact symmetry: validity unchanged
            rows.append(dict(base, condition="scalefix_hungarian", **path_stats(fa, match(fa, fb), X, y, mid, crit)))
            rows.append(dict(base, condition="scalefix_random_perm", **path_stats(fa, match(fa, fb, nrng), X, y, mid, crit)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(R, "Z4b_paths.csv"), index=False)
    summ = {}
    for cond, d in df.groupby("condition"):
        summ[cond] = dict(n_pairs=int(len(d)), frac_pairs_valid_path=float((d.frac_valid_path >= 0.9).mean()), frac_valid_at_end=float(d.valid_at_end.mean()),
                          median_frac_valid=float(d.frac_valid_path.median()), median_first_break=float(d.first_break.median()))
    # paired comparison, gauge-fixed Hungarian vs its random-assignment null
    h = df[df.condition == "scalefix_hungarian"].frac_valid_path.values; n = df[df.condition == "scalefix_random_perm"].frac_valid_path.values
    from scipy.stats import wilcoxon
    summ["paired_hungarian_vs_null"] = dict(mean_diff=float((h - n).mean()), frac_hungarian_better=float((h > n).mean()), frac_null_better=float((h < n).mean()),
                                            wilcoxon_p=float(wilcoxon(h, n, zero_method="zsplit").pvalue) if (h != n).any() else 1.0)
    json.dump(summ, open(os.path.join(R, "Z4b_summary.json"), "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
