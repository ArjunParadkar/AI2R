"""Z4 (explanation-space path connectivity on the F specimens) and Z5 (covariance law of persistence
under scaling, from F10's audit table). SPEC §Z4–Z5. Writes results/Z4_paths.csv, Z5_scaling_law.csv."""
import os, sys, json
import numpy as np, pandas as pd
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "F_gauge"); sys.path.insert(0, F)
from flib.models import MLP
from flib.functions import truth_table
from flib.enumerate import mask_from_id, ablated_forward
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(R, exist_ok=True)


def hungarian_match(m_a, m_b):
    """Permute m_b's hidden units layer by layer to best match m_a (incoming weight vectors)."""
    from scipy.optimize import linear_sum_assignment
    from flib.gauge import apply_gauge
    perms = []
    for l in range(m_a.depth):
        A = np.vstack([m_a.Ws[l], m_a.bs[l][None]]); B = np.vstack([m_b.Ws[l], m_b.bs[l][None]])
        cost = ((A.T[:, None, :] - B.T[None, :, :]) ** 2).sum(-1); ri, ci = linear_sum_assignment(cost); p = np.empty(A.shape[1], int); p[ri] = ci; perms.append(p)
        # NB: layer l+1's rows are permuted by apply_gauge consistently
    return apply_gauge(m_b, perms, None)


def valid(model, X, y, mid, scheme):
    means = [a.mean(0) for a in model.hidden_acts(X)]; lg, _ = ablated_forward(model, X, mask_from_id(mid, sum(model.widths))[None], scheme, means)
    return bool(((lg[0] > 0).astype(np.int8) == y).all())


def z4():
    E = pd.read_csv(os.path.join(F, "results", "explanations.csv"), dtype={"key6": str, "key4": str})
    sp = E.specimen.str.extract(r"^(?P<function>[A-Z0-9_]+)_d(?P<depth>\d)_w(?P<width>\d+)_s(?P<seed>\d+)$"); E = pd.concat([E, sp], axis=1)
    E["cell"] = E.function + "_d" + E.depth + "_w" + E.width; rows = []; rng = np.random.default_rng(0)
    # within-orbit pairs (same key6, different specimens)
    dup = E[E.duplicated(["cell", "criterion", "key6"], keep=False)]
    n_within = int(len(dup) // 2)
    for (cell, crit), g in E[E.interpretable == 1].groupby(["cell", "criterion"]):
        specs = g.specimen.unique()
        if len(specs) < 2: continue
        for t in range(10):     # 10 cross-specimen pairs per cell: a valid explanation of A, interpolated toward B (after unit matching)
            a, b = rng.choice(specs, 2, replace=False); ra = g[g.specimen == a].sample(1, random_state=t).iloc[0]
            fn = ra.function; X, y = truth_table(fn)
            ma = MLP.load(os.path.join(F, "results", "networks", f"{a}.npz")); mb = hungarian_match(ma, MLP.load(os.path.join(F, "results", "networks", f"{b}.npz")))
            mid = int(ra.circuit_id); ok = []
            for s in np.linspace(0, 1, 50):
                m = MLP([(1 - s) * Wa + s * Wb for Wa, Wb in zip(ma.Ws, mb.Ws)], [(1 - s) * ba + s * bb for ba, bb in zip(ma.bs, mb.bs)])
                ok.append(valid(m, X, y, mid, crit))
            ok = np.array(ok); rows.append(dict(cell=cell, criterion=crit, spec_a=a, spec_b=b, circuit_id=mid, frac_valid_path=float(ok.mean()),
                                              valid_at_end=bool(ok[-1]), first_break=float(np.argmin(ok) / 49) if not ok.all() else 1.0))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(R, "Z4_paths.csv"), index=False)
    summ = dict(n_within_orbit_pairs=n_within, note="within-orbit pairs across specimens" if n_within else "no within-orbit pairs exist (K = N): question vacuous; cross-orbit connectivity reported",
                frac_pairs_valid_path=float((df.frac_valid_path >= 0.9).mean()) if len(df) else np.nan, frac_valid_at_end=float(df.valid_at_end.mean()) if len(df) else np.nan,
                median_frac_valid=float(df.frac_valid_path.median()) if len(df) else np.nan, n_pairs=int(len(df)))
    json.dump(summ, open(os.path.join(R, "Z4_summary.json"), "w"), indent=1); print(summ)


def z5():
    A = pd.read_csv(os.path.join(F, "results", "audit_table.csv"))
    rows = []
    for metric, g in A.groupby("metric"):
        g = g[g.error.isna() | (g.error == "")]
        rows.append(dict(metric=metric, n_specimens=len(g), verdict_majority=g.verdict.mode().iloc[0] if len(g) else "", frac_invariant=float((g.verdict == "invariant").mean()),
                         frac_covariant=float((g.verdict == "covariant").mean()), exponent_median=float(g.scaling_exponent.median()), exponent_iqr=float(g.scaling_exponent.quantile(.75) - g.scaling_exponent.quantile(.25)),
                         r2_median=float(g.scaling_r2.median()), cv_scale_median=float(g.cv_scale.median()), cv_perm_median=float(g.cv_perm.median())))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(R, "Z5_scaling_law.csv"), index=False); print(df.to_string())
    std = df[df.metric == "ph_std_probe"]
    print("Z5 pre-registered (ph_std_probe invariant on >= 99 % of specimens):", float(std.frac_invariant.iloc[0]) if len(std) else None)


if __name__ == "__main__":
    (z5 if len(sys.argv) > 1 and sys.argv[1] == "z5" else z4)()
