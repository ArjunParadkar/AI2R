"""SPEC amendment 2: shuffle-seed replicates and without-replacement subsample CIs for Test 1."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6lib import mp_outliers, feature_shuffle
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache"); RES = os.path.join(HERE, "results")
rows, subs = [], []
for L in (2, 6, 10):
    z = np.load(os.path.join(CACHE, f"L{L}.npz")); X, E, seq = z["X"].astype(np.float64), z["E"].astype(np.float64), z["seq"]
    for s in range(20):
        r = mp_outliers(feature_shuffle(E, seed=100 + s), "correlation")
        rows.append(dict(layer=L, seed=100 + s, n_outliers=r.n_outliers, structured_fraction=r.structured_fraction, lambda_max=float(r.eigenvalues[0]), tw_threshold=r.tw_threshold))
    ug = np.unique(seq); rng = np.random.default_rng(0)
    for name, C in (("E", E), ("X", X)):
        for s in range(20):
            pick = rng.choice(ug, size=int(0.8 * len(ug)), replace=False); idx = np.isin(seq, pick)
            r = mp_outliers(C[idx], "correlation")
            subs.append(dict(layer=L, cloud=name, draw=s, n=int(idx.sum()), n_outliers=r.n_outliers, structured_fraction=r.structured_fraction, excess_fraction=r.excess_fraction))
    print("L", L, "done", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "e6_test1_shuffle_seeds.csv"), index=False)
pd.DataFrame(subs).to_csv(os.path.join(RES, "e6_test1_subsample_ci.csv"), index=False)
d = pd.DataFrame(rows); print(d.groupby("layer").n_outliers.agg(["max", "mean"]))
s = pd.DataFrame(subs); print(s.groupby(["layer", "cloud"]).n_outliers.describe(percentiles=[.025, .975]))
