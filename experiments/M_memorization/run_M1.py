"""M1 — audit the v1 memorization features on the cached battery clouds (SPEC §M1).
Gauge action on a cached post-ReLU/residual cloud: column permutations × positive column scalings
(CNN); GPT-2 residual clouds: invariant by construction (perm/scale still applied to columns to
document the float floor, but labelled 'residual: no internal renaming acts'). Writes results/M1_audit.csv."""
import os, sys, glob, json, csv
from multiprocessing import Pool
import numpy as np
from whiteboxtopo.signals.memorization import memorization_features
from whiteboxtopo.topology import standardized_cloud
from whiteboxtopo.invariants.check import per_component_cv, INVARIANT_CV, _fit_powerlaw, UNIFORM_ALPHAS
LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "results")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(R, exist_ok=True)
N = 30   # AMENDMENTS #1


def feats(A, std):
    A = standardized_cloud(A) if std else A
    f = memorization_features(A, n_components=30, n_shuffle=3, seed=0)
    keys = sorted(f); return keys, np.array([f[k] for k in keys], dtype=np.float64)


def job(args):
    fam, path, key = args
    A = np.load(path)[key].astype(np.float64); rows = []
    rng = np.random.default_rng(abs(hash((path, key))) % (1 << 31))
    for std in (False, True):
        keys, base = feats(A, std)
        samp = {}
        for kind in ("perm", "scale", "combined"):
            vals = [base]
            for r in range(1, N):
                B = A.copy()
                if kind in ("scale", "combined"): B = B * np.exp(rng.uniform(np.log(0.1), np.log(10), size=B.shape[1]))[None, :]
                if kind in ("perm", "combined"): B = B[:, rng.permutation(B.shape[1])]
                vals.append(feats(B, std)[1])
            samp[kind] = np.stack(vals)
        sweep = np.stack([feats(A * a, std)[1] for a in UNIFORM_ALPHAS]); ks, r2s = _fit_powerlaw(UNIFORM_ALPHAS, sweep)
        cv = {k: per_component_cv(v) for k, v in samp.items()}
        for j, name in enumerate(keys):
            comb = cv["combined"][j]
            verdict = "invariant" if comb < INVARIANT_CV else ("covariant" if cv["perm"][j] < INVARIANT_CV and r2s[j] >= 0.99 and abs(ks[j]) > 1e-3 else "artifact")
            rows.append(dict(family=fam, model=os.path.basename(path)[:-4], layer=key, standardized=int(std), feature=name,
                             base=base[j], cv_perm=cv["perm"][j], cv_scale=cv["scale"][j], cv_combined=comb, exponent=ks[j], r2=r2s[j],
                             verdict=verdict, note="residual stream: no internal renaming acts (invariant by construction)" if fam == "gpt2" else ""))
    return rows


def main():
    jobs = []
    for fam in ("cifar", "gpt2"):
        for p in sorted(glob.glob(os.path.join(LIB, f"battery_{fam}", "*.npz"))):
            for key in np.load(p).files:
                jobs.append((fam, p, key))
    print(len(jobs), "clouds", flush=True)
    out = os.path.join(R, "M1_audit.csv"); w = None
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool, open(out, "w", newline="") as f:
        for i, rows in enumerate(pool.imap_unordered(job, jobs)):
            if w is None: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
            for r in rows: w.writerow(r)
            f.flush()
            if i % 10 == 0: print(i, rows[0]["model"], rows[0]["layer"], flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
