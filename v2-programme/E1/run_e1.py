"""E1 main run. See SPEC.md. Usage: python run_e1.py [--smoke] [--cells XOR3:2:4,...]"""
import argparse, os, sys, time, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1lib.functions import truth_table, FUNCTIONS
from e1lib.models import train_mlp, MLP
from e1lib.enumerate import enumerate_valid, minimal_ids
from e1lib.gauge import explanation_keys, orbit_counts, random_gauge
from e1lib.funcdim import functional_dimension

EPS = [0.01, 0.02, 0.05, 0.1, 0.2]
MAX_LEAVES = 1_000_000
N_SEEDS = 24
CELLS = [(f, d, w) for f in FUNCTIONS for d in (2, 3) for w in (4, 6, 8, 10) if d * w <= 20]
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run_network(model, X, y, tag):
    rows, pooled = [], {}
    for scheme in ("zero", "mean"):
        valid = enumerate_valid(model, X, y, scheme)
        mins = minimal_ids(valid)
        recs = [explanation_keys(model, X, int(c), scheme, EPS) for c in mins]
        oc = orbit_counts(recs, EPS)
        truncated = 0
        # jitter check (SPEC control 4): 1e-4 relative noise, recompute orbit counts
        rng = np.random.default_rng(0)
        jit = model.copy()
        jit.Ws = [W * (1 + 1e-4 * rng.normal(size=W.shape)) for W in jit.Ws]
        jit.bs = [b * (1 + 1e-4 * rng.normal(size=b.shape)) for b in jit.bs]
        jrecs = [explanation_keys(jit, X, int(c), scheme, EPS) for c in mins] if (jit.predict(X) == y).all() else recs
        joc = orbit_counts(jrecs, EPS)
        row = {**tag, "scheme": scheme, "n_valid": len(valid), "n_minimal": len(mins),
               "n_uninterpretable": sum(1 for c, _, _ in recs if c == 0), "n_truncated": truncated,
               "all_valid_expl": None, **oc,
               **{f"jitter_orbits_eps{e}": joc[f"orbits_eps{e}"] for e in EPS}}
        row["ratio_eps0.05"] = oc["orbits_eps0.05"] / oc["n_expl"] if oc["n_expl"] else np.nan
        rows.append(row)
        pooled[scheme] = recs
    return rows, pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cells", default=None)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--suffix", default="", help="output CSV suffix (for resumed partial runs; aggregate_e1 merges e1_*<suffix>.csv)")
    a = ap.parse_args()
    cells = CELLS
    if a.cells:
        cells = [(c.split(":")[0], int(c.split(":")[1]), int(c.split(":")[2])) for c in a.cells.split(",")]
    if a.smoke:
        cells, a.seeds = [("XOR3", 2, 4), ("MAJ3", 2, 6)], 3
    os.makedirs(os.path.join(RES, "networks"), exist_ok=True)
    rows, pooled_rows, funcdim_rows = [], [], []
    for (fname, d, w) in cells:
        X, y = truth_table(fname)
        pooled = {"zero": [], "mean": []}
        t0 = time.time()
        for seed in range(a.seeds):
            model, conv, steps = train_mlp(X, y, d, w, seed)
            tag = {"function": fname, "depth": d, "width": w, "seed": seed, "converged": conv, "steps": steps}
            if not conv:
                rows.append({**tag, "scheme": "none"})
                continue
            model.save(os.path.join(RES, "networks", f"{fname}_d{d}_w{w}_s{seed}.npz"))
            r, p = run_network(model, X, y, tag)
            rows += r
            with open(os.path.join(RES, "networks", f"{fname}_d{d}_w{w}_s{seed}.recs.pkl"), "wb") as fh:
                pickle.dump(p, fh)
            for s in p:
                pooled[s] += p[s]
            fd = functional_dimension(model, np.random.default_rng(seed))
            funcdim_rows.append({**tag, **fd})
            # untrained control (SPEC 5): same shape, random init, count valid circuits
            rng = np.random.default_rng(1000 + seed)
            dims = [X.shape[1]] + [w] * d + [1]
            rnd = MLP([rng.normal(size=(dims[i], dims[i + 1])) / np.sqrt(dims[i]) for i in range(len(dims) - 1)],
                      [rng.normal(size=dims[i + 1]) * 0.1 for i in range(len(dims) - 1)])
            rows.append({**tag, "seed": seed, "scheme": "untrained_zero", "converged": bool((rnd.predict(X) == y).all()),
                         "n_valid": len(enumerate_valid(rnd, X, y, "zero"))})
        for s in pooled:
            oc = orbit_counts(pooled[s], EPS)
            pooled_rows.append({"function": fname, "depth": d, "width": w, "scheme": s, **oc})
        print(f"{fname} d={d} w={w}: {time.time()-t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"e1_per_network{a.suffix}.csv"), index=False)
        pd.DataFrame(pooled_rows).to_csv(os.path.join(RES, f"e1_pooled{a.suffix}.csv"), index=False)
        pd.DataFrame(funcdim_rows).to_csv(os.path.join(RES, f"e1_funcdim{a.suffix}.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
