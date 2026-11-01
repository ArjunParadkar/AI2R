"""E2 runner: orbit-sample every E1 network under perm / scale / combined, compute per-network
CVs of every SPEC quantity, the uniform-scaling sweep, and the untrained null. Writes
results/e2_per_network.csv, results/e2_scaling.csv (+ *_untrained). See SPEC.md."""
import argparse, glob, os, re, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e2lib import audit_metrics, all_quantities, scaling_sweep, GROUPS  # noqa: E402
from e2lib.quantities import gate_agreement, fit_exponent, box_inputs  # noqa: E402
from e2lib.core import sample_orbit  # noqa: E402
from e1lib.models import MLP  # noqa: E402
from e1lib.functions import truth_table  # noqa: E402

NET_DIR = os.path.join(HERE, "..", "E1", "results", "networks")
RES = os.path.join(HERE, "results")
PAT = re.compile(r"(?P<function>[A-Z0-9_]+)_d(?P<depth>\d)_w(?P<width>\d+)_s(?P<seed>\d+)\.npz$")


def parse_name(path):
    m = PAT.search(os.path.basename(path))
    return {"function": m["function"], "depth": int(m["depth"]), "width": int(m["width"]), "seed": int(m["seed"])}


def untrained_mlp(n_in, d, w, seed):
    rng = np.random.default_rng(1000 + seed)
    dims = [n_in] + [w] * d + [1]
    return MLP([rng.normal(size=(dims[i], dims[i + 1])) / np.sqrt(dims[i]) for i in range(len(dims) - 1)],
               [rng.normal(size=dims[i + 1]) * 0.1 for i in range(len(dims) - 1)])


def process(job):
    tag, model, n_reps, net_index = job
    X, y = truth_table(tag["function"])
    Xb = box_inputs(X.shape[1], seed=net_index)
    rows, sweep_rows = [], []
    conv = bool((model.predict(X) == y).all() and (np.abs(model.logits(X)) >= 0.5).all())
    base_q = all_quantities(model, X, Xb)
    for gi, group in enumerate(GROUPS):
        seed = 100000 * gi + net_index
        # gates need the raw (R, N, n_hidden) stack for the per-index agreement, so re-collect
        gate_stack = []
        def qf(mm):
            q = all_quantities(mm, X, Xb)
            gate_stack.append(q["gates"])
            return q
        res, dists, _ = audit_metrics(qf, model, n_reps, group, seed=seed)
        for k, r in res.items():
            rows.append({**tag, "converged": conv, "group": group, "quantity": k, "cv": r["cv"],
                         "agreement": r["agreement"], "max_abs_dev": r["max_abs_dev"],
                         "param_dist_median": float(np.median(dists[1:])),
                         "param_dist_min": float(np.min(dists[1:])),
                         "base_norm": float(np.linalg.norm(np.asarray(base_q[k], dtype=np.float64)))})
        rows.append({**tag, "converged": conv, "group": group, "quantity": "gates_index_agreement",
                     "cv": float("nan"), "agreement": gate_agreement(gate_stack), "max_abs_dev": float("nan"),
                     "param_dist_median": float(np.median(dists[1:])), "param_dist_min": float(np.min(dists[1:])),
                     "base_norm": float("nan")})
    for r in scaling_sweep(model, X, Xb):
        sweep_rows.append({**tag, **r})
    # per-layer exponent fits
    fits = []
    sw = pd.DataFrame(sweep_rows)
    for l, g in sw.groupby("layer"):
        for col in [c for c in sw.columns if c.split("_")[0] in ("tc0", "tc1", "b0", "b1", "pe0", "pe1")]:
            k, r2 = fit_exponent(g["alpha"].values, g[col].values)
            fits.append({**tag, "layer": l, "quantity": col, "exponent": k, "r2": r2,
                         "all_positive": bool((g[col].values > 0).all()),
                         "constant": bool(np.allclose(g[col].values, g[col].values[0]))})
    base_scalars = {k: np.asarray(v, dtype=np.float64).ravel() for k, v in base_q.items()}
    return rows, sweep_rows, fits, {**tag, **{f"base::{k}": v for k, v in base_scalars.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--untrained", action="store_true", help="run the untrained null instead")
    ap.add_argument("--untrained-seeds", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(NET_DIR, "*.npz")))
    if a.limit:
        paths = paths[: a.limit]
    suffix = "_untrained" if a.untrained else ""
    if a.untrained:
        cells = sorted({(t["function"], t["depth"], t["width"]) for t in map(parse_name, paths)})
        jobs = []
        i = 0
        for f, d, w in cells:
            n_in = truth_table(f)[0].shape[1]
            for s in range(a.untrained_seeds):
                jobs.append(({"function": f, "depth": d, "width": w, "seed": s}, untrained_mlp(n_in, d, w, s), a.n_reps, i))
                i += 1
    else:
        jobs = [(parse_name(p), MLP.load(p), a.n_reps, i) for i, p in enumerate(paths)]
    print(f"{len(jobs)} networks, n_reps={a.n_reps}, workers={a.workers}, untrained={a.untrained}", flush=True)
    t0 = time.time()
    rows, sweep, fits, bases = [], [], [], []
    with Pool(a.workers) as pool:
        for n, (r, s, f, b) in enumerate(pool.imap_unordered(process, jobs, chunksize=2)):
            rows += r; sweep += s; fits += f; bases.append(b)
            if (n + 1) % 50 == 0 or n + 1 == len(jobs):
                print(f"  {n+1}/{len(jobs)} done, {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(RES, f"e2_per_network{suffix}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"e2_per_network{suffix}.csv"), index=False)
    pd.DataFrame(sweep).to_csv(os.path.join(RES, f"e2_sweep_raw{suffix}.csv"), index=False)
    pd.DataFrame(fits).to_csv(os.path.join(RES, f"e2_scaling{suffix}.csv"), index=False)
    pd.to_pickle(bases, os.path.join(RES, f"e2_base_values{suffix}.pkl"))
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
