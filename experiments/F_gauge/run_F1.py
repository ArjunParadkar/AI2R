"""F1 — train the specimens (SPEC §F1). Writes results/specimens.csv and results/networks/*.npz.
Resumable: existing rows are skipped."""
import os, sys, csv, time, argparse
from multiprocessing import Pool
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from flib.functions import truth_table, NAMES3, NAMES4
from flib.models import train_mlp

R = os.path.join(os.path.dirname(__file__), "results")
NET = os.path.join(R, "networks")
DEPTHS, WIDTHS, SEEDS = (2, 3), (4, 6, 8, 10), list(range(24))


def job(args):
    fn, d, w, s = args
    torch.set_num_threads(1)
    path = os.path.join(NET, f"{fn}_d{d}_w{w}_s{s}.npz")
    X, y = truth_table(fn)
    t0 = time.time()
    model, conv, steps = train_mlp(X, y, d, w, s)
    model.save(path)
    return dict(function=fn, depth=d, width=w, seed=s, converged=int(conv), steps=steps,
                wall=round(time.time() - t0, 2), path=os.path.relpath(path, R), n_in=X.shape[1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=16); a = ap.parse_args()
    os.makedirs(NET, exist_ok=True)
    out = os.path.join(R, "specimens.csv")
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for r in csv.DictReader(f):
                done.add((r["function"], int(r["depth"]), int(r["width"]), int(r["seed"])))
    jobs = [(fn, d, w, s) for fn in NAMES3 + NAMES4 for d in DEPTHS for w in WIDTHS for s in SEEDS if (fn, d, w, s) not in done]
    print(f"{len(jobs)} networks to train ({len(done)} done)", flush=True)
    fields = ["function", "depth", "width", "seed", "converged", "steps", "wall", "path", "n_in"]
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f, Pool(a.procs) as pool:
        wr = csv.DictWriter(f, fieldnames=fields)
        if new: wr.writeheader()
        for i, row in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            wr.writerow(row); f.flush()
            if i % 100 == 0: print(f"{i}/{len(jobs)} {row}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
