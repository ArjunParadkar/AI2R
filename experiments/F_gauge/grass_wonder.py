"""F3 — recompute results/symmetry.csv with the amended (block-null) accounting for every converged specimen."""
import os, sys, csv
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flib.models import MLP
from flib.functions import truth_table
from whiteboxtopo import symmetry as S
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def job(r):
    import torch; torch.set_num_threads(1)
    spec = f'{r["function"]}_d{r["depth"]}_w{r["width"]}_s{r["seed"]}'
    m = MLP.load(os.path.join(R, r["path"])); X, y = truth_table(r["function"]); h = S.hidden(m, inputs=X, seed=0, keep_directions=False)
    return dict(specimen=spec, n_params=h.n_params, rank=h.rank, fiber_dim=h.corank, known_dim=h.known_dim, excess=h.excess,
                explained=h.identified["explained_total"], unexplained=h.unexplained, dead=h.identified["dead"], all_active=h.identified["all_active"],
                shared_pattern=h.identified["shared_pattern"], G_incomplete=int(h.G_incomplete), dead_units=len(h.dead_units), constant_units=len(h.constant_units),
                n_groups=len(h.groups), block_null=str(h.identified["block_null_dirs"]))


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(R, "specimens.csv"))) if int(r["converged"])]
    out = os.path.join(R, "symmetry.csv")
    with Pool(3) as pool, open(out, "w", newline="") as f:
        w = None
        for i, r in enumerate(pool.imap_unordered(job, rows, chunksize=8)):
            if w is None: w = csv.DictWriter(f, fieldnames=list(r)); w.writeheader()
            w.writerow(r); f.flush()
            if i % 200 == 0: print(i, r, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
