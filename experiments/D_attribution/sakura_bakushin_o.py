"""D1 — attribution anchor on the Block-F specimens (SPEC §D1)."""
import os, sys, csv
from multiprocessing import Pool
import numpy as np
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "F_gauge"); sys.path.insert(0, F)
from flib.models import MLP
from flib.functions import truth_table
from whiteboxtopo.invariants import check_invariance, as_relu_mlp
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def grads(m, X):
    """d logit / d x (N, n_in) and per-unit a_u * d logit/d a_u (n_hidden,) mean over X."""
    m = as_relu_mlp(m); masks, acts, out = m.forward(X)
    g = np.ones((X.shape[0], 1)); unit = []
    for l in range(m.depth, 0, -1):
        g = g @ m.Ws[l].T                       # d logit / d a_{l-1}
        unit.append((acts[l - 1] * g).mean(0)); g = g * masks[l - 1]
    gx = g @ m.Ws[0].T
    return gx, np.concatenate(unit[::-1])


def ig(m, X, steps=64):
    m = as_relu_mlp(m); base = np.zeros_like(X); tot = np.zeros_like(X)
    for k in range(steps):
        a = (k + 0.5) / steps; Xa = base + a * (X - base)
        masks, acts, out = m.forward(Xa); g = np.ones((X.shape[0], 1))
        for l in range(m.depth, 0, -1):
            g = (g @ m.Ws[l].T) * masks[l - 1]
        tot += g @ m.Ws[0].T
    return tot * (X - base) / steps


def job(args):
    spec, path, fn = args
    import torch; torch.set_num_threads(1)
    m = MLP.load(os.path.join(F, "results", path)); X, y = truth_table(fn); P = np.random.default_rng(0).uniform(-1, 2, size=(256, X.shape[1]))
    rows = []
    for cloud_name, C in (("truthtable", X), ("probe", P)):
        M = {"integrated_gradients_input": lambda mm: ig(mm, C).ravel(),
             "gradxinput_input": lambda mm: (grads(mm, C)[0] * C).ravel(),
             "neuron_attribution_values": lambda mm: grads(mm, C)[1],
             "neuron_attribution_sorted": lambda mm: np.sort(grads(mm, C)[1]),
             "neuron_ablation_effect": lambda mm: np.concatenate([[(as_relu_mlp(mm).logits(C)[:, 0] - _abl(mm, C, l, u)).mean() for u in range(w)] for l, w in enumerate(as_relu_mlp(mm).widths)])}
        for name, fn_ in M.items():
            r = check_invariance(fn_, m, n=100, name=name, keep_samples=False, uniform_sweep=True)
            row = r.row(); row.update(specimen=spec, cloud=cloud_name); rows.append(row)
    return rows


def _abl(mm, C, l, u):
    m = as_relu_mlp(mm); h = C
    for i, (W, b) in enumerate(zip(m.Ws[:-1], m.bs[:-1])):
        h = np.maximum(h @ W + b, 0.0)
        if i == l:
            h = h.copy(); h[:, u] = h[:, u].mean()
    return (h @ m.Ws[-1] + m.bs[-1])[:, 0]


def main():
    spec = [r for r in csv.DictReader(open(os.path.join(F, "results", "specimens.csv"))) if int(r["converged"]) and int(r["seed"]) < 5]
    jobs = [(f'{r["function"]}_d{r["depth"]}_w{r["width"]}_s{r["seed"]}', r["path"], r["function"]) for r in spec]
    out = os.path.join(R, "D1_audit.csv"); done = {r["specimen"] for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    jobs = [j for j in jobs if j[0] not in done]; print(len(jobs), "specimens", flush=True)
    fields = None
    with Pool(3) as pool, open(out, "a", newline="") as f:
        w = None
        for i, rows in enumerate(pool.imap_unordered(job, jobs)):
            if w is None:
                fields = list(rows[0]); w = csv.DictWriter(f, fieldnames=fields)
                if not done: w.writeheader()
            for r in rows: w.writerow({k: r.get(k, "") for k in fields})
            f.flush()
            if i % 20 == 0: print(i, rows[0]["specimen"], flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
