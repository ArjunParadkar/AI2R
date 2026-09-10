"""Z1 (zigzag across training), Z2 (scale x training-fraction Hilbert function), Z3 (graph PH of V2
graphs) on the V1 checkpoints. SPEC §Z1–Z3. Writes results/Z1_zigzag.csv, Z2_multiparam.csv, Z3_graphph.csv."""
import os, sys, csv, json, argparse
import numpy as np, pandas as pd, torch
from scipy.spatial.distance import pdist
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "V_legibility"); sys.path.insert(0, V)
from mejiro_dober import ARCHS, CK, STEPS, N_CK, build
from mejiro_dober import R as VR
from twin_turbo import load
from whiteboxtopo.legibility.tasks import Task
from whiteboxtopo import topology as T, graphs
from whiteboxtopo.core.persistence import compute_diagrams
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(R, exist_ok=True)
N_SUB = 150
SIMPLEX_BUDGET = 2_500_000   # Z AMENDMENTS #2: the union complex is counted before the zigzag; N_SUB is lowered until it fits


def normalise(c):
    """Centre a cloud and scale it to unit median pairwise distance (Z AMENDMENTS #2)."""
    c = c - c.mean(0); return c / max(float(np.median(pdist(c))), 1e-12)


def informative_scale(X):
    """The Rips scale at which the (unit-median) final cloud has the most H1 classes; quantile 0.1 of its distances when it
    has no H1 bar at all. Returns (scale, distance quantile of the scale, max beta1)."""
    D = pdist(X); dg = np.asarray(compute_diagrams(X, maxdim=1, backend="ripser", thresh=1.0).dgms[1]).reshape(-1, 2)
    if len(dg) == 0: return float(np.quantile(D, 0.1)), 0.1, 0
    grid = np.linspace(float(D.min()), 1.0, 200); b1 = np.array([np.sum((dg[:, 0] <= s) & (dg[:, 1] > s)) for s in grid])
    s = float(grid[int(np.argmax(b1))]); return s, float(np.mean(D <= s)), int(b1.max())


def count_simplices(sub, scale, cap):
    import dionysus as d
    tot = 0
    for i in range(len(sub) - 1):
        tot += len(d.fill_rips(np.vstack([sub[i], sub[i + 1]]), 2, float(scale)))
        if tot > cap: return tot, True
    return tot, False


@torch.no_grad()
def penultimate(m, x):
    m.forward(x, record=True); outs = m._node_out
    last = [o for o, l in zip(outs, m.node_layer) if l == max(m.node_layer)]
    A = torch.cat([o.reshape(o.shape[0], -1) if o.dim() == 2 else o[:, -1].reshape(o.shape[0], -1) for o in last], 1)
    return A.float().cpu().numpy()


def clouds_for(row, dev="cuda"):
    task = Task(row["task"], random_labels=bool(row["random_labels"])); x, y = task.batch(512, 7, "eval"); x = x.to(dev)
    steps = sorted(int(f[5:11]) for f in os.listdir(os.path.join(CK, row["name"])) if f.startswith("step_"))
    clouds = []
    for st in steps:
        m, _ = load(row, st, dev); clouds.append(penultimate(m, x))
    Xf = clouds[-1]; mu = Xf.mean(0); U, s, Vt = np.linalg.svd(Xf - mu, full_matrices=False); P = Vt[:20].T
    return steps, [(c - mu) @ P for c in clouds]


def film_events(diags, tol=0.15):
    """Greedy bar matching between consecutive diagrams: count births + deaths the film reports."""
    ev = 0
    for a, b in zip(diags[:-1], diags[1:]):
        a = np.asarray(a).reshape(-1, 2); b = np.asarray(b).reshape(-1, 2); used = set(); matched = 0
        for i in range(len(a)):
            if len(b) == 0: break
            d = np.abs(b - a[i]).max(1); d[list(used)] = np.inf; j = int(np.argmin(d))
            if d[j] < tol * max(a[i, 1] - a[i, 0], 1e-9): used.add(j); matched += 1
        ev += (len(a) - matched) + (len(b) - len(used))
    return ev


def seiun_sky(rows, dev):
    out = os.path.join(R, "Z1_zigzag.csv"); done = set(pd.read_csv(out).name) if os.path.exists(out) else set(); res = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    for row in rows:
        if row["name"] in done or row["untrained"]: continue
        steps, clouds = clouds_for(row, dev); clouds = [normalise(c) for c in clouds]          # Z AMENDMENTS #2
        rng = np.random.default_rng(0); pick = rng.choice(len(clouds[0]), N_SUB, replace=False)   # the SAME probe examples at every checkpoint
        for n_sub in (N_SUB, 100, 75, 50):
            sub = [c[pick[:n_sub]] for c in clouds]; scale, scale_q, b1max = informative_scale(sub[-1])
            n_simp, over = count_simplices(sub, scale, SIMPLEX_BUDGET)
            if not over: break
        print(row["name"], "n_sub", n_sub, "scale", round(scale, 3), "q", round(scale_q, 3), "beta1max", b1max, "simplices", n_simp, flush=True)
        z = T.zigzag(sub, scale, maxdim=1)
        diags = [np.asarray(compute_diagrams(c, maxdim=1, backend="ripser", thresh=scale).dgms[1]).reshape(-1, 2) for c in sub]
        bars = z.bars[1]; long = bars[(np.where(np.isfinite(bars[:, 1]), bars[:, 1], z.n_positions) - bars[:, 0]) >= 3] if len(bars) else bars
        film_ev = film_events(diags); zz_ev = int(2 * len(bars) - np.sum(~np.isfinite(bars[:, 1]))) if len(bars) else 0
        r = dict(name=row["name"], task=row["task"], arch=row["arch"], random_labels=row["random_labels"], n_ckpt=len(steps), scale=scale,
                 n_sub=n_sub, scale_quantile=scale_q, beta1_final_max=b1max, budget_reduced=int(n_sub < N_SUB),
                 n_zigzag_bars_H1=int(len(bars)), n_long_bars=int(len(long)), film_events=film_ev, zigzag_events=zz_ev,
                 mislabelled_fraction=float(max(film_ev - zz_ev, 0) / max(film_ev, 1)), n_simplices=z.n_simplices,
                 betti1_curve=json.dumps(z.betti_curve(1).tolist()))
        res.append(r); pd.DataFrame(res).to_csv(out, index=False); print(r["name"], r["n_zigzag_bars_H1"], r["film_events"], r["mislabelled_fraction"], flush=True)


def manhattan_cafe(rows, dev):
    out = os.path.join(R, "Z2_multiparam.csv"); done = set(pd.read_csv(out).name) if os.path.exists(out) else set(); res = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    for row in rows:
        if row["name"] in done or row["untrained"]: continue
        steps, clouds = clouds_for(row, dev); Xf = clouds[-1]; diam = float(np.max(np.linalg.norm(Xf - Xf.mean(0), axis=1)) * 2)
        # per point: first checkpoint after which it moves < 10 % of the diameter
        moves = np.stack([np.linalg.norm(clouds[i + 1] - clouds[i], axis=1) for i in range(len(clouds) - 1)])   # (T-1, n)
        stable_from = np.array([next((i for i in range(len(moves)) if (moves[i:, j] < 0.1 * diam).all()), len(moves)) for j in range(Xf.shape[0])], dtype=float)
        rng = np.random.default_rng(1); sub = rng.choice(len(Xf), 300, replace=False)
        H = T.hilbert_function(Xf[sub], stable_from[sub], params=np.arange(len(clouds)), n_grid=20, maxdim=1)
        tp = compute_diagrams(Xf[sub], maxdim=1, backend="ripser").dgms[1]; tp = np.asarray(tp).reshape(-1, 2); tp1 = float(np.sum((tp[:, 1] - tp[:, 0]) ** 2))
        r = dict(name=row["name"], task=row["task"], arch=row["arch"], hilbert_mass_H0=H.hilbert_mass[0], hilbert_mass_H1=H.hilbert_mass[1],
                 argmax_H1_param=H.argmax[1][0], argmax_H1_scale=H.argmax[1][1], total_persistence_H1_final=tp1, frac_stable_by_half=float((stable_from <= len(moves) / 2).mean()))
        res.append(r); pd.DataFrame(res).to_csv(out, index=False); print(r, flush=True)


def agnes_tachyon(rows, dev):
    out = os.path.join(R, "Z3_graphph.csv"); done = set(pd.read_csv(out).name) if os.path.exists(out) else set(); res = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    for row in rows:
        if row["name"] in done: continue
        task = Task(row["task"], random_labels=bool(row["random_labels"])); m, _ = load(row, STEPS, dev); x, y = task.batch(512, 0, "eval")
        E = graphs.attribution_graph(m, (x.to(dev), y.to(dev)), raw=True); A = graphs.threshold_density(graphs.symmetrise(E), 0.1)
        g = T.graph_ph(A, maxdim=1)
        r = dict(name=row["name"], task=row["task"], arch=row["arch"], betti1_finite=g.betti1_finite, betti1_essential=g.betti1_essential,
                 total_persistence_H0=g.total_persistence[0], total_persistence_H1=g.total_persistence[1], n_edges=g.n_edges)
        res.append(r); pd.DataFrame(res).to_csv(out, index=False); print(r, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=["seiun_sky", "manhattan_cafe", "agnes_tachyon"]); a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(VR, "V1_models.csv")))]
    for r in rows:
        for k in ("random_labels", "untrained", "seed"): r[k] = int(r[k])
    if a.which in ("z1", "seiun_sky"):
        sel = [r for r in rows if (r["task"] == "modadd" and r["arch"] in ("mlp128", "tf2")) or r["random_labels"]]; seiun_sky(sel, "cuda")
    elif a.which in ("z2", "manhattan_cafe"):
        sel = [r for r in rows if r["task"] == "modadd" and not r["random_labels"]][:20]; manhattan_cafe(sel, "cuda")
    else:
        agnes_tachyon(rows, "cuda")
