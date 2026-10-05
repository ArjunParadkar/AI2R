"""Gardner et al. 2022 grid-module recordings (CC0; figshare 16764508).  Level D: spikes + position only; no loss imposed by anyone.  Authors: Arjun Paradkar and Reva Paradkar.
(1) module recovery: pool the neurons of every module in a file, response route (ratemaps -> symmetry profiles -> clusters) WITHOUT the module labels; compare with the authors' labels.
(2) torus: per module, population activity in the open field -> PCA(6) -> density-filtered farthest-first subsample -> Rips H1/H2 vs a per-neuron circular-shift null."""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata
D = 'data/gardner/Toroidal_topology_grid_cell_data/'
FILES = {'rat_q': ('rat_q_grid_modules_1_2', (27826, 31223)), 'rat_r_day1': ('rat_r_day1_grid_modules_1_2_3', (7457, 14778)), 'rat_s': ('rat_s_grid_modules_1', (9939, 12363))}


def load(name):
    f, win = FILES[name]; d = np.load(D + f + '.npz', allow_pickle=True); mods = {k: d[k].item() for k in d.keys() if k.startswith('spikes_mod')}
    t, x, y = d['t'], d['x'], d['y']; m = (t >= win[0]) & (t <= win[1]); return mods, t[m], x[m], y[m], win


def ratemaps(mods, t, x, y, win, bins=40, arena=1.5, sigma=1.2):
    """-> list of (module key, neuron id), array (N, bins, bins).  Occupancy-normalised, Gaussian smoothed; unvisited bins filled with the neuron's mean."""
    xi = np.clip(((x + arena / 2) / arena * bins).astype(int), 0, bins - 1); yi = np.clip(((y + arena / 2) / arena * bins).astype(int), 0, bins - 1)
    occ = np.zeros((bins, bins)); np.add.at(occ, (xi, yi), 0.01); occs = gaussian_filter(occ, sigma); ids, maps = [], []
    for k, sp in mods.items():
        for n, st in sp.items():
            st = st[(st >= win[0]) & (st <= win[1])]; idx = np.clip(np.searchsorted(t, st), 0, len(t) - 1); cnt = np.zeros((bins, bins)); np.add.at(cnt, (xi[idx], yi[idx]), 1)
            rm = gaussian_filter(cnt, sigma) / np.maximum(occs, 1e-3); rm[occ < 0.05] = rm[occ >= 0.05].mean() if (occ >= 0.05).any() else 0; ids.append((k, n)); maps.append(rm)
    return ids, np.array(maps)


def population_matrix(sp_mod, win, dt=0.1, smooth=2.0):
    n = len(sp_mod); edges = np.arange(win[0], win[1] + dt, dt); X = np.zeros((len(edges) - 1, n))
    for j, (_, st) in enumerate(sp_mod.items()): X[:, j] = np.histogram(st, edges)[0]
    X = gaussian_filter(X, (smooth, 0)); X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9); return X


def subsample(Z, n_pts=800, density_frac=0.2, k=20, seed=0):
    from scipy.spatial import cKDTree
    tree = cKDTree(Z); dk = tree.query(Z, k)[0][:, -1]; keep = np.argsort(dk)[:max(n_pts * 2, int(len(Z) * density_frac))]; Zk = Z[keep]
    idx = [int(np.random.default_rng(seed).integers(len(Zk)))]; d = np.linalg.norm(Zk - Zk[idx[0]], axis=1)
    for _ in range(n_pts - 1): i = int(d.argmax()); idx.append(i); d = np.minimum(d, np.linalg.norm(Zk - Zk[i], axis=1))
    return Zk[idx]


def topology(X, ncomp=6, n_pts=800, maxdim=2, seed=0):
    from ripser import ripser
    X0 = X - X.mean(0); U, S, Vt = np.linalg.svd(X0, full_matrices=False); Z = U[:, :ncomp] * S[:ncomp]; P = subsample(Z, n_pts, seed=seed)
    dg = ripser(P, maxdim=maxdim)['dgms']; life = lambda d: np.sort(d[:, 1] - d[:, 0])[::-1] if len(d) else np.array([0.0])
    return dict(h1=life(dg[1]), h2=life(dg[2]) if len(dg) > 2 else np.array([0.0]), scale=float(np.percentile(np.linalg.norm(P[:, None] - P[None], axis=2), 95)))


def shuffle_null(X, n_null=20, seed=0, **kw):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_null):
        Xs = np.stack([np.roll(X[:, j], int(rng.integers(len(X) // 10, len(X) - len(X) // 10))) for j in range(X.shape[1])], 1); out.append(topology(Xs, seed=seed, **kw))
    return out


def torus_call(res, null):
    """H1: number of bars whose lifetime exceeds the maximum over the shuffle null (p <= 1/(n_null+1) each); H2: longest bar vs null max."""
    nm1 = max(n['h1'][0] for n in null); nm2 = max(n['h2'][0] for n in null)
    return dict(b1=int((res['h1'] > nm1).sum()), h1_top2=res['h1'][:2].tolist(), null_h1_max=float(nm1), h2_top=float(res['h2'][0]), null_h2_max=float(nm2), h2_above_null=bool(res['h2'][0] > nm2),
                p_h1_top2=float((1 + sum(n['h1'][min(1, len(n['h1']) - 1)] >= res['h1'][min(1, len(res['h1']) - 1)] for n in null)) / (len(null) + 1)))
