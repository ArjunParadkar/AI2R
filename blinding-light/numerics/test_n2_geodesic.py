import numpy as np, torch
from torch.func import vmap, jacrev
from common import *
from whiteboxtopo.numerics import geodesic as G

TH, PH = (0.55, 2.59), (0.0, 2.0)
CORE_TH, CORE_PH = (1.0, 2.14), (0.4, 1.6)


def sample(n_max=4000, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.uniform(np.cos(TH[1]), np.cos(TH[0]), n_max); th = np.arccos(c); ph = rng.uniform(*PH, n_max)
    return np.stack([th, ph], 1)


def embed(P): return np.stack([np.sin(P[:, 0]) * np.cos(P[:, 1]), np.sin(P[:, 0]) * np.sin(P[:, 1]), np.cos(P[:, 0])], 1)


def chart_metric(P):
    Pt = torch.as_tensor(P, dtype=T64)
    Jm = vmap(jacrev(sphere_embed))(Pt)
    return (Jm.transpose(1, 2) @ Jm).numpy()


def core_mask(P):
    return (P[:, 0] > CORE_TH[0]) & (P[:, 0] < CORE_TH[1]) & (P[:, 1] > CORE_PH[0]) & (P[:, 1] < CORE_PH[1])


def run(n, k, mode, P_all=None, ncore=350):
    P = (sample() if P_all is None else P_all)[:n]
    core = np.where(core_mask(P))[0][:ncore]
    if mode == 'ambient': D = G.graph_geodesics(embed(P), k, None, sources=core)
    else: D = G.graph_geodesics(P, k, chart_metric, sources=core)
    Dg = D[:, core]
    E = embed(P)[core]; Dt = np.arccos(np.clip(E @ E.T, -1, 1))
    mask = Dt >= 0.3
    return Dg, Dt, mask


def stats(Dg, Dt, mask):
    r = np.abs(Dg[mask] - Dt[mask]) / Dt[mask]
    return float(r.mean()), float(np.quantile(r, 0.95)), float((Dg[mask] / Dt[mask]).min())


def test_n2_01_04(cell):
    out = {}; Ds = {}
    for mode in ('ambient', 'chart'):
        Dg, Dt, mask = run(4000, 12, mode)
        out[mode] = stats(Dg, Dt, mask); Ds[mode] = (Dg, mask)
    mask = Ds['ambient'][1]
    diff = float((np.abs(Ds['ambient'][0][mask] - Ds['chart'][0][mask]) / Ds['ambient'][0][mask]).mean())
    cell('T-N2-03', {'n': 4000, 'k': 12}, 0, {'mean_abs_rel_diff': diff}, {'<=0.02': diff <= 0.02})
    mn = min(out[m][2] for m in out)
    # structural checks on a small graph
    P = sample()[:600]
    D = G.graph_geodesics(P, 12, chart_metric)
    sym = float(np.abs(D - D.T).max()); diag = float(np.abs(np.diag(D)).max())
    tri = max(float((D[:, [k]] + D[[k], :] - D).min() * -1) for k in range(len(D)))
    cell('T-N2-04', {'n': 600}, 0, {'sym': sym, 'diag': diag, 'triangle_violation': tri, 'min_ratio': mn},
         {'sym<=1e-12': sym <= 1e-12 and diag == 0, 'tri<=1e-9': tri <= 1e-9, 'ratio>=.995': mn >= 0.995})

    # the headline cell is asserted last so the other cells in this test are recorded even if it fails
    cell('T-N2-01', {'n': 4000, 'k': 12}, 0, {m: {'mean': v[0], 'p95': v[1]} for m, v in out.items()},
         {f'{m}: mean<=.03,p95<=.08': out[m][0] <= 0.03 and out[m][1] <= 0.08 for m in out})

def test_n2_02_scaling(cell):
    P_all = sample()
    byk = {k: stats(*run(4000, k, 'chart', P_all))[0] for k in (6, 12, 24)}
    byn = {n: stats(*run(n, 12, 'chart', P_all))[0] for n in (500, 1000, 2000, 4000)}
    cell('T-N2-02', {'ks': [6, 12, 24], 'ns': [500, 1000, 2000, 4000]}, 0, {'err_vs_k': byk, 'err_vs_n': byn},
         {'k_nonincreasing': byk[12] <= 1.1 * byk[6] and byk[24] <= 1.1 * byk[12], 'n_not_worse': byn[4000] <= 1.1 * byn[500]})
