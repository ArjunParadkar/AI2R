"""Analyse a trained GridRNN: response route (population symmetry profiles -> modules -> lattices -> characters), per-cell classical lattice fits (the 'ratemap' reference
for H6.1), connectivity route, and the cross-route consistency check.  Authors: Arjun Paradkar and Reva Paradkar."""
import numpy as np, torch
from bench.bio import gridlib as G, gridrnn as R


def alive(rm, mask, min_std=1e-3):
    v = rm[:, mask].std(1); return np.where(v > min_std)[0]


def percell_lattice(rm, arena, idx, max_cells=None):
    bs = arena / rm.shape[1]; out = []
    for i in idx[:max_cells]:
        ac = G.autocorr(rm[i]); L = G.lattice_from_profile(ac, bs)
        if L is not None: out.append(dict(i=int(i), spacing=L['spacing'], orientation=L['orientation'], r2=G.hex_template_r2(rm[i], L['spacing'], arena)))
    return out


def analyse(m, bins=44, n_cells_r2=24, seed=0, device='cuda'):
    rm, mask = R.ratemaps(m, bins=bins, device=device, seed=seed); idx = alive(rm, mask); out = dict(n_alive=int(len(idx)))
    if len(idx) < 20: out['response'] = None; return out, rm, idx
    rmf = rm[idx].copy(); rmf = np.where(mask[None], rmf, 0.0)
    res = G.response_route(rmf, m.L); pc = percell_lattice(rmf, m.L, np.arange(len(rmf)))
    out['response'] = dict(n_modules=int(res['n_modules']), silhouette=float(res['silhouette']),
                           modules=[{k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in mm.items() if k in ('n', 'spacing', 'orientation', 'r2_characters')} for mm in res['modules']],
                           labels=res['labels'].tolist(), mean_r2_all=float(np.mean([p['r2'] for p in pc])) if pc else None, frac_gridlike=float(np.mean([p['r2'] >= 0.7 for p in pc])) if pc else 0.0,
                           percell=pc[:200])
    # agreement of the population route with per-cell classical fits (H6.1): module spacing within 10 %, orientation within 7 deg
    ag = []; labs = res['labels']
    for mi, mm in enumerate(res['modules']):
        if 'spacing' not in mm: continue
        cells = [p for p in pc if labs[p['i']] == mi]
        if len(cells) < 5: continue
        sp = np.array([c['spacing'] for c in cells]); ori = np.array([c['orientation'] for c in cells]); dori = np.abs((ori - mm['orientation'] + 30) % 60 - 30)
        ag.append(dict(module=mi, n=len(cells), spacing_rel_diff=float(abs(np.median(sp) - mm['spacing']) / mm['spacing']), orient_diff=float(np.median(dori)), spacing_cv=float(sp.std() / sp.mean())))
    out['agreement'] = ag
    # connectivity route on W restricted to alive units
    W = m.W.detach().cpu().numpy()[np.ix_(idx, idx)]; W = np.maximum(W, 0)           # excitatory part (distances need a monotone weight)
    cr = G.connectivity_route(W); out['connectivity'] = dict(k=int(cr['k']), modules=[dict(n=mm['n'], b1=mm['b1'], life=mm['life']) for mm in cr['modules']])
    # cross-route: within-module corr between connectivity weight and spatial-phase distance
    phase = G_phase(rmf, res, m.L); cc = []
    for mi in range(res['n_modules']):
        ids = np.where(res['labels'] == mi)[0]
        if len(ids) < 10 or mi >= len(res['modules']) or 'spacing' not in res['modules'][mi]: continue
        ph = phase[ids]; d = np.abs(ph[:, None] - ph[None]); d = np.minimum(d, 1 - d); dd = np.sqrt((d ** 2).sum(-1)); Wm = ((m.W.detach().cpu().numpy()[np.ix_(idx, idx)])[np.ix_(ids, ids)]); iu = np.triu_indices(len(ids), 1)
        from scipy.stats import spearmanr
        cc.append(dict(module=mi, n=int(len(ids)), rho=float(spearmanr(Wm[iu] + Wm.T[iu], dd[iu]).statistic)))
    out['cross_route'] = cc
    return out, rm, idx


def G_phase(rmf, res, arena):
    """spatial phase of every unit on its module's lattice torus: coordinates of the ratemap's Fourier peak phases at the module's two lattice reciprocal vectors, in [0,1)^2."""
    bins = rmf.shape[1]; g = (np.arange(bins) + 0.5) / bins * arena; X, Y = np.meshgrid(g, g, indexing='ij'); ph = np.zeros((len(rmf), 2))
    for mi, mm in enumerate(res['modules']):
        if 'a1' not in mm: continue
        B = np.stack([mm['a1'], mm['a2']], 1); Binv = np.linalg.inv(B)                       # reciprocal vectors: rows of inv(B) (cycles per metre)
        for i in np.where(res['labels'] == mi)[0]:
            z = rmf[i] - rmf[i].mean()
            for j in range(2):
                k = 2 * np.pi * Binv[j]; c = (z * np.exp(-1j * (k[0] * X + k[1] * Y))).sum(); ph[i, j] = (np.angle(c) / (2 * np.pi)) % 1.0
    return ph
