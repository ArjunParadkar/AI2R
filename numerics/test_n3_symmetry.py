import math, numpy as np, torch
from common import *
from whiteboxtopo.numerics import symmetry as S


def gl(d):
    out = []
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d)); E[i, j] = 1; out.append(E)
    return out


def so3():
    L = []
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        A = np.zeros((3, 3)); A[i, j] = 1; A[j, i] = -1; L.append(A / math.sqrt(2))
    return L


def pts(n, d, seed): return np.random.default_rng(seed).standard_normal((n, d))


def test_n3_01_02(cell):
    X3 = pts(2000, 3, 0)
    r = S.search_lie(lambda x: torch.linalg.norm(x).reshape(1), X3, gl(3))
    gap = r['eigenvalues'][3] / max(r['eigenvalues'][2], 1e-300)
    sine = S.principal_sine(r['basis'], so3()) if r['dim'] == 3 else 1.0
    cell('T-N3-01', {'n': 2000, 'candidates': 'gl(3)'}, 0, {'dim': r['dim'], 'lam3': r['eigenvalues'][2], 'lam4': r['eigenvalues'][3], 'ratio': gap, 'sine': sine},
         {'dim=3': r['dim'] == 3, 'gap>=1e6': gap >= 1e6, 'sine<=1e-8': sine <= 1e-8})
    cases = {
        'i_norm': (lambda x: torch.linalg.norm(x).reshape(1), 3, False, r),
        'ii_proj12': (lambda x: x[:2], 3, False, None),
        'iii_x1x2sq_x3': (lambda x: torch.stack([x[0] ** 2 + x[1] ** 2, x[2]]), 1, True, None),
        'iv_identity': (lambda x: x, 0, True, None)}
    res = {}; conds = {}
    for name, (F, dim, ab, pre) in cases.items():
        rr = pre or S.search_lie(F, X3, gl(3))
        res[name] = {'dim': rr['dim'], 'abelian': rr['abelian'], 'closure_res': rr['closure_residual'], 'max_bracket': rr['max_bracket']}
        want_dim = {'i_norm': 3, 'ii_proj12': 3}.get(name, dim)
        conds[name + ' dim'] = rr['dim'] == want_dim
        conds[name + ' abelian flag'] = (rr['abelian'] == ab) if rr['dim'] > 1 else True
        conds[name + ' closure<=1e-8'] = rr['closure_residual'] <= 1e-8
    cell('T-N3-02', {'n': 2000}, 0, res, conds)


def test_n3_03_finite_group(cell):
    X = pts(500, 3, 1)
    F = lambda X: np.stack([X.sum(1), (X ** 2).sum(1), (X ** 3).sum(1)], 1)
    perm = lambda p: np.eye(3)[list(p)]
    gens = [perm((1, 0, 2)), perm((1, 2, 0))]
    els = S.group_closure(gens); ab = S.group_is_abelian(els)
    d_true = max(S.delta_exact(F, X, lambda Y, P=P: Y @ P.T) for P in els)
    rng = np.random.default_rng(2)
    decoys = [np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(10)]
    d_dec = min(S.delta_exact(F, X, lambda Y, P=P: Y @ P.T) for P in decoys)
    cell('T-N3-03', {'n': 500}, 1, {'closure_size': len(els), 'abelian': ab, 'delta_true_max': d_true, 'delta_decoy_min': d_dec},
         {'size=6': len(els) == 6, 'nonabelian': not ab, 'true<=1e-12': d_true <= 1e-12, 'decoy>=0.1': d_dec >= 0.1})


def test_n3_04_calibration(cell):
    X = pts(3000, 2, 3); Fv = np.linalg.norm(X, axis=1)
    dd = S.DatasetDelta(X, k=3)
    rot = lambda t: (lambda Y: Y @ np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]).T)
    Ts = {**{f'rot{t:.2f}': (rot(t), True) for t in (0.3, 1.0, 2.0, math.pi)},
          'scale2': (lambda Y: 2 * Y, False), 'shift3': (lambda Y: Y + np.array([3.0, 0.0]), False)}
    rng = np.random.default_rng(4); rng2 = np.random.default_rng(5)
    meas = {}; conds = {}; fpr_max = 0.0
    for name, (T, should) in Ts.items():
        idx = dd.prepare(T); obs = dd.delta(Fv, idx)
        null = S.null_deltas(dd, Fv, T, 400, rng)
        pt, lo, hi, eps = S.calibrate(null, 0.05, 0.95)
        held = S.null_deltas(dd, Fv, T, 400, rng2)
        fpr = float((held <= lo).mean()); fpr_max = max(fpr_max, fpr)
        meas[name] = {'delta': obs, 'thr_point': pt, 'thr_lo': lo, 'thr_hi': hi, 'dkw_eps': eps, 'fpr_heldout': fpr}
        conds[name + (' accepted' if should else ' rejected')] = (obs <= lo) == should
    conds['fpr<=0.10'] = fpr_max <= 0.10
    cell('T-N3-04', {'n': 3000, 'R': 400, 'alpha': 0.05, 'conf': 0.95}, 3, meas, conds)
