"""N5 persistent homology — T-N5-01..03 (thresholds frozen in TOLERANCES.md)."""
import numpy as np
from scipy.spatial.distance import cdist
from whiteboxtopo.numerics import homology as H

EPS = 1e-6


def circle(n, rng, R=1.0):
    t = rng.uniform(0, 2 * np.pi, n); return R * np.c_[np.cos(t), np.sin(t)]


def sphere(n, rng):
    v = rng.normal(size=(n, 3)); return v / np.linalg.norm(v, axis=1, keepdims=True)


def clifford(n, rng, R=1.0):
    a, b = rng.uniform(0, 2 * np.pi, (2, n)); return R * np.c_[np.cos(a), np.sin(a), np.cos(b), np.sin(b)] / np.sqrt(2)


def test_n5_01_stability(cell):
    rng = np.random.default_rng(0); X = circle(150, rng); D = cdist(X, X)
    dg0 = H.persistence(D=D, maxdim=1); ok = {}; worst = {}
    for eta in (0.01, 0.05, 0.1):
        res = []
        for t in range(20):
            Dp = H.perturb_distance_matrix(D, eta, np.random.default_rng(100 + t))
            dg = H.persistence(D=Dp, maxdim=1)
            res.append([H.bottleneck(dg0[k], dg[k]) for k in (0, 1)])
        res = np.array(res); worst[eta] = res.max(0).tolist(); ok[f'eta={eta}'] = bool((res <= eta + EPS).all())
    cell('T-N5-01', {'n': 150, 'etas': [0.01, 0.05, 0.1], 'trials': 20}, 0, {'worst_bottleneck_H0_H1': worst}, ok)


def _cert(X, ref, ref_sp, r, dim, maxdim):
    dg = H.persistence(X=X, maxdim=maxdim)
    eta_c = 2 * H.hausdorff_to_manifold(X, ref, ref_sp)      # d_B <= 2 d_H
    return dg, eta_c, H.certified_alive(dg[dim], eta_c, r), len(H.uncertified_long(dg[dim], eta_c))


def test_n5_02_certified_betti(cell):
    rng = np.random.default_rng(1); m, conds = {}, {}
    ref = circle(20000, rng); sp = 0.004   # conservative: max gap of 20000 random points on the circle ~ ln(n)/n * 2pi
    dg, e, b1, nlong = _cert(circle(200, rng), ref, sp, 1.0, 1, 1)
    conds['circle b1=1'] = b1 == 1; conds['circle no extra'] = nlong == 1; m['circle'] = dict(eta_c=e, b1=b1, nlong=nlong)
    # two circles, centres 6 apart
    refc = np.r_[circle(20000, rng), circle(20000, rng) + [6, 0]]
    X = np.r_[circle(150, rng), circle(150, rng) + [6, 0]]
    dg = H.persistence(X=X, maxdim=1); e = 2 * H.hausdorff_to_manifold(X, refc, sp)
    b0 = H.certified_alive(np.c_[dg[0][:, 0], np.where(np.isfinite(dg[0][:, 1]), dg[0][:, 1], 1e9)], e, 1.0)
    b1 = H.certified_alive(dg[1], e, 1.0)
    conds['two circles b0=2'] = b0 == 2; conds['two circles b1=2'] = b1 == 2; m['two_circles'] = dict(eta_c=e, b0=b0, b1=b1)
    # sphere
    refs = sphere(40000, rng); sps = 0.04   # conservative: random-sample gap bound for 40000 points on S^2
    X = sphere(300, rng); dg = H.persistence(X=X, maxdim=2); e = 2 * H.hausdorff_to_manifold(X, refs, sps)
    b2 = H.certified_alive(dg[2], e, 1.0); b1 = H.certified_alive(dg[1], e, 1.0)
    conds['sphere b2=1'] = b2 == 1; conds['sphere b1=0'] = b1 == 0; m['sphere'] = dict(eta_c=e, b2=b2, b1=b1)
    # clifford torus H1
    reft = clifford(60000, rng); spt = 0.08   # conservative: random-sample gap bound for 60000 points on the Clifford torus
    X = clifford(500, rng); dg = H.persistence(X=X, maxdim=2); e = 2 * H.hausdorff_to_manifold(X, reft, spt)
    b1 = H.certified_alive(dg[1], e, 0.9); nl = len(H.uncertified_long(dg[1], e))
    conds['torus b1=2'] = b1 == 2; conds['torus no extra H1'] = nl == 2
    m['torus'] = dict(eta_c=e, b1=b1, nlong=nl, H2_reported=int(H.certified_alive(dg[2], e, 0.9)))
    cell('T-N5-02', {'seeds': 1}, 1, m, conds)


def test_n5_03_subsampling(cell):
    rng = np.random.default_rng(2); X = clifford(1500, rng) + 0.01 * rng.normal(size=(1500, 4))
    dgX = H.persistence(X=X, maxdim=1); holds = []; err = {}
    for kind in ('random', 'maxmin'):
        for mm in (50, 100, 200, 400):
            es = []
            for s in range(5):
                if kind == 'random': idx = np.random.default_rng(10 + s).choice(len(X), mm, replace=False); cr = H.covering_radius(X[idx], X)
                else: idx, cr = H.maxmin_subsample(X, mm, seed=s)
                dg = H.persistence(X=X[idx], maxdim=1)
                e = max(H.bottleneck(dgX[k], dg[k]) for k in (0, 1)); es.append(e); holds.append(e <= 2 * cr + EPS)
            err[f'{kind}_{mm}'] = float(np.mean(es))
    cell('T-N5-03', {'n': 1500, 'm': [50, 100, 200, 400], 'seeds': 5}, 2, {'mean_err': err},
         {'bound holds 100%': all(holds), 'maxmin err(400)<err(50)': err['maxmin_400'] < err['maxmin_50']})
