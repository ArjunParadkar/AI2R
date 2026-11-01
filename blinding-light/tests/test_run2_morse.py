"""whiteboxtopo.morse: discrete Morse-Smale complex, separatrix distance, boundary curvature -- synthetic ground truth.

Planted landscapes on a uniform 2-D cloud with the library's own k=15 graph (Euclidean = pullback of the identity):
 * two Gaussian bumps, straight boundary x = 0: two basins, separatrix edges straddle x = 0, d_sep ~ |x|,
   ||II|| ~ 0;
 * a circular boundary |x| = R: ||II|| ~ 1/R at three radii.
Measured values are printed (run with -s) and written to results/run2/blockC_tests.json.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from whiteboxtopo.manifold import knn_graph
from whiteboxtopo.morse import (ascending_flow, morse_smale, separatrix_distance, boundary_curvature, rho_prime)

OUT = Path(__file__).resolve().parent.parent / 'results' / 'run2' / 'blockC_tests.json'
REC = {}


def _cloud(n=4000, half=2.0, seed=0):
    return np.random.default_rng(seed).uniform(-half, half, (n, 2))


def _bumps(X, c0=(-1.0, 0.0), c1=(1.0, 0.0), s=0.8):
    l0 = 5 * np.exp(-((X - np.array(c0)) ** 2).sum(1) / (2 * s * s))
    l1 = 5 * np.exp(-((X - np.array(c1)) ** 2).sum(1) / (2 * s * s))
    return np.stack([l0, l1], 1)


@pytest.fixture(scope='module')
def bumps():
    X = _cloud()
    lg = _bumps(X)
    _, _, W = knn_graph(X, k=15)
    ms = morse_smale(W, lg.max(1), lg.argmax(1))
    return X, lg, W, ms


def test_flow_is_ascending_and_terminates_on_maxima(bumps):
    X, lg, W, ms = bumps
    f = lg.max(1)
    succ, basin = ascending_flow(W, f)
    moved = succ != np.arange(len(X))
    assert (f[succ[moved]] > f[moved]).all()
    assert (succ[basin] == basin).all()                      # every basin label is a fixed point of the flow
    for m in np.unique(basin):                               # and a fixed point has no higher neighbour
        nb = W.indices[W.indptr[m]:W.indptr[m + 1]]
        assert (f[nb] <= f[m]).all()


def test_two_bumps_give_two_basins_at_the_bump_centres(bumps):
    X, lg, W, ms = bumps
    assert ms['n_maxima'] == 2
    cm = X[ms['maxima']]
    cm = cm[np.argsort(cm[:, 0])]
    err = np.abs(cm - np.array([[-1.0, 0.0], [1.0, 0.0]])).max()
    REC['two_bumps_max_location_error'] = float(err)
    assert err < 0.1
    # the discrete flow's basin boundary follows x = 0 only to within the graph scale, so basins are not exactly
    # label-pure; the agreement between 'label of my basin's maximum' and 'my label' is measured instead
    lab = lg.argmax(1)
    agree = float((lab[ms['basin']] == lab).mean())
    REC['two_bumps_basin_label_agreement'] = agree
    assert agree > 0.97
    assert (ms['basin'] == ms['maxima'][0]).sum() > 1500


def test_separatrix_edges_straddle_the_known_boundary(bumps):
    X, lg, W, ms = bumps
    e = ms['sep_edges']
    assert len(e) > 20
    assert (X[e[:, 0], 0] * X[e[:, 1], 0] < 0).all()         # every separatrix edge crosses x = 0
    midx = np.abs(0.5 * (X[e[:, 0], 0] + X[e[:, 1], 0]))
    REC['two_bumps_sep_midpoint_abs_x_max'] = float(midx.max())
    REC['two_bumps_n_sep_edges'] = int(len(e))
    assert midx.max() < 0.12                                 # graph scale: median edge ~0.06


def test_d_sep_is_the_distance_to_the_boundary(bumps):
    X, lg, W, ms = bumps
    d, nearest = separatrix_distance(W, ms['sep_edges'], ms['sep_w'])
    assert np.isfinite(d).all() and (nearest >= 0).all()
    true = np.abs(X[:, 0])
    inner = (np.abs(X[:, 1]) < 1.5) & (true < 1.5)           # away from the cloud's corners
    ratio = d[inner] / np.maximum(true[inner], 1e-9)
    big = inner & (true > 0.3)
    REC['d_sep_over_true_median'] = float(np.median(d[big] / true[big]))
    REC['d_sep_abs_err_median'] = float(np.median(np.abs(d[inner] - true[inner])))
    REC['d_sep_pearson'] = float(np.corrcoef(d[inner], true[inner])[0, 1])
    assert REC['d_sep_pearson'] > 0.99
    # two opposite biases of a few %: graph geodesics over-read straight lines, separatrix midpoints sit within a
    # graph edge of x = 0 rather than on it (so the nearest one can be nearer than the line)
    assert 0.93 < REC['d_sep_over_true_median'] < 1.10
    assert REC['d_sep_abs_err_median'] < 0.06
    assert ratio.min() > 0.0


def test_half_edge_convention():
    """Path graph 0-1-2-3, f increasing outwards from the middle edge, labels change on edge (1,2) of length 0.4:
    d_sep = 0.2 at its endpoints, and 0.2 + the path length elsewhere."""
    from scipy.sparse import csr_matrix
    r = [0, 1, 2]; c = [1, 2, 3]; w = [1.0, 0.4, 2.0]
    W = csr_matrix((w + w, (r + c, c + r)), shape=(4, 4))
    f = np.array([3.0, 1.0, 1.5, 4.0]); lab = np.array([0, 0, 1, 1])
    ms = morse_smale(W, f, lab)
    assert ms['n_maxima'] == 2 and ms['sep_edges'].tolist() == [[1, 2]]
    d, nearest = separatrix_distance(W, ms['sep_edges'], ms['sep_w'])
    np.testing.assert_allclose(d, [1.2, 0.2, 0.2, 2.2])
    assert (nearest == 0).all()


def test_label_condition_is_required():
    """Two basins inside ONE class (two bumps, same argmax) produce no separatrix edge; d_sep is infinite."""
    X = _cloud(n=1500, seed=3)
    lg = _bumps(X)
    _, _, W = knn_graph(X, k=15)
    ms = morse_smale(W, lg.max(1), np.zeros(len(X), int))
    assert ms['n_maxima'] == 2 and ms['n_edges_diff_basin'] > 0 and ms['n_sep'] == 0
    d, nearest = separatrix_distance(W, ms['sep_edges'], ms['sep_w'])
    assert np.isinf(d).all() and (nearest == -1).all()
    assert np.isinf(rho_prime(d, nearest, np.zeros(0))).all()


def test_straight_boundary_has_zero_curvature():
    """Linear readout g = 3x: the fit is exact and Q = 0."""
    X = _cloud()
    lg = np.stack([-1.5 * X[:, 0], 1.5 * X[:, 0]], 1) + 5 * np.exp(-(X[:, 1] ** 2))[:, None]
    _, _, W = knn_graph(X, k=15)
    ms = morse_smale(W, lg.max(1), lg.argmax(1))
    bc = boundary_curvature(X, lg, ms['sep_edges'], lg.argmax(1), d=2)
    REC['straight_II_max'] = float(np.nanmax(bc['II']))
    assert ms['n_sep'] > 10 and np.nanmax(bc['II']) < 1e-8
    np.testing.assert_allclose(bc['bnorm'], 3.0, rtol=1e-8)


def test_bump_boundary_is_flat_to_tolerance(bumps):
    """The two-bump boundary x = 0 is straight, but g is not linear: ||II|| must still be small."""
    X, lg, W, ms = bumps
    e = ms['sep_edges']
    keep = np.abs(0.5 * (X[e[:, 0], 1] + X[e[:, 1], 1])) < 1.5
    bc = boundary_curvature(X, lg, e[keep], lg.argmax(1), d=2)
    REC['bump_boundary_II_median'] = float(np.nanmedian(bc['II']))
    assert np.nanmedian(bc['II']) < 0.05


@pytest.mark.parametrize('R', [0.5, 1.0, 1.5])
def test_circular_boundary_curvature_is_one_over_R(R):
    """psi_0 = -r^2 (peak at the origin), psi_1 = -(r - 2R)^2 (a ridge at r = 2R): argmax changes on r = R and
    g = psi_0 - psi_1 = 4R^2 - 4R r, whose zero set has curvature 1/R."""
    X = _cloud(n=5000, half=2.2 * max(R, 1.0), seed=int(10 * R))
    r = np.linalg.norm(X, axis=1)
    lg = np.stack([-r ** 2, -(r - 2 * R) ** 2], 1)
    lab = lg.argmax(1)
    _, _, W = knn_graph(X, k=15)
    ms = morse_smale(W, lg.max(1), lab)
    e = ms['sep_edges']
    rm = np.linalg.norm(0.5 * (X[e[:, 0]] + X[e[:, 1]]), axis=1)
    assert len(e) > 20 and np.abs(rm - R).max() < 0.15 * max(R, 1.0)
    bc = boundary_curvature(X, lg, e, lab, d=2)
    med = float(np.nanmedian(bc['II']))
    iqr = np.nanpercentile(bc['II'], [25, 75])
    REC[f'circle_R{R}'] = dict(expected=1 / R, median=med, rel_err=abs(med * R - 1), q25=float(iqr[0]), q75=float(iqr[1]),
                               n_sep=int(len(e)), n_maxima=ms['n_maxima'])
    assert med == pytest.approx(1 / R, rel=0.10), med
    d, nearest = separatrix_distance(W, e, ms['sep_w'])
    inside = r < 0.8 * R
    assert np.corrcoef(d[inside], (R - r)[inside])[0, 1] > 0.98
    rp = rho_prime(d, nearest, bc['II'])
    ok = np.isfinite(rp)
    np.testing.assert_allclose(rp[ok], d[ok] / (1 + bc['II'][nearest[ok]]))


def test_underdetermined_fit_is_flagged():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((400, 12))
    lg = np.stack([X[:, 0], -X[:, 0]], 1) + 3 * np.exp(-(X[:, 1:] ** 2).sum(1))[:, None]
    _, _, W = knn_graph(X, k=15)
    ms = morse_smale(W, lg.max(1), lg.argmax(1))
    bc = boundary_curvature(X, lg, ms['sep_edges'], lg.argmax(1), d=8, k=30)
    assert bc['n_params'] == 45 and bc['underdetermined']
    assert np.nanmax(bc['resid']) < 1e-6                     # 45 unknowns interpolate 30 points
    assert not boundary_curvature(X, lg, ms['sep_edges'], lg.argmax(1), d=6, k=30)['underdetermined']


def test_zz_write_record():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(REC, indent=1))
    print(json.dumps(REC, indent=1))
