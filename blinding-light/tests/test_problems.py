"""whiteboxtopo.problems: LayerData and the seven problems, on clouds with analytic readouts."""
import numpy as np
import pytest

from conftest import make_circle, arc_classifier, offplane_direction, random_orthogonal
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.nulls import label_permute, feature_shuffle
from whiteboxtopo.topology import persistence, circular_correlation, circular_rank_correlation
from whiteboxtopo.problems import (LayerData, fiber_beta0, p1_inventory, p2_explain, p4_ignorance,
                                   p5_failure, p6_geodesic_generalization, p3_spurious,
                                   p7_formation)
from whiteboxtopo.problems.core import path_product_check, _coordinate_tangent, _softmax_conf, _cname

TWO_PI = 2 * np.pi


@pytest.fixture(scope='module')
def g_fit(circle, circle_readout):
    return fit_manifold(circle.X, J=circle_readout.J, metric='g')


@pytest.fixture(scope='module')
def null_band(circle):
    h1 = persistence(feature_shuffle(circle.X, seed=0), do_cocycles=False)['dgms'][1]
    return float((h1[:, 1] - h1[:, 0]).max())


@pytest.fixture(scope='module')
def inventory(circle_layer, g_fit, null_band):
    return p1_inventory(circle_layer, g_fit, noise_band=null_band)


# ------------------------------------------------------------------ LayerData
def test_layerdata_pred_and_defaults(circle_layer, circle_readout):
    assert (circle_layer.pred == circle_readout.logits.argmax(1)).all()
    # the analytic head really does cut the circle into its four contiguous arcs
    assert (circle_layer.pred == circle_readout.true_arc).mean() > 0.99
    ld = LayerData('x', np.zeros((3, 2)), np.zeros((3, 1, 2)), np.zeros((3, 1)), np.zeros(3))
    assert ld.variables == {} and ld.modular_period is None


# ------------------------------------------------------------------ P1
def test_p1_inventory_of_a_circle(circle, inventory, g_fit):
    inv = inventory
    assert inv['layer'] == 'circle'
    assert inv['beta0'] == 1 and inv['beta1'] == 1
    assert inv['m'] == g_fit.m and inv['m'] > 0.9
    assert abs(inv['d_intrinsic'] - 1) < 0.2
    assert inv['tildeC'] > 1.0
    assert len(inv['coords']) == 1 and inv['independent_coords'] == [0] and inv['dependent_coords'] == []
    c = inv['coords'][0]
    assert circular_correlation(c['theta'], circle.theta) > 0.95
    assert c['lifetime'] > 1.3
    nm = c['naming']
    assert nm['named'] is not None and nm['named']['name'] == 'cos_theta' and nm['named']['r'] > 0.95
    junk = [d for d in nm['all'] if d['name'] == 'junk'][0]
    assert junk['r'] < 0.3
    assert inv['dgm1'].shape[1] == 2


def test_p1_fibers_of_contiguous_arcs_are_contractible(inventory, circle_layer):
    fib = inventory['fibers']
    assert set(fib) == {0, 1, 2, 3}
    for y, f in fib.items():
        assert f['n'] == (circle_layer.labels == y).sum()
        assert f['beta1'] == 0, (y, f['beta1'])
        assert 'coord' not in f


def test_p1_inventory_gap_fallback_and_options(circle_layer, g_fit):
    inv = p1_inventory(circle_layer, g_fit, do_fibers=False)
    assert 'fibers' not in inv
    assert inv['beta1'] == 1 and len(inv['coords']) == 1
    strict = p1_inventory(circle_layer, g_fit, do_fibers=False, name_threshold=1.01)
    assert strict['coords'][0]['naming']['named'] is None


def test_p1_inventory_two_circles(two_circles):
    n = two_circles.n
    ld = LayerData('two', two_circles.X, np.zeros((n, 1, two_circles.D)), np.zeros((n, 1)),
                   two_circles.which, variables={'which': two_circles.which.astype(float)})
    fit = fit_manifold(ld.X)
    inv = p1_inventory(ld, fit, fiber_min=40)
    assert inv['beta0'] == 2 and inv['beta1'] == 2
    assert len(inv['coords']) == 2
    # each class's coordinate is the angle of ONE circle and constant on the other.
    # The harmonic coordinate is a monotone warp of arc length on a randomly sampled circle
    # (measured linear r = 0.986 and 0.942 on the two circles, rank r = 0.9999 on both), so the
    # cyclic ORDER is asserted tightly and the linear agreement at 0.9.
    owners = []
    for c in inv['coords']:
        sel = [two_circles.which == w for w in (0, 1)]
        r = [circular_correlation(c['theta'][m], two_circles.theta[m]) for m in sel]
        w = int(np.argmax(r))
        owners.append(w)
        assert r[w] > 0.9 and r[1 - w] < 0.3, r
        assert circular_rank_correlation(c['theta'][sel[w]], two_circles.theta[sel[w]]) > 0.99
    assert sorted(owners) == [0, 1]
    # each fiber is a whole circle: one loop, with a coordinate
    for y in (0, 1):
        f = inv['fibers'][y]
        assert f['beta1'] == 1
        sel = two_circles.which == y
        assert circular_correlation(f['coord']['theta'], two_circles.theta[sel]) > 0.9
        assert circular_rank_correlation(f['coord']['theta'], two_circles.theta[sel]) > 0.99


def test_p1_inventory_on_noise_reports_a_low_manifold_score(noise):
    n = noise.n
    ld = LayerData('noise', noise.X, np.zeros((n, 1, noise.D)), np.zeros((n, 1)), np.zeros(n, int))
    inv = p1_inventory(ld, fit_manifold(noise.X), do_fibers=False)
    assert inv['m'] < 0.1
    assert inv['d_intrinsic'] > 5


# ------------------------------------------------------------------ P6 / fiber_beta0
def test_p6_geodesic_generalization_is_one_for_a_consistent_arc_classifier(circle_layer, g_fit):
    out = p6_geodesic_generalization(circle_layer, g_fit)
    assert out['G'] > 0.98, out['G']
    assert out['G_interior'] > 0.98, out['G_interior']
    assert set(out['per_class']) == {0, 1, 2, 3}
    assert all(out['understood'].values())
    assert out['m'] == g_fit.m and out['n_pairs'] > 600
    assert 0 <= out['frac_direct'] < 0.5


def test_p6_drops_to_chance_under_label_permutation(circle_layer, g_fit):
    perm = LayerData('perm', circle_layer.X, circle_layer.J, circle_layer.logits,
                     label_permute(circle_layer.labels, seed=1))
    out = p6_geodesic_generalization(perm, g_fit)
    # four equal arcs: a path node carries a random label's class with probability 1/4
    assert abs(out['G'] - 0.25) < 0.1, out['G']
    assert abs(out['G_interior'] - 0.25) < 0.1, out['G_interior']
    assert not any(out['understood'].values())


def test_p6_is_deterministic_and_skips_unlabelled_points(circle_layer, g_fit):
    a = p6_geodesic_generalization(circle_layer, g_fit, seed=3, pairs_per_class=50)
    b = p6_geodesic_generalization(circle_layer, g_fit, seed=3, pairs_per_class=50)
    assert a['G'] == b['G'] and a['n_pairs'] == b['n_pairs']
    labels = circle_layer.labels.copy()
    labels[labels == 2] = -1
    part = LayerData('part', circle_layer.X, circle_layer.J, circle_layer.logits, labels)
    out = p6_geodesic_generalization(part, g_fit)
    assert set(out['per_class']) == {0, 1, 3}


def test_p6_disconnected_pairs_score_zero(two_circles):
    """One class spread over two components: about half of the pairs have no geodesic."""
    n = two_circles.n
    logits = np.zeros((n, 1))
    ld = LayerData('two', two_circles.X, np.zeros((n, 1, two_circles.D)), logits, np.zeros(n, int))
    out = p6_geodesic_generalization(ld, fit_manifold(two_circles.X))
    assert out['G'] == pytest.approx(0.5, abs=0.1), out['G']


def test_fiber_beta0(circle_layer, g_fit, two_circles):
    assert fiber_beta0(circle_layer, g_fit) == {0: 1, 1: 1, 2: 1, 3: 1}
    perm = LayerData('perm', circle_layer.X, circle_layer.J, circle_layer.logits,
                     label_permute(circle_layer.labels, seed=1))
    b = fiber_beta0(perm, g_fit)
    assert all(v > 3 for v in b.values()), b        # scattered labels: the fiber falls apart
    n = two_circles.n
    ld = LayerData('two', two_circles.X, None, None, np.zeros(n, int))
    assert fiber_beta0(ld, fit_manifold(two_circles.X)) == {0: 2}


# ------------------------------------------------------------------ P2
def test_p2_margin_is_the_pullback_arc_length_to_the_boundary(circle, circle_layer, circle_readout, g_fit):
    inputs = [10, 111, 222, 333]
    psi = lambda pts: np.broadcast_to(circle_readout.W, (len(pts),) + circle_readout.W.shape)
    out = p2_explain(circle_layer, g_fit, inputs, psi=psi)
    assert [r['i'] for r in out] == inputs
    q = np.pi / 2
    # along the circle the head moves the logits at |W t| = scale * sqrt(2) per unit arc
    speed = 5.0 * np.sqrt(2)
    for r in out:
        th = circle.theta[r['i']]
        to_b = min(np.mod(th, q), q - np.mod(th, q))
        assert r['pred'] == circle_layer.pred[r['i']]
        assert abs(r['flips_to'] - r['pred']) in (1, 3)              # an adjacent arc
        assert r['path'][0] == r['i'] and circle_layer.pred[r['path'][-1]] == r['flips_to']
        assert (circle_layer.pred[r['path'][:-1]] == r['pred']).all()
        assert r['n_steps'] == len(r['path']) - 1 and r['component'] == 0
        assert r['margin'] == pytest.approx(speed * to_b, rel=0.1, abs=0.4), (r['margin'], speed * to_b)
        assert r['euclid_margin'] <= r['margin'] / speed * 1.05 + 0.02
        np.testing.assert_allclose(r['tangent'], circle.X[r['path'][1]] - circle.X[r['path'][0]])
        # linear head: path product attribution == integrated gradients, exactly
        assert r['ig_r'] == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(r['lin_attr'], r['ig_attr'], atol=1e-10)


def test_p2_coords_passthrough_and_no_other_class(circle_layer, g_fit, inventory):
    out = p2_explain(circle_layer, g_fit, [5], coords=inventory['coords'])
    key = f"c{inventory['coords'][0]['class_idx']}"
    assert out[0]['coords'][key] == pytest.approx(float(inventory['coords'][0]['theta'][5]))
    one = LayerData('one', circle_layer.X, circle_layer.J, np.ones((len(circle_layer.X), 2)) * [1, 0],
                    circle_layer.labels)
    res = p2_explain(one, g_fit, [5])
    assert res[0]['margin'] == np.inf and res[0]['path'] is None


def test_path_product_check_on_a_nonlinear_head(circle):
    """psi = V tanh(U h): over one short graph edge the local linear map and IG must agree."""
    rng = np.random.default_rng(50)
    U, V = 0.7 * rng.standard_normal((16, circle.D)), rng.standard_normal((4, 16))
    jac = lambda P: np.einsum('oh,nh,hd->nod', V, 1 - np.tanh(P @ U.T) ** 2, U)
    ld = LayerData('mlp', circle.X, jac(circle.X), np.tanh(circle.X @ U.T) @ V.T, np.zeros(circle.n, int))
    out = path_product_check(ld, jac, np.array([0, 1]))
    assert out['ig_r'] > 0.999, out['ig_r']
    # IG is exact: it equals the true logit change (completeness)
    true = ld.logits[1] - ld.logits[0]
    np.testing.assert_allclose(out['ig_attr'], true, rtol=1e-3, atol=1e-7)


# ------------------------------------------------------------------ P4
@pytest.fixture(scope='module')
def p4_setup(circle):
    v = offplane_direction(circle)
    ro = arc_classifier(circle, extra_rows=v)
    fit = fit_manifold(circle.X, J=ro.J, metric='g', dim_cap=1)
    rng = np.random.default_rng(60)
    th = rng.uniform(0, TWO_PI, 300)
    P = np.stack([np.cos(th), np.sin(th)], 1) @ circle.E.T
    hold = P + circle.sigma * rng.standard_normal(P.shape)
    ld = LayerData('c', circle.X, ro.J, ro.logits, ro.logits.argmax(1))
    return dict(ld=ld, fit=fit, hold=hold, P=P, v=v, W=ro.W)


def test_p4_ignorance_separates_off_manifold_inputs(p4_setup):
    s = p4_setup
    ood = {'radial': (s['hold'] + 0.5 * s['P'], None),
           'offplane': (s['hold'] + 0.5 * s['v'][None], None),
           'in_dist': (s['hold'][::-1].copy(), None)}
    out = p4_ignorance(s['ld'], s['fit'], s['hold'], ood)
    assert out['m'] == s['fit'].m and out['reach'] > 0 and out['holdout_dg'].shape == (300,)
    for nm in ('radial', 'offplane'):
        r = out['sets'][nm]
        assert r['auc_ignorance'] > 0.99, (nm, r['auc_ignorance'])
        assert r['auc_euclid_knn'] > 0.99
        assert r['frac_outside'] == 1.0
        assert 'auc_softmax' not in r
    r = out['sets']['in_dist']                # the same points: no detector can tell them apart
    assert r['auc_ignorance'] == pytest.approx(0.5, abs=1e-9)
    assert r['frac_outside'] <= 0.02


def test_p4_softmax_baseline_is_reported_when_holdout_logits_exist(p4_setup):
    s = p4_setup
    ld = LayerData('c', s['ld'].X, s['ld'].J, s['ld'].logits, s['ld'].labels)
    ld.logits_holdout = s['hold'] @ s['W'].T
    off = s['hold'] + 0.5 * s['v'][None]
    out = p4_ignorance(ld, s['fit'], s['hold'], {'offplane': (off, off @ s['W'].T)})
    r = out['sets']['offplane']
    assert 0.0 <= r['auc_softmax'] <= 1.0
    # pushing along one logit's direction changes the confidence only mildly; the geometric
    # detector must beat it
    assert r['auc_ignorance'] > r['auc_softmax']


def test_softmax_conf():
    z = np.array([[0.0, 0.0], [1000.0, 0.0], [np.log(3.0), 0.0]])
    np.testing.assert_allclose(_softmax_conf(z), [0.5, 1.0, 0.75])


# ------------------------------------------------------------------ P5
def _sphere4(n, R, seed):
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n, 5))
    return R * P / np.linalg.norm(P, axis=1, keepdims=True)


def test_p5_failure_risk_tracks_curvature_on_a_4d_manifold():
    """Two 4-spheres (R=1 and R=3), a constant linear head: rho = ||II|| * Lip is 3x larger on
    the small sphere, where a step of fixed size leaves the manifold sooner.  Take the failure
    radius to be R."""
    n = 600
    E = random_orthogonal(8, 5, 70)
    X = np.concatenate([_sphere4(n, 1.0, 71) @ E.T, _sphere4(n, 3.0, 72) @ E.T + 20.0])
    radius = np.r_[np.full(n, 1.0), np.full(n, 3.0)]
    W = np.random.default_rng(73).standard_normal((3, 8))
    J = np.broadcast_to(W, (2 * n, 3, 8)).copy()
    ld = LayerData('s4', X, J, X @ W.T, np.zeros(2 * n, int))
    fit = fit_manifold(X, J=J, metric='g')
    out = p5_failure(ld, fit, radius)
    assert out['n'] == 2 * n and out['m'] == fit.m
    np.testing.assert_allclose(out['lip'], np.linalg.svd(W, compute_uv=False)[0], rtol=1e-8)
    np.testing.assert_allclose(out['rho'], out['curv'] * out['lip'])
    ratio = np.median(out['curv'][:n]) / np.median(out['curv'][n:])
    assert ratio == pytest.approx(3.0, rel=0.1), ratio
    # a two-valued radius caps |Spearman| at sqrt(3)/2 = 0.866
    assert out['spearman_rho'] < -0.8, out['spearman_rho']
    assert out['p_rho'] < 1e-20
    assert -1 <= out['spearman_softmax_margin'] <= 1
    assert out['saddles'].dtype.kind == 'i'


def test_p5_ignores_points_without_a_failure_radius():
    n = 300
    X = _sphere4(n, 1.0, 74)
    W = np.random.default_rng(75).standard_normal((3, 5))
    J = np.broadcast_to(W, (n, 3, 5)).copy()
    ld = LayerData('s4', X, J, X @ W.T, np.zeros(n, int))
    radius = np.random.default_rng(76).uniform(1, 2, n)
    radius[::3] = np.inf
    out = p5_failure(ld, fit_manifold(X, J=J), radius)
    assert out['n'] == np.isfinite(radius).sum()
    assert np.isfinite(out['spearman_rho'])


def test_p5_failure_risk_tracks_curvature_on_a_1d_manifold():
    n = 400
    c1 = make_circle(n=n, R=1.0, sigma=0.0, seed=80)
    c3 = make_circle(n=n, R=3.0, sigma=0.0, seed=80)
    X = np.concatenate([c1.X, c3.X + 20.0])
    radius = np.r_[np.full(n, 1.0), np.full(n, 3.0)]
    W = np.random.default_rng(81).standard_normal((3, c1.D))
    J = np.broadcast_to(W, (2 * n, 3, c1.D)).copy()
    ld = LayerData('circles', X, J, X @ W.T, np.zeros(2 * n, int))
    out = p5_failure(ld, fit_manifold(X, J=J), radius)
    ratio = np.median(out['curv'][:n]) / np.median(out['curv'][n:])
    assert np.median(out['curv'][:n]) == pytest.approx(0.5, rel=0.1), np.median(out['curv'][:n])
    assert ratio == pytest.approx(3.0, rel=0.1), ratio
    assert out['spearman_rho'] < -0.8, out['spearman_rho']


# ------------------------------------------------------------------ P3
def test_p3_flags_a_coordinate_that_tracks_a_nuisance_the_output_reads(circle, circle_layer, g_fit, inventory):
    rng = np.random.default_rng(90)
    nuisance = {'sin_angle': np.sin(circle.theta), 'unrelated': rng.standard_normal(circle.n)}
    out = p3_spurious(circle_layer, g_fit, nuisance, coords=inventory['coords'])
    assert out['m'] == g_fit.m and out['tau'] is None
    assert out['n_spurious'] == 1 and len(out['classes']) == 1
    c = out['classes'][0]
    assert c['best_nuisance'] == 'sin_angle' and c['best_corr'] > 0.95
    assert c['corr']['unrelated'] < 0.3
    # the head moves at scale * sqrt(sum_c sin^2(theta - centre_c)) = 5 * sqrt(2) along the circle
    assert c['sensitivity'] == pytest.approx(5 * np.sqrt(2), rel=0.05), c['sensitivity']
    assert c['lifetime'] == inventory['coords'][0]['lifetime']


def test_p3_requires_both_correlation_and_output_dependence(circle, circle_layer, g_fit, inventory):
    rng = np.random.default_rng(91)
    only_junk = p3_spurious(circle_layer, g_fit, {'unrelated': rng.standard_normal(circle.n)},
                            coords=inventory['coords'])
    assert only_junk['n_spurious'] == 0
    high_tau = p3_spurious(circle_layer, g_fit, {'sin_angle': np.sin(circle.theta)},
                           coords=inventory['coords'], tau=100.0)
    assert high_tau['n_spurious'] == 0 and high_tau['classes'][0]['best_corr'] > 0.95
    # a head that does not read the circle at all: correlated, but the output does not depend on it
    J0 = np.zeros_like(circle_layer.J)
    fit0 = fit_manifold(circle.X)
    fit0.J = J0
    blind = p3_spurious(circle_layer, fit0, {'sin_angle': np.sin(circle.theta)},
                        coords=inventory['coords'], tau=0.1)
    assert blind['classes'][0]['sensitivity'] == 0.0 and blind['n_spurious'] == 0


def test_p3_modular_nuisance_and_internal_inventory():
    """coords=None: P3 runs P1 itself.  Nuisance is a Z_24 variable wound 3 times."""
    p = 24
    c = make_circle(n=312, sigma=0.003, seed=92)
    z = np.floor(np.mod(3 * c.theta, TWO_PI) / TWO_PI * p).astype(int) % p     # 3*theta, discretised
    ro = arc_classifier(c)
    ld = LayerData('mod', c.X, ro.J, ro.logits, ro.logits.argmax(1), modular_period=p)
    fit = fit_manifold(c.X, J=ro.J)
    out = p3_spurious(ld, fit, {'z': z, 'other': np.random.default_rng(93).integers(0, p, c.n)})
    assert len(out['classes']) == 1
    cl = out['classes'][0]
    # theta is z at frequency 1/3 -- not an integer frequency of z, so no character matches ...
    assert cl['corr']['z'] < 0.3 and cl['corr']['other'] < 0.3
    assert out['n_spurious'] == 0
    # ... whereas the variable a = floor(theta) itself matches at k=1
    a = np.floor(c.theta / TWO_PI * p).astype(int) % p
    out2 = p3_spurious(ld, fit, {'a': a})
    assert out2['classes'][0]['best_corr'] > 0.95 and out2['n_spurious'] == 1


def test_coordinate_tangent_is_the_unit_tangent_of_the_circle(circle, g_fit, inventory):
    v = _coordinate_tangent(g_fit, inventory['coords'][0]['theta'])
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-9)
    true_t = np.stack([-np.sin(circle.theta), np.cos(circle.theta)], 1) @ circle.E.T
    cos = np.einsum('nD,nD->n', v, true_t)
    assert np.median(np.abs(cos)) > 0.99
    assert abs(np.sign(cos).mean()) > 0.98, 'orientation must be consistent around the circle'


# ------------------------------------------------------------------ P7
def _coord(birth, death, theta, name=None):
    naming = dict(named=dict(name=name, freq=None) if name else None,
                  best=dict(name=name or 'junk', r=0.4, freq=None))
    return dict(birth=birth, death=death, lifetime=death - birth, theta=theta, naming=naming)


def test_p7_formation_tracks_birth_order():
    rng = np.random.default_rng(95)
    tA, tB, tC = (rng.uniform(0, TWO_PI, 300) for _ in range(3))
    jit = lambda t: np.mod(t + 0.05 * rng.standard_normal(300) + 0.4, TWO_PI)
    inv = [dict(coords=[]),
           dict(coords=[_coord(0.10, 1.00, tA, 'angle')]),
           dict(coords=[_coord(0.30, 1.50, tB), _coord(0.11, 1.02, jit(tA), 'angle')]),
           dict(coords=[_coord(0.12, 1.05, jit(tA), 'angle'), _coord(0.31, 1.52, jit(tB)),
                        _coord(2.00, 2.60, tC)]),
           dict(coords=[_coord(0.12, 1.06, jit(tA), 'angle'), _coord(0.30, 1.55, jit(tB))])]
    out = p7_formation(inv, steps=[0, 100, 200, 300, 400])
    tr = out['tracks']
    assert len(tr) == 3
    assert out['n_surviving'] == 2
    assert sorted(out['birth_order']) == [100, 200]
    a = [t for t in tr if t['name'] == 'angle'][0]
    assert a['birth_step'] == 100 and a['survives'] and len(a['lifetimes']) == 4
    dead = [t for t in tr if not t['survives']][0]
    assert dead['birth_step'] == 300 and dead['last'] == 3
    assert 'angle' in out['names'] and any(n.startswith('unnamed') for n in out['names'])


def test_p7_formation_with_nothing_to_track():
    out = p7_formation([dict(coords=[]), dict(coords=[])], steps=[0, 1])
    assert out == dict(tracks=[], n_surviving=0, birth_order=[], names=[])


def test_cname():
    assert _cname(dict(naming=dict(named=dict(name='a', freq=17)))) == 'a@freq17'
    assert _cname(dict(naming=dict(named=dict(name='x', freq=None)))) == 'x'
    assert _cname(dict(naming=dict(named=None, best=dict(name='b', r=0.456)))) == 'unnamed(best b r=0.46)'
    assert _cname({}) == 'unnamed'
