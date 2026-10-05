"""whiteboxtopo.topology: persistence, circular coordinates, circular statistics, fibers,
class matching, and the isotypic (equivariant) decomposition."""
import itertools

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from conftest import make_circle
from whiteboxtopo.nulls import feature_shuffle
from whiteboxtopo.topology import (persistence, significant_classes, normalized_persistence,
                                   circular_coordinates, circular_correlation,
                                   circular_rank_correlation, circular_linear_correlation,
                                   name_coordinate, toroidal_decorrelate, fiber_topology,
                                   match_classes, induced_shift_operator, isotypic_planes,
                                   isotypic_decomposition)

TWO_PI = 2 * np.pi


def lifetimes(dgm):
    l = dgm[:, 1] - dgm[:, 0]
    return np.sort(l[np.isfinite(l)])[::-1]


def null_band(X, seed=0):
    """The band open_model uses: longest H1 bar of one feature-shuffled copy."""
    h1 = persistence(feature_shuffle(X, seed=seed), maxdim=1, do_cocycles=False)['dgms'][1]
    return float((h1[:, 1] - h1[:, 0]).max()) if len(h1) else 0.0


# ------------------------------------------------------------------ persistence
def test_persistence_bookkeeping(circle, circle_ph):
    r = circle_ph
    d = pdist(circle.X)
    assert r['thresh'] == pytest.approx(np.percentile(d, 95))
    assert r['median_dist'] == pytest.approx(np.median(d))
    assert r['D'].shape == (circle.n, circle.n)
    assert len(r['dgms']) == 2
    assert len(r['cocycles'][1]) == len(r['dgms'][1])
    assert len(r['dgms'][0]) == circle.n          # one H0 bar per point
    assert np.isinf(r['dgms'][0][:, 1]).sum() == 1, 'the 95th-pct complex of a circle is connected'


def test_persistence_accepts_a_distance_matrix(circle_ph):
    ix = np.arange(0, circle_ph['D'].shape[0], 3)          # every third point: still a circle
    sub = circle_ph['D'][np.ix_(ix, ix)]
    a = persistence(sub, distance_matrix=True, do_cocycles=False)
    assert a['D'] is sub
    assert lifetimes(a['dgms'][1])[0] > 1.0


def test_persistence_circle_has_exactly_one_dominant_h1_class(circle_ph):
    h1 = circle_ph['dgms'][1]
    life = lifetimes(h1)
    assert life[0] > 10 * life[1], life[:3]
    sig, band = significant_classes(h1)
    assert len(sig) == 1
    birth, death = h1[sig[0]]
    # Rips H1 of the unit circle dies at sqrt(3) (inscribed equilateral triangle) or a bit
    # earlier on a finite sample; it is born at about the largest sample gap.
    assert 1.4 < death <= np.sqrt(3) + 0.02, death
    assert birth < 0.15, birth


def test_persistence_torus_has_exactly_two_dominant_h1_classes(torus_ph):
    h1 = torus_ph['dgms'][1]
    life = lifetimes(h1)
    assert life[1] > 0.9 * life[0], life[:3]          # the two generators are equally long
    assert life[1] > 4 * life[2], life[:4]
    sig, _ = significant_classes(h1)
    assert len(sig) == 2


def test_persistence_two_circles_two_h1_classes_and_two_components(two_circles_ph):
    sig, _ = significant_classes(two_circles_ph['dgms'][1])
    assert len(sig) == 2
    h0 = two_circles_ph['dgms'][0]
    l0 = h0[:, 1] - h0[:, 0]
    big = (l0[np.isfinite(l0)] > 1.0).sum() + np.isinf(l0).sum()
    assert big == 2, 'one essential component plus one bar of length ~ the gap between the circles'


def test_persistence_noise_has_no_dominant_class(noise, noise_ph, circle, circle_ph):
    """Relative to the matched feature-shuffle null, the circle's class is far outside the band
    and the noise cloud's longest bar is of the same size as the band."""
    top_noise = lifetimes(noise_ph['dgms'][1])[0]
    band_noise = null_band(noise.X)
    top_circle = lifetimes(circle_ph['dgms'][1])[0]
    band_circle = null_band(circle.X)
    assert top_circle > 4 * band_circle, (top_circle, band_circle)
    assert top_noise < 1.5 * band_noise, (top_noise, band_noise)
    # scale-free: longest bar / median distance
    assert top_noise / noise_ph['median_dist'] < 0.25
    assert top_circle / circle_ph['median_dist'] > 1.0


def test_no_significant_class_on_noise_with_the_open_model_band(noise, noise_ph):
    """The band open_model uses: 1.5x the longest H1 bar over two independent feature shuffles.
    (The bare longest bar of ONE shuffle is exchangeable with the data's on a structureless
    cloud and flagged 6 of 12 Gaussian clouds; that procedure was replaced.)"""
    from whiteboxtopo.nulls import feature_shuffle
    tops = []
    for sd in (0, 1):
        d1 = persistence(feature_shuffle(noise.X, seed=sd), do_cocycles=False)['dgms'][1]
        lf = d1[:, 1] - d1[:, 0]; tops.append(float(lf[np.isfinite(lf)].max()))
    sig, band = significant_classes(noise_ph['dgms'][1], noise_band=1.5 * max(tops))
    assert len(sig) == 0, (len(sig), band)


# ------------------------------------------------------------------ significant_classes
def test_significant_classes_on_constructed_diagrams():
    dgm = np.array([[0.0, 0.1], [0.0, 0.12], [0.1, 0.2], [0.05, 1.5], [0.0, 0.11]])
    sig, band = significant_classes(dgm)
    assert list(sig) == [3] and band == pytest.approx(0.12)
    # an explicit noise band overrides the gap heuristic
    sig, band = significant_classes(dgm, noise_band=0.105)
    assert set(sig) == {1, 3, 4} and band == 0.105
    sig, _ = significant_classes(dgm, noise_band=2.0)
    assert len(sig) == 0
    # min_lifetime_frac raises the band to a fraction of the longest bar
    sig, band = significant_classes(dgm, noise_band=0.0, min_lifetime_frac=0.5)
    assert list(sig) == [3] and band == pytest.approx(0.725)


def test_significant_classes_degenerate_inputs():
    sig, band = significant_classes(np.zeros((0, 2)))
    assert len(sig) == 0 and band == 0.0
    sig, band = significant_classes(np.array([[0.0, 1.0]]))            # single bar
    assert list(sig) == [0] and band == pytest.approx(0.5)
    sig, band = significant_classes(np.array([[0.0, 1.0], [0.0, 0.1]]))
    assert list(sig) == [0]


def test_significant_classes_h0_of_a_connected_cloud_is_one(circle_ph):
    # H0: the essential bar is the component; finite bars are judged by the 2x ratio-gap rule
    sig, _ = significant_classes(circle_ph['dgms'][0], essential_always=True)
    assert len(sig) == 1, len(sig)


# ------------------------------------------------------------------ normalized_persistence
def test_normalized_persistence_is_scale_invariant():
    c = make_circle(n=150, sigma=0.003, seed=31)
    vals = []
    for s in (1.0, 7.0, 0.01):
        r = persistence(s * c.X, do_cocycles=False)
        vals.append(normalized_persistence(r['dgms'][1], r['median_dist']))
    assert vals[0] > 0.5
    # (relative 1e-4: ripser works in float32, so the diagrams scale only to ~1e-6)
    np.testing.assert_allclose(vals, vals[0], rtol=1e-4)


def test_normalized_persistence_value_and_degenerate_cases():
    dgm = np.array([[0.0, 3.0], [1.0, 5.0], [0.0, np.inf]])
    assert normalized_persistence(dgm, 2.0) == pytest.approx((1.5 ** 2) + (2.0 ** 2))
    assert normalized_persistence(np.zeros((0, 2)), 1.0) == 0.0
    assert normalized_persistence(dgm, 0.0) == 0.0


def test_normalized_persistence_is_carried_by_one_bar_on_a_circle_and_by_none_on_noise(circle_ph, noise_ph):
    """NOTE: the total itself does NOT separate structure from noise -- measured 1.26 on the
    circle and 1.29 on the Gaussian blob, because hundreds of short noise bars add up in
    quadrature.  What differs is how the total is distributed, which is what is asserted."""
    c = normalized_persistence(circle_ph['dgms'][1], circle_ph['median_dist'])
    z = normalized_persistence(noise_ph['dgms'][1], noise_ph['median_dist'])
    top_c = (lifetimes(circle_ph['dgms'][1])[0] / circle_ph['median_dist']) ** 2
    top_z = (lifetimes(noise_ph['dgms'][1])[0] / noise_ph['median_dist']) ** 2
    assert c > 1.0, c                  # one bar of length ~1.6 over a median distance of ~1.41
    assert top_c / c > 0.95, top_c / c
    assert top_z / z < 0.1, top_z / z


# ------------------------------------------------------------------ circular_coordinates
def test_circular_coordinate_recovers_the_angle_of_a_circle(circle, circle_ph):
    h1 = circle_ph['dgms'][1]
    sig, _ = significant_classes(h1)
    cc = circular_coordinates(circle.X, h1, circle_ph['cocycles'][1], D=circle_ph['D'],
                              class_idx=int(sig[0]))
    th = cc['theta']
    assert th.shape == (circle.n,) and (th >= 0).all() and (th < TWO_PI + 1e-9).all()
    r = circular_correlation(th, circle.theta)
    assert r > 0.95, r
    assert circular_rank_correlation(th, circle.theta) > 0.99
    assert cc['birth'] <= cc['alpha'] < cc['death']
    assert cc['lifetime'] == pytest.approx(cc['death'] - cc['birth'])
    assert cc['n_edges'] > circle.n
    assert 0 <= cc['harmonic_residual'] <= 1.0 + 1e-9
    # the coordinate winds exactly once: sorted by true angle it is monotone up to one wrap
    step = np.mod(np.diff(th[np.argsort(circle.theta)]) + np.pi, TWO_PI) - np.pi
    assert abs(abs(step.sum()) - TWO_PI) < 0.5, step.sum()


def test_circular_coordinate_computes_its_own_distance_matrix():
    c = make_circle(n=150, sigma=0.003, seed=32)
    r = persistence(c.X)
    sig, _ = significant_classes(r['dgms'][1])
    cc = circular_coordinates(c.X, r['dgms'][1], r['cocycles'][1], class_idx=int(sig[0]))
    assert circular_correlation(cc['theta'], c.theta) > 0.95


def test_circular_coordinates_of_a_torus_are_an_integer_basis_of_its_angles(torus, torus_ph):
    """H^1(T^2; Z) = Z^2.  Each persistent class's coordinate must be k1*a + k2*b for small
    integers, and the two integer vectors must be linearly independent."""
    h1 = torus_ph['dgms'][1]
    sig, _ = significant_classes(h1)
    assert len(sig) == 2
    combos, thetas = [], []
    for ci in sig:
        cc = circular_coordinates(torus.X, h1, torus_ph['cocycles'][1], D=torus_ph['D'],
                                  class_idx=int(ci))
        best = max(((circular_correlation(cc['theta'], np.mod(k1 * torus.a + k2 * torus.b, TWO_PI)),
                     k1, k2) for k1, k2 in itertools.product(range(-2, 3), repeat=2)
                    if (k1, k2) != (0, 0)), key=lambda t: t[0])
        assert best[0] > 0.95, best
        combos.append(best[1:]); thetas.append(cc['theta'])
    det = combos[0][0] * combos[1][1] - combos[0][1] * combos[1][0]
    assert abs(det) == 1, combos
    kept, dropped = toroidal_decorrelate(thetas)
    assert kept == [0, 1] and dropped == []


def test_circular_coordinates_returns_none_without_edges():
    X = np.array([[0.0, 0], [1, 0], [0, 1], [5, 5]])
    dgm = np.array([[0.1, 0.2]])
    out = circular_coordinates(X, dgm, [np.array([[0, 1, 1]])], class_idx=0)
    assert out is None


# ------------------------------------------------------------------ circular statistics
def test_circular_correlation_invariances():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, TWO_PI, 2000)
    assert circular_correlation(a, a) == pytest.approx(1.0)
    assert circular_correlation(a, np.mod(a + 1.234, TWO_PI)) == pytest.approx(1.0)      # rotation
    assert circular_correlation(a, np.mod(-a + 0.4, TWO_PI)) == pytest.approx(1.0)       # reflection
    assert circular_correlation(a, np.mod(-a + 0.4, TWO_PI), allow_reflection=False) < 0.08
    b = rng.uniform(0, TWO_PI, 2000)
    assert circular_correlation(a, b) < 0.08                                            # unrelated
    # symmetric in its arguments
    assert circular_correlation(a, b) == pytest.approx(circular_correlation(b, a))
    # a doubled angle is a different coordinate
    assert circular_correlation(a, np.mod(2 * a, TWO_PI)) < 0.08


def test_circular_correlation_degrades_smoothly_with_noise():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, TWO_PI, 4000)
    sig = 0.5
    r = circular_correlation(a, a + sig * rng.standard_normal(4000))
    assert r == pytest.approx(np.exp(-sig ** 2 / 2), abs=0.03)      # |E exp(i N(0, s^2))|


def test_circular_rank_correlation_removes_a_monotone_warp():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, TWO_PI, 1500)
    warped = TWO_PI * (a / TWO_PI) ** 3                 # monotone, same cyclic order
    lin = circular_correlation(a, warped)
    rank = circular_rank_correlation(a, warped)
    assert lin < 0.8, lin
    assert rank > 0.999, rank
    assert circular_rank_correlation(a, rng.uniform(0, TWO_PI, 1500)) < 0.1
    assert circular_rank_correlation(a, np.mod(-a + 2.0, TWO_PI)) > 0.999


def test_circular_linear_correlation():
    rng = np.random.default_rng(3)
    th = rng.uniform(0, TWO_PI, 2000)
    assert circular_linear_correlation(th, np.cos(th)) == pytest.approx(1.0, abs=1e-6)
    assert circular_linear_correlation(th, 3 * np.sin(th + 0.7) - 2) == pytest.approx(1.0, abs=1e-6)
    assert circular_linear_correlation(th, rng.standard_normal(2000)) < 0.08
    assert circular_linear_correlation(th, np.ones(2000)) == 0.0
    assert circular_linear_correlation(th, np.cos(2 * th)) < 0.08


# ------------------------------------------------------------------ name_coordinate
def test_name_coordinate_finds_the_frequency_of_a_modular_variable():
    rng = np.random.default_rng(4)
    p = 36
    a = rng.integers(0, p, 600)
    b = rng.integers(0, p, 600)
    theta = np.mod(TWO_PI * 3 * a / p + 0.7 + 0.05 * rng.standard_normal(600), TWO_PI)
    out = name_coordinate(theta, {'a': a, 'b': b}, modular_period=p)
    assert out['named'] is not None
    assert out['named']['name'] == 'a' and out['named']['freq'] == 3 and out['named']['kind'] == 'modular'
    assert out['named']['r'] > 0.99
    assert out['best'] is out['all'][0]
    other = [d for d in out['all'] if d['name'] == 'b'][0]
    assert other['r'] < 0.3


def test_name_coordinate_linear_variable_and_threshold():
    rng = np.random.default_rng(5)
    theta = rng.uniform(0, TWO_PI, 800)
    out = name_coordinate(theta, {'x': np.cos(theta - 0.3), 'junk': rng.standard_normal(800)})
    assert out['named']['name'] == 'x' and out['named']['kind'] == 'linear' and out['named']['freq'] is None
    nothing = name_coordinate(theta, {'junk': rng.standard_normal(800)})
    assert nothing['named'] is None and nothing['best']['name'] == 'junk'
    strict = name_coordinate(theta, {'x': np.cos(theta) + 0.8 * rng.standard_normal(800)}, threshold=0.95)
    assert strict['named'] is None and strict['threshold'] == 0.95
    empty = name_coordinate(theta, {})
    assert empty['best'] is None and empty['named'] is None and empty['all'] == []


# ------------------------------------------------------------------ toroidal_decorrelate
def test_toroidal_decorrelate_drops_integer_combinations():
    rng = np.random.default_rng(6)
    a = rng.uniform(0, TWO_PI, 1000)
    b = rng.uniform(0, TWO_PI, 1000)
    thetas = [a, b, np.mod(a + b, TWO_PI), np.mod(2 * a, TWO_PI), np.mod(a - 2 * b + 1.0, TWO_PI)]
    kept, dropped = toroidal_decorrelate(thetas)
    assert kept == [0, 1]
    by_idx = {d['idx']: d for d in dropped}
    assert set(by_idx) == {2, 3, 4}
    assert by_idx[2]['by'] == [0, 1] and by_idx[2]['coeffs'][0] == by_idx[2]['coeffs'][1]
    assert abs(by_idx[2]['coeffs'][0]) == 1
    assert by_idx[3]['by'] == [0] and abs(by_idx[3]['coeffs'][0]) == 2
    c = by_idx[4]['coeffs']
    assert by_idx[4]['by'] == [0, 1] and c[0] * c[1] < 0 and sorted(map(abs, c)) == [1, 2]


def test_toroidal_decorrelate_keeps_independent_and_handles_empty():
    rng = np.random.default_rng(7)
    ths = [rng.uniform(0, TWO_PI, 1000) for _ in range(3)]
    kept, dropped = toroidal_decorrelate(ths)
    assert kept == [0, 1, 2] and dropped == []
    assert toroidal_decorrelate([]) == ([], [])


# ------------------------------------------------------------------ fiber_topology
@pytest.fixture(scope='module')
def two_circle_fibers(two_circles):
    labels = two_circles.which.copy()
    labels[:20] = 7                       # a fiber too small to analyse
    return fiber_topology(two_circles.X, labels, min_points=40), labels


def test_fiber_topology_each_circle_fiber_has_one_loop(two_circle_fibers):
    fib, labels = two_circle_fibers
    assert set(fib) == {0, 1, 7}
    assert fib[7] == dict(n=20, skipped=True)
    for y in (0, 1):
        f = fib[y]
        assert f['n'] == (labels == y).sum()
        assert (labels[f['idx']] == y).all()
        assert f['beta1'] == 1, f['beta1']
        h1 = f['res']['dgms'][1]
        assert f['tildeC'] == pytest.approx(normalized_persistence(h1, f['res']['median_dist']))
        assert f['tildeC'] >= (lifetimes(h1)[0] / f['res']['median_dist']) ** 2 > 0.5
        assert 'cocycles' in f['res']


def test_fiber_topology_beta0_of_a_connected_fiber_is_one(two_circle_fibers):
    fib, _ = two_circle_fibers
    got = {y: fib[y]['beta0'] for y in (0, 1)}
    assert got == {0: 1, 1: 1}, got


def test_fiber_topology_with_a_noise_band_kills_arc_loops(circle, circle_readout):
    """A contiguous arc is contractible: with the matched null band it has no H1."""
    labels = circle_readout.true_arc
    band = null_band(circle.X)
    fib = fiber_topology(circle.X, labels, noise_band=band, min_points=40)
    assert set(fib) == {0, 1, 2, 3}
    assert [fib[y]['beta1'] for y in range(4)] == [0, 0, 0, 0]
    # (an empty H1 diagram reports band 0.0)
    assert all(fib[y]['band1'] in (band, 0.0) for y in range(4))


# ------------------------------------------------------------------ match_classes
def test_match_classes_identity_and_new_class():
    a = np.array([[0.1, 1.5], [0.2, 0.9]])
    pairs, un_a, un_b = match_classes(a, a.copy())
    assert [(i, j) for i, j, _ in pairs] == [(0, 0), (1, 1)] and un_a == [] and un_b == []
    assert all(c == 0 for _, _, c in pairs)
    b = np.array([[0.2, 0.92], [0.1, 1.48], [0.3, 2.4]])          # permuted, jittered, one new
    pairs, un_a, un_b = match_classes(a, b)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (1, 0)]
    assert un_a == [] and un_b == [2]


def test_match_classes_empty_and_vanished():
    a = np.array([[0.1, 1.5], [0.2, 0.9]])
    assert match_classes(np.zeros((0, 2)), a) == ([], [], [0, 1])
    assert match_classes(a, np.zeros((0, 2))) == ([], [0, 1], [])
    far = np.array([[5.0, 9.0]])                       # nothing comparable: no match
    pairs, un_a, un_b = match_classes(a, far)
    assert pairs == [] and un_a == [0, 1] and un_b == [0]


def test_match_classes_coordinates_break_ties():
    rng = np.random.default_rng(8)
    t1, t2 = rng.uniform(0, TWO_PI, 500), rng.uniform(0, TWO_PI, 500)
    a = np.array([[0.1, 1.0], [0.1, 1.0]])
    b = np.array([[0.12, 1.02], [0.12, 1.02]])          # diagram distance cannot tell them apart
    pairs, _, _ = match_classes(a, b, [t1, t2], [np.mod(t2 + 0.3, TWO_PI), np.mod(-t1, TWO_PI)])
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (1, 0)]
    pairs, _, _ = match_classes(a, b, [t1, None], [None, t2])      # None coordinates tolerated
    assert len(pairs) == 2


# ------------------------------------------------------------------ equivariant
@pytest.fixture(scope='module')
def z36():
    """A cloud carrying the characters k=2 and k=5 of Z_36 in two NON-orthogonal planes."""
    rng = np.random.default_rng(12)
    p, reps, D = 36, 12, 12
    t = np.repeat(np.arange(p), reps)
    B = rng.standard_normal((D, 4))
    B[:, 2] += 0.6 * B[:, 0]                           # make the two planes overlap
    B /= np.linalg.norm(B, axis=0)
    def cloud(noise_rng):
        P = np.stack([np.cos(TWO_PI * 2 * t / p), np.sin(TWO_PI * 2 * t / p),
                      0.6 * np.cos(TWO_PI * 5 * t / p), 0.6 * np.sin(TWO_PI * 5 * t / p)], 1)
        return P @ B.T + 0.02 * noise_rng.standard_normal((len(t), D)) + 3.0
    return dict(X=cloud(rng), X_eval=cloud(np.random.default_rng(13)), t=t, p=p, D=D)


def _angle(Y):
    return np.arctan2(Y[:, 1], Y[:, 0])


def test_induced_shift_operator_advances_the_group_coordinate(z36):
    T, basis, M, idx, r = induced_shift_operator(z36['X'], z36['t'], z36['p'])
    D, p = z36['D'], z36['p']
    assert T.shape == (D, D) and M.shape == (p, D) and basis.shape == (r, D)
    assert r == 4, r                                    # two planes
    assert list(idx) == list(range(p))
    np.testing.assert_allclose(M.mean(0), 0, atol=1e-9)
    pred = M @ T.T
    err = np.linalg.norm(pred - np.roll(M, -1, axis=0)) / np.linalg.norm(M)
    assert err < 0.05, err
    w = np.linalg.eigvals(basis @ T @ basis.T)
    freqs = sorted(np.round(np.abs(np.angle(w)) / TWO_PI * p).astype(int))
    assert freqs == [2, 2, 5, 5], freqs
    np.testing.assert_allclose(np.abs(w), 1.0, atol=0.03)      # a group action is unitary


def test_isotypic_planes_recover_frequencies_and_circles(z36):
    planes = isotypic_planes(z36['X'], z36['t'], z36['p'])
    assert sorted(d['freq'] for d in planes) == [2, 5]
    assert planes[0]['freq'] == 2 and planes[0]['var'] > planes[1]['var']       # sorted by variance
    for d in planes:
        assert d['plane'].shape == (z36['D'], 2) and d['proj'].shape == (len(z36['t']), 2)
        np.testing.assert_allclose(d['plane'].T @ d['plane'], np.eye(2), atol=1e-8)
        assert d['modulus'] == pytest.approx(1.0, abs=0.03)
        assert d['signed_freq'] in (d['freq'], z36['p'] - d['freq'])
        r = circular_correlation(_angle(d['proj']), TWO_PI * d['freq'] * z36['t'] / z36['p'])
        assert r > 0.99, (d['freq'], r)


@pytest.mark.parametrize('detect', ['action', 'dft'])
def test_isotypic_decomposition_in_sample(z36, detect):
    comps, freqs = isotypic_decomposition(z36['X'], z36['t'], z36['p'], n_freq=2, detect=detect)
    assert sorted(freqs) == [2, 5]
    assert [c['freq'] for c in comps] == [2, 5]          # by variance
    for c in comps:
        assert c['recon_r2'] > 0.99, c['recon_r2']
        r = circular_correlation(_angle(c['proj']), TWO_PI * c['freq'] * z36['t'] / z36['p'])
        assert r > 0.99, (c['freq'], r)
        # the dual coordinates are clean circles: radius ~1 for every point
        rad = np.linalg.norm(c['proj'], axis=1)
        assert np.std(rad) / np.mean(rad) < 0.1


def test_isotypic_decomposition_out_of_sample_transfers_and_the_null_does_not(z36):
    comps, _ = isotypic_decomposition(z36['X'], z36['t'], z36['p'], n_freq=2, detect='dft',
                                      X_eval=z36['X_eval'])
    for c in comps:
        r = circular_correlation(_angle(c['proj']), TWO_PI * c['freq'] * z36['t'] / z36['p'])
        assert r > 0.99, (c['freq'], r)
    # feature-shuffled cloud: planes fitted on one shuffle say nothing about another
    Xs, Xs_eval = feature_shuffle(z36['X'], seed=1), feature_shuffle(z36['X_eval'], seed=2)
    comps, _ = isotypic_decomposition(Xs, z36['t'], z36['p'], n_freq=2, detect='dft',
                                      X_eval=Xs_eval, var_floor=0.0)
    for c in comps:
        r = circular_correlation(_angle(c['proj']), TWO_PI * c['freq'] * z36['t'] / z36['p'])
        assert r < 0.3, (c['freq'], r)
