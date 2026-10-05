"""whiteboxtopo.geometry: curvature, Lipschitz field, reach / ignorance / tubular distance,
confidence saddles, manifold margin."""
import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from conftest import make_circle, make_sphere, arc_classifier, offplane_direction
from whiteboxtopo.manifold import knn_graph, tangent_pca, fit_manifold
from whiteboxtopo.geometry import (second_fundamental_form, lipschitz_field, reach, ignorance,
                                   failure_risk, confidence_saddles, manifold_margin,
                                   tubular_distance)


# ------------------------------------------------------------------ second_fundamental_form
@pytest.mark.parametrize('R', [0.5, 1.0, 3.0])
def test_curvature_of_a_circle_is_one_over_2R(R):
    """Docstring: Frobenius norm of the fitted quadratic = sqrt(d) / (2R); d = 1 for a circle."""
    c = make_circle(n=600, R=R, sigma=0.0, seed=40)
    kappa = second_fundamental_form(c.X, k=20, dim_cap=1)
    assert kappa.shape == (c.n,)
    assert np.median(kappa) == pytest.approx(1 / (2 * R), rel=0.05), np.median(kappa)
    assert np.std(kappa) / np.mean(kappa) < 0.1          # a circle bends the same everywhere


def test_curvature_scales_exactly_like_one_over_R():
    """The estimator is scale-equivariant: curv(R * X) = curv(X) / R, point by point."""
    c = make_circle(n=500, R=1.0, sigma=0.0, seed=41)
    k1 = second_fundamental_form(c.X, k=20, dim_cap=1)
    k4 = second_fundamental_form(4.0 * c.X, k=20, dim_cap=1)
    np.testing.assert_allclose(k4, k1 / 4.0, rtol=1e-6)


@pytest.mark.parametrize('R', [1.0, 2.5])
def test_curvature_of_a_sphere_is_sqrt2_over_2R(R):
    s = make_sphere(n=1200, R=R, seed=42)
    kappa = second_fundamental_form(s.X, k=30, dim_cap=2)
    assert np.median(kappa) == pytest.approx(np.sqrt(2) / (2 * R), rel=0.05), np.median(kappa)


def test_curvature_of_a_plane_is_zero(plane):
    kappa = second_fundamental_form(plane.X, k=30, dim_cap=2)
    assert kappa.max() < 1e-8, kappa.max()


def test_curvature_sees_where_an_ellipse_bends():
    """Ellipse a=2, b=1: kappa = ab / (a^2 sin^2 t + b^2 cos^2 t)^(3/2), from 1/4 to 2."""
    n, a, b = 1000, 2.0, 1.0
    t = np.sort(np.random.default_rng(43).uniform(0, 2 * np.pi, n))
    X = np.stack([a * np.cos(t), b * np.sin(t), np.zeros(n)], 1)
    true = a * b / (a ** 2 * np.sin(t) ** 2 + b ** 2 * np.cos(t) ** 2) ** 1.5
    est = second_fundamental_form(X, k=16, dim_cap=1)
    from scipy.stats import spearmanr
    assert spearmanr(est, true).statistic > 0.95
    assert np.median(est / (true / 2)) == pytest.approx(1.0, abs=0.1)


def test_curvature_returns_zeros_when_the_fit_is_underdetermined():
    X = np.random.default_rng(44).standard_normal((50, 6))
    out = second_fundamental_form(X, k=8, dim_cap=4)       # 10 monomials > 8 neighbours
    assert out.shape == (50,) and (out == 0).all()


def test_curvature_default_dim_cap_absorbs_the_normal_of_a_low_dimensional_manifold():
    """Documents a consequence of the default dim_cap=4: on a noise-free circle (d=1, spanning 2
    ambient directions locally) the 'tangent' fit swallows the normal direction and the
    curvature reads 0 regardless of R.  dim_cap must match the intrinsic dimension."""
    c = make_circle(n=600, R=0.5, sigma=0.0, seed=40)
    assert np.median(second_fundamental_form(c.X, k=30)) < 1e-6
    assert np.median(second_fundamental_form(c.X, k=30, dim_cap=1)) > 0.9


# ------------------------------------------------------------------ lipschitz_field / failure_risk
def test_lipschitz_field_is_the_spectral_norm():
    rng = np.random.default_rng(45)
    n, o, D = 60, 4, 9
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    s = rng.uniform(0.1, 5.0, n)
    J = s[:, None, None] * Q[:o][None]               # orthonormal rows scaled by s_i
    np.testing.assert_allclose(lipschitz_field(J), s, rtol=1e-10)
    J2 = rng.standard_normal((n, o, D))
    expect = np.array([np.linalg.svd(j, compute_uv=False)[0] for j in J2])
    np.testing.assert_allclose(lipschitz_field(J2), expect, rtol=1e-10)
    # it bounds the network's response to any unit step
    v = rng.standard_normal((n, D)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    assert (np.linalg.norm(np.einsum('nod,nd->no', J2, v), axis=1) <= lipschitz_field(J2) + 1e-12).all()


def test_failure_risk_is_the_product():
    curv, lip = np.array([0.0, 1.0, 2.0]), [3.0, 0.5, 2.0]
    np.testing.assert_array_equal(failure_risk(curv, lip), [0.0, 0.5, 4.0])


# ------------------------------------------------------------------ reach / ignorance / tubular
@pytest.fixture(scope='module')
def tube(circle):
    """Probe circle with 1-D tangents, a head that reads the circle's plane plus one off-plane
    direction `v`, and a second off-plane direction `blind` that the head ignores."""
    v = offplane_direction(circle, seed=77)
    blind = offplane_direction(circle, seed=78)
    blind -= v * (v @ blind); blind /= np.linalg.norm(blind)
    ro = arc_classifier(circle, extra_rows=v)
    idx, _, _ = knn_graph(circle.X, k=15)
    bases, _ = tangent_pca(circle.X, idx, dim_cap=1)
    rng = np.random.default_rng(99)
    th = rng.uniform(0, 2 * np.pi, 300)                # held-out: fresh angles, fresh noise
    H = np.stack([np.cos(th), np.sin(th)], 1) @ circle.E.T + circle.sigma * rng.standard_normal((300, circle.D))
    radial = np.stack([np.cos(th), np.sin(th)], 1) @ circle.E.T
    return dict(X=circle.X, bases=bases, J=ro.J, W=ro.W, holdout=H, v=v, blind=blind,
                radial=radial, sigma=circle.sigma)


def test_reach_is_the_percentile_of_holdout_distances(tube):
    eps, dg, nn = reach(tube['X'], tube['bases'], tube['J'], tube['holdout'])
    assert dg.shape == (300,) and (dg >= 0).all()
    assert eps == pytest.approx(np.percentile(dg, 99))
    assert isinstance(nn, NearestNeighbors)
    eps50, _, _ = reach(tube['X'], tube['bases'], tube['J'], tube['holdout'], percentile=50)
    assert eps50 < eps
    # the holdout is on-manifold: d_g is of the order of (noise * |J|), here ~ 0.003 * 5 * few
    assert eps < 0.2, eps


def test_ignorance_small_on_manifold_large_off_manifold(tube):
    eps, _, nn = reach(tube['X'], tube['bases'], tube['J'], tube['holdout'])
    ig_in, dg_in = ignorance(tube['holdout'], tube['X'], tube['bases'], tube['J'], eps, nn)
    assert (ig_in <= 1).mean() >= 0.98                 # by construction of the 99th percentile
    np.testing.assert_allclose(ig_in, dg_in / eps)
    for name, disp in (('radial', 0.5 * tube['radial']), ('offplane', 0.5 * tube['v'][None])):
        ig_out, dg_out = ignorance(tube['holdout'] + disp, tube['X'], tube['bases'], tube['J'], eps)
        assert ig_out.min() > 5, (name, ig_out.min())
        # analytic: a displacement delta normal to the manifold is seen as |J delta|
        expect = np.linalg.norm(np.atleast_2d(disp) @ tube['W'].T, axis=1)
        assert np.median(dg_out / expect) == pytest.approx(1.0, abs=0.1), name


def test_ignorance_grows_linearly_with_the_displacement(tube):
    eps, _, nn = reach(tube['X'], tube['bases'], tube['J'], tube['holdout'])
    med = [np.median(ignorance(tube['X'] + s * tube['v'][None], tube['X'], tube['bases'],
                               tube['J'], eps, nn)[1]) for s in (0.25, 0.5, 1.0)]
    assert med[1] / med[0] == pytest.approx(2.0, rel=0.05)
    assert med[2] / med[1] == pytest.approx(2.0, rel=0.05)


def test_pullback_ignorance_is_blind_where_the_head_is_but_tubular_distance_is_not(tube):
    """The motivating claim in tubular_distance's docstring, on an analytic head."""
    eps, _, nn = reach(tube['X'], tube['bases'], tube['J'], tube['holdout'])
    off = tube['holdout'] + 0.5 * tube['blind'][None]
    ig_blind, _ = ignorance(off, tube['X'], tube['bases'], tube['J'], eps, nn)
    assert np.median(ig_blind) < 2.0, np.median(ig_blind)          # d_g calls it 'known'
    t_in = tubular_distance(tube['holdout'], tube['X'], tube['bases'])
    t_off = tubular_distance(off, tube['X'], tube['bases'])
    assert np.median(t_off) > 20 * np.median(t_in), (np.median(t_off), np.median(t_in))


def test_tubular_distance_units_and_separation(tube):
    t_in = tubular_distance(tube['holdout'], tube['X'], tube['bases'])
    assert t_in.shape == (300,) and (t_in >= 0).all()
    # in units of the local normal thickness: on-manifold points sit at O(1)
    assert 0.3 < np.median(t_in) < 3.0, np.median(t_in)
    for disp in (0.5 * tube['radial'], 0.5 * tube['v'][None]):
        t_out = tubular_distance(tube['holdout'] + disp, tube['X'], tube['bases'])
        assert t_out.min() > 5 * np.percentile(t_in, 99), (t_out.min(), np.percentile(t_in, 99))
    # probe points themselves are at distance exactly 0
    assert tubular_distance(tube['X'][:50], tube['X'], tube['bases']).max() < 1e-9
    # displacement ALONG the manifold's tangent does not count as off-manifold
    idx = np.arange(50)
    tang = tube['bases'][idx, :, 0]
    t_tan = tubular_distance(tube['X'][idx] + 0.004 * tang, tube['X'], tube['bases'])
    assert np.median(t_tan) < 0.5 * np.median(t_in), (np.median(t_tan), np.median(t_in))


def test_tubular_distance_accepts_a_prefitted_index(tube):
    nn = NearestNeighbors(n_neighbors=11).fit(tube['X'])
    a = tubular_distance(tube['holdout'], tube['X'], tube['bases'], nn=nn)
    b = tubular_distance(tube['holdout'], tube['X'], tube['bases'])
    np.testing.assert_allclose(a, b)


# ------------------------------------------------------------------ confidence_saddles
@pytest.fixture(scope='module')
def grid():
    g = np.arange(-8, 9)
    P = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2).astype(float)
    _, _, A = knn_graph(P, k=8)
    interior = (np.abs(P) < 7).all(1)
    return P, A, interior


def test_confidence_saddles_finds_the_saddle_of_a_monkey_free_saddle(grid):
    P, A, interior = grid
    f = P[:, 0] ** 2 - P[:, 1] ** 2
    saddles, ncomp = confidence_saddles(A, f)
    origin = int(np.where((P == 0).all(1))[0][0])
    assert origin in saddles
    assert ncomp[origin] == 2
    assert ncomp.shape == (len(P),)
    assert set(saddles[interior[saddles]]) == {origin}, P[saddles[interior[saddles]]]


def test_confidence_saddles_none_on_monotone_or_single_peak_functions(grid):
    P, A, interior = grid
    for f in (P[:, 0] + 0.37 * P[:, 1], -(P ** 2).sum(1) + 0.01 * P[:, 0]):
        saddles, _ = confidence_saddles(A, f)
        assert interior[saddles].sum() == 0, P[saddles[interior[saddles]]]


def test_confidence_saddles_between_two_peaks(grid):
    P, A, interior = grid
    f = np.exp(-((P[:, 0] - 5) ** 2 + P[:, 1] ** 2) / 18) + np.exp(-((P[:, 0] + 5) ** 2 + P[:, 1] ** 2) / 18)
    saddles, _ = confidence_saddles(A, f)
    inner = saddles[interior[saddles]]
    assert len(inner) >= 1
    assert np.abs(P[inner, 0]).max() <= 1, P[inner]       # on the ridge x = 0 between the peaks
    assert np.linalg.norm(P[inner], axis=1).min() == 0    # and the pass itself is found


# ------------------------------------------------------------------ manifold_margin
def test_manifold_margin_is_the_arc_distance_to_the_decision_boundary(even_circle):
    c = even_circle
    f = fit_manifold(c.X)
    pred = np.floor(c.theta / (np.pi / 2)).astype(int) % 4
    margin = manifold_margin(f.W, pred)
    to_boundary = np.minimum(np.mod(c.theta, np.pi / 2), np.pi / 2 - np.mod(c.theta, np.pi / 2))
    spacing = 2 * np.pi / c.n
    # nearest differently-labelled POINT is between 0 and one spacing past the boundary
    err = margin - to_boundary
    assert err.min() > -0.02, err.min()
    assert err.max() < spacing + 0.02 + 0.02 * to_boundary.max(), err.max()
    assert np.corrcoef(margin, to_boundary)[0, 1] > 0.999


def test_manifold_margin_single_class_and_disconnected(two_circles):
    f = fit_manifold(two_circles.X)
    assert np.isinf(manifold_margin(f.W, np.zeros(two_circles.n, int))).all()
    # each circle its own class: the other class is unreachable along the manifold
    assert np.isinf(manifold_margin(f.W, two_circles.which)).all()


def test_manifold_margin_under_the_pullback_metric_scales_with_the_jacobian(even_circle):
    c = even_circle
    pred = (c.theta > np.pi).astype(int)
    J = np.broadcast_to(3.0 * np.eye(c.D), (c.n, c.D, c.D))
    m1 = manifold_margin(fit_manifold(c.X).W, pred)
    m3 = manifold_margin(fit_manifold(c.X, J=J).W, pred)
    np.testing.assert_allclose(m3, 3.0 * m1, rtol=1e-6)
