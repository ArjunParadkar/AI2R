"""whiteboxtopo.manifold: graph, tangents, dimension, manifold score, metric, geodesics."""
import numpy as np
import pytest
from scipy.sparse import issparse

from conftest import make_circle, circ_dist
from whiteboxtopo.manifold import (knn_graph, tangent_pca, intrinsic_dim, manifold_score,
                                   pullback_metric_lengths, secant_metric_lengths, geodesic,
                                   geodesic_distances, ManifoldFit, fit_manifold)


# ------------------------------------------------------------------ knn_graph
def test_knn_graph_shapes_symmetry_and_exactness(circle):
    k = 15
    idx, dist, A = knn_graph(circle.X, k=k)
    n = circle.n
    assert idx.shape == (n, k) and dist.shape == (n, k)
    assert issparse(A) and A.shape == (n, n)
    assert not (idx == np.arange(n)[:, None]).any(), 'a point is listed as its own neighbour'
    assert (np.diff(dist, axis=1) >= 0).all(), 'neighbour distances must be ascending'
    assert abs(A - A.T).max() == 0, 'adjacency must be symmetric'
    assert A.nnz >= n * k
    # against brute force
    for i in (0, 57, 399):
        d = np.linalg.norm(circle.X - circle.X[i], axis=1)
        d[i] = np.inf
        brute = np.argsort(d)[:k]
        assert set(brute) == set(idx[i])
        np.testing.assert_allclose(np.sort(d)[:k], dist[i], rtol=1e-10)
    # edge weights are the Euclidean lengths
    i, j = 3, idx[3, 0]
    assert A[i, j] == pytest.approx(np.linalg.norm(circle.X[i] - circle.X[j]))


def test_knn_graph_k_is_clipped_to_n_minus_1():
    X = np.random.default_rng(0).standard_normal((5, 3))
    idx, dist, A = knn_graph(X, k=15)
    assert idx.shape == (5, 4)
    assert A.shape == (5, 5)


def test_knn_graph_neighbours_on_a_circle_are_angular_neighbours(circle):
    idx, _, _ = knn_graph(circle.X, k=15)
    ang = circ_dist(circle.theta[idx], circle.theta[:, None])
    spacing = 2 * np.pi / circle.n
    # 15 neighbours, ~7.5 each side; a random sample has gaps, allow 3x
    assert ang.max() < 3 * 7.5 * spacing, ang.max()


def test_knn_graph_metric_g_and_secant_need_their_inputs(circle):
    with pytest.raises(ValueError):
        knn_graph(circle.X, metric='g')
    with pytest.raises(ValueError):
        knn_graph(circle.X, metric='secant')


def test_knn_graph_metric_g_ignores_directions_the_jacobian_kills():
    """A circle in (x, y) with a large nuisance z the network does not read.  Euclidean
    neighbours are contaminated by z; pullback neighbours are the exact angular neighbours."""
    rng = np.random.default_rng(5)
    n, k = 400, 10
    th = np.sort(rng.uniform(0, 2 * np.pi, n))
    X = np.stack([np.cos(th), np.sin(th), 0.05 * rng.standard_normal(n)], 1)
    P = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    J = np.broadcast_to(P, (n, 2, 3)).copy()
    idx_g, dist_g, _ = knn_graph(X, k=k, J=J, metric='g')
    idx_true, dist_true, _ = knn_graph(X[:, :2], k=k)
    idx_e, _, _ = knn_graph(X, k=k)
    same = np.mean([set(a) == set(b) for a, b in zip(idx_g, idx_true)])
    assert same == 1.0, same
    np.testing.assert_allclose(dist_g, dist_true, atol=1e-12)
    ang_g = circ_dist(th[idx_g], th[:, None]).mean()
    ang_e = circ_dist(th[idx_e], th[:, None]).mean()
    assert ang_g < ang_e, (ang_g, ang_e)
    # ... and the Euclidean graph really was contaminated, so the re-ranking did something
    same_e = np.mean([set(a) == set(b) for a, b in zip(idx_e, idx_true)])
    assert same_e < 0.5, same_e


def test_knn_graph_secant_equals_g_for_a_linear_readout(circle, circle_readout):
    """psi linear => ||psi(h_j) - psi(h_i)|| == ||J (h_j - h_i)|| exactly."""
    out = circle_readout.logits
    ig, dg, Ag = knn_graph(circle.X, k=12, J=circle_readout.J, metric='g')
    isec, dsec, Asec = knn_graph(circle.X, k=12, out=out, metric='secant')
    np.testing.assert_allclose(dg, dsec, rtol=1e-8, atol=1e-10)
    assert (ig == isec).mean() > 0.999
    assert abs(Ag - Asec).max() < 1e-8


# ------------------------------------------------------------------ tangent_pca
def test_tangent_pca_shapes_and_orthonormality(circle):
    idx, _, _ = knn_graph(circle.X, k=15)
    bases, spectra = tangent_pca(circle.X, idx, dim_cap=3)
    assert bases.shape == (circle.n, circle.D, 3)
    assert spectra.shape == (circle.n, min(15, circle.D))
    G = np.einsum('nDa,nDb->nab', bases, bases)
    np.testing.assert_allclose(G, np.broadcast_to(np.eye(3), G.shape), atol=1e-5)
    assert (np.diff(spectra, axis=1) <= 1e-9).all(), 'spectra must be descending'
    assert (spectra >= 0).all()
    # dim_cap=None -> capped by min(k, D)
    b2, _ = tangent_pca(circle.X, idx)
    assert b2.shape == (circle.n, circle.D, min(15, circle.D))


def test_tangent_pca_recovers_the_true_tangent_of_a_circle(circle):
    idx, _, _ = knn_graph(circle.X, k=15)
    bases, spectra = tangent_pca(circle.X, idx, dim_cap=1)
    true_t = np.stack([-np.sin(circle.theta), np.cos(circle.theta)], 1) @ circle.E.T
    cos = np.abs(np.einsum('nD,nD->n', bases[:, :, 0], true_t))
    assert np.median(cos) > 0.995, np.median(cos)
    assert cos.min() > 0.97, cos.min()
    # one dominant direction: lambda_1 / lambda_2 large everywhere
    assert np.median(spectra[:, 0] / spectra[:, 1]) > 20


def test_tangent_pca_recovers_the_tangent_plane_of_a_torus(torus):
    idx, _, _ = knn_graph(torus.X, k=15)
    bases, _ = tangent_pca(torus.X, idx, dim_cap=2)
    ta = np.stack([-np.sin(torus.a), np.cos(torus.a), 0 * torus.a, 0 * torus.a], 1) @ torus.E.T
    tb = np.stack([0 * torus.b, 0 * torus.b, -np.sin(torus.b), np.cos(torus.b)], 1) @ torus.E.T
    for t in (ta, tb):       # each true tangent vector must lie in the estimated plane
        captured = np.linalg.norm(np.einsum('nDd,nD->nd', bases, t), axis=1)
        assert np.median(captured) > 0.98, np.median(captured)


# ------------------------------------------------------------------ intrinsic_dim
def test_intrinsic_dim_circle_is_one(circle):
    d_local, info = intrinsic_dim(circle.X, k=15)
    assert d_local.shape == (circle.n,)
    assert info['ranks'] == (7, 15)
    assert abs(info['d_global'] - 1.0) < 0.15, info
    assert abs(np.nanmedian(d_local) - 1.0) < 0.2, np.nanmedian(d_local)


def test_intrinsic_dim_torus_is_two(torus):
    d_local, info = intrinsic_dim(torus.X, k=15)
    assert abs(info['d_global'] - 2.0) < 0.4, info
    assert abs(np.nanmedian(d_local) - 2.0) < 0.4, np.nanmedian(d_local)


def test_intrinsic_dim_sphere_is_two(sphere):
    _, info = intrinsic_dim(sphere.X, k=15)
    assert abs(info['d_global'] - 2.0) < 0.3, info


def test_intrinsic_dim_noise_is_high(noise):
    _, info = intrinsic_dim(noise.X, k=15)
    assert info['d_global'] > 6.0, info          # isotropic Gaussian in R^10


def test_intrinsic_dim_orders_the_manifolds(circle, torus, noise):
    d = [intrinsic_dim(o.X)[1]['d_global'] for o in (circle, torus, noise)]
    assert d[0] < d[1] < d[2], d


def test_intrinsic_dim_custom_ranks_and_given_idx(circle):
    idx, _, _ = knn_graph(circle.X, k=15)
    d1, i1 = intrinsic_dim(circle.X, idx=idx, k=15)
    d2, i2 = intrinsic_dim(circle.X, k=15)
    np.testing.assert_allclose(d1, d2, equal_nan=True)
    _, fine = intrinsic_dim(circle.X, k=15, ranks=(1, 2))
    assert fine['ranks'] == (1, 2)
    assert fine['d_global'] == pytest.approx(i2['d_global_fine_scale'], rel=1e-9)


def test_intrinsic_dim_noise_free_circle_both_scales_agree():
    c = make_circle(n=500, sigma=0.0, seed=21)
    _, info = intrinsic_dim(c.X)
    assert abs(info['d_global'] - 1) < 0.12, info
    assert abs(info['d_global_fine_scale'] - 1) < 0.12, info


def test_intrinsic_dim_fine_scale_sees_the_noise_ball(circle):
    """The reason the estimator is evaluated at the graph scale: at ranks (1,2) a noisy circle
    reports the noise dimension."""
    _, info = intrinsic_dim(circle.X)
    assert info['d_global_fine_scale'] > 2 * info['d_global'], info


@pytest.mark.xfail(strict=False, reason=(
    "intrinsic_dim docstring (manifold/core.py:94-95) states 'a 1-D circle in R^10 with "
    "sigma=0.01 noise and spacing 0.008 returns d ~ 7.5 at ranks (1,2) and d ~ 1 at the graph "
    "scale'.  Measured with exactly those parameters (n=785) at the default k=15, ranks (7,15): "
    "fine scale 6.90 (consistent) but graph scale d_global = 2.28, not ~1 (fit_manifold then "
    "scores this circle m = 0.18).  The claim only holds at a wider graph: k=30 -> 1.30, "
    "k=60 -> 1.07.  The noise displacement in R^10 (~0.045) exceeds r_7 (~0.03), so the default "
    "ranks are still inside the noise ball."))
def test_intrinsic_dim_docstring_claim_noisy_circle_graph_scale():
    n = int(round(2 * np.pi / 0.008))
    c = make_circle(n=n, D=10, sigma=0.01, seed=0)
    _, info = intrinsic_dim(c.X)
    assert info['d_global_fine_scale'] > 5, info
    assert abs(info['d_global'] - 1.0) < 0.3, info


# ------------------------------------------------------------------ manifold_score
def test_manifold_score_on_constructed_spectra():
    clean = np.tile([1.0, 0.01, 0.008, 0.006, 0.005], (50, 1))
    s = manifold_score(clean, np.ones(50), D=10)
    assert s['m'] == 1.0 and (s['pca_dim'] == 1).all() and s['D'] == 10
    flat = np.tile([1.0, 0.9, 0.8, 0.7, 0.6], (50, 1))
    assert manifold_score(flat, np.ones(50), D=10)['m'] == 0.0
    # mixture: m is the fraction
    mix = np.concatenate([clean[:30], flat[:10]])
    assert manifold_score(mix, np.ones(40), D=10)['m'] == pytest.approx(0.75)


def test_manifold_score_dimension_limit_and_twonn_gate():
    three_d = np.tile([1.0, 0.9, 0.8, 0.01, 0.009], (20, 1))        # gap after 3 directions
    assert manifold_score(three_d, np.full(20, 3.0), D=10)['m'] == 0.0     # 3 > 0.2 * 10
    assert manifold_score(three_d, np.full(20, 3.0), D=20)['m'] == 1.0     # 3 <= 0.2 * 20
    # PCA says 1-D but TwoNN says 9-D: not a low-dimensional manifold
    clean = np.tile([1.0, 0.01, 0.008], (20, 1))
    assert manifold_score(clean, np.full(20, 9.0), D=10)['m'] == 0.0
    # NaN local dimension never counts
    assert manifold_score(clean, np.full(20, np.nan), D=10)['m'] == 0.0


def test_manifold_score_ignores_the_numerically_zero_tail():
    """A k-point neighbourhood has rank <= k-1; the zero tail must not be read as a gap."""
    flat_rank3 = np.tile([1.0, 0.9, 0.8, 0.0, 0.0, 0.0], (10, 1))
    s = manifold_score(flat_rank3, np.ones(10), D=30)
    assert s['m'] == 0.0 and (s['gap'] < 3).all()
    rank1 = np.tile([1.0, 0.0, 0.0], (10, 1))
    s = manifold_score(rank1, np.ones(10), D=30)
    assert s['m'] == 1.0 and (s['pca_dim'] == 1).all() and np.isinf(s['gap']).all()


# ------------------------------------------------------------------ fit_manifold
def test_fit_manifold_circle(circle, circle_fit):
    f = circle_fit
    assert isinstance(f, ManifoldFit)
    assert f.m > 0.9, f.m
    assert f.m == f.score['m']
    assert abs(f.d_global - 1) < 0.15
    assert f.n_components() == 1
    assert f.W is f.A, 'with no Jacobian the g-graph is the Euclidean graph'
    assert {'knn', 'tangent_pca', 'intrinsic_dim', 'pullback'} <= set(f.timings)
    assert all(v >= 0 for v in f.timings.values())
    assert f.bases.shape == (circle.n, circle.D, 8)
    assert f.score['score_k'] == 15 and f.score['D'] == circle.D


def test_fit_manifold_torus_scores_high(torus):
    f = fit_manifold(torus.X)
    assert f.m > 0.9, f.m
    assert abs(f.d_global - 2) < 0.4, f.d_global
    assert f.n_components() == 1


def test_fit_manifold_noise_scores_low(noise):
    f = fit_manifold(noise.X)
    assert f.m < 0.1, f.m
    assert f.score['score_k'] > 15, 'high TwoNN dimension must widen the scoring neighbourhood'
    assert 'score' in f.timings


def test_fit_manifold_two_circles_has_two_components(two_circles):
    f = fit_manifold(two_circles.X)
    assert f.n_components() == 2
    assert f.m > 0.9, f.m


def test_fit_manifold_with_jacobian_uses_the_pullback_metric(circle, circle_readout):
    f = fit_manifold(circle.X, J=circle_readout.J, metric='g')
    W = f.W.tocoo()
    expect = np.linalg.norm((circle.X[W.col] - circle.X[W.row]) @ circle_readout.W.T, axis=1)
    np.testing.assert_allclose(W.data, expect, rtol=1e-6, atol=1e-12)
    assert f.J is circle_readout.J
    fs = fit_manifold(circle.X, metric='secant', out=circle_readout.logits)
    Ws = fs.W.tocoo()
    expect = np.linalg.norm(circle_readout.logits[Ws.col] - circle_readout.logits[Ws.row], axis=1)
    np.testing.assert_allclose(Ws.data, expect, rtol=1e-6, atol=1e-10)


def test_fit_manifold_explicit_score_k(circle):
    f = fit_manifold(circle.X, score_k=40)
    assert f.score['score_k'] == 40
    assert f.m > 0.9, f.m


# ------------------------------------------------------------------ metric lengths
def test_pullback_lengths_identity_jacobian_is_euclidean(circle):
    _, _, A = knn_graph(circle.X, k=10)
    J = np.broadcast_to(np.eye(circle.D), (circle.n, circle.D, circle.D))
    W = pullback_metric_lengths(circle.X, J, A)
    assert abs(W - A).max() < 1e-9
    assert abs(W - W.T).max() < 1e-12


def test_pullback_lengths_is_the_endpoint_averaged_quadratic_form():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 5))
    J = rng.standard_normal((40, 3, 5))
    _, _, A = knn_graph(X, k=5)
    W = pullback_metric_lengths(X, J, A).tocoo()
    for e in range(0, W.nnz, 17):
        i, j = W.row[e], W.col[e]
        d = X[j] - X[i]
        G = 0.5 * (J[i].T @ J[i] + J[j].T @ J[j])
        assert W.data[e] == pytest.approx(np.sqrt(d @ G @ d), rel=1e-9)


def test_pullback_agrees_with_secant_to_first_order_on_a_two_layer_net(circle):
    """psi(h) = V tanh(U h): J is analytic.  For neighbouring points the chord in output space
    equals the pullback length up to O(|delta|^2)."""
    rng = np.random.default_rng(8)
    U = rng.standard_normal((16, circle.D)) * 0.7
    V = rng.standard_normal((4, 16))
    Z = np.tanh(circle.X @ U.T)
    out = Z @ V.T
    J = np.einsum('oh,nh,hd->nod', V, 1 - Z ** 2, U)
    _, _, A = knn_graph(circle.X, k=8)
    Wg = pullback_metric_lengths(circle.X, J, A).tocoo()
    Ws = secant_metric_lengths(out, A).tocoo()
    assert (Wg.row == Ws.row).all() and (Wg.col == Ws.col).all()
    rel = np.abs(Wg.data - Ws.data) / Ws.data
    assert np.median(rel) < 0.01, np.median(rel)
    assert rel.max() < 0.1, rel.max()


def test_secant_lengths_are_output_chords(circle, circle_readout):
    _, _, A = knn_graph(circle.X, k=6)
    W = secant_metric_lengths(circle_readout.logits, A)
    assert W.shape == A.shape and W.nnz == A.nnz
    C = W.tocoo()
    expect = np.linalg.norm(circle_readout.logits[C.col] - circle_readout.logits[C.row], axis=1)
    np.testing.assert_allclose(C.data, expect, atol=1e-10)


# ------------------------------------------------------------------ geodesics
def test_geodesic_on_circle_is_arc_length(circle, circle_fit):
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(25):
        i, j = rng.choice(circle.n, 2, replace=False)
        arc = circle.R * circ_dist(circle.theta[i], circle.theta[j])
        if arc < 0.5:
            continue
        path, L = geodesic(circle_fit.W, i, j)
        assert path[0] == i and path[-1] == j
        ratios.append(L / arc)
        # the path walks monotonically away from i along the circle
        prog = circ_dist(circle.theta[path], circle.theta[i])
        assert (np.diff(prog) > -0.02).all()
        # and its length is the sum of its hops
        hops = np.linalg.norm(np.diff(circle.X[path], axis=0), axis=1).sum()
        assert L == pytest.approx(hops, rel=1e-9)
    ratios = np.array(ratios)
    assert len(ratios) > 10
    assert np.abs(ratios - 1).max() < 0.05, (ratios.min(), ratios.max())


def test_geodesic_scales_with_radius():
    Ls = []
    for R in (1.0, 3.0):
        c = make_circle(n=400, R=R, sigma=0.0, seed=9, even=True)
        f = fit_manifold(c.X)
        _, L = geodesic(f.W, 0, 200)             # antipodal
        assert L == pytest.approx(np.pi * R, rel=0.01), (R, L)
        Ls.append(L)
    # (not exact: on an evenly spaced circle the 15th neighbour is a tie, broken differently)
    assert Ls[1] / Ls[0] == pytest.approx(3.0, rel=1e-3)


def test_geodesic_trivial_and_disconnected(two_circles):
    f = fit_manifold(two_circles.X)
    path, L = geodesic(f.W, 5, 5)
    assert list(path) == [5] and L == 0.0
    path, L = geodesic(f.W, 0, two_circles.n - 1)       # different circles
    assert path is None and np.isinf(L)
    path, L = geodesic(f.W, 0, 100)                     # same circle
    assert path is not None and np.isfinite(L)


def test_geodesic_distances_matrix(circle, circle_fit):
    Dg = geodesic_distances(circle_fit.W)
    assert Dg.shape == (circle.n, circle.n)
    np.testing.assert_allclose(Dg, Dg.T, atol=1e-9)
    assert (np.diag(Dg) == 0).all() and np.isfinite(Dg).all()
    arc = circ_dist(circle.theta[:, None], circle.theta[None, :])
    far = arc > 0.5
    ratio = Dg[far] / arc[far]
    # endpoint noise (|eps| ~ 0.01 each) against arcs >= 0.5 moves the ratio by up to ~4%
    assert 0.95 < ratio.min() and ratio.max() < 1.05, (ratio.min(), ratio.max())
    assert abs(np.median(ratio) - 1) < 0.01, np.median(ratio)
    # graph distance can never undercut the Euclidean chord
    chord = np.linalg.norm(circle.X[:, None] - circle.X[None], axis=2)
    assert (Dg >= chord - 1e-9).all()
    # subset of sources, predecessors
    sub = geodesic_distances(circle_fit.W, sources=[3, 50])
    np.testing.assert_allclose(sub, Dg[[3, 50]])
    d, pred = geodesic_distances(circle_fit.W, sources=[3], return_pred=True)
    assert pred.shape == (1, circle.n) and pred[0, 3] < 0


def test_geodesic_distances_inf_between_components(two_circles):
    f = fit_manifold(two_circles.X)
    Dg = geodesic_distances(f.W, sources=[0])
    assert np.isfinite(Dg[0, two_circles.which == 0]).all()
    assert np.isinf(Dg[0, two_circles.which == 1]).all()


def test_pullback_geodesic_scales_with_the_jacobian(circle):
    """g = c^2 * I  =>  every geodesic is c times longer."""
    c = 2.5
    J = np.broadcast_to(c * np.eye(circle.D), (circle.n, circle.D, circle.D))
    f1 = fit_manifold(circle.X)
    f2 = fit_manifold(circle.X, J=J)
    d1 = geodesic_distances(f1.W, sources=[0])
    d2 = geodesic_distances(f2.W, sources=[0])
    np.testing.assert_allclose(d2, c * d1, rtol=1e-6)
