"""whiteboxtopo.nulls: the controls every positive result has to survive."""
import numpy as np
import pytest

from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.nulls import feature_shuffle, label_permute, shuffle_predictor
from whiteboxtopo.problems import LayerData, p6_geodesic_generalization
from whiteboxtopo.topology import persistence


def top_h1(X):
    h1 = persistence(X, do_cocycles=False)['dgms'][1]
    return float((h1[:, 1] - h1[:, 0]).max()) if len(h1) else 0.0


# ------------------------------------------------------------------ feature_shuffle
def test_feature_shuffle_preserves_every_marginal_exactly(circle):
    Y = feature_shuffle(circle.X, seed=0)
    assert Y.shape == circle.X.shape and Y.dtype == circle.X.dtype
    for j in range(circle.D):
        assert np.array_equal(np.sort(Y[:, j]), np.sort(circle.X[:, j])), j      # exact, not approx
    np.testing.assert_allclose(Y.mean(0), circle.X.mean(0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(Y.std(0), circle.X.std(0), rtol=1e-12)


def test_feature_shuffle_does_not_touch_its_input_and_is_seeded(circle):
    before = circle.X.copy()
    a = feature_shuffle(circle.X, seed=5)
    assert np.array_equal(circle.X, before)
    assert not np.shares_memory(a, circle.X)
    assert np.array_equal(a, feature_shuffle(circle.X, seed=5))
    assert not np.array_equal(a, feature_shuffle(circle.X, seed=6))
    assert not np.array_equal(a, circle.X)


def test_feature_shuffle_shuffles_columns_independently(circle):
    """One shared permutation would preserve the joint structure; each column needs its own."""
    Y = feature_shuffle(circle.X, seed=0)
    rows = {tuple(r) for r in circle.X.round(12)}
    surviving = sum(tuple(r) in rows for r in Y.round(12))
    assert surviving == 0
    # joint structure gone: the cross-covariance collapses
    C0 = np.cov(circle.X.T); C1 = np.cov(Y.T)
    off = ~np.eye(circle.D, dtype=bool)
    assert np.abs(C1[off]).max() < 0.25 * np.abs(C0[off]).max()


def test_feature_shuffle_destroys_h1_and_the_manifold(circle, circle_ph):
    h1 = circle_ph['dgms'][1]
    top_real = float((h1[:, 1] - h1[:, 0]).max())
    top_null = top_h1(feature_shuffle(circle.X, seed=0))
    assert top_real > 1.3
    assert top_null < 0.25 * top_real, (top_null, top_real)
    f_real, f_null = fit_manifold(circle.X), fit_manifold(feature_shuffle(circle.X, seed=0))
    assert f_real.m > 0.9 and f_null.m < 0.2, (f_real.m, f_null.m)
    assert f_null.d_global > 3 * f_real.d_global


def test_feature_shuffle_destroys_both_loops_of_the_torus(torus, torus_ph):
    h1 = torus_ph['dgms'][1]
    second = np.sort(h1[:, 1] - h1[:, 0])[-2]
    assert top_h1(feature_shuffle(torus.X, seed=0)) < 0.4 * second


# ------------------------------------------------------------------ label_permute
def test_label_permute_preserves_the_label_histogram(circle_layer):
    y = circle_layer.labels
    before = y.copy()
    p = label_permute(y, seed=0)
    assert np.array_equal(np.bincount(p), np.bincount(y))
    assert np.array_equal(np.sort(p), np.sort(y))
    assert np.array_equal(y, before), 'input must not be modified'
    assert (p != y).mean() > 0.5                               # 4 balanced classes: ~75% move
    assert np.array_equal(p, label_permute(y, seed=0))
    assert not np.array_equal(p, label_permute(y, seed=1))
    assert np.array_equal(label_permute(list(y), seed=0), p)   # accepts a list


def test_label_permute_breaks_the_label_geometry_relation(circle, circle_layer, circle_readout):
    fit = fit_manifold(circle.X, J=circle_readout.J, metric='g')
    real = p6_geodesic_generalization(circle_layer, fit)['G']
    perm = LayerData('perm', circle_layer.X, circle_layer.J, circle_layer.logits,
                     label_permute(circle_layer.labels, seed=2))
    null = p6_geodesic_generalization(perm, fit)['G']
    assert real > 0.98 and null < 0.4, (real, null)


# ------------------------------------------------------------------ shuffle_predictor
def test_shuffle_predictor_keeps_the_values_and_kills_the_correlation():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(3000)
    target = v + 0.1 * rng.standard_normal(3000)
    s = shuffle_predictor(v, seed=0)
    assert np.array_equal(np.sort(s), np.sort(v))
    assert np.corrcoef(v, target)[0, 1] > 0.99
    assert abs(np.corrcoef(s, target)[0, 1]) < 0.06             # ~ 3 / sqrt(n)
    assert np.array_equal(s, shuffle_predictor(v, seed=0))
    assert not np.array_equal(s, shuffle_predictor(v, seed=1))
