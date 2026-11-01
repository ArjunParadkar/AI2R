"""Synthetic ground truth for whiteboxtopo.depthprofile (SPEC2 block A)."""
import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from whiteboxtopo.depthprofile import (euclidean_distance_matrix, single_linkage, mst_largest_edge,
                                       robust_scale, beta0_at, beta0_curve, fiber_beta0_profile,
                                       depth_of_connection, untangling_index,
                                       pullback_distance_matrix, gw_fibers)


def two_blobs(n=60, gap=10.0, D=5, seed=0):
    rng = np.random.default_rng(seed)
    a = 0.1 * rng.standard_normal((n, D))
    b = 0.1 * rng.standard_normal((n, D)); b[:, 0] += gap
    return np.concatenate([a, b]), np.r_[np.zeros(n, int), np.ones(n, int)]


def test_two_blobs_beta0_below_and_above_gap():
    X, _ = two_blobs()
    D = euclidean_distance_matrix(X)
    assert 9.0 < mst_largest_edge(D) < 10.5                # the MST's largest edge IS the gap
    assert beta0_at(D, 5.0) == 2
    assert beta0_at(D, 11.0) == 1
    assert beta0_at(D, 0.0) == len(X)
    c = beta0_curve(D, np.linspace(0, 12, 40))
    assert np.all(np.diff(c) <= 0) and c[0] == len(X) and c[-1] == 1


def test_beta0_matches_brute_force_components():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 3))
    D = euclidean_distance_matrix(X)
    for eps in (0.2, 0.5, 0.8, 1.5):
        ref = connected_components(D <= eps, directed=False)[0]
        assert beta0_at(D, eps) == ref


def test_robust_scale_ignores_one_outlier():
    X, _ = two_blobs(n=100, gap=0.0)
    X = np.concatenate([X, [[50.0, 0, 0, 0, 0]]])          # 1 outlier in 201 points (< 1%)
    D = euclidean_distance_matrix(X)
    assert mst_largest_edge(D) > 40
    assert robust_scale(D, 0.99) < 1.0


def test_fiber_profile_dstar_and_U():
    # layer 1: class 0 split in two far pieces, class 1 compact; layer 2: both compact
    rng = np.random.default_rng(2)
    lab = np.r_[np.zeros(40, int), np.ones(40, int)]
    L1 = 0.05 * rng.standard_normal((80, 4)); L1[:20, 0] += 4.0; L1[40:, 1] += 6.0
    L2 = 0.05 * rng.standard_normal((80, 4)); L2[40:, 1] += 6.0
    b = []
    for X in (L1, L2):
        D = euclidean_distance_matrix(X)
        eps = 0.5 * mst_largest_edge(D)
        cls, B, C = fiber_beta0_profile(D, lab, [eps], eps_grid=np.linspace(0, 8, 9))
        assert C.shape == (2, 9) and list(cls) == [0, 1]
        b.append(B[:, 0])
    b = np.stack(b, 1)                                       # (class, layer)
    assert b.tolist() == [[2, 1], [1, 1]]
    d, never = depth_of_connection(b)
    assert d.tolist() == [2, 1] and not never.any()
    assert untangling_index(d, 2) == pytest.approx(1 - (2 / 2 + 1 / 2) / 2)
    d2, never2 = depth_of_connection(np.array([[3, 2, 2]]))
    assert d2.tolist() == [3] and never2.tolist() == [True]
    assert untangling_index(d2, 3) == 0.0


def test_gw_rotated_copy_is_zero_and_different_shape_is_not():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 3)) * np.array([3.0, 1.0, 0.3])
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    Y = 7.0 * (X @ Q.T) + 5.0                                # rotated, scaled (median-normalised away), shifted
    Dx, Dy = euclidean_distance_matrix(X), euclidean_distance_matrix(Y)
    g_same, _ = gw_fibers(Dx, Dy)
    assert g_same < 1e-6
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    Dc = euclidean_distance_matrix(np.stack([np.cos(t), np.sin(t)], 1))
    g_diff, _ = gw_fibers(Dx, Dc)
    assert g_diff > 100 * max(g_same, 1e-8) and g_diff > 1e-3


def test_pullback_distance_matrix_is_a_finite_symmetric_matrix():
    # two blobs 5 apart: the k-NN candidate pool is Euclidean, so the graph has two components and
    # the cross-component fill (2 x largest finite distance) must make the matrix finite
    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, (300, 2)).astype(np.float32); X[150:, 1] += 5.0
    W = np.array([[1.0, 0.0]], dtype=np.float32)
    out = X @ W.T
    De = euclidean_distance_matrix(X)
    assert beta0_at(De, 0.5 * mst_largest_edge(De)) == 2
    Ds, _ = pullback_distance_matrix(X, out=out, k=15)
    assert Ds.shape == (300, 300) and np.isfinite(Ds).all() and np.allclose(Ds, Ds.T)
    J = np.broadcast_to(W[None], (300, 1, 2)).copy()
    Dg, _ = pullback_distance_matrix(X, J=J, k=15)
    assert np.isfinite(Dg).all() and (Dg >= 0).all()


def test_chunked_pullback_lengths_equal_the_library():
    from whiteboxtopo.manifold import knn_graph
    from whiteboxtopo.manifold.core import pullback_metric_lengths
    from whiteboxtopo.depthprofile import _pullback_lengths_chunked
    rng = np.random.default_rng(5)
    X = rng.standard_normal((120, 6)).astype(np.float32)
    J = rng.standard_normal((120, 4, 6)).astype(np.float32)
    _, _, A = knn_graph(X, 10, J=J, metric='g')
    ref = pullback_metric_lengths(X, J, A).toarray()
    got = _pullback_lengths_chunked(X, J, A, chunk_bytes=4 * 6 * 4 * 7).toarray()      # 7 edges per chunk
    assert np.allclose(ref, got, rtol=1e-5, atol=1e-7)
