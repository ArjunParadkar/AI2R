"""Synthetic ground truth for whiteboxtopo.kernel (run 2, block B).

Run with  pytest --noconftest tests/test_run2_kernel.py  so the run-1 timing record
(results/test_timings.json, written by conftest) is left alone.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from whiteboxtopo.kernel import (row_space, kernel_split, invisible_displacement, iota, knn_sets,
                                 jaccard_rows, kappa, combined, auroc)

D, SEEN = 12, 4          # a linear head that reads coordinates 0..3 and ignores 4..11


def head(seed=0):
    rng = np.random.default_rng(seed)
    J = np.zeros((6, D))
    J[:, :SEEN] = rng.standard_normal((6, SEEN))
    return J


def test_rank_and_kernel_dimension():
    V, r = row_space(head())
    assert r == SEEN
    assert np.allclose(V[:r] @ V[:r].T, np.eye(r), atol=1e-10)


def test_displacement_inside_ignored_subspace_is_all_invisible():
    rng = np.random.default_rng(1)
    delta = np.zeros((50, D)); delta[:, SEEN:] = rng.standard_normal((50, D - SEEN))
    par, perp, rank = kernel_split(delta, head())
    assert np.allclose(perp, delta, atol=1e-10)
    assert np.allclose(par, 0, atol=1e-10)
    assert (rank == SEEN).all()


def test_displacement_inside_read_subspace_is_all_visible():
    rng = np.random.default_rng(2)
    delta = np.zeros((50, D)); delta[:, :SEEN] = rng.standard_normal((50, SEEN))
    par, perp, _ = kernel_split(delta, head())
    assert np.allclose(par, delta, atol=1e-10)
    assert np.allclose(perp, 0, atol=1e-10)


def test_split_is_orthogonal_and_head_cannot_see_perp():
    rng = np.random.default_rng(3)
    J = head(); delta = rng.standard_normal((40, D))
    par, perp, _ = kernel_split(delta, J)
    assert np.allclose(par + perp, delta)
    assert np.allclose((par * perp).sum(1), 0, atol=1e-10)
    assert np.allclose(perp @ J.T, 0, atol=1e-9)                     # J delta_perp = 0
    assert np.allclose(par @ J.T, delta @ J.T, atol=1e-9)            # J delta_par = J delta


def test_rcond_truncates_small_singular_values():
    J = np.diag([1.0, 0.5, 1e-5])                                    # third direction below 1e-3 * sigma_max
    _, r = row_space(J, rcond=1e-3)
    assert r == 2
    par, perp, _ = kernel_split(np.array([[0.0, 0.0, 1.0]]), J)
    assert np.allclose(perp, [[0, 0, 1]]) and np.allclose(par, 0)


def test_per_point_jacobians_match_shared_and_callable():
    rng = np.random.default_rng(4)
    Xtr = rng.standard_normal((30, D)); H = Xtr[:10] + 0.01 * rng.standard_normal((10, D))
    J = head()
    shared = invisible_displacement(H, Xtr, J)
    stack = invisible_displacement(H, Xtr, np.broadcast_to(J, (30,) + J.shape))
    called = []
    def fn(ids):
        called.append(np.asarray(ids)); return np.broadcast_to(J, (len(ids),) + J.shape)
    lazy = invisible_displacement(H, Xtr, fn)
    for k in ('perp', 'par', 'ratio'):
        assert np.allclose(shared[k], stack[k], atol=1e-10) and np.allclose(shared[k], lazy[k], atol=1e-10)
    assert (shared['nn_idx'] == np.arange(10)).all()
    assert set(called[0].tolist()) == set(range(10))                 # only the needed Jacobians were requested
    assert (shared['kernel_dim'] == D - SEEN).all()


def test_per_point_jacobians_differ_per_point():
    # point 0's head reads coordinate 0, point 1's head reads coordinate 1
    Xtr = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    J = np.zeros((2, 1, 3)); J[0, 0, 0] = 1; J[1, 0, 1] = 1
    H = Xtr + np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    r = invisible_displacement(H, Xtr, J)
    assert np.allclose(r['perp'], [0.0, 1.0]) and np.allclose(r['par'], [1.0, 0.0])


def test_iota_is_one_at_the_in_distribution_median_and_detects_invisible_shift():
    rng = np.random.default_rng(5)
    Xtr = rng.standard_normal((400, D)) * 0.1
    Hin = rng.standard_normal((200, D)) * 0.1
    Hout = Hin.copy(); Hout[:, SEEN:] += 3.0                         # shifted only where the head is blind
    J = head()
    a, b = invisible_displacement(Hin, Xtr, J), invisible_displacement(Hout, Xtr, J)
    assert np.isclose(np.median(iota(a['perp'], a['perp'])), 1.0)
    assert auroc(iota(a['perp'], a['perp']), iota(b['perp'], a['perp'])) > 0.99
    # the invisible fraction of the displacement goes to ~1 for the shifted points
    assert np.median(b['ratio']) > 0.99 > np.median(a['ratio'])
    # with the nearest neighbour held fixed, what the head sees of the shift is exactly nothing
    par_in, _, _ = kernel_split(Hin - Xtr[a['nn_idx']], J)
    par_out, _, _ = kernel_split(Hout - Xtr[a['nn_idx']], J)
    assert np.allclose(par_in, par_out, atol=1e-10)


def test_jaccard_identical_disjoint_and_half():
    A = np.arange(30).reshape(2, 15)
    assert np.allclose(jaccard_rows(A, A), 1.0)
    assert np.allclose(jaccard_rows(A, A[:, ::-1]), 1.0)             # order does not matter
    assert np.allclose(jaccard_rows(A, A + 100), 0.0)
    B = A.copy(); B[:, :5] += 1000                                   # 10 shared of 20 in the union
    assert np.allclose(jaccard_rows(A, B), 0.5)


def test_kappa_identical_layers_is_one_and_means_over_consecutive_pairs():
    rng = np.random.default_rng(6)
    Xtr = rng.standard_normal((200, 5)); H = rng.standard_normal((40, 5))
    s = knn_sets(Xtr, H, 15)
    assert s.shape == (40, 15)
    k, per = kappa([s, s, s])
    assert np.allclose(k, 1.0) and per.shape == (2, 40)
    # an isometry of a layer (rotation + scale) keeps every neighbourhood
    Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    k2, _ = kappa([s, knn_sets(3.0 * Xtr @ Q, 3.0 * H @ Q, 15)])
    assert np.allclose(k2, 1.0)
    # an unrelated layer destroys it; consecutive-pair mean of (1, ~0) is ~0.5
    s3 = knn_sets(rng.standard_normal((200, 5)), rng.standard_normal((40, 5)), 15)
    k3, per3 = kappa([s, s, s3])
    assert np.allclose(per3[0], 1.0) and per3[1].mean() < 0.15
    assert np.allclose(k3, per3.mean(0))
    with pytest.raises(ValueError):
        kappa([s])


def test_combined_and_auroc_orientation():
    assert np.allclose(combined([2.0, 3.0], [1.0, 0.5]), [0.0, 1.5])
    assert auroc([0, 0.1, 0.2], [1, 2, 3]) == 1.0
    assert auroc([1, 2, 3], [0, 0.1, 0.2]) == 0.0
