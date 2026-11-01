import numpy as np
import pytest
import whiteboxtopo as wb
from whiteboxtopo.metrics.tc import persistent_entropy
from whiteboxtopo.metrics.sv import sv_trajectory
from conftest import circle

D1 = np.array([[0.0, 1.0], [0.2, 0.5], [0.1, 0.15]])
EMPTY = np.empty((0, 2))


def test_tc_known():
    assert wb.topological_complexity(D1) == pytest.approx(1.0 + 0.09 + 0.0025)
    assert wb.topological_complexity(EMPTY) == 0.0


def test_persistent_entropy_bounds():
    assert persistent_entropy(EMPTY) == 0.0
    assert persistent_entropy(np.array([[0, 1.0]])) == pytest.approx(0.0)
    eq = np.array([[0, 1.0], [0, 1.0], [0, 1.0], [0, 1.0]])
    assert persistent_entropy(eq) == pytest.approx(np.log(4))


def test_betti_known():
    assert wb.betti_at_scale(D1, 0.3) == 2
    assert wb.betti_at_scale(D1, 0.7) == 1
    assert wb.betti_at_scale(D1, 2.0) == 0
    assert wb.betti_at_scale(EMPTY, 0.5) == 0
    assert list(wb.betti_curve(D1, [0.05, 0.3, 0.7])) == [1, 2, 1]
    assert wb.median_scale(D1) == pytest.approx(np.median([0.5, 0.35, 0.125]))


def test_sv_properties():
    assert wb.structural_velocity(D1, D1) == 0.0
    assert wb.structural_velocity(EMPTY, EMPTY) == 0.0
    a = wb.structural_velocity(D1, EMPTY); b = wb.structural_velocity(EMPTY, D1)
    assert a > 0 and a == pytest.approx(b)
    # a single bar vs empty: W2 to the diagonal = lifetime/sqrt(2)
    assert wb.structural_velocity(np.array([[0.0, 2.0]]), EMPTY) == pytest.approx(2 / np.sqrt(2), rel=1e-3)
    traj = sv_trajectory([D1, D1, EMPTY])
    assert traj[0] == 0.0 and traj[1] == 0.0 and traj[2] > 0


def test_sv_detects_circle_collapse():
    Dc = wb.compute_diagrams(circle(150))[1]
    Db = wb.compute_diagrams(np.random.default_rng(0).standard_normal((150, 2)) * 0.1)[1]
    assert wb.structural_velocity(Dc, Db) > 0.3


def test_fpl():
    assert wb.top_k_lifetime(D1, 3) == pytest.approx((1.0 + 0.3 + 0.05) / 3)
    assert wb.top_k_lifetime(D1, 1) == pytest.approx(1.0)
    assert wb.top_k_lifetime(EMPTY) == 0.0
    D2 = np.array([[0.0, 1.0], [0.0, 0.95], [0.0, 0.05]])
    film = [D2, D2, D2, D1, D1, D1]
    assert wb.feature_persistence_runs(film).tolist() == [3]
    # ties at the median never count as 'above' (strict >): all-identical film -> no run
    assert wb.feature_persistence_runs([D1, D1, D1]).tolist() == [0]
    assert wb.feature_persistence_runs([]).tolist() == [0]


def test_biastop_permutation_null_for_identical_groups():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 6)); r = wb.fit_reducer(X, 4)
    obs, p, null = wb.bias_top_permutation_test(X[:60], X[60:], r, n_permutations=30, rng=rng)
    assert null.shape == (30,) and 0 < p <= 1
    assert p > 0.05  # same distribution -> no detectable topological bias
