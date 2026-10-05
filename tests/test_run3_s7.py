"""Planted truth for S7 (ID-profile classes, fiber connection scale) and the small S3 helpers."""
import numpy as np, pytest
from scipy.spatial.distance import pdist, squareform
from analysis.run3_blockP64 import id_profile, conn_slope
from whiteboxtopo.depthprofile import fiber_connection_scales
from whiteboxtopo.symmetry_finite import element_order, cycle_coordinate


def test_flat_plane_has_a_plateau_profile_near_its_dimension():
    rng = np.random.default_rng(0); X = np.zeros((1500, 12), np.float32); X[:, :3] = rng.uniform(-1, 1, (1500, 3))
    r = id_profile(X, n_boot=5); assert r['profile'] == 'plateau' and 2.2 < r['id_curve'][2] < 3.6


def test_tight_clusters_have_a_decaying_profile():
    rng = np.random.default_rng(1); C = rng.standard_normal((10, 20)) * 10
    X = (C[rng.integers(0, 10, 1500)] + 0.05 * rng.standard_normal((1500, 20))).astype(np.float32)
    assert id_profile(X, n_boot=5)['profile'] == 'decay'                 # beyond the cluster size the estimate collapses


def test_noise_around_a_curve_grows_with_scale_or_is_not_a_plateau_at_small_k():
    rng = np.random.default_rng(2); t = rng.uniform(0, 2 * np.pi, 1500)
    X = (np.c_[np.cos(t), np.sin(t), np.zeros((1500, 8))] + 0.02 * rng.standard_normal((1500, 10))).astype(np.float32)
    ids = id_profile(X, n_boot=3)['id_curve']; assert ids[0] > ids[-1] and ids[-1] < 1.6    # noise ball at small k, the curve at large k


def test_connection_scale_is_the_gap_between_two_blobs_of_one_class():
    rng = np.random.default_rng(3); A = 0.1 * rng.standard_normal((40, 2)); B = 0.1 * rng.standard_normal((40, 2)) + np.array([5.0, 0])
    X = np.r_[A, B, 0.1 * rng.standard_normal((40, 2)) + np.array([0, 9.0])]; y = np.r_[np.zeros(80, int), np.ones(40, int)]
    s = fiber_connection_scales(squareform(pdist(X)), y); assert 4.0 < s[0] < 5.5 and s[1] < 0.5


def test_connection_slope_is_negative_when_fibers_tighten_with_depth():
    rng = np.random.default_rng(4); y = np.repeat(np.arange(5), 40); C = rng.standard_normal((5, 6)) * 4
    clouds = [(C[y] * a + rng.standard_normal((200, 6))).astype(np.float32) for a in (0.0, 0.5, 2.0)]      # classes separate and tighten relative to the cloud
    assert conn_slope(clouds, y)['slope'] < -0.2


def test_permutation_helpers():
    p = np.array([1, 2, 0, 4, 3]); assert element_order(p) == 6
    s = (np.arange(7) + 3) % 7; t = cycle_coordinate(s); assert t[0] == 0 and t[3] == 1 and sorted(t) == list(range(7))
