"""Planted truth for S3: regular polygons must give the dihedral group; random clouds the trivial one; a simplex is flagged degenerate."""
import numpy as np, pytest
from whiteboxtopo.symmetry_finite import automorphisms, full_cycle, cycle_coordinate, characters, element_order
from scipy.spatial.distance import pdist, squareform


def polygon(p, noise=0.0, seed=0, freqs=(1,)):
    rng = np.random.default_rng(seed); a = 2 * np.pi * np.arange(p) / p
    X = np.concatenate([np.c_[np.cos(k * a), np.sin(k * a)] for k in freqs], 1)
    return X + noise * rng.standard_normal(X.shape)


@pytest.mark.parametrize('p', [12, 31, 113])
def test_regular_polygon_gives_dihedral_group(p):
    r = automorphisms(squareform(pdist(polygon(p))))
    assert r['order'] == 2 * p and r['has_full_cycle'] and r['has_involution'] and r['closure'] == 1.0 and not r['degenerate']


def test_noisy_multifrequency_embedding_of_Z113_and_relabelling():
    X = polygon(113, noise=0.02, seed=1, freqs=(3, 26, 50)); perm = np.random.default_rng(2).permutation(113)
    r = automorphisms(squareform(pdist(X[perm])))                      # points presented in scrambled order: nothing is supplied
    assert r['order'] == 226 and r['has_full_cycle']
    s = full_cycle(r); t = cycle_coordinate(s); assert element_order(s) == 113 and sorted(t) == list(range(113))
    ch = characters(X[perm], t, top=3)                                  # discovered frequencies equal the planted ones up to ONE unit u
    true = {3, 26, 50}; fold = lambda k: min(k % 113, 113 - k % 113)
    assert any({fold(u * c['freq']) for c in ch} == true for u in range(1, 57))


def test_random_cloud_is_trivial_and_simplex_is_degenerate():
    r = automorphisms(squareform(pdist(np.random.default_rng(0).standard_normal((60, 5)))))
    assert r['order'] == 1 and not r['has_full_cycle']
    assert automorphisms(np.ones((40, 40)) - np.eye(40))['degenerate']


def test_two_point_swap_symmetry_is_found():
    X = np.random.default_rng(3).standard_normal((30, 4)); X[1] = X[0] * np.array([1, 1, 1, -1]); X[:, 3][2:] = 0   # reflection fixes all but 0<->1
    r = automorphisms(squareform(pdist(X)), tol=0.02); assert r['order'] == 2 and r['has_involution']
