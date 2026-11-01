"""S5 tests: zigzag recovers a planted appearing/disappearing loop; graph PH counts planted
cycles; Hilbert function and Mapper run on a circle."""
import numpy as np, pytest
from whiteboxtopo import topology as T, nulls


def test_zigzag_recovers_planted_loop_interval():
    pytest.importorskip("dionysus")
    rng = np.random.default_rng(0); clouds = []
    for t in range(8):
        if 2 <= t <= 5:
            X, _ = nulls.planted("circle", 40, seed=t, noise=0.03)       # loop present in clouds 2..5
        else:
            X = rng.normal(size=(12, 2)) * 0.05                            # tight blob, no loop
        clouds.append(X)
    z = T.zigzag(clouds, scale=0.7, maxdim=1)
    long = [b for b in z.bars[1] if (np.inf if not np.isfinite(b[1]) else b[1]) - b[0] >= 4]
    assert len(long) == 1
    first, last = z.bars_in_cloud_index(1)[[i for i, b in enumerate(z.bars[1]) if b is long[0] or np.array_equal(b, long[0])][0]]
    assert (first, last) == (2, 5)
    assert z.betti_curve(1)[4:11].min() == 1 and z.betti_curve(1)[0] == 0 and z.betti_curve(1)[-1] == 0


def test_graph_ph_counts_planted_cycles():
    n = 12; A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1.0       # one 12-cycle, never filled
    r = T.graph_ph(A)
    assert r.betti1_essential == 1 and r.betti1_finite == 0
    A[0, 6] = A[6, 0] = 0.5                                  # chord -> second cycle, also unfilled (no triangles)
    r = T.graph_ph(A); assert r.betti1_essential == 2


def test_hilbert_function_circle():
    X, _ = nulls.planted("circle", 150, seed=0, noise=0.03); f = np.linspace(0, 1, 150)
    m = T.hilbert_function(X, f, n_grid=6)
    assert m.hilbert.shape == (2, 6, 6) and m.hilbert[1].max() >= 1


def test_mapper_circle_has_cycle():
    X, _ = nulls.planted("circle", 300, seed=1, noise=0.03)
    r = T.mapper(X, X[:, 0], n_cubes=8, overlap=0.4)
    assert r.n_components == 1 and r.cycle_rank >= 1


def test_normalized_persistence_scale_invariant():
    rng = np.random.default_rng(2); A = rng.normal(size=(200, 6)) * rng.uniform(0.1, 10, size=6)
    a = T.normalized_persistence(A); b = T.normalized_persistence(A * np.exp(rng.normal(size=6))[None, :])
    for k in a:
        assert np.isclose(a[k], b[k], rtol=1e-8, atol=1e-8)
