import numpy as np
import pytest
import whitebox as wb
from conftest import circle, blob


def test_circle_has_one_loop(circle_cloud):
    D = wb.compute_diagrams(circle_cloud, maxdim=1, thresh_percentile=100)
    lt = D.lifetimes(1)
    assert lt.size >= 1
    # one dominant bar, everything else is noise
    top = np.sort(lt)[::-1]
    assert top[0] > 1.0 and (top[1] < 0.2 if top.size > 1 else True)
    assert wb.betti_at_scale(D[1], 1.0) == 1


def test_two_circles_two_loops():
    X = np.vstack([circle(150, seed=1), circle(150, seed=2) + np.array([5.0, 0.0])])
    D = wb.compute_diagrams(X, maxdim=1, thresh_percentile=100)
    assert wb.betti_at_scale(D[1], 1.0) == 2


def test_blob_has_less_tc_than_circle(circle_cloud, blob_cloud):
    Dc = wb.compute_diagrams(circle_cloud); Db = wb.compute_diagrams(blob_cloud)
    assert wb.topological_complexity(Dc[1]) > 5 * wb.topological_complexity(Db[1])


def test_backends_agree(circle_cloud):
    if "giotto-ph" not in wb.available_backends():
        pytest.skip("giotto-ph not installed")
    a = wb.compute_diagrams(circle_cloud, backend="ripser")
    b = wb.compute_diagrams(circle_cloud, backend="giotto-ph")
    assert a.thresh == pytest.approx(b.thresh)
    for k in (0, 1):
        A = a[k][np.lexsort(a[k].T)]; B = b[k][np.lexsort(b[k].T)]
        assert A.shape == B.shape
        assert np.allclose(A, B, atol=1e-6)


def test_reducer_fixed_and_clipped():
    X = np.random.default_rng(0).standard_normal((20, 50))
    r = wb.fit_reducer(X, n_components=30)
    assert r.n_components == 19
    assert r.transform(X).shape == (20, 19)
    assert np.allclose(r.transform(X), r.transform(X))


def test_tiny_cloud_empty():
    D = wb.compute_diagrams(np.zeros((2, 3)))
    assert D[0].shape == (0, 2) and D[1].shape == (0, 2)


def test_film_from_activations():
    rng = np.random.default_rng(0)
    acts = [{"a": rng.standard_normal((60, 8)), "b": circle(60, seed=t)} for t in range(3)]
    film = wb.film_from_activations(acts, n_components=5)
    assert film.layers == ["a", "b"] and len(film) == 3
    assert film.reducers["a"].n_components == 5
    assert all(D.n_points == 60 for D in film.frames["b"])
    assert len(film.diagrams("b", 1)) == 3
