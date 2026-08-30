import numpy as np
import pytest
import whiteboxtopo as wb
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


def test_landmark_backend_preserves_circle():
    X = circle(600)
    D = wb.compute_diagrams(X, backend="landmark", n_landmarks=100, thresh_percentile=100)
    assert D.n_points == 600 and len(D.meta["landmarks"]) == 100
    assert wb.betti_at_scale(D[1], 1.0) == 1


def test_gpu_distance_matches_cpu(circle_cloud):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    a = wb.compute_diagrams(circle_cloud, distance_device="cpu")
    b = wb.compute_diagrams(circle_cloud, distance_device="cuda")
    assert abs(a.thresh - b.thresh) < 1e-4
    assert abs(wb.topological_complexity(a[1]) - wb.topological_complexity(b[1])) < 1e-3


def test_film_bootstrap_and_sv_pairing():
    rng = np.random.default_rng(0)
    base = circle(120, seed=3)
    acts = [{"h": base + 0.01 * t * rng.standard_normal(base.shape)} for t in range(3)]
    film = wb.film_from_activations(acts, n_components=2, n_boot=8, subsample=60)
    tc = film.tc("h"); sv = film.sv("h")
    assert len(tc) == 3 and tc[0].n_boot == 8 and tc[0].ci_low <= tc[0].ci_high
    assert sv[0].point == 0.0 and sv[1].n_boot == 8
    s = film.summary()
    assert set(s["h"]) == {"TC", "SV", "B1", "FPL"} and len(s["h"]["TC"]["point"]) == 3


def test_topological_film_snapshot_torch():
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3))
    probe = [(torch.randn(40, 4), None) for _ in range(2)]
    film = wb.topological_film(net, probe, ["1", "2"], n_components=3, n_boot=4, subsample=40)
    assert len(film) == 1 and film.layers == ["1", "2"]
    ck = [(e, net.state_dict()) for e in (1, 2)]
    film2 = wb.topological_film(net, probe, ["2"], checkpoints=ck, n_components=3, n_boot=0)
    assert film2.epochs == [1, 2] and film2.sv("2")[1].point == pytest.approx(0.0, abs=1e-9)
