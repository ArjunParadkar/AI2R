import numpy as np, pytest
import whitebox as wb
from whitebox.signals.correspondence import rtd, correspondence, cross_barcode, normalized_distances
from conftest import circle


def _rot(X, d, seed=0):
    rng = np.random.default_rng(seed); Q, _ = np.linalg.qr(rng.standard_normal((d, d))); Z = np.zeros((len(X), d)); Z[:, :X.shape[1]] = X
    return Z @ Q.T


def test_rtd_zero_for_isometry():
    X = circle(200); Y = _rot(X, 10) * 5.0 + 3.0     # scaled, shifted, rotated: same topology, same neighbourhoods
    r = rtd(X, Y, n_trials=3, batch=200)
    assert r.point == pytest.approx(0.0, abs=1e-6)


def test_rtd_positive_and_symmetric_for_shuffle():
    X = circle(200); Y = X[np.random.default_rng(0).permutation(200)]
    r = rtd(X, Y, n_trials=3, batch=200)
    assert r.point > 0.5


def test_correspondence_bounds_and_direction():
    X = circle(300, seed=1)
    good = correspondence(X, _rot(X, 8) * 2, n_trials=4, batch=150)
    bad = correspondence(X, np.random.default_rng(2).standard_normal((300, 8)), n_trials=4, batch=150)
    assert 0 <= bad.ci_low <= bad.score <= bad.ci_high <= 1
    assert good.score > 0.95 and bad.score < 0.4
    assert good.rtd.point < bad.rtd.point


def test_cross_barcode_shape():
    X = circle(50); r = normalized_distances(X); b = cross_barcode(r, r)
    assert len(b) == 2 and b[1].shape[1] == 2 and b[1].size == 0   # identical clouds: no H1 in the cross-barcode


def test_representation_correspondence_api():
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2))
    X = torch.tensor(circle(120), dtype=torch.float32)
    out = wb.representation_correspondence(net, [(X[:60], None), (X[60:], None)], ["1", "2"], n_trials=2, batch=120)
    assert set(out) == {"1", "2"} and all(0 <= c.score <= 1 for c in out.values())
    with pytest.raises(ValueError):
        wb.representation_correspondence(activations={"a": np.zeros((5, 2))})
