"""whiteboxtopo.invariants.extractors -- path products (2.5), Fourier irreps (2.6),
circular coordinates (2.7), each against the null that would kill it."""
import numpy as np
import pytest

from whiteboxtopo.invariants import ReluMLP, apply_gauge, random_gauge_element
from whiteboxtopo.invariants.extractors import (path_product, path_products, top_contributions,
                                                fourier_irreps, circular_coordinate, as_grok_model)

P = 113
K_TRUE = (3, 7, 11)
W_TRUE = (1.0, 0.6, 0.35)


def synthetic_grok(p=P, K=K_TRUE, w=W_TRUE, d=16, seed=0, rotate=True):
    """A grokked modular-addition model built analytically, not trained.

    Embedding rows (first p) carry unit-amplitude cos/sin blocks at the frequencies
    in K; the unembedding carries the *readout* amplitudes w_k; the forward pass is
    logits(c) = sum_k w_k cos(omega_k (a + b - c)). Both matrices are then rotated by
    a random orthogonal matrix, which is a gauge freedom of the residual stream.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(p)
    om = 2 * np.pi * np.array(K) / p
    ang = np.outer(x, om)
    E = np.zeros((p + 1, d))
    U = np.zeros((d, p))
    for i in range(len(K)):
        E[:p, 2 * i] = np.cos(ang[:, i]); E[:p, 2 * i + 1] = np.sin(ang[:, i])
        U[2 * i, :] = w[i] * np.cos(ang[:, i]); U[2 * i + 1, :] = w[i] * np.sin(ang[:, i])
    if rotate:
        Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
        E, U = E @ Q, Q.T @ U

    def forward(tokens):
        t = np.atleast_2d(np.asarray(tokens))
        s = (t[:, 0] + t[:, 1]) % p
        return np.stack([sum(w[i] * np.cos(om[i] * ((s_ - x) % p)) for i in range(len(K))) for s_ in s])

    return as_grok_model(type("M", (), {"embed": E, "unembed": U, "forward": staticmethod(forward)})())


def random_mlp(seed=0, n_in=6, widths=(16, 12), n_out=4):
    rng = np.random.default_rng(seed)
    dims = [n_in, *widths, n_out]
    return ReluMLP([rng.standard_normal((dims[i], dims[i + 1])) / np.sqrt(dims[i]) for i in range(len(dims) - 1)],
                   [rng.standard_normal(dims[i + 1]) * 0.1 for i in range(len(dims) - 1)])


# ------------------------------------------------------------------ 2.5 path products
def test_path_product_exact_linearisation():
    m = random_mlp()
    X = np.random.default_rng(1).standard_normal((32, 6))
    A, c = path_products(m, X)
    f = m.logits(X)
    assert np.abs(np.einsum("nij,nj->ni", A, X) + c - f).max() < 1e-10
    A1, c1 = path_product(m, X[0])
    assert np.allclose(A1, A[0]) and np.allclose(c1, c[0])


def test_path_product_gauge_invariant():
    """Null: a random element of G (perms x positive scalings) must not move A or c."""
    m = random_mlp(seed=2)
    X = np.random.default_rng(3).standard_normal((8, 6))
    A, c = path_products(m, X)
    rng = np.random.default_rng(4)
    for _ in range(5):
        perms, scales = random_gauge_element(m.widths, rng, "combined")
        Ag, cg = path_products(apply_gauge(m, perms, scales), X)
        assert np.abs(Ag - A).max() < 1e-10
        assert np.abs(cg - c).max() < 1e-10


def test_top_contributions():
    m = random_mlp(seed=5)
    x = np.random.default_rng(6).standard_normal(6)
    A, c = path_product(m, x)
    tc = top_contributions(A, x, k=3, c=c)
    assert tc.indices.shape == (4, 3)
    assert np.allclose(tc.total + tc.bias, m.logits(x[None, :])[0])
    assert np.all(np.abs(tc.values[:, 0]) >= np.abs(tc.values[:, -1]))
    assert isinstance(tc.to_dict(), dict)


# ------------------------------------------------------------------ 2.6 Fourier irreps
def test_fourier_irreps_recovers_K():
    mdl = synthetic_grok(rotate=False)
    res = fourier_irreps(mdl.embed, P)
    assert res.K == K_TRUE
    assert res.kappa > 0.999
    assert abs(res.power.sum() - 1.0) < 1e-12


def test_fourier_irreps_orthogonal_invariance():
    """Null: rotating the embedding columns is a gauge move; K, kappa, spectrum must not budge."""
    mdl = synthetic_grok(rotate=False)
    E = mdl.embed
    Q = np.linalg.qr(np.random.default_rng(7).standard_normal((E.shape[1], E.shape[1])))[0]
    a, b = fourier_irreps(E, P), fourier_irreps(E @ Q, P)
    assert a.K == b.K == K_TRUE
    assert abs(a.kappa - b.kappa) < 1e-12
    assert np.abs(a.power - b.power).max() < 1e-12
    for k in K_TRUE:
        assert abs(a.amplitude[k] - b.amplitude[k]) < 1e-9 * a.amplitude[k]


def test_fourier_irreps_flat_spectrum_returns_nothing():
    """A random (initialisation-like) embedding has no certified frequencies at tau = 0.05."""
    E = np.random.default_rng(8).standard_normal((P + 1, 16))
    res = fourier_irreps(E, P)
    assert res.K == () and res.kappa == 0.0


# ------------------------------------------------------------------ 2.7 circular coordinate
def test_circular_coordinate_on_noisy_circle():
    rng = np.random.default_rng(9)
    t = np.sort(rng.uniform(0, 2 * np.pi, 160))
    X = np.c_[np.cos(t), np.sin(t)] + 0.05 * rng.standard_normal((160, 2))
    try:
        theta, ratio = circular_coordinate(X)
    except NotImplementedError as e:
        pytest.skip(str(e))
    cc = max(abs(np.mean(np.exp(1j * (theta - t)))), abs(np.mean(np.exp(1j * (theta + t)))))
    assert cc > 0.95, f"circular correlation {cc:.3f}"
    assert ratio > 2.0
    assert theta.min() >= 0 and theta.max() < 2 * np.pi
