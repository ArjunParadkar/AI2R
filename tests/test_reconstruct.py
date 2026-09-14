"""whiteboxtopo.reconstruct -- Block R rebuilds (R1, R2, R3) and the coverage gap (R4)."""
import numpy as np
import pytest

from whiteboxtopo.invariants.extractors import fourier_irreps
from whiteboxtopo.reconstruct import (from_irreps, fit_tangent_dynamics, from_attractor,
                                      from_circuits, residual, coverage_gap)
from test_extractors import P, K_TRUE, synthetic_grok, random_mlp


def _inventory(mdl, p=P):
    """The certified inventory: frequencies + per-frequency weights, nothing else."""
    emb = fourier_irreps(mdl.embed, p)
    read = fourier_irreps(mdl.unembed.T, p)
    return emb, read


# ------------------------------------------------------------------ R1
def test_from_irreps_rebuilds_modular_addition():
    mdl = synthetic_grok()
    emb, read = _inventory(mdl)
    assert emb.K == K_TRUE == read.K
    rec = from_irreps(emb.K, emb.amplitude, P, readout=read.amplitude)

    a, b, Lh = rec.all_pairs()
    L = mdl.forward(np.stack([a, b, np.full_like(a, P)], 1))
    assert (Lh.argmax(1) == (a + b) % P).mean() == 1.0
    assert (L.argmax(1) == (a + b) % P).mean() == 1.0
    z1, z2 = L - L.mean(), Lh - Lh.mean()
    corr = float((z1.ravel() @ z2.ravel()) / (np.linalg.norm(z1) * np.linalg.norm(z2)))
    assert corr > 0.98, corr
    assert rec.to_dict()["K"] == list(K_TRUE)


def test_from_irreps_uses_only_the_inventory():
    """Two models differing only by a gauge rotation give the identical rebuild."""
    m0, m1 = synthetic_grok(seed=0, rotate=False), synthetic_grok(seed=3, rotate=True)
    rec0 = from_irreps(_inventory(m0)[0].K, _inventory(m0)[1].amplitude, P)
    rec1 = from_irreps(_inventory(m1)[0].K, _inventory(m1)[1].amplitude, P)
    assert np.abs(rec0.w - rec1.w).max() < 1e-9
    assert np.abs(rec0.readout - rec1.readout).max() < 1e-9


def test_from_irreps_rejects_empty_K():
    with pytest.raises(ValueError):
        from_irreps((), {}, P)


# ------------------------------------------------------------------ R2
def _true_g(theta, u):
    return 0.8 + 0.5 * u + 0.3 * np.sin(theta)


def test_from_attractor_integrates_like_the_original():
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 2000)
    u = rng.uniform(-1, 1, 2000)
    dyn = fit_tangent_dynamics(th, u, _true_g(th, u) + 0.01 * rng.standard_normal(2000))
    assert dyn.r2 > 0.99
    rec = from_attractor(th, dyn)

    dt, n = 0.01, 100
    useq = 0.4 * np.ones(n)
    got = rec.integrate(0.3, useq, dt)
    # ground truth: RK4 on the true tangent field
    true = [0.3]
    for uu in useq:
        x = true[-1]
        k1 = _true_g(x, uu); k2 = _true_g(x + .5 * dt * k1, uu)
        k3 = _true_g(x + .5 * dt * k2, uu); k4 = _true_g(x + dt * k3, uu)
        true.append((x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)) % (2 * np.pi))
    true = np.array(true)
    err = np.abs(np.angle(np.exp(1j * (got - true))))
    travel = abs(np.angle(np.exp(1j * (true[-1] - true[0]))))
    assert err.max() < 0.1 * travel, (err.max(), travel)
    assert set(rec.to_dict()) == {"dynamics", "n_states"}


def test_fit_tangent_dynamics_autonomous_and_nulls():
    rng = np.random.default_rng(1)
    th = rng.uniform(0, 2 * np.pi, 800)
    dyn = fit_tangent_dynamics(th, None, 1.0 + 0.4 * np.cos(th))
    assert dyn.r2 > 0.999
    assert np.abs(dyn.predict(th) - (1.0 + 0.4 * np.cos(th))).max() < 1e-6
    noise = fit_tangent_dynamics(th, None, rng.standard_normal(800))   # null: no tangent structure
    assert noise.r2 < 0.05


# ------------------------------------------------------------------ R3
def test_from_circuits_mean_ablation():
    m = random_mlp(seed=11)
    X = np.random.default_rng(12).standard_normal((64, 6))
    full = from_circuits(m, [np.ones(w, dtype=bool) for w in m.widths], X)
    assert np.abs(full(X) - m.logits(X)).max() < 1e-12          # null: keeping everything is the model
    empty = from_circuits(m, [np.zeros(w, dtype=bool) for w in m.widths], X)
    out = empty(X)
    assert np.abs(out - out[0]).max() < 1e-12                   # everything ablated -> constant
    half = from_circuits(m, {0: range(8), 1: range(6)}, X)
    assert half(X).shape == (64, 4)
    assert 0 < np.abs(half(X) - m.logits(X)).max()
    assert half.to_dict()["n_kept"] == [8, 6]
    with pytest.raises(ValueError):
        from_circuits(m, [np.ones(w, dtype=bool) for w in m.widths], X, ablation="median")


# ------------------------------------------------------------------ R4
def test_residual_shapes():
    A = np.zeros((10, 3)); B = np.ones((10, 3))
    assert residual(A, B).shape == (10, 3) and residual(A, B).max() == -1.0
    with pytest.raises(ValueError):
        residual(A, np.ones((10, 4)))


def test_coverage_gap_gaussian_is_zero():
    R = np.random.default_rng(0).standard_normal((600, 60))
    cg = coverage_gap(R)
    assert cg.gamma_cov < 0.02, cg.to_dict()
    assert cg.n_outliers == 0
    assert 0.8 < cg.sigma2 < 1.2
    assert cg.gamma_null_gaussian < 0.02 and cg.gamma_null_shuffle < 0.02


def test_coverage_gap_detects_planted_structure():
    rng = np.random.default_rng(1)
    n, d, k = 600, 60, 3
    V = np.linalg.qr(rng.standard_normal((d, k)))[0]
    R = rng.standard_normal((n, d)) + (4.0 * rng.standard_normal((n, k))) @ V.T
    cg = coverage_gap(R)
    assert cg.n_outliers >= k
    assert cg.gamma_cov > 0.3, cg.to_dict()
    # an isotropically random planted subspace also heterogenises the marginals, so the
    # matched-marginal Gaussian null is NOT zero here; the gap must clear it by a margin
    assert cg.gamma_excess > 0.1, cg.to_dict()
    assert isinstance(cg.to_dict()["gamma_cov"], float)


def test_coverage_gap_nulls_are_flat_for_equal_loadings():
    """Planted structure with uniform per-coordinate loading: both nulls go to ~0."""
    rng = np.random.default_rng(2)
    n, d = 600, 60
    j = np.arange(d)
    V = np.sqrt(2.0 / d) * np.stack([np.cos(2 * np.pi * j / d), np.sin(2 * np.pi * j / d),
                                     np.cos(4 * np.pi * j / d), np.sin(4 * np.pi * j / d)], 1)
    R = rng.standard_normal((n, d)) + (4.0 * rng.standard_normal((n, 4))) @ V.T
    cg = coverage_gap(R)
    assert cg.gamma_cov > 0.3
    assert cg.gamma_null_gaussian < 0.05 and cg.gamma_null_shuffle < 0.05, cg.to_dict()
