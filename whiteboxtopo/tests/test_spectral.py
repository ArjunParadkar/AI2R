"""whiteboxtopo.spectral -- SPEC 2.10 / 6.5 / 6.8.

Each test is the null or the known-answer case the corresponding docstring
claims: a noisy circle (the diffusion embedding must recover the circular
coordinate), circle vs radius-3 points (P4 must separate them, AUC > 0.9),
and a blob splitting into two blobs (the small-eigenvalue count must follow
the component count, not the point count).
"""
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from conftest import circle, blob
from whiteboxtopo.spectral import (laplacian, diffusion_coords, nystrom, kernel_params, default_eps,
                                   out_of_manifold, random_projection_null, OOMResult,
                                   eigenflow, EigenflowResult)


def ring(n=60, r=3.0, seed=7):
    """Points at radius r -- out-of-distribution for the unit circle."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    return np.c_[r * np.cos(t), r * np.sin(t)]


def circular_agreement(theta_hat, theta_true):
    """Resultant length after the best alignment (rotation x reflection): 1 = perfect."""
    return max(abs(np.mean(np.exp(1j * (theta_hat - s * theta_true)))) for s in (1.0, -1.0))


# --------------------------------------------------------------------------- laplacian
def test_laplacian_is_a_normalized_laplacian():
    X = circle(200)
    L, W = laplacian(X, k=15)
    assert sp.issparse(L) and sp.issparse(W)
    assert L.shape == W.shape == (200, 200)
    assert abs((W - W.T)).max() == 0.0                      # symmetrised
    assert W.diagonal().max() == 0.0                        # no self-loops
    assert 0.0 < W.data.max() <= 1.0
    lam = np.linalg.eigvalsh(L.toarray())
    assert lam[0] == pytest.approx(0.0, abs=1e-10)          # connected graph: one trivial eigenvalue
    assert lam[-1] <= 2.0 + 1e-10
    assert lam[1] > 1e-6


def test_default_eps_is_the_median_squared_knn_distance():
    X = circle(200)
    eps = default_eps(X, k=15)
    L1, _ = laplacian(X, k=15)
    L2, _ = laplacian(X, k=15, eps=eps)
    assert abs(L1 - L2).max() == pytest.approx(0.0, abs=1e-15)
    assert eps > 0


# --------------------------------------------------------------------------- diffusion coords
def test_circle_diffusion_coords_trace_a_closed_loop():
    X = circle(200, noise=0.02, seed=0)
    L, _ = laplacian(X, k=15)
    Psi, lam, phi = diffusion_coords(L, t=1, n_eigs=10)
    assert Psi.shape == (200, 10) and lam.shape == (10,) and phi.shape == (200, 10)
    assert np.all(np.diff(lam) >= -1e-12)                   # ascending, trivial one dropped
    assert lam[0] > 1e-8
    theta_hat = np.arctan2(Psi[:, 1], Psi[:, 0])
    theta_true = np.arctan2(X[:, 1], X[:, 0])
    assert circular_agreement(theta_hat, theta_true) > 0.95
    # a closed loop: the first two coordinates have roughly constant radius
    # (0.3, not 0.05: conftest's circle samples angles at random, so the local
    # sampling density -- and with it the diffusion-map radius -- fluctuates)
    r = np.hypot(Psi[:, 0], Psi[:, 1])
    assert r.std() / r.mean() < 0.3
    assert r.min() > 0.3 * r.max()
    # and every angular sector is occupied (no collapsed arc)
    counts, _ = np.histogram(theta_hat, bins=12, range=(-np.pi, np.pi))
    assert counts.min() > 0


def test_diffusion_coords_are_deterministic():
    X = circle(150, seed=3)
    L, _ = laplacian(X, k=12)
    a = diffusion_coords(L, t=2, n_eigs=5)[0]
    b = diffusion_coords(L, t=2, n_eigs=5)[0]
    assert np.array_equal(a, b)


def test_nystrom_reproduces_the_eigenvectors_on_the_reference_points():
    X = circle(200)
    L, _ = laplacian(X, k=15)
    _, lam, phi = diffusion_coords(L, t=1, n_eigs=6)
    phi_new = nystrom(X, kernel_params(X, k=15), X, phi, lam)
    assert phi_new.shape == phi.shape
    for i in range(3):
        assert abs(np.corrcoef(phi_new[:, i], phi[:, i])[0, 1]) > 0.99


def test_nystrom_sends_far_points_toward_the_origin():
    X = circle(200)
    L, _ = laplacian(X, k=15)
    _, lam, phi = diffusion_coords(L, t=1, n_eigs=6)
    p = kernel_params(X, k=15)
    near = np.linalg.norm(nystrom(X, p, X[:20], phi, lam), axis=1)
    far = np.linalg.norm(nystrom(X, p, ring(20, r=5.0), phi, lam), axis=1)
    assert far.max() < near.min()


# --------------------------------------------------------------------------- P4
def test_out_of_manifold_separates_radius_3_from_the_circle():
    X, ood = circle(200), ring(60, r=3.0)
    res = out_of_manifold(X, ood, k=15, t=1, n_eigs=10, seed=0)
    assert isinstance(res, OOMResult)
    assert res.d_M.shape == (60,) and res.flags.shape == (60,)
    y = np.r_[np.zeros(res.d_M_in.size), np.ones(res.d_M.size)]
    auc = roc_auc_score(y, np.r_[res.d_M_in, res.d_M])
    assert auc > 0.9                                        # SPEC 6.5 pass condition
    assert res.flags.mean() > 0.9                           # OOD flagged
    held_flags = res.d_M_in > res.threshold
    assert held_flags.mean() <= 0.05                        # held-out in-distribution not flagged
    d = res.to_dict()
    assert d["embedding"] == "diffusion" and d["n_flagged"] == int(res.flags.sum())
    assert set(["d_M", "threshold", "flags"]).issubset(d)


def test_out_of_manifold_is_deterministic_given_seed():
    X, ood = circle(200), ring(30, r=3.0)
    a = out_of_manifold(X, ood, seed=1)
    b = out_of_manifold(X, ood, seed=1)
    assert np.array_equal(a.d_M, b.d_M) and a.threshold == b.threshold


def test_random_projection_null_returns_the_same_shape():
    X, ood = circle(200), ring(60, r=3.0)
    res = out_of_manifold(X, ood, seed=0)
    null = random_projection_null(X, ood, d=2, seed=0)
    assert isinstance(null, OOMResult)
    assert null.d_M.shape == res.d_M.shape
    assert null.flags.shape == res.flags.shape
    assert null.d_M_in.shape == res.d_M_in.shape
    assert null.embedding == "random_projection"
    assert np.isfinite(null.threshold)
    assert set(null.to_dict()) == set(res.to_dict())


# --------------------------------------------------------------------------- P7
def test_eigenflow_tracks_the_component_count():
    """One blob -> two separated blobs: the small-eigenvalue count goes 1 -> 2.

    The threshold is set to 0.01 rather than the 0.1 default: a single finite
    Gaussian blob with k=15 already has several non-trivial eigenvalues below
    0.1, so 0.1 counts sampling noise as components here (documented failure
    boundary of the count).
    """
    base = blob(150, 2, seed=0)
    clouds = []
    for s in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0):
        A = base.copy(); A[75:, 0] += s; clouds.append(A)
    res = eigenflow(clouds, k=15, n_eigs=8, threshold=0.01)
    assert isinstance(res, EigenflowResult)
    assert res.eigenvalues.shape == (6, 8) and res.gap.shape == (6,)
    assert res.n_small[0] == 1
    assert res.n_small[-1] == 2
    assert np.all(np.diff(res.n_small) >= 0)                # monotone as the blobs separate
    assert res.gap[-1] < res.gap[0]                         # the gap closes as a component splits off
    assert np.allclose(res.eigenvalues[:, 0], 0.0, atol=1e-8)
    d = res.to_dict()
    assert d["n_checkpoints"] == 6 and len(d["gap"]) == 6


def test_crystallization_index_is_the_first_half_final_gap_crossing():
    """Reverse direction: two blobs merging into one, so the gap opens."""
    base = blob(150, 2, seed=0)
    clouds = []
    for s in (16.0, 8.0, 4.0, 2.0, 1.0, 0.0):
        A = base.copy(); A[75:, 0] += s; clouds.append(A)
    res = eigenflow(clouds, k=15, n_eigs=8)
    idx = res.crystallization_index()
    assert 0 <= idx < len(clouds)
    assert res.gap[idx] >= 0.5 * res.gap[-1]
    assert all(g < 0.5 * res.gap[-1] for g in res.gap[:idx])
    assert res.crystallization_index(0.99) >= idx
