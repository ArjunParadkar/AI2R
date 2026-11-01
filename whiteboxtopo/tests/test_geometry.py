"""whiteboxtopo.geometry -- SPEC 2.11 / 6.6.

Known-answer cases and nulls: a unit circle embedded in R^3 (principal
curvature must be 1), a flat plane (the null -- curvature must be ~0), the
finite-difference Jacobian on a constant-gate region (the anchor for the exact
path product) and the ReLU-MLP gauge group (Lip must be exactly invariant).
"""
import numpy as np
import pytest

from whiteboxtopo.geometry import (tangent_pca, curvature, CurvatureResult, lipschitz_field, path_products,
                                   failure_predictor, perturbation_failure_radius, softmax_margin_baseline,
                                   spearman_test)
from whiteboxtopo.invariants.model import ReluMLP, apply_gauge, random_gauge_element


def circle3d(n=300, r=1.0, noise=1e-4, seed=0):
    """Unit circle in the xy-plane of R^3 with tiny isotropic noise."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    return np.c_[r * np.cos(t), r * np.sin(t), np.zeros(n)] + noise * rng.standard_normal((n, 3))


def plane3d(n=300, noise=1e-4, seed=1):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-1, 1, (n, 2)), np.zeros(n)] + noise * rng.standard_normal((n, 3))


def sphere3d(n=500, r=2.0, noise=1e-4, seed=2):
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n, 3))
    return r * P / np.linalg.norm(P, axis=1, keepdims=True) + noise * rng.standard_normal((n, 3))


def toy_mlp(seed=0, n_in=3, widths=(8, 6), n_out=3):
    rng = np.random.default_rng(seed)
    dims = [n_in] + list(widths) + [n_out]
    return ReluMLP([rng.normal(size=(dims[i], dims[i + 1])) for i in range(len(dims) - 1)],
                   [rng.normal(size=dims[i + 1]) for i in range(len(dims) - 1)])


def hand_mlp():
    """Hand-built 2 -> 3 -> 2 ReLU MLP with a large-margin interior region around the origin+1."""
    W1 = np.array([[1.0, 0.5, -1.0], [0.0, 1.0, 0.5]])
    b1 = np.array([1.0, 1.0, 2.0])
    W2 = np.array([[1.0, -2.0], [0.5, 1.0], [-1.0, 0.25]])
    b2 = np.array([0.1, -0.3])
    return ReluMLP([W1, W2], [b1, b2])


# --------------------------------------------------------------------------- tangent / curvature
def test_tangent_pca_finds_a_1d_tangent_on_a_circle():
    X = circle3d()
    bases, explained = tangent_pca(X, k=15)
    assert bases.shape == (300, 3, 1) and explained.shape == (300, 1)
    assert explained.min() > 0.9
    # the tangent at (cos t, sin t, 0) is (-sin t, cos t, 0): |<T, tangent>| ~ 1
    t = np.arctan2(X[:, 1], X[:, 0])
    tan = np.c_[-np.sin(t), np.cos(t), np.zeros(len(t))]
    align = np.abs(np.einsum("nd,nd->n", bases[:, :, 0], tan))
    assert np.median(align) > 0.999
    # orthonormal columns
    G = np.einsum("ndm,ndl->nml", bases, bases)
    assert np.allclose(G, np.eye(1)[None], atol=1e-10)


def test_tangent_pca_finds_a_2d_tangent_on_a_plane():
    bases, explained = tangent_pca(plane3d(), k=15)
    assert bases.shape == (300, 3, 2)
    assert explained.sum(axis=1).min() > 0.9


def test_curvature_of_a_unit_circle_is_one():
    res = curvature(circle3d(n=300, r=1.0, noise=1e-4), k=15)
    assert isinstance(res, CurvatureResult)
    assert res.dim == 1 and res.principal.shape == (300, 1) and res.norm.shape == (300,)
    med = float(np.median(np.abs(res.principal)))
    assert abs(med - 1.0) < 0.25                       # within 25 % of the true curvature 1/r = 1
    assert float(np.median(res.norm)) == pytest.approx(med, rel=1e-12)
    assert np.median(res.residual) < 0.1
    d = res.to_dict()
    assert d["dim"] == 1 and len(d["norm"]) == 300


def test_curvature_of_a_radius_2_circle_is_one_half():
    res = curvature(circle3d(n=300, r=2.0, noise=1e-4), k=15)
    assert abs(float(np.median(np.abs(res.principal))) - 0.5) < 0.25 * 0.5


def test_curvature_of_a_flat_plane_is_near_zero():
    flat = curvature(plane3d(n=300, noise=1e-4), k=15)
    curved = curvature(circle3d(n=300, r=1.0, noise=1e-4), k=15)
    assert float(np.median(flat.norm)) < 0.05
    assert float(np.median(flat.norm)) < 0.1 * float(np.median(curved.norm))


def test_curvature_sign_convention_is_positive_on_a_sphere():
    res = curvature(sphere3d(n=500, r=2.0), k=20)
    assert res.dim == 2
    assert float(np.median(res.mean_curvature)) > 0     # oriented toward the local neighbour mean
    assert abs(float(np.median(res.norm)) - 0.5) < 0.3   # 1/r = 0.5, loose: coarse sampling


# --------------------------------------------------------------------------- Lipschitz field
def test_lipschitz_equals_the_finite_difference_jacobian_on_a_constant_gate_region():
    m = hand_mlp()
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.3, 0.3, (40, 2))
    res = lipschitz_field(m, X, k=5)
    assert res.A.shape == (40, 2, 2) and res.lip.shape == (40,)
    masks, _, _ = m.forward(X)
    pre_margin = np.min(np.abs(X @ m.Ws[0] + m.bs[0]), axis=1)
    i = int(np.argmax(pre_margin))                      # deepest inside its gate region
    h = 1e-6
    x = X[i]
    J = np.stack([(m.logits((x + h * np.eye(2)[j])[None])[0] - m.logits((x - h * np.eye(2)[j])[None])[0]) / (2 * h)
                  for j in range(2)], axis=1)
    A = res.A[i]
    assert np.linalg.norm(J - A) / np.linalg.norm(A) < 1e-6
    assert abs(np.linalg.svd(J, compute_uv=False)[0] - res.lip[i]) / res.lip[i] < 1e-6


def test_lipschitz_is_exactly_gauge_invariant():
    m = toy_mlp(seed=1)
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, (60, 3))
    base = lipschitz_field(m, X, k=5)
    worst = 0.0
    for _ in range(10):
        perms, scales = random_gauge_element(m.widths, rng, group="combined")
        g = apply_gauge(m, perms, scales)
        got = lipschitz_field(g, X, k=5)
        worst = max(worst, float(np.abs(got.lip - base.lip).max() / base.lip.max()),
                    float(np.abs(got.A - base.A).max() / np.abs(base.A).max()))
    assert worst < 1e-9


def test_lipschitz_variation_is_zero_inside_one_gate_region():
    m = hand_mlp()
    rng = np.random.default_rng(3)
    X = rng.uniform(-0.01, 0.01, (20, 2))               # all in one region: A is constant
    res = lipschitz_field(m, X, k=4)
    assert res.variation.max() < 1e-9
    assert res.to_dict()["n"] == 20


def test_path_products_match_lipschitz_field():
    m = toy_mlp(seed=4)
    X = np.random.default_rng(5).uniform(-1, 1, (25, 3))
    assert np.array_equal(path_products(m, X), lipschitz_field(m, X, k=5).A)


# --------------------------------------------------------------------------- P5
def test_failure_predictor_is_the_product_of_its_components():
    m = toy_mlp(seed=6)
    X = np.random.default_rng(7).uniform(-1, 1, (80, 3))
    fp = failure_predictor(m, X, k=15)
    assert fp.rho.shape == (80,)
    assert np.allclose(fp.rho, fp.curvature_norm * fp.lip)
    assert np.all(fp.rho >= 0)
    d = fp.to_dict()
    assert len(d["rho"]) == 80 and d["hidden_dim"] == fp.dim


def test_perturbation_failure_radius_is_monotone_and_vectorised():
    m = toy_mlp(seed=8)
    X = np.random.default_rng(9).uniform(-1, 1, (60, 3))
    radii = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
    r = perturbation_failure_radius(m, X, None, radii=radii, n_dirs=8, seed=0)
    assert r.shape == (60,)
    assert set(np.unique(r[np.isfinite(r)])).issubset(set(radii))
    assert np.array_equal(r, perturbation_failure_radius(m, X, None, radii=radii, n_dirs=8, seed=0))
    # more directions can only find a flip earlier
    r16 = perturbation_failure_radius(m, X, None, radii=radii, n_dirs=64, seed=0)
    assert np.all(np.nan_to_num(r16, posinf=1e9) <= np.nan_to_num(r, posinf=1e9) + 1e-12)
    # a bigger grid finds at least as many failures
    r_big = perturbation_failure_radius(m, X, None, radii=radii + (1.0, 2.0), n_dirs=8, seed=0)
    assert np.isfinite(r_big).sum() >= np.isfinite(r).sum()


def test_perturbation_failure_radius_marks_misclassified_inputs_with_y():
    m = toy_mlp(seed=10)
    X = np.random.default_rng(11).uniform(-1, 1, (40, 3))
    pred = np.argmax(m.logits(X), axis=1)
    y = pred.copy(); y[:5] = (y[:5] + 1) % 3
    r = perturbation_failure_radius(m, X, y, n_dirs=4, seed=0)
    assert np.all(np.isnan(r[:5]))
    assert not np.any(np.isnan(r[5:]))


def test_softmax_margin_baseline_is_the_top1_top2_logit_gap():
    m = toy_mlp(seed=12)
    X = np.random.default_rng(13).uniform(-1, 1, (30, 3))
    margin = softmax_margin_baseline(m, X)
    Z = m.logits(X)
    S = np.sort(Z, axis=1)
    assert margin.shape == (30,)
    assert np.allclose(margin, S[:, -1] - S[:, -2])
    assert np.all(margin >= 0)


def test_spearman_test_sign_convention_and_pass_flag():
    n = 200
    rng = np.random.default_rng(14)
    rho = rng.uniform(0, 1, n)
    radius = 1.0 - rho + 0.05 * rng.standard_normal(n)      # perfectly anti-correlated by construction
    out = spearman_test(rho, radius)
    assert out["spearman_rho"] < -0.5 and out["p_value"] < 0.01
    assert out["pass_spec"] is True and out["n"] == n
    # the shuffled null must not pass
    assert spearman_test(rng.permutation(rho), radius)["pass_spec"] is False
    # the wrong sign must not pass either
    assert spearman_test(-rho, radius)["pass_spec"] is False
    assert spearman_test([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])["pass_spec"] is False


def test_spearman_test_handles_inf_and_nan_radii():
    m = toy_mlp(seed=15)
    X = np.random.default_rng(16).uniform(-1, 1, (50, 3))
    fp = failure_predictor(m, X, k=15)
    r = perturbation_failure_radius(m, X, None, n_dirs=8, seed=0)
    out = spearman_test(fp.rho, r)
    assert np.isfinite(out["spearman_rho"]) and out["n"] == 50
    assert isinstance(out["pass_spec"], bool)
