"""S2 tests: weight product -> invariant; per-unit activation -> artifact; singular values ->
invariant (perm) / covariant (scale); gauge alignment repairs a unit-indexed metric."""
import numpy as np
from whiteboxtopo.invariants import ReluMLP, check_invariance, weight_product, INVARIANT_CV


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    m = ReluMLP([rng.normal(size=(3, 8)), rng.normal(size=(8, 1))], [rng.normal(size=8), rng.normal(size=1)])
    return m, rng.uniform(-1, 2, size=(64, 3))


def test_weight_product_invariant():
    m, X = _setup()
    r = check_invariance(weight_product, m, n=40)
    assert r.verdict == "invariant" and max(r.median_cv.values()) < 1e-10 and abs(r.scaling_exponent) < 1e-6


def test_per_unit_activation_artifact():
    m, X = _setup()
    r = check_invariance(lambda mm: mm.hidden_acts(X)[0].mean(0), m, n=40)
    assert r.verdict == "artifact" and r.median_cv["perm"] > INVARIANT_CV


def test_singular_values_covariant_with_exponent_one():
    m, X = _setup()
    r = check_invariance(lambda mm: np.linalg.svd(mm.Ws[0], compute_uv=False), m, n=40)
    assert r.verdict == "covariant" and r.median_cv["perm"] < 1e-10 and abs(r.scaling_exponent - 1.0) < 1e-6 and r.scaling_r2 > 0.999


def test_gauge_alignment_makes_unit_metric_invariant():
    m, X = _setup()
    r = check_invariance(lambda mm: mm.hidden_acts(X)[0].mean(0), m, n=40, align=True, unit_key=0, scale_power=1.0)
    assert r.verdict == "artifact" and r.verdict_aligned == "invariant"


def test_hungarian_alignment_repairs_permutation_only():
    m, X = _setup()
    r = check_invariance(lambda mm: mm.hidden_acts(X)[0].mean(0)[:, None], m, n=40, kinds=("perm",), align="hungarian")
    assert r.median_cv["perm"] > INVARIANT_CV and r.cv_aligned["perm"].max() < 1e-10


def test_row_has_all_columns():
    m, X = _setup()
    row = check_invariance(weight_product, m, n=10).row()
    for k in ("cv_perm", "cv_scale", "cv_combined", "verdict", "scaling_exponent"):
        assert k in row
