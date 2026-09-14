"""whiteboxtopo.geometry -- curvature and Lipschitz structure (SPEC 2.11, 6.6).

    from whiteboxtopo.geometry import curvature, lipschitz_field, failure_predictor
    T, expl = tangent_pca(cloud, k=15)          # local tangent spaces
    cur = curvature(cloud, k=15)                # eigenvalues of the fitted Q_h
    lip = lipschitz_field(model, X)             # exact path-product Lipschitz field
    fp  = failure_predictor(model, X)           # rho = ||Q_h|| * Lip
    r   = perturbation_failure_radius(model, X, y)
    spearman_test(fp.rho, r)["pass_spec"]       # SPEC 6.6 verdict

Instruments, not signals. ``rho`` is an *ordering* of inputs by predicted
fragility with no calibration attached; the only claim any of these functions
supports is the one ``spearman_test`` returns, against the failure radii you
actually measured and against ``softmax_margin_baseline``. Each docstring
states the null it was validated against and its failure boundary.
"""
from .tangent import tangent_pca, curvature, CurvatureResult, EXPLAINED_TARGET, MAX_INTRINSIC_DIM
from .lipschitz import lipschitz_field, path_products, LipResult
from .failure import (failure_predictor, FailurePredictor, perturbation_failure_radius,
                      softmax_margin_baseline, spearman_test, SPEC_RHO_MAX, SPEC_P_MAX)

__all__ = ["tangent_pca", "curvature", "CurvatureResult", "EXPLAINED_TARGET", "MAX_INTRINSIC_DIM",
           "lipschitz_field", "path_products", "LipResult",
           "failure_predictor", "FailurePredictor", "perturbation_failure_radius",
           "softmax_margin_baseline", "spearman_test", "SPEC_RHO_MAX", "SPEC_P_MAX"]
