"""whiteboxtopo.invariants — gauge-invariant observables for ReLU MLPs, and an audit tool.

An interpretability quantity is a statement about the *function* a network
computes only if it is constant on the network's gauge orbit. For a ReLU MLP
the known gauge group G is hidden-unit permutations x per-unit positive
scalings. This module ships (1) the quantities that E2 (whitebox-v2,
2026-08-30) measured to be exactly invariant under G, each re-checked by its
own orbit-sampling unit test, and (2) ``check_invariance``, which runs the
same audit on any metric you write.

    from whiteboxtopo.invariants import check_invariance
    res = check_invariance(lambda m: my_metric(m, X), model, n=100)      # handbook S2 auditor
    res.verdict          # 'invariant' | 'covariant' | 'artifact'   (median-component CV < 0.01 ...)
    res.median_cv        # {'perm': ..., 'scale': ..., 'combined': ...}
    res.scaling_exponent # k in q(alpha) ~ alpha^k when covariant

``check_invariance_e2`` is the earlier (E2) auditor with its ``exact`` /
``approximate`` / ``not_invariant`` thresholds (1e-6 / 0.05), kept for the
shipped E2 invariant column. Both sample the group in
``whiteboxtopo.symmetry``: ReLU MLPs, the toy transformer, and GPT-2.

Instrument, not signal: the audit measures dispersion under G; it makes no
claim about what an invariant quantity *means*. See VALIDATION.md.
"""
from .model import ReluMLP, as_relu_mlp, apply_gauge, sample_orbit, random_gauge_element, param_distance, GROUPS
from .core import (check_invariance as check_invariance_e2, audit_metrics, InvarianceReport, cv_of, agreement_of, verdict_for,
                   EXACT_CV, BROKEN_CV, FLOAT_FLOOR)
from .check import check_invariance, audit_table, audit_quantities, AuditResult, per_component_cv, global_cv, INVARIANT_CV, COVARIANT_R2
from .extractors import path_product, fourier_irreps, circular_coordinate
from .metrics import (logits, input_jacobian, weight_product, standardized_tc, standardize_units,
                      sorted_gradxact, sorted_ablation, gate_multiset, gate_bits, gradxact, ablation,
                      invariant_quantities, INVARIANTS)

__all__ = ["ReluMLP", "as_relu_mlp", "apply_gauge", "sample_orbit", "random_gauge_element", "param_distance", "GROUPS",
           "check_invariance", "check_invariance_e2", "audit_table", "audit_quantities", "AuditResult", "per_component_cv", "global_cv", "INVARIANT_CV", "COVARIANT_R2", "audit_metrics", "InvarianceReport", "cv_of", "agreement_of", "verdict_for",
           "EXACT_CV", "BROKEN_CV", "FLOAT_FLOOR",
           "logits", "input_jacobian", "weight_product", "standardized_tc", "standardize_units",
           "sorted_gradxact", "sorted_ablation", "gate_multiset", "gate_bits", "gradxact", "ablation",
           "invariant_quantities", "INVARIANTS",
           "path_product", "fourier_irreps", "circular_coordinate"]
