"""whiteboxtopo.invariants — gauge-invariant observables for ReLU MLPs, and an audit tool.

An interpretability quantity is a statement about the *function* a network
computes only if it is constant on the network's gauge orbit. For a ReLU MLP
the known gauge group G is hidden-unit permutations x per-unit positive
scalings. This module ships (1) the quantities that E2 (whitebox-v2,
2026-08-30) measured to be exactly invariant under G, each re-checked by its
own orbit-sampling unit test, and (2) ``check_invariance``, which runs the
same audit on any metric you write.

    from whiteboxtopo.invariants import check_invariance, standardized_tc
    rep = check_invariance(lambda m: my_metric(m, X), model, n_reps=100)
    rep.verdict      # 'exact' | 'approximate' | 'not_invariant'
    rep.broken_by    # 'none' | 'perm' | 'scale' | 'perm+scale' | 'combined-only'

Verdict thresholds (E2 pre-registered): exact CV < 1e-6, not_invariant
CV > 0.05. Scope: ReLU MLPs (numpy ``.Ws/.bs`` containers or a torch
``nn.Sequential`` of Linear/ReLU). Transformers, LayerNorm, gated units,
convolutions have other symmetry groups; an ``exact`` verdict here says
nothing about those.

Instrument, not signal: the audit measures dispersion under G; it makes no
claim about what an invariant quantity *means*. See VALIDATION.md.
"""
from .model import ReluMLP, as_relu_mlp, apply_gauge, sample_orbit, random_gauge_element, param_distance, GROUPS
from .core import (check_invariance, audit_metrics, InvarianceReport, cv_of, agreement_of, verdict_for,
                   EXACT_CV, BROKEN_CV, FLOAT_FLOOR)
from .metrics import (logits, input_jacobian, weight_product, standardized_tc, standardize_units,
                      sorted_gradxact, sorted_ablation, gate_multiset, gate_bits, gradxact, ablation,
                      invariant_quantities, INVARIANTS)

__all__ = ["ReluMLP", "as_relu_mlp", "apply_gauge", "sample_orbit", "random_gauge_element", "param_distance", "GROUPS",
           "check_invariance", "audit_metrics", "InvarianceReport", "cv_of", "agreement_of", "verdict_for",
           "EXACT_CV", "BROKEN_CV", "FLOAT_FLOOR",
           "logits", "input_jacobian", "weight_product", "standardized_tc", "standardize_units",
           "sorted_gradxact", "sorted_ablation", "gate_multiset", "gate_bits", "gradxact", "ablation",
           "invariant_quantities", "INVARIANTS"]
