"""E2 — gauge-invariant observables. See ../SPEC.md.

Reusable core: ``check_invariance(metric_fn, model, n_reps, group)`` audits any
metric by sampling the gauge orbit of a ReLU MLP (E1's ``e1lib`` models/gauge).
"""
import os
import sys

_E1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "E1")
if _E1 not in sys.path:
    sys.path.insert(0, _E1)

from .core import (check_invariance, audit_metrics, sample_orbit, InvarianceReport,  # noqa: E402
                   GROUPS, param_distance, cv_of, agreement_of)
from .quantities import all_quantities, scaling_sweep, QUANTITY_INFO  # noqa: E402

__all__ = ["check_invariance", "audit_metrics", "sample_orbit", "InvarianceReport", "GROUPS",
           "param_distance", "cv_of", "agreement_of", "all_quantities", "scaling_sweep",
           "QUANTITY_INFO"]
