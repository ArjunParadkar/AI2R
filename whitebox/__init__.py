"""WHITEBOX — topological interpretability of neural-network training.

Layering (each level only depends on the one below it):

    core     persistence computation (point cloud -> persistence diagrams)
    metrics  the instruments: SV, TC, Betti, FPL, BiasTop  (claim-independent)
    signals  calibrated answers built on the instruments     (Phase 2+)
    viz      plotting of films and diagrams

Public API is re-exported here. Anything not listed here is internal.
"""
from .core.reduce import fit_reducer, Reducer
from .core.persistence import compute_diagrams, Diagrams, available_backends
from .core.film import Film, film_from_activations
from .metrics.sv import structural_velocity
from .metrics.tc import topological_complexity
from .metrics.betti import betti_at_scale, betti_curve, median_scale
from .metrics.fpl import top_k_lifetime, feature_persistence_runs
from .metrics.biastop import bias_top, bias_top_permutation_test

__version__ = "0.0.1"

__all__ = [
    "fit_reducer", "Reducer",
    "compute_diagrams", "Diagrams", "available_backends",
    "Film", "film_from_activations",
    "structural_velocity", "topological_complexity",
    "betti_at_scale", "betti_curve", "median_scale",
    "top_k_lifetime", "feature_persistence_runs",
    "bias_top", "bias_top_permutation_test",
]
