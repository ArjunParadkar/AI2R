"""WHITEBOX — topological interpretability of neural-network training.

Layering (each level only depends on the one below it):

    core     persistence computation (point cloud -> persistence diagrams)
    metrics  the instruments: SV, TC, Betti, FPL, BiasTop  (claim-independent)
    signals  calibrated answers built on the instruments     (Phase 2+)
    viz      plotting of films and diagrams
    invariants  (v2) gauge-invariant observables of ReLU MLPs + check_invariance audit
    legibility  (v2) attribution-graph structure profile; legibility_score is NOT calibrated
    darkmatter  (v2) SAE reconstruction-error structure: MP/TW spectral null + topology null
    gluing      (v2) explanation-obstruction instrument; NO "distinct mechanisms" verdict

Public API is re-exported here. Anything not listed here is internal.
"""
from .core.reduce import fit_reducer, Reducer
from .core.persistence import compute_diagrams, Diagrams, available_backends
from .core.film import Film, FilmRecorder, film_from_activations, topological_film
from .core.bootstrap import bootstrap_metric, BootEstimate
from .metrics.sv import structural_velocity
from .metrics.tc import topological_complexity
from .metrics.betti import betti_at_scale, betti_curve, median_scale
from .metrics.fpl import top_k_lifetime, feature_persistence_runs
from .metrics.biastop import bias_top, bias_top_permutation_test
from .signals import memorization_signal, memorization_features, MemorizationSignal
from .signals import representation_correspondence, correspondence, rtd, Correspondence
from . import invariants, legibility, darkmatter, gluing
from .invariants import check_invariance, InvarianceReport
from .legibility import legibility_profile, legibility_score
from .darkmatter import residual_structure, DarkMatterReport
from .gluing import explanation_obstruction, GluingReport

__version__ = "0.1.0"

__all__ = [
    "fit_reducer", "Reducer",
    "compute_diagrams", "Diagrams", "available_backends",
    "Film", "FilmRecorder", "film_from_activations", "topological_film",
    "bootstrap_metric", "BootEstimate",
    "structural_velocity", "topological_complexity",
    "betti_at_scale", "betti_curve", "median_scale",
    "top_k_lifetime", "feature_persistence_runs",
    "bias_top", "bias_top_permutation_test",
    "memorization_signal", "memorization_features", "MemorizationSignal",
    "representation_correspondence", "correspondence", "rtd", "Correspondence",
    "invariants", "legibility", "darkmatter", "gluing", "check_invariance", "InvarianceReport", "legibility_profile",
    "legibility_score", "residual_structure", "DarkMatterReport", "explanation_obstruction", "GluingReport",
]
