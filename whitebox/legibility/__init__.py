"""whitebox.legibility — structural measures of a model's attribution graph (instrument), and an
explicitly UNCALIBRATED ``legibility_score`` (raises).

    from whitebox.legibility import legibility_profile, NodeMLP
    prof = legibility_profile(model, (x, targets), discovery_batches=[(x1, t1), (x2, t2)])
    prof["Q10"], prof["Q10_excess"], prof["tw10_norm"], prof["sparsity"], prof["S"], prof["jaccard"]

Ported from whitebox-v2 E4 (``e4lib``). Framework: Adolfi, Vilas & Wareham (ICLR 2025) on the
complexity of circuit discovery; Golechha et al. (arXiv 2502.02470) clusterability; Syed, Rager
& Conmy (2023) attribution patching. E4's verdict (87 small models): none of these measures
predicts discovery success within an architecture — the profile is a measurement, not a prediction.
Scope: small node-instrumented models. Optional dependencies: networkx, python-igraph, leidenalg, torch.
"""
from .graph import (symmetrise, threshold_density, modularity_q, modularity_sweep, treewidth_bound, shuffled_q_null,
                    sparsity, weight_sparsity, DENSITIES)
from .discovery import discover_circuit, faithfulness, jaccard, logprob_metric
from .models import NodeMLP, node_importance
from .profile import legibility_profile, legibility_score, NotCalibratedError, E4_CALIBRATION_ATTEMPT

__all__ = ["symmetrise", "threshold_density", "modularity_q", "modularity_sweep", "treewidth_bound", "shuffled_q_null",
           "sparsity", "weight_sparsity", "DENSITIES", "discover_circuit", "faithfulness", "jaccard", "logprob_metric",
           "NodeMLP", "node_importance", "legibility_profile", "legibility_score", "NotCalibratedError",
           "E4_CALIBRATION_ATTEMPT"]
