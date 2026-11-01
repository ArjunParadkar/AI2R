"""E4 legibility core (whitebox-v2). Graph measures, discovery pipeline, legibility_profile.

References: Adolfi, Vilas & Wareham (ICLR 2025, arXiv 2410.08025) — complexity of circuit
discovery; Golechha, Chaudhary, Velja, Abate & Schoots (arXiv 2502.02470) — clusterability
loss (re-implemented in cluster_loss.py, theirs not ours); Syed, Rager & Conmy (2023) —
edge attribution patching.
"""
from .graph import modularity_q, modularity_sweep, treewidth_bound, sparsity, weight_sparsity, shuffled_q_null
from .discovery import discover_circuit, faithfulness, jaccard
from .legibility import legibility_profile
from .cluster_loss import clusterability, clusterability_loss, block_mask_
