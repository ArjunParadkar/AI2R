"""whitebox.gluing -- does the explanation glue? A model-free measurement
instrument, NOT a validated "distinct mechanisms" detector:

    from whitebox.gluing import explanation_obstruction
    rep = explanation_obstruction(scores, embedding=emb, k=10, k_e=200)
    rep.omega, rep.omega_rel, rep.null_lo, rep.null_hi, rep.p_null
    rep.significant_vs_null   # p_null <= 0.025 -- NOT "distinct mechanisms" by itself
    rep.blocking_pairs        # top frustrated cluster pairs, with bootstrap CIs
    rep.cyclomatic            # the nerve's cyclomatic number (frustration needs a cycle)
    rep.caveat                # the E3 control-failure account -- read it

Ported from whitebox-v2 E3 (2026-08-30, GPT-2 small, IOI vs an induction
control -- see benchmarks/GLUING.md and E3/SPEC.md + E3/FINDINGS.md in the
whitebox-v2 experiment tree). Framework: sheaf-of-explanations (arXiv
2603.17141) and the sheaf/holonomy atlas of "A Gauge Theory of Superposition".
This module takes plain numpy arrays: (n_examples, n_edges) attribution/
explanation scores, plus either a precomputed cover (membership matrix) or an
embedding to build one from -- no transformer_lens, no EAP, no torch
dependency.

**Verdict discipline (mirrors whitebox.darkmatter / whitebox.legibility): NO
scored "distinct mechanisms" detector ships.** E3's own pre-registered
matched single-mechanism control (induction) FAILED its own null-band
criterion (Omega above the shuffle null in 34/40 grid cells, including the
summary config), and IOI showed *less* raw obstruction than the control in
40/40 grid cells -- the opposite of the pre-registered direction. The one
IOI blocking pair that did clear its own null did not survive a second-
embedding stability check. `explanation_obstruction` therefore returns the
measurement (Omega, its null band, p_null, blocking pairs with bootstrap
CIs, the nerve's cyclomatic number) and a `significant_vs_null` boolean, but
NO field or docstring anywhere claims that a positive reading means "your
explanation doesn't generalize" or "these are distinct mechanisms". See
`GluingReport.caveat` and benchmarks/GLUING.md for the full account.
"""
from .report import explanation_obstruction, GluingReport, CAVEAT, blocking_pairs
from .cover import cover_from_embedding, preprocess, fcm, soft_cover, hard_cover, knn_centre_cover, composition
from .core import (nerve, section, restrict, jaccard, cover_means, agreements_from_means, pair_agreements,
                   bottleneck, h0, obstruction, triangle_frustration, brute_force_obs)

__all__ = [
    "explanation_obstruction", "GluingReport", "CAVEAT", "blocking_pairs",
    "cover_from_embedding", "preprocess", "fcm", "soft_cover", "hard_cover", "knn_centre_cover", "composition",
    "nerve", "section", "restrict", "jaccard", "cover_means", "agreements_from_means", "pair_agreements",
    "bottleneck", "h0", "obstruction", "triangle_frustration", "brute_force_obs",
]
