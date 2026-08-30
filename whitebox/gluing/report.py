"""explanation_obstruction -- the whitebox.gluing public API.

Ported from whitebox-v2 E3 (2026-08-30, GPT-2 small, IOI vs an induction
control; see /home/god/whitebox-v2/E3/SPEC.md and E3/FINDINGS.md in the
experiment tree). Framework: sheaf-of-explanations (arXiv 2603.17141:
explanations as local sections over a cover of input space, gluing
counterexamples, the obstruction to a global explanation is cohomological)
and "A Gauge Theory of Superposition" (sheaf atlas of feature charts with
holonomy obstructions -- E3's "disagreement around a cycle of the nerve" is
the discrete analogue of holonomy around a loop of charts). This module is
only the measurement instrument: cover -> local circuits (sections) ->
restrictions -> agreement -> frustrated-cycle obstruction on the nerve.

WHAT THIS MODULE DOES NOT SHIP, AND WHY (read before using `significant_vs_null`
for anything): E3 pre-registered a matched single-mechanism control (an
induction task) that was supposed to give Omega approx 0 and land inside its
own prompt-shuffle null band -- the sanity anchor the whole "distinct
mechanisms" reading depends on. It did not. At the pre-registered summary
config (k=10, k_e=200, layer-11 embedding) the control's Omega = 1.517 sat
*outside* its own null band [0.564, 0.991] (p_null = 0.000), and this held in
34/40 grid cells swept. Worse: IOI (the task believed to have genuinely
multiple template-specific mechanisms) showed *smaller* raw Omega than the
single-mechanism control in 40/40 grid cells -- the direction of the
pre-registered comparison was wrong in every cell tested. And the one
blocking pair that did clear IOI's own null (p_null=0.005) did not survive a
second-embedding robustness check (E3 FINDINGS.md S3): the top pair's
template/order composition does not appear, by identity or by composition,
among the top-3 blocking pairs found under a different embedding layer.

Read `explanation_obstruction`'s docstring and `GluingReport.caveat` before
treating a positive `significant_vs_null` as evidence of "distinct mechanisms
located in prompt space". E3's own battery says it usually is not, absent two
checks this module cannot perform for you (a matched single-mechanism control
on your data, and a second-embedding stability check on the blocking pair).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import core as sh
from .cover import cover_from_embedding

CAVEAT = (
    "This is a diagnostic instrument, not a validated 'distinct mechanisms' "
    "detector. On E3's own pre-registered battery (whitebox-v2/E3, GPT-2 "
    "small): the matched single-mechanism control (induction) FAILED its own "
    "null-band criterion -- Omega=1.517 outside null band [0.564, 0.991], "
    "p_null=0.000 at the summary config, and outside the null in 34/40 swept "
    "grid cells -- and raw Omega was LARGER for the control than for IOI in "
    "40/40 grid cells (the pre-registered direction was wrong everywhere "
    "tested). The one IOI blocking pair that cleared its own null "
    "(p_null=0.005) did not survive a second-embedding check: its "
    "template/order composition did not reappear among the top-3 blocking "
    "pairs found under a different embedding layer. A positive "
    "`significant_vs_null` here only supports 'distinct mechanisms located in "
    "example space' if BOTH (a) a matched single-mechanism control run on "
    "the SAME architecture/embedding also lands inside ITS null band, and "
    "(b) the top blocking pair's identity or composition survives a second, "
    "independent embedding choice. Neither held in E3's own battery. Do not "
    "read a bare 'Omega above null' as 'your explanation doesn't "
    "generalize' -- report the control and the second-embedding check "
    "alongside it, or do not make the claim."
)


@dataclass
class GluingReport:
    """One cover's obstruction reading. See the module docstring and
    `caveat` before treating `significant_vs_null` as a "distinct mechanisms"
    finding -- it is not, on its own."""
    # --- integrated quantities ---
    omega: float
    omega_rel: float
    omega_per_edge: float
    omega_ci: Tuple[float, float]           # bootstrap CI on omega
    mean_J: float
    n_frustrated_edges: int                 # edges with m_e > 0 at the finest tau grid
    cyclomatic: int                         # |E| - |V| + #components of the nerve

    # --- null (prompt/example-label shuffle) ---
    null_lo: float
    null_hi: float
    null_mean: float
    p_null: float                           # P(null Omega >= observed Omega)
    significant_vs_null: bool               # p_null <= 0.025 (observed Omega above the
                                             # 97.5th percentile of the shuffle null) --
                                             # NOT by itself a "distinct mechanisms" claim

    # --- per-edge (nerve edges, i.e. cluster pairs with overlap >= n_min) ---
    edges: List[Tuple[int, int]]
    J: np.ndarray
    J_raw: np.ndarray
    J_bott: np.ndarray
    m_e: np.ndarray
    J_ci: Tuple[np.ndarray, np.ndarray]     # bootstrap CI per edge
    m_e_ci: Tuple[np.ndarray, np.ndarray]

    # --- cover / nerve ---
    n_clusters: int
    n_tris: int
    cover: Dict

    # --- blocking pairs (top frustrated cluster pairs) ---
    blocking_pairs: List[Dict]

    # --- curves (per tau) ---
    taus: np.ndarray
    obs: np.ndarray
    h0: np.ndarray
    triangle_frustration: np.ndarray

    caveat: str = field(default=CAVEAT)
    meta: Dict = field(default_factory=dict)

    def __repr__(self):
        return (f"GluingReport(omega={self.omega:.3g} (CI {self.omega_ci}), "
                f"null=[{self.null_lo:.3g},{self.null_hi:.3g}], p_null={self.p_null:.3g}, "
                f"significant_vs_null={self.significant_vs_null}, "
                f"n_edges={len(self.edges)}, cyclomatic={self.cyclomatic}, "
                f"n_frustrated={self.n_frustrated_edges})")


def _analyse(scores, members, edges, tris, k_e, restrict_frac=0.5, taus=None,
            n_shuffle=200, n_boot=200, seed=1):
    """Everything after the cover is fixed: agreement, obstruction, the
    example-label shuffle null, and bootstrap CIs. Bootstrap = row counts as
    weights (no copies); nerve fixed; memberships travel with rows. Mirrors
    E3's `e3lib/pipeline.py::analyse_cover` (single k_e)."""
    n_v = members.shape[1]
    N = scores.shape[0]
    M, O = sh.cover_means(scores, members, edges)
    J, Jraw, (secs, sets) = sh.agreements_from_means(M, O, edges, k_e, restrict_frac, True)
    ob = sh.obstruction(n_v, edges, J, taus)
    taus = ob["taus"]
    tf = sh.triangle_frustration(tris, edges, J, taus)

    rng = np.random.default_rng(seed)
    null_omega = np.empty(n_shuffle)
    for s in range(n_shuffle):                      # (i) permute example rows of the membership matrix
        Mn, On = sh.cover_means(scores, members[rng.permutation(N)], edges)
        Jn, _ = sh.agreements_from_means(Mn, On, edges, k_e, restrict_frac)
        null_omega[s] = sh.obstruction(n_v, edges, Jn, taus)["omega"]

    boot_J = np.empty((n_boot, len(edges))) if edges else np.empty((n_boot, 0))
    boot_m = np.empty((n_boot, len(edges))) if edges else np.empty((n_boot, 0))
    boot_omega = np.empty(n_boot)
    for b in range(n_boot):                          # (iii) bootstrap rows with replacement
        w = np.bincount(rng.integers(0, N, N), minlength=N)
        Mb, Ob = sh.cover_means(scores, members, edges, w)
        Jb, _ = sh.agreements_from_means(Mb, Ob, edges, k_e, restrict_frac)
        obb = sh.obstruction(n_v, edges, Jb, taus)
        boot_J[b] = Jb; boot_m[b] = obb["m_e"]; boot_omega[b] = obb["omega"]

    q1 = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")   # 1-D (omega, null)
    q2 = lambda a, p, n: np.percentile(a, p, axis=0) if len(a) else np.full(n, float("nan"))  # per-edge
    return dict(J=J, J_raw=Jraw, sections=secs, restricted_sets=sets, obstruction=ob, taus=taus,
                triangle_frustration=tf,
                null_omega=null_omega,
                null_band=(q1(null_omega, 2.5), q1(null_omega, 97.5)),
                null_mean=float(null_omega.mean()) if n_shuffle else float("nan"),
                p_null=float((null_omega >= ob["omega"]).mean()) if n_shuffle else float("nan"),
                boot_J_ci=(q2(boot_J, 2.5, len(edges)), q2(boot_J, 97.5, len(edges))),
                boot_m_ci=(q2(boot_m, 2.5, len(edges)), q2(boot_m, 97.5, len(edges))),
                boot_omega_ci=(q1(boot_omega, 2.5), q1(boot_omega, 97.5)))


def blocking_pairs(res, edges, members, top=10):
    """Rank nerve edges by frustrated lifetime m_e (ties by disagreement),
    with sizes/overlap and the bootstrap CI on m_e and J."""
    ob = res["obstruction"]
    if not edges:
        return []
    order = np.lexsort((-(1 - res["J"]), -ob["m_e"]))
    rows = []
    for r in order[:top]:
        i, j = edges[r]
        ov = members[:, i] & members[:, j]
        rows.append(dict(
            pair=(int(i), int(j)), size_i=int(members[:, i].sum()), size_j=int(members[:, j].sum()),
            overlap=int(ov.sum()), J=float(res["J"][r]), J_raw=float(res["J_raw"][r]),
            J_bott=float(ob["J_bott"][r]), m_e=float(ob["m_e"][r]),
            m_e_ci=(float(res["boot_m_ci"][0][r]), float(res["boot_m_ci"][1][r])),
            J_ci=(float(res["boot_J_ci"][0][r]), float(res["boot_J_ci"][1][r])),
        ))
    return rows


def explanation_obstruction(scores, cover=None, embedding=None, k=10, k_e=200, theta=0.5,
                            n_pca=20, n_min=10, restrict_frac=0.5, n_shuffle=200, n_boot=200,
                            cover_method="knn2", taus=None, top_pairs=10, seed=0) -> GluingReport:
    """Measure the obstruction to gluing per-cluster explanations into one global
    explanation.

    Parameters
    ----------
    scores : (n_examples, n_edges) array
        Per-example attribution/explanation weights over "edges" -- whatever a
        caller's explanation method assigns weight to (attention-head/MLP edges
        as in E3's EAP graph, feature attributions, saliency weights, anything
        with one row per example and a fixed set of columns across examples).
    cover : (n_examples, n_clusters) bool array, optional
        A precomputed overlapping cover (membership matrix). If given,
        `embedding`/`k`/`theta`/`n_pca`/`cover_method` are ignored.
    embedding : (n_examples, d) array, optional
        Used to build a cover via `whitebox.gluing.cover_from_embedding` if
        `cover` is not given. Exactly one of `cover`/`embedding` is required.
    k : number of clusters (ignored if `cover` is given).
    k_e : section size -- each cluster's local explanation is its top-k_e
        edges by |mean score| (top-k rather than an absolute threshold, so
        Jaccard between equal-size sets is comparable across clusters).
    theta, n_pca, n_min, cover_method : passed to `cover_from_embedding`
        (`cover_method="knn2"`, the default, is E3's Amendment-1 2-nearest-
        centre cover; `"fcm"` is the originally pre-registered fuzzy c-means +
        relative-membership rule, degenerate on E3's own embeddings -- see
        `cover.py`'s docstring).
    restrict_frac : fraction of k_e kept when a cluster's section is re-scored
        on an overlap's rows (0.5 = E3's default: keep the top half).
    n_shuffle : number of example-label permutations for the null band (0 to skip).
    n_boot : number of row-resamples-with-replacement for bootstrap CIs (0 to skip).
    taus : optional array of thresholds for the obs(tau)/H0(tau) curves
        (default: 0, 0.02, ..., 1.0).
    top_pairs : how many blocking pairs to report (ranked by frustrated lifetime m_e).
    seed : shuffle/bootstrap RNG seed.

    Returns
    -------
    GluingReport -- see its docstring and the module docstring for what
    `significant_vs_null` does and does not license you to claim.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"scores must be (n_examples, n_edges); got shape {scores.shape}")
    if (cover is None) == (embedding is None):
        raise ValueError("pass exactly one of cover= (membership bool array) or embedding=")

    if cover is not None:
        members = np.asarray(cover, dtype=bool)
        if members.shape[0] != scores.shape[0]:
            raise ValueError(f"cover has {members.shape[0]} rows, scores has {scores.shape[0]}")
        edges, tris, _ov = sh.nerve(members, n_min)
        cover_dict = dict(members=members, edges=edges, tris=tris, overlap=_ov, theta=None, fcm_obj=None, U=None)
    else:
        embedding = np.asarray(embedding, dtype=float)
        if embedding.shape[0] != scores.shape[0]:
            raise ValueError(f"embedding has {embedding.shape[0]} rows, scores has {scores.shape[0]}")
        cover_dict = cover_from_embedding(embedding, k, theta, n_pca, seed, n_min, method=cover_method)
        members, edges, tris = cover_dict["members"], cover_dict["edges"], cover_dict["tris"]

    k_e = min(k_e, scores.shape[1])
    res = _analyse(scores, members, edges, tris, k_e, restrict_frac, taus, n_shuffle, n_boot, seed + 1)
    ob = res["obstruction"]
    bp = blocking_pairs(res, edges, members, top_pairs)
    null_lo, null_hi = res["null_band"]

    return GluingReport(
        omega=ob["omega"], omega_rel=ob["omega_rel"], omega_per_edge=ob["omega_per_edge"],
        omega_ci=res["boot_omega_ci"], mean_J=float(res["J"].mean()) if len(res["J"]) else float("nan"),
        n_frustrated_edges=int((ob["m_e"] > 0).sum()), cyclomatic=ob["cyclomatic"],
        null_lo=null_lo, null_hi=null_hi, null_mean=res["null_mean"], p_null=res["p_null"],
        significant_vs_null=bool(res["p_null"] <= 0.025) if n_shuffle else False,
        edges=list(edges), J=res["J"], J_raw=res["J_raw"], J_bott=ob["J_bott"], m_e=ob["m_e"],
        J_ci=res["boot_J_ci"], m_e_ci=res["boot_m_ci"],
        n_clusters=members.shape[1], n_tris=len(tris), cover=cover_dict,
        blocking_pairs=bp, taus=res["taus"], obs=ob["obs"], h0=ob["h0"],
        triangle_frustration=res["triangle_frustration"],
        meta=dict(k=k, k_e=k_e, n_min=n_min, restrict_frac=restrict_frac, n_shuffle=n_shuffle, n_boot=n_boot,
                  seed=seed, cover_method=cover_method if cover is None else "given"),
    )
