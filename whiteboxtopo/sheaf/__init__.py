"""whiteboxtopo.sheaf — Čech cohomology of explanations (FINAL_SPEC §2.12, §4).

A thin, spec-named layer over :mod:`whiteboxtopo.gluing`, which already implements
the Čech machinery (nerve, sections, restrictions, Jaccard agreement, the
frustrated-cycle obstruction). Nothing mathematical is re-implemented here; this
module only supplies the §4 API names and the §2.12 reading of the numbers:

    cover(X, k, overlap)              overlapping cover of the inputs + its nerve
    local_explanations(score_fn, cv)  E_i: one explanation vector per region
    restrict(E, section, overlap, k)  E_i|_{ij} by re-scoring on the overlap
    cech_H0_H1(cover, E, tau, k_e)    H0 (consistent families), H1 (obstruction
                                      cocycles), boundaries (mechanism boundaries)
    sanity_gate(result)               §2.12 gate: a single-mechanism task must
                                      return H1 == 0

Reading of the numbers (§2.12)
------------------------------
* ``H0`` = number of connected components of the agreement graph
  ``{nerve edge e : J_e >= tau}`` = number of *consistent explanation families*.
  Two regions in different components do not share one explanation at level tau.
* ``H1`` = the number of nerve edges that are inconsistent (``J_e < tau``) but
  whose endpoints are still joined at level tau by another path — a cycle of
  overlaps with unresolvable disagreement (``gluing.core.obstruction``'s
  ``obs(tau)``). A *clean* split into two families is an H0 phenomenon and
  contributes nothing to H1; frustration is what H1 counts.
* ``boundaries`` = the disagreeing overlaps themselves (``J_e < tau``), i.e. the
  localized mechanism boundaries. A nonempty boundary set with ``H1 == 0`` means
  the explanation splits cleanly; ``H1 > 0`` means it cannot be split at all.

Null and failure boundary
-------------------------
This module ships **no** "distinct mechanisms" verdict. The whitebox-v2 E3
experiment's own pre-registered single-mechanism control failed its null band
(see ``whiteboxtopo.gluing.CAVEAT`` and benchmarks/GLUING.md); the shipped
instrument is the measurement plus the sanity gate. For a null, run the same
pipeline on shuffled scores (``whiteboxtopo.nulls.feature_shuffle``) or use
``whiteboxtopo.gluing.explanation_obstruction``, which carries the shuffle null,
bootstrap CIs and ``p_null``. H0/H1 here are *descriptive numbers at a chosen
tau*, not a test. Both are gauge-invariant only to the extent the caller's
``score_fn`` is: the feature axis may be permuted by a gauge element, but the
Jaccard of top-k sets is unchanged when the same relabelling is applied to every
region — a per-region relabelling would break it.

Determinism: every entry point takes ``seed``; k-means++ centres are the only
stochastic ingredient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..gluing import core as _core
from ..gluing import cover as _cover_mod

__all__ = ["Cover", "LocalExplanation", "CechResult", "cover", "cover_from_members",
           "local_explanations", "restrict", "cech_H0_H1", "sanity_gate"]


# --------------------------------------------------------------------- cover
@dataclass
class Cover:
    """An overlapping cover of the example set plus its nerve.

    ``members`` is a bool (N, k) membership matrix (example x region);
    ``edges`` the nerve's 1-simplices (region pairs sharing >= ``n_min``
    examples), ``triangles`` its 2-simplices, ``overlap`` the (k, k) matrix of
    overlap sizes. ``method="knn2"`` produces empty triangles by construction
    (E3 Amendment 1) — reported, not hidden.
    """
    members: np.ndarray
    edges: List[tuple]
    triangles: List[tuple]
    overlap: np.ndarray
    method: str = "given"
    seed: int = 0
    n_min: int = 10
    params: dict = field(default_factory=dict)

    @property
    def n_examples(self) -> int:
        return int(self.members.shape[0])

    @property
    def n_regions(self) -> int:
        return int(self.members.shape[1])

    def indices(self, i: int) -> np.ndarray:
        """Row indices of the examples in region ``i``."""
        return np.where(self.members[:, i])[0]

    def overlap_indices(self, i: int, j: int) -> np.ndarray:
        """Row indices of the examples in ``U_i ∩ U_j``."""
        return np.where(self.members[:, i] & self.members[:, j])[0]

    def sizes(self) -> np.ndarray:
        return self.members.sum(0).astype(int)

    def to_dict(self) -> dict:
        return {"n_examples": self.n_examples, "n_regions": self.n_regions,
                "n_edges": len(self.edges), "n_triangles": len(self.triangles),
                "sizes": self.sizes().tolist(), "edges": [tuple(map(int, e)) for e in self.edges],
                "method": self.method, "seed": self.seed, "n_min": self.n_min, "params": dict(self.params)}


def cover(X, k: int, overlap: float = 0.5, seed: int = 0, n_min: int = 10,
          method: str = "knn2", n_pca: int = 20) -> Cover:
    """Cover the examples ``X`` (N, d) by ``k`` overlapping regions and build the nerve.

    ``overlap`` (§4 signature) is interpreted per method:

    * ``method="knn2"`` (default, E3 Amendment 1): k-means++ centres, each example
      belongs to its ``n_nearest = max(2, round(1 / (1 - overlap)))`` nearest
      centres (``overlap=0.5`` -> 2, the validated default; ``overlap=2/3`` -> 3).
      Overlap is structural, so no threshold can degenerate.
    * ``method="fcm"``: the originally pre-registered fuzzy c-means cover, where
      ``overlap`` is the relative-membership threshold theta (example in region i
      iff ``u_i >= overlap * max_j u_j``). Degenerate on some real embeddings —
      see ``whiteboxtopo.gluing.cover``.

    Null / failure boundary: a cover whose nerve has no cycle cannot show
    frustration at all (``cyclomatic == 0`` => ``H1 == 0`` trivially), and a
    cover with regions smaller than ``n_min`` simply loses those nerve edges.
    Both are visible in :meth:`Cover.to_dict`. Deterministic given ``seed``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        X = X.reshape(len(X), -1)
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must lie in [0, 1), got {overlap}")
    if method == "knn2":
        n_nearest = max(2, int(round(1.0 / (1.0 - overlap)))) if overlap > 0 else 1
        n_nearest = min(n_nearest, k)
        Xp = _cover_mod.preprocess(X, n_pca)
        members, _, centres = _cover_mod.knn_centre_cover(Xp, k, n_nearest=n_nearest, seed=seed)
        edges, tris, ov = _core.nerve(members, n_min)
        return Cover(members=members, edges=edges, triangles=tris, overlap=ov, method="knn2",
                     seed=seed, n_min=n_min, params={"n_nearest": n_nearest, "n_pca": n_pca,
                                                     "overlap": overlap, "k": int(k)})
    if method == "fcm":
        out = _cover_mod.cover_from_embedding(X, k, theta=overlap, n_pca=n_pca, seed=seed,
                                              n_min=n_min, method="fcm")
        return Cover(members=out["members"], edges=out["edges"], triangles=out["tris"],
                     overlap=out["overlap"], method="fcm", seed=seed, n_min=n_min,
                     params={"theta": out["theta"], "n_pca": n_pca, "k": int(k)})
    raise ValueError(f"method must be 'knn2' or 'fcm', got {method!r}")


def cover_from_members(members, n_min: int = 10, method: str = "given", seed: int = 0) -> Cover:
    """Wrap a caller-supplied bool (N, k) membership matrix in a :class:`Cover`
    (nerve computed by ``gluing.core.nerve``). Used for hand-built covers and tests."""
    members = np.asarray(members, dtype=bool)
    edges, tris, ov = _core.nerve(members, n_min)
    return Cover(members=members, edges=edges, triangles=tris, overlap=ov, method=method,
                 seed=seed, n_min=n_min, params={"k": int(members.shape[1])})


# ------------------------------------------------------------ explanations
@dataclass
class LocalExplanation:
    """The certified local explanation E_i of one region: the region's score
    vector and its top-``k_e`` support (the section)."""
    region: int
    size: int
    scores: np.ndarray
    section: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {"region": self.region, "size": self.size,
                "section": None if self.section is None else self.section.tolist()}


def local_explanations(score_fn: Callable[[np.ndarray], np.ndarray], cov: Cover,
                       k_e: Optional[int] = None) -> List[LocalExplanation]:
    """E_i for every region of ``cov``.

    ``score_fn(indices) -> ndarray`` is supplied by the caller and must return a
    **1-D score vector of fixed length ``n_features``** — one score per
    explanation atom (attribution-graph edge, feature, unit, SAE latent: whatever
    the caller's explanation assigns weight to) — for the sub-population
    ``indices`` (an int array of row indices into the example set). The same
    feature axis must be used for every region; otherwise the restrictions and
    Jaccard agreements below are meaningless.

    If ``k_e`` is given, each explanation also carries its section = the top-k_e
    features by ``|score|``.

    Returns one :class:`LocalExplanation` per region, in region order. Null: a
    ``score_fn`` that ignores ``indices`` makes every section identical and every
    agreement 1.0 — the trivial-explanation control.
    """
    out = []
    n_feat = None
    for i in range(cov.n_regions):
        idx = cov.indices(i)
        s = np.asarray(score_fn(idx), dtype=float).ravel()
        if n_feat is None:
            n_feat = s.size
        elif s.size != n_feat:
            raise ValueError(f"score_fn returned {s.size} features for region {i}, {n_feat} for region 0; "
                             "the feature axis must be fixed across regions")
        sec = None
        if k_e is not None and s.size:
            kk = int(min(k_e, s.size))
            sec = np.sort(np.argpartition(-np.abs(s), kk - 1)[:kk])
        out.append(LocalExplanation(region=i, size=int(idx.size), scores=s, section=sec))
    return out


def restrict(E, section, overlap_idx, k_r):
    """E_i|_{ij}: re-score the section's features on the overlap's rows and keep the
    ``k_r`` strongest (``gluing.core.restrict``).

    ``E`` is the (N, n_features) per-example score matrix, ``section`` the region's
    candidate feature set (int array, e.g. ``LocalExplanation.section``),
    ``overlap_idx`` the rows of ``U_i ∩ U_j``. The result is always a subset of
    ``section``. Restriction needs per-example scores: a per-region summary cannot
    be re-scored on a sub-population, which is why ``cech_H0_H1`` takes ``E`` and
    not the list from :func:`local_explanations`.
    """
    return _core.restrict(np.asarray(E, dtype=float), np.asarray(section, dtype=int),
                          np.asarray(overlap_idx, dtype=int), int(k_r))


# ------------------------------------------------------------------- Čech
@dataclass
class CechResult:
    """H0/H1 of the explanation presheaf at threshold ``tau`` (§2.12)."""
    H0: int
    H1: int
    boundaries: List[dict]
    tau: float
    k_e: int
    n_regions: int
    n_edges: int
    cyclomatic: int
    J: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    J_bott: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    J_raw: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    omega: float = 0.0
    omega_rel: float = 0.0
    n_triangles: int = 0
    restrict_frac: float = 0.5
    notes: str = ""

    @property
    def single_mechanism(self) -> bool:
        """The §2.12 sanity gate as a property: no unresolvable disagreement."""
        return self.H1 == 0

    def to_dict(self) -> dict:
        return {"H0": int(self.H0), "H1": int(self.H1), "tau": float(self.tau), "k_e": int(self.k_e),
                "n_regions": int(self.n_regions), "n_edges": int(self.n_edges),
                "n_triangles": int(self.n_triangles), "cyclomatic": int(self.cyclomatic),
                "omega": float(self.omega), "omega_rel": float(self.omega_rel),
                "restrict_frac": float(self.restrict_frac),
                "boundaries": [dict(b) for b in self.boundaries],
                "J": np.asarray(self.J).tolist(), "J_bott": np.asarray(self.J_bott).tolist(),
                "single_mechanism": bool(self.single_mechanism), "notes": self.notes}


def cech_H0_H1(cov: Cover, E, tau: float = 0.5, k_e: int = 32, restrict_frac: float = 0.5) -> CechResult:
    """H0, H1 and the mechanism boundaries of the explanation presheaf on ``cov``.

    ``E`` is the (N, n_features) per-example score matrix (same feature axis for
    every example; see :func:`local_explanations`). Sections are the top-``k_e``
    features per region; restrictions keep ``ceil(k_e * restrict_frac)`` of them on
    each overlap; two regions *agree* on an overlap iff the Jaccard of their
    restricted sections is ``>= tau``.

    * ``H0`` — connected components of the agreement graph (consistent explanation families).
    * ``H1`` — ``obs(tau)`` from ``gluing.core.obstruction``: inconsistent overlaps
      whose endpoints remain connected at level tau (frustrated cycles).
    * ``boundaries`` — every nerve edge with ``J < tau``, with its J, its
      widest-path agreement ``J_bott`` and whether it is frustrated.

    Failure boundary: with no nerve edges (``n_edges == 0``) H0 is the region count
    and H1 is 0 by vacuity; with ``cyclomatic == 0`` H1 is 0 by construction, not by
    evidence — check ``cyclomatic`` before reading ``H1 == 0`` as a result. No null
    is computed here (see the module docstring).
    """
    E = np.asarray(E, dtype=float)
    if E.ndim != 2:
        raise ValueError(f"E must be (n_examples, n_features), got shape {E.shape}")
    if E.shape[0] != cov.n_examples:
        raise ValueError(f"E has {E.shape[0]} rows, cover has {cov.n_examples} examples")
    k_e = int(max(1, min(k_e, E.shape[1])))
    k_v, edges = cov.n_regions, cov.edges
    notes = ""
    if not edges:
        notes = "nerve has no edges (regions never overlap by >= n_min): H1 vacuous"
        return CechResult(H0=k_v, H1=0, boundaries=[], tau=float(tau), k_e=k_e, n_regions=k_v,
                          n_edges=0, cyclomatic=0, n_triangles=len(cov.triangles),
                          restrict_frac=restrict_frac, notes=notes)
    J, J_raw = _core.pair_agreements(E, cov.members, edges, k_e, restrict_frac=restrict_frac)
    obs = _core.obstruction(k_v, edges, J, taus=np.array([float(tau)]))
    H0 = int(_core.h0(k_v, edges, J, float(tau)))
    H1 = int(obs["obs"][0])
    Jb = np.asarray(obs["J_bott"], dtype=float)
    boundaries = [{"i": int(edges[e][0]), "j": int(edges[e][1]), "J": float(J[e]),
                   "J_raw": float(J_raw[e]), "J_bott": float(Jb[e]),
                   "n_overlap": int(cov.overlap[edges[e][0], edges[e][1]]),
                   "frustrated": bool(J[e] < tau <= Jb[e])}
                  for e in range(len(edges)) if J[e] < tau]
    if int(obs["cyclomatic"]) == 0:
        notes = "nerve is acyclic (cyclomatic == 0): H1 == 0 holds by construction, not by evidence"
    return CechResult(H0=H0, H1=H1, boundaries=boundaries, tau=float(tau), k_e=k_e, n_regions=k_v,
                      n_edges=len(edges), cyclomatic=int(obs["cyclomatic"]), J=np.asarray(J),
                      J_bott=Jb, J_raw=np.asarray(J_raw), omega=float(obs["omega"]),
                      omega_rel=float(obs["omega_rel"]), n_triangles=len(cov.triangles),
                      restrict_frac=restrict_frac, notes=notes)


def sanity_gate(result: CechResult) -> bool:
    """The §2.12 single-mechanism gate: ``True`` iff ``H1 == 0``.

    A known single-mechanism task must pass. Passing is necessary, never
    sufficient: an acyclic nerve passes vacuously (``result.notes`` says so), and
    H1 counts frustration only — a clean two-family split passes the gate while
    ``H0 == 2`` records the split.
    """
    return int(result.H1) == 0
