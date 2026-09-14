"""whiteboxtopo.align — cross-fiber circuit alignment and the mechanism count
(FINAL_SPEC §2.9, §4, §8.3).

Two circuits found in two independently trained networks are the *same*
mechanism iff some gauge element carries one onto the other:

    C_i ~ C_j  iff  exists g in G with  g . (theta_i|C_i) ~= theta_j|C_j   (tol 1e-4)

    match_circuits(theta_i, C_i, theta_j, C_j)  -> (GaugeMatch g, residual r)
    count_mechanisms(seeds, circuits, tau)      -> MechanismCount (K, classes, K/N)
    relabel_kill(theta, C, seed)                -> the §8.3 planted kill

Method (the F4a construction, self-contained)
---------------------------------------------
1. **Scale gauge-fix.** Each hidden unit's incoming vector ``(w_in, b)`` is unit-
   normalised and its norm pushed into the outgoing weights; the output row is
   unit-normalised too. This kills the ``(R_>0)^n`` factor exactly.
2. **Permutation match.** Layer by layer (parents first), the Hungarian algorithm
   matches A's units to B's on squared distance between normalised incoming
   vectors, B's input columns having first been reordered by the parent layer's
   match. This kills the ``S_n`` factor.
3. **Residual.** ``r = sqrt(2 * sum ||a - b||^2 / (sum ||a||^2 + sum ||b||^2))``
   over all matched incoming vectors and the output row: 0 for an exact
   relabelling, 2 for anti-aligned, scale-free.

Convention (inherited from F4a): the output row is normalised, so two circuits
differing only by an overall output gain match; the gain is reported separately
as ``GaugeMatch.output_gain``. Within a single network, distinct circuits are
never gauge-equivalent (Proposition 2.1), so ``count_mechanisms`` compares
*across* seeds only by default.

Null and failure boundary
-------------------------
* Planted positive (§8.3): a literal gauge relabelling must give ``r < 1e-8`` and
  ``K = 1`` — :func:`relabel_kill`.
* Planted negative: seeds trained to different named algorithms must give
  ``K = 2``; unrelated random circuits give ``r`` of order 1.
* ``r = inf`` when the two circuits have different layer sizes: no gauge element
  can relate them, and they are placed in different classes.
* The greedy layer-by-layer Hungarian is **not** guaranteed to find the global
  optimum over the product of permutations (it is exact when the parent match is
  correct, which an exact relabelling makes true). A large ``r`` is therefore an
  upper bound on the true gauge distance: this module can miss a match, it cannot
  invent one.
* Functional degeneracies (dead units, units sharing an activation pattern) are
  *not* collapsed here; the F-block experiment's collapse step
  (``experiments/F_gauge/flib/crossseed.effective_subimpl``) needs activation
  patterns on a dataset. Use :func:`flib_crossseed` to reach it when the
  experiment tree is present; without the collapse, two circuits that differ only
  by a split of one effective unit into two will *not* match.

Deterministic: no randomness anywhere except the caller-supplied ``seed`` in
:func:`relabel_kill`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["GaugeMatch", "MechanismCount", "match_circuits", "count_mechanisms", "relabel_kill",
           "effective_circuit", "residual", "flib_crossseed", "MATCH_TOL"]

MATCH_TOL = 1e-4          # §2.9: weights on the support match to tolerance 1e-4


# ------------------------------------------------------------------ adapters
def _as_weights(theta) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """(Ws, bs) with ``Ws[l]`` of shape (fan_in, fan_out) from a ReluMLP, a
    ``(Ws, bs)`` pair, any object carrying ``.Ws``/``.bs``, or a torch Sequential."""
    if isinstance(theta, (tuple, list)) and len(theta) == 2 and not hasattr(theta, "Ws"):
        Ws, bs = theta
        return [np.asarray(W, float) for W in Ws], [np.asarray(b, float) for b in bs]
    if hasattr(theta, "Ws") and hasattr(theta, "bs"):
        return [np.asarray(W, float) for W in theta.Ws], [np.asarray(b, float) for b in theta.bs]
    from ..invariants.model import as_relu_mlp
    m = as_relu_mlp(theta)
    return [np.asarray(W, float) for W in m.Ws], [np.asarray(b, float) for b in m.bs]


def _circuit_units(C, widths: Sequence[int]) -> List[np.ndarray]:
    """Per-hidden-layer unit index arrays for a circuit spec.

    Accepted: ``None`` / ``"all"`` (the whole network); a sequence of index
    arrays, one per hidden layer; a dict with key ``"units"``; any object with a
    ``.units`` attribute. Indices are into the hidden layer's units.
    """
    if C is None or (isinstance(C, str) and C == "all"):
        return [np.arange(w) for w in widths]
    if isinstance(C, dict):
        C = C.get("units", None)
        if C is None:
            return [np.arange(w) for w in widths]
    elif hasattr(C, "units"):
        C = C.units
    units = [np.sort(np.asarray(u, dtype=int)) for u in C]
    if len(units) != len(widths):
        raise ValueError(f"circuit has {len(units)} hidden layers, model has {len(widths)}")
    for l, u in enumerate(units):
        if u.size == 0 or u.min() < 0 or u.max() >= widths[l]:
            raise ValueError(f"circuit layer {l}: unit indices out of range for width {widths[l]}")
    return units


def _algorithm_of(C) -> Optional[str]:
    if isinstance(C, dict):
        a = C.get("algorithm")
    else:
        a = getattr(C, "algorithm", None)
    return None if a is None else str(a)


def _subnetwork(Ws, bs, units) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Restrict the network to the circuit's units (rows/columns dropped elsewhere)."""
    Ws = [W.copy() for W in Ws]
    bs = [b.copy() for b in bs]
    for l, u in enumerate(units):
        Ws[l] = Ws[l][:, u]
        bs[l] = bs[l][u]
        Ws[l + 1] = Ws[l + 1][u, :]
    return Ws, bs


def _normalize(Ws, bs):
    """Gauge-fix the scalings: unit-normalise each hidden unit's incoming (w, b) and
    push the norm downstream; unit-normalise the output row. Returns
    ``(Ws, bs, norms, output_gain)`` — the same construction as F4a's
    ``flib.crossseed._normalize``, with the norms kept so the gauge element can be
    reconstructed."""
    Ws = [W.copy() for W in Ws]
    bs = [b.copy() for b in bs]
    norms = []
    for l in range(len(Ws) - 1):
        V = np.vstack([Ws[l], bs[l][None, :]])
        n = np.linalg.norm(V, axis=0)
        n = np.where(n == 0, 1.0, n)
        Ws[l] = Ws[l] / n[None, :]
        bs[l] = bs[l] / n
        Ws[l + 1] = Ws[l + 1] * n[:, None]
        norms.append(n)
    on = float(np.sqrt((Ws[-1] ** 2).sum() + (bs[-1] ** 2).sum())) or 1.0
    Ws[-1] = Ws[-1] / on
    bs[-1] = bs[-1] / on
    return Ws, bs, norms, on


def effective_circuit(theta, C=None):
    """The gauge-fixed effective sub-implementation of circuit ``C`` in ``theta``.

    Returns ``(inc, out, norms, output_gain)`` where ``inc[l]`` is the
    (k_l, fan_in + 1) matrix of unit-normalised incoming vectors of hidden layer
    ``l`` and ``out = (W_out, b_out)`` the unit-normalised output map (``W_out``
    of shape (k_last, n_out), rows indexed by the last hidden layer's units).
    Every entry is a deterministic function of gauge invariants, so a literal
    relabelling changes only the row order of ``inc`` and ``W_out``.
    """
    Ws, bs = _as_weights(theta)
    widths = [W.shape[1] for W in Ws[:-1]]
    units = _circuit_units(C, widths)
    Ws, bs = _subnetwork(Ws, bs, units)
    Ws, bs, norms, gain = _normalize(Ws, bs)
    inc = [np.vstack([Ws[l], bs[l][None, :]]).T for l in range(len(Ws) - 1)]
    out = (Ws[-1], bs[-1])
    return inc, out, norms, gain


def residual(sa, sb):
    """Gauge residual between two effective sub-implementations
    ``(inc, (W_out, b_out))`` (layer-by-layer Hungarian matching, parents first).

    Returns ``(r, perms)``; ``r = inf`` and ``perms = None`` if the layer sizes
    differ. ``perms[l][a] = b`` means A's unit ``a`` is matched to B's unit ``b``.
    """
    inc_a, (Wa, ba) = sa[0], sa[1]
    inc_b, (Wb, bb) = sb[0], sb[1]
    if len(inc_a) != len(inc_b) or any(a.shape != b.shape for a, b in zip(inc_a, inc_b)):
        return float("inf"), None
    if Wa.shape != Wb.shape or ba.shape != bb.shape:
        return float("inf"), None
    perms, num, den, prev = [], 0.0, 0.0, None
    for A, B in zip(inc_a, inc_b):
        Bp = B if prev is None else np.hstack([B[:, :-1][:, prev], B[:, -1:]])
        cost = ((A[:, None, :] - Bp[None, :, :]) ** 2).sum(-1)
        ri, ci = linear_sum_assignment(cost)
        p = np.empty(A.shape[0], int)
        p[ri] = ci
        perms.append(p)
        num += float(cost[ri, ci].sum())
        den += float((A ** 2).sum() + (Bp ** 2).sum())
        prev = p
    Wbp = Wb if prev is None else Wb[prev]
    num += float(((Wa - Wbp) ** 2).sum() + ((ba - bb) ** 2).sum())
    den += float((Wa ** 2).sum() + (ba ** 2).sum() + (Wbp ** 2).sum() + (bb ** 2).sum())
    return (float(np.sqrt(2 * num / den)) if den > 0 else 0.0), perms


# -------------------------------------------------------------------- match
@dataclass
class GaugeMatch:
    """The gauge element carrying ``theta_j|C_j`` onto ``theta_i|C_i`` (or the best
    attempt), plus the residual distance."""
    residual: float
    matched: bool
    tol: float = MATCH_TOL
    perms: Optional[List[np.ndarray]] = field(default=None, repr=False)
    scales: Optional[List[np.ndarray]] = field(default=None, repr=False)
    output_gain: float = 1.0
    size_signature: Tuple[int, ...] = ()
    reason: str = ""

    def is_identity_up_to_tol(self) -> bool:
        return self.matched

    def to_dict(self) -> dict:
        return {"residual": float(self.residual), "matched": bool(self.matched), "tol": float(self.tol),
                "output_gain": float(self.output_gain), "size_signature": list(self.size_signature),
                "perms": None if self.perms is None else [p.tolist() for p in self.perms],
                "scales": None if self.scales is None else [np.asarray(s).tolist() for s in self.scales],
                "reason": self.reason}


def match_circuits(theta_i, C_i, theta_j, C_j, tol: float = MATCH_TOL, backend: str = "self"):
    """Align two circuits across fibers (§2.9). Returns ``(g, dist)``.

    ``theta_*`` is a ReLU-MLP-style parameter object (a :class:`ReluMLP`, a
    ``(Ws, bs)`` pair, an object with ``.Ws``/``.bs``, or a torch ``Sequential``);
    ``C_*`` is a circuit spec (``None`` = the whole network, a per-hidden-layer
    list of unit indices, or a dict/object with ``units`` and optionally
    ``algorithm``). ``g`` is a :class:`GaugeMatch` carrying the per-layer
    permutation and per-unit scaling that maps ``theta_j|C_j`` to ``theta_i|C_i``,
    and ``dist`` is the residual ``r`` (``inf`` if the layer sizes differ, so the
    circuits cannot be gauge-related at all).

    ``backend="flib"`` routes to the F-block experiment implementation
    (:func:`flib_crossseed`), which additionally collapses shared-activation-pattern
    groups; it needs the experiment tree and F-block model objects.

    Null: two independent random circuits give ``r`` of order 1;
    :func:`relabel_kill` is the planted positive.
    """
    if backend == "flib":
        return flib_crossseed(theta_i, C_i, theta_j, C_j, tol=tol)
    if backend != "self":
        raise ValueError(f"backend must be 'self' or 'flib', got {backend!r}")
    inc_a, out_a, norms_a, gain_a = effective_circuit(theta_i, C_i)
    inc_b, out_b, norms_b, gain_b = effective_circuit(theta_j, C_j)
    sig = tuple(int(a.shape[0]) for a in inc_a)
    r, perms = residual((inc_a, out_a), (inc_b, out_b))
    if perms is None:
        return GaugeMatch(residual=r, matched=False, tol=tol, size_signature=sig,
                          reason="layer sizes differ: no gauge element can relate these circuits"), r
    scales = [np.asarray(norms_a[l], float) / np.asarray(norms_b[l], float)[perms[l]]
              for l in range(len(perms))]
    g = GaugeMatch(residual=r, matched=bool(r < tol), tol=tol, perms=perms, scales=scales,
                   output_gain=float(gain_a / gain_b) if gain_b else float("nan"),
                   size_signature=sig,
                   reason="" if r < tol else f"residual {r:.3g} >= tol {tol:g}")
    return g, r


def flib_crossseed(theta_i, C_i, theta_j, C_j, tol: float = MATCH_TOL,
                   f_gauge_dir: Optional[str] = None):
    """Route :func:`match_circuits` through the F-block experiment implementation
    (``experiments/F_gauge/flib/crossseed``), which adds the functional-dimension
    collapse. ``C_*`` must be dicts carrying ``mid`` (the circuit id), ``X`` (the
    probe batch) and ``scheme``; ``theta_*`` must be F-block model objects.

    Raises ``ImportError`` if the experiment tree is not importable — the library
    itself never depends on it.
    """
    mod = _import_flib(f_gauge_dir)
    need = ("mid", "X", "scheme")
    for C in (C_i, C_j):
        if not isinstance(C, dict) or any(k not in C for k in need):
            raise ValueError(f"backend='flib' needs circuit dicts with keys {need}")
    sa = mod.effective_subimpl(theta_i, C_i["X"], C_i["mid"], C_i["scheme"])
    sb = mod.effective_subimpl(theta_j, C_j["X"], C_j["mid"], C_j["scheme"])
    r, perms = mod.residual(sa, sb)
    g = GaugeMatch(residual=r, matched=bool(r < tol), tol=tol, perms=perms,
                   size_signature=tuple(mod.size_signature(sa)), reason="flib backend")
    return g, r


def _import_flib(f_gauge_dir: Optional[str] = None):
    """Import ``flib.crossseed`` from the experiment tree by ``sys.path`` insertion
    (inside the function, so the library has no experiment dependency)."""
    if f_gauge_dir is None:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        f_gauge_dir = os.path.join(here, "experiments", "F_gauge")
    if f_gauge_dir not in sys.path:
        sys.path.insert(0, f_gauge_dir)
    import importlib
    return importlib.import_module("flib.crossseed")


# ----------------------------------------------------------------- counting
class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[a] = b
        return a != b


def _as_circuit_list(C) -> list:
    """Is ``C`` one circuit or several? A list/tuple counts as *several* only when
    every element is itself a circuit spec (None, a dict, or an object with
    ``.units``); a bare list of per-layer index arrays is one circuit."""
    if isinstance(C, (list, tuple)) and len(C) > 0 and all(
            c is None or isinstance(c, dict) or hasattr(c, "units") for c in C):
        return list(C)
    return [C]


@dataclass
class MechanismCount:
    """The cross-fiber mechanism count (§2.9 step 5 of ``open()``)."""
    K: int
    classes: List[List[Tuple[Any, int]]]
    N: int
    K_over_N: float
    tau: float
    K_alg: Optional[int] = None
    algorithms: Optional[Dict[str, int]] = None
    residuals: Dict[str, float] = field(default_factory=dict, repr=False)
    n_pairs: int = 0
    within_seed: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return {"K": int(self.K), "N": int(self.N), "K_over_N": float(self.K_over_N),
                "tau": float(self.tau), "K_alg": self.K_alg, "algorithms": self.algorithms,
                "classes": [[[str(s), int(c)] for s, c in cl] for cl in self.classes],
                "residuals": {k: float(v) for k, v in self.residuals.items()},
                "n_pairs": int(self.n_pairs), "within_seed": bool(self.within_seed),
                "notes": self.notes}


def count_mechanisms(seeds: Sequence, circuits: Sequence, tau: float = 0.1,
                     labels: Optional[Sequence] = None, within_seed: bool = False) -> MechanismCount:
    """|pi_0(C(F)/G)| estimate: union-find over cross-seed pairs with ``r < tau``.

    ``seeds[i]`` is the i-th trained parameter object; ``circuits[i]`` is that
    seed's circuit spec, or a list of circuit specs. Circuit ids are
    ``(label, circuit_index)`` with ``label = labels[i]`` (default ``i``).

    Pairs are compared *across seeds only* by default: within one generic network
    distinct circuits are never gauge-equivalent (Proposition 2.1), so
    within-network comparisons only ever return K/N = 1 and are noise here. Pass
    ``within_seed=True`` to include them (the redundancy finding).

    ``K_alg`` is the number of distinct ``algorithm`` labels carried by the
    circuits, when they carry one — the ground truth for the §8.3 negative
    (seeds trained to different named algorithms must give ``K = 2``).

    ``tau`` (default 0.1) is the residual at which two circuits are called the
    same mechanism; ``r < 1e-8`` for a literal relabelling and of order 1 for
    unrelated circuits, so the count is insensitive to tau over more than a
    decade — check ``residuals`` before trusting a K obtained near the boundary.
    """
    if len(seeds) != len(circuits):
        raise ValueError(f"{len(seeds)} seeds but {len(circuits)} circuit entries")
    labels = list(labels) if labels is not None else list(range(len(seeds)))
    items = []          # (seed_pos, label, circuit_index, theta, C)
    for i, (th, C) in enumerate(zip(seeds, circuits)):
        for j, c in enumerate(_as_circuit_list(C)):
            items.append((i, labels[i], j, th, c))
    n = len(items)
    uf = _UF(n)
    res: Dict[str, float] = {}
    n_pairs = 0
    for a in range(n):
        for b in range(a + 1, n):
            if items[a][0] == items[b][0] and not within_seed:
                continue
            n_pairs += 1
            _, r = match_circuits(items[a][3], items[a][4], items[b][3], items[b][4], tol=tau)
            res[f"{items[a][1]}:{items[a][2]}|{items[b][1]}:{items[b][2]}"] = float(r)
            if r < tau:
                uf.union(a, b)
    groups: Dict[int, List[Tuple[Any, int]]] = {}
    for idx, it in enumerate(items):
        groups.setdefault(uf.find(idx), []).append((it[1], it[2]))
    classes = [sorted(v, key=lambda t: (str(t[0]), t[1])) for _, v in sorted(groups.items())]
    algs = [_algorithm_of(it[4]) for it in items]
    named = [a for a in algs if a is not None]
    alg_counts = {a: named.count(a) for a in sorted(set(named))} if named else None
    K = len(classes)
    return MechanismCount(K=K, classes=classes, N=n, K_over_N=K / n if n else float("nan"),
                          tau=float(tau), K_alg=len(alg_counts) if alg_counts else None,
                          algorithms=alg_counts, residuals=res, n_pairs=n_pairs,
                          within_seed=within_seed,
                          notes="cross-seed pairs only (Proposition 2.1)" if not within_seed
                                else "all pairs, within-seed included")


# ------------------------------------------------------------- planted kill
def relabel_kill(theta, C=None, seed: int = 0, tol: float = 1e-8, kind: str = "combined") -> dict:
    """The §8.3 planted kill: build a literal gauge relabelling of ``theta`` and
    check that :func:`match_circuits` recovers it exactly.

    Applies a random permutation + positive scaling (``kind`` in
    ``{"perm", "scale", "combined"}``) to ``theta``, relabels ``C`` accordingly,
    and asserts ``r < tol`` (default 1e-8) and ``K == 1``. Returns a dict with
    ``passed``, ``residual``, ``K``, ``tol``, ``seed`` and ``kind`` — it *returns*
    the flag rather than raising, so a failure is a deliverable (Operating
    Constitution), and the caller decides.
    """
    from ..invariants.model import as_relu_mlp, apply_gauge
    m = as_relu_mlp(theta)
    widths = m.widths
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(w) if kind in ("perm", "combined") else np.arange(w) for w in widths]
    scales = [np.exp(rng.uniform(-1.0, 1.0, w)) if kind in ("scale", "combined") else np.ones(w)
              for w in widths]
    theta2 = apply_gauge(m, perms, scales)
    units = _circuit_units(C, widths)
    # apply_gauge puts old unit perms[l][i] at new position i, so old unit u sits at inv[u]
    units2 = []
    for l, p in enumerate(perms):
        inv = np.empty_like(p)
        inv[p] = np.arange(len(p))
        units2.append(np.sort(inv[units[l]]))
    g, r = match_circuits(theta, C, theta2, {"units": units2}, tol=tol)
    cnt = count_mechanisms([theta, theta2], [C, {"units": units2}], tau=max(tol, 1e-8))
    return {"passed": bool(r < tol and cnt.K == 1), "residual": float(r), "K": int(cnt.K),
            "tol": float(tol), "seed": int(seed), "kind": kind,
            "size_signature": list(g.size_signature),
            "note": "planted positive (FINAL_SPEC 8.3): a literal relabelling must align exactly"}
