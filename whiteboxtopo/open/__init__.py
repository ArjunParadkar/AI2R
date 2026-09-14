"""whiteboxtopo.open — the ``open()`` algorithm (FINAL_SPEC §3, §4, §5, §6).

    from whiteboxtopo.open import open, Behavior, grid
    card = open(model, Behavior("modadd", X, y, kind="group", p=113), seeds=[t1, t2])
    print(card.to_markdown()); print(grid([card]))

``open(model, behavior)`` runs the nine steps of §3 as a pipeline of named step
functions over a mutable :class:`OpenState`:

    1 step_gauge      G: permutations, scalings, hidden (Jacobian corank)
    2 step_gate       P0: attribution-graph legibility verdict
    3 step_extract    P1-P7 candidates from every available method
    4 step_certify    check_invariance on every candidate; artifacts discarded
    5 step_count      cross-fiber mechanism count (needs seeds)
    6 step_glue       cover -> Cech H0/H1 -> mechanism boundaries
    7 step_reconstruct  rebuild from certified objects; residual; gamma_cov
    8 step_verify     behavioral equivalence verdict
    9 step_report     the Scorecard

Operating Constitution: **negatives are deliverables**. Every step is wrapped so
that an exception marks the cells that step owns ``"not_reached"`` with the error
in ``provenance`` instead of aborting the run; a module that has not been written
yet gives ``"not reached: <module> unavailable"``. A Scorecard with eight
``not_reached`` cells is a valid, honest result — it is never an exception.

Cell statuses: ``certified`` (the answer is invariant under G and its null was
run), ``partial`` (produced but not fully certified, or certified under a stated
gauge-fixing), ``failed`` (produced and rejected — e.g. ``check_invariance``
called it an artifact, or a pre-registered threshold was missed),
``not_reached`` (no method available for this model/behavior).

Failure boundary of this module itself: the general-MLP paths implemented here
are the ReLU-MLP ones (gauge-fixed weight graph, path products, activation-based
explanation scores). Group/attractor extraction, reconstruction, verification,
spectral and geometric cells are delegated to ``whiteboxtopo.invariants.extractors``,
``whiteboxtopo.reconstruct``, ``whiteboxtopo.verify``, ``whiteboxtopo.spectral``
and ``whiteboxtopo.geometry``, imported lazily inside the step that needs them.
Deterministic given ``config["seed"]``.
"""
from __future__ import annotations

import importlib
import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["Behavior", "CellResult", "Scorecard", "OpenState", "open", "grid", "step",
           "CELLS", "PROBLEMS", "STATUSES", "DEFAULT_STEPS",
           "step_gauge", "step_gate", "step_extract", "step_certify", "step_count",
           "step_glue", "step_reconstruct", "step_verify", "step_report"]

CELLS = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")
PROBLEMS = {
    "P0": "is it explainable",
    "P1": "what it learned",
    "P2": "why this decision",
    "P3": "when bias entered",
    "P4": "what it does not know",
    "P5": "where it will fail",
    "P6": "memorizing vs learning",
    "P7": "how representations evolve",
}
STATUSES = ("certified", "not_reached", "failed", "partial")

# §2.14 / §6.1. PROVISIONAL: to be replaced by the V-block phase diagram.
GATE_RULE = ("tractable iff Q > 0.3 and treewidth_bound <= 8; intractable iff treewidth_bound > 32; "
             "else marginal — PROVISIONAL, to be replaced by the V-block phase diagram "
             "(V8 found no calibrated structure->tractability map, see legibility.profile.V8_CALIBRATION_ATTEMPT)")
GATE_Q, GATE_TW_OK, GATE_TW_BAD = 0.3, 8, 32


# ------------------------------------------------------------------ objects
@dataclass
class Behavior:
    """The behavior B being opened: a task, an input distribution and a metric.

    ``kind`` selects the extraction family: ``"group"`` (a finite group acts —
    Fourier/irrep extraction, ``p`` is the modulus), ``"attractor"`` (continuous
    attractor — persistent H1 + circular coordinate), ``"general"`` (everything
    else — path products, certified circuits). ``X`` is the input batch
    (array-like), ``y`` the targets if any, ``metric`` an optional
    ``metric(model, X, y) -> float``.
    """
    name: str
    X: Any
    y: Any = None
    metric: Optional[Callable] = None
    kind: str = "general"
    p: Optional[int] = None

    def __post_init__(self):
        if self.kind not in ("group", "attractor", "general"):
            raise ValueError(f"kind must be group|attractor|general, got {self.kind!r}")

    @property
    def n(self) -> int:
        return int(np.asarray(self.X).shape[0])

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "p": self.p, "n": self.n,
                "has_y": self.y is not None, "has_metric": self.metric is not None}


@dataclass
class CellResult:
    """One cell of the scorecard: the answer to one of P0-P7 for this model."""
    problem: str
    status: str = "not_reached"
    value: Any = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    null: Dict[str, Any] = field(default_factory=dict)
    provenance: str = ""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")

    def short(self) -> str:
        if self.value is None:
            return self.status
        v = self.value
        if isinstance(v, float):
            v = f"{v:.3g}"
        s = str(v)
        return f"{self.status} ({s[:28]})"

    def to_dict(self) -> dict:
        return {"problem": self.problem, "question": PROBLEMS.get(self.problem, ""),
                "status": self.status, "value": _jsonable(self.value),
                "evidence": _jsonable(self.evidence), "null": _jsonable(self.null),
                "provenance": self.provenance}


@dataclass
class Scorecard:
    """The output of :func:`open`: eight cells, a reconstruction verdict, and the
    coverage gap gamma_cov (the structured fraction of the unexplained residue).

    ``degree_of_opening = 1 - gamma_cov`` is the headline number (§1.1); it is
    ``None`` when no reconstruction was reached — an unknown gap is never
    reported as a zero gap.
    """
    model: str
    behavior: Optional[dict] = None
    cells: Dict[str, CellResult] = field(default_factory=dict)
    reconstruction: Any = None
    gamma_cov: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def degree_of_opening(self) -> Optional[float]:
        return None if self.gamma_cov is None else float(1.0 - self.gamma_cov)

    def status_counts(self) -> Dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for c in self.cells.values():
            out[c.status] += 1
        return out

    def to_grid(self, values: bool = False) -> list:
        """One row of the §5 headline grid: ``[model, P0, ..., P7, degree]``."""
        deg = self.degree_of_opening
        row = [self.model]
        for p in CELLS:
            c = self.cells.get(p)
            row.append("not_reached" if c is None else (c.short() if values else c.status))
        row.append("—" if deg is None else f"{deg:.3f}")
        return row

    def to_markdown(self) -> str:
        head = [f"### open({self.model}) — {self.behavior['name'] if self.behavior else 'behavior'}", ""]
        deg = self.degree_of_opening
        head.append(f"gamma_cov = {'unknown' if self.gamma_cov is None else f'{self.gamma_cov:.4f}'}; "
                    f"degree of opening 1 - gamma_cov = {'unknown' if deg is None else f'{deg:.4f}'}")
        head += ["", "| cell | question | status | value | provenance |", "|---|---|---|---|---|"]
        for p in CELLS:
            c = self.cells.get(p) or CellResult(p)
            v = c.value
            v = f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))
            prov = c.provenance.replace("|", "/").replace("\n", " ")
            head.append(f"| {p} | {PROBLEMS[p]} | {c.status} | {_md(v)} | {_md(prov)} |")
        if self.reconstruction is not None:
            head += ["", f"reconstruction: {self.reconstruction}"]
        if self.errors:
            head += ["", "errors:"] + [f"* {e}" for e in self.errors]
        return "\n".join(head)

    def to_dict(self) -> dict:
        return {"model": self.model, "behavior": self.behavior,
                "cells": {p: (self.cells[p].to_dict() if p in self.cells else CellResult(p).to_dict())
                          for p in CELLS},
                "reconstruction": _jsonable(self.reconstruction), "gamma_cov": self.gamma_cov,
                "degree_of_opening": self.degree_of_opening, "status_counts": self.status_counts(),
                "errors": list(self.errors), "provenance": _jsonable(self.provenance),
                "config": _jsonable(self.config)}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


class OpenState(dict):
    """The mutable context threaded through the nine steps (a dict with helpers).

    Keys: ``model``, ``behavior``, ``seeds``, ``checkpoints``, ``config``,
    ``cells`` (``{P#: CellResult}``), ``candidates``, ``errors``, ``gamma_cov``,
    ``reconstruction``, ``provenance``.
    """

    def cell(self, p: str) -> CellResult:
        return self["cells"].setdefault(p, CellResult(p))

    def set_cell(self, p: str, status: str, value=None, evidence=None, null=None, provenance="") -> CellResult:
        c = CellResult(p, status, value, dict(evidence or {}), dict(null or {}), provenance)
        self["cells"][p] = c
        return c

    def not_reached(self, p: str, why: str) -> CellResult:
        return self.set_cell(p, "not_reached", provenance=f"not reached: {why}")

    def note(self, key: str, value) -> None:
        self["provenance"][key] = value


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "to_dict"):
        try:
            return _jsonable(o.to_dict())
        except Exception:
            return str(o)
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    return str(o)


def _md(s: str) -> str:
    s = str(s)
    return s if len(s) <= 90 else s[:87] + "..."


# -------------------------------------------------------------- step plumbing
def step(name: str, cells: Sequence[str] = ()):
    """Decorator: name a step and declare the cells it owns.

    A step owns a cell if its failure is the reason that cell has no answer. When
    a step raises, every owned cell that is still unanswered becomes
    ``not_reached`` with the error in ``provenance``; an owned cell that an
    earlier step already answered is downgraded ``certified -> partial`` and the
    error appended. Nothing propagates out of :func:`open`.
    """
    def deco(fn):
        fn.step_name = name
        fn.cells = tuple(cells)
        return fn
    return deco


def _fail_cells(state: OpenState, cells: Sequence[str], msg: str) -> None:
    for p in cells:
        cur = state["cells"].get(p)
        if cur is None or cur.status == "not_reached":
            state.set_cell(p, "not_reached", provenance=msg)
        else:
            if cur.status == "certified":
                cur.status = "partial"
            cur.provenance = (cur.provenance + "; " if cur.provenance else "") + msg


def _run_step(fn, state: OpenState) -> OpenState:
    name = getattr(fn, "step_name", getattr(fn, "__name__", "step"))
    cells = tuple(getattr(fn, "cells", ()))
    try:
        out = fn(state)
        return out if isinstance(out, OpenState) else state
    except Exception as e:                                    # negatives are deliverables
        msg = f"not reached: {name} raised {type(e).__name__}: {e}"
        state["errors"].append(msg)
        state["provenance"].setdefault("tracebacks", {})[name] = traceback.format_exc(limit=3)
        _fail_cells(state, cells, msg)
        return state


def _lazy(modpath: str):
    """Import a module that another agent may still be writing. Returns the module
    or raises ``ImportError`` — callers turn that into ``not reached: <module> unavailable``."""
    return importlib.import_module(modpath)


def _pending(state: OpenState, cells, modpath: str, exc: Optional[BaseException] = None,
             what: str = "") -> None:
    """Mark cells ``not_reached``. An :class:`ImportError` means the module is still
    being written ("unavailable"); anything else is a real failure of an existing
    module and is reported as such — the two must not be confused."""
    if exc is None or isinstance(exc, (ImportError, ModuleNotFoundError)):
        msg = f"not reached: {modpath} unavailable (module pending)"
        if exc is not None and not isinstance(exc, (ImportError, ModuleNotFoundError)):
            msg += f": {exc}"
    else:
        msg = f"not reached: {modpath}{'.' + what if what else ''} raised {type(exc).__name__}: {exc}"
    for p in cells:
        state.set_cell(p, "not_reached", provenance=msg)


# --------------------------------------------------------- MLP helper pieces
def _as_mlp(model):
    from ..invariants.model import as_relu_mlp
    return as_relu_mlp(model)


def _gauge_fixed_weights(model):
    """Unit-normalised incoming weights with the norms pushed downstream — the
    scale gauge-fixing of ``whiteboxtopo.align`` (a function of gauge invariants)."""
    from ..align import _normalize
    m = _as_mlp(model)
    Ws, bs, _, _ = _normalize(m.Ws, m.bs)
    return Ws, bs


def _weight_graph(model) -> np.ndarray:
    """Gauge-fixed weight-magnitude attribution graph of a ReLU MLP: nodes are
    inputs, hidden units and outputs in layer order, edge weight ``|W|`` in the
    gauge-fixed frame. Permutation-invariant (isomorphism) and scaling-invariant
    (the frame is fixed) — used when the model is not node-instrumented for EAP.
    """
    Ws, _ = _gauge_fixed_weights(model)
    sizes = [Ws[0].shape[0]] + [W.shape[1] for W in Ws]
    off = np.cumsum([0] + sizes)
    N = int(off[-1])
    A = np.zeros((N, N))
    for l, W in enumerate(Ws):
        A[off[l]:off[l + 1], off[l + 1]:off[l + 2]] = np.abs(W)
    return A + A.T


def _mlp_explanation_scores(model, X) -> np.ndarray:
    """Per-example, per-hidden-unit explanation scores ``a_u(x) * ||w_out(u)||``.

    Gauge-invariant: a per-unit scaling multiplies the activation by alpha and
    divides the outgoing row by alpha, so the product is unchanged; a permutation
    permutes the feature axis identically for every example, which leaves every
    Jaccard agreement in the sheaf computation unchanged. Shape (N, n_units).
    """
    m = _as_mlp(model)
    _, acts, _ = m.forward(np.asarray(X, dtype=float))
    cols = []
    for l, h in enumerate(acts):
        w = np.linalg.norm(m.Ws[l + 1], axis=1)         # outgoing norm per unit of layer l
        cols.append(np.asarray(h) * w[None, :])
    return np.concatenate(cols, axis=1)


# ------------------------------------------------------------------- 1 gauge
@step("step_gauge", ())
def step_gauge(state: OpenState) -> OpenState:
    """§3.1 — the gauge group G: permutations, scalings, and the hidden excess via
    the Jacobian corank. Records exact vs approximate symmetries; owns no cell but
    every ``certified`` below means "invariant under this G"."""
    from .. import symmetry as S
    model = state["model"]
    desc = S.describe(model)
    rec: Dict[str, Any] = {"type": desc.get("type"), "known_dim": desc.get("known_dim"),
                           "n_params": desc.get("n_params"), "widths": desc.get("widths"),
                           "exact": desc.get("type") == "relu_mlp",
                           "note": "ReLU MLP: permutations x positive scalings are exact. "
                                   "Non-homogeneous activations / LayerNorm make MLP scalings "
                                   "approximate — see symmetry.transformer_gauge."}
    cfg = state["config"]
    want_hidden = cfg.get("hidden_symmetry")
    if want_hidden is None:
        want_hidden = bool(desc.get("type") == "relu_mlp" and (desc.get("n_params") or 0) <= 4000)
    if want_hidden:
        rep = S.hidden(model, np.asarray(state["behavior"].X, dtype=float), seed=cfg.get("seed", 0))
        rec.update(corank=int(rep.corank), excess=int(rep.excess), unexplained=int(rep.unexplained),
                   identified=dict(rep.identified), G_incomplete=bool(rep.G_incomplete))
    state["gauge"] = rec
    state.note("gauge", rec)
    return state


# -------------------------------------------------------------------- 2 gate
@step("step_gate", ("P0",))
def step_gate(state: OpenState) -> OpenState:
    """§3.2 / §6.1 — P0. Attribution graph -> (Q, treewidth bound, sparsity) ->
    legibility verdict. Uses EAP (``whiteboxtopo.graphs.attribution_graph``) when the
    model is node-instrumented, otherwise the gauge-fixed weight graph.

    Verdict rule (PROVISIONAL, see :data:`GATE_RULE`): tractable iff Q > 0.3 and
    tw <= 8; intractable iff tw > 32; else marginal. Null: degree-preserving
    rewiring of the same graph. If the verdict is ``intractable`` the pipeline
    continues with partial extraction only (§2.14).
    """
    from ..legibility.graph import modularity_q, treewidth_bound, threshold_density, sparsity
    model, cfg = state["model"], state["config"]
    density = float(cfg.get("gate_density", 0.1))
    source = "gauge_fixed_weight_graph"
    if hasattr(model, "node_means") and hasattr(model, "edge_scores"):
        from ..graphs import attribution_graph
        A = attribution_graph(model, (state["behavior"].X, state["behavior"].y), threshold=density)
        source = "eap_attribution_graph"
    else:
        A = threshold_density(_weight_graph(model), density)
    Q, _ = modularity_q(A, seed=cfg.get("seed", 0))
    tw = int(treewidth_bound(A))
    sp = float(sparsity(A[np.triu_indices(A.shape[0], 1)]))
    verdict = "tractable" if (Q > GATE_Q and tw <= GATE_TW_OK) else ("intractable" if tw > GATE_TW_BAD else "marginal")
    null = {}
    if cfg.get("gate_null", A.shape[0] <= 64):
        try:
            from ..graphs import rewired_null
            mu, sd = rewired_null(A, n=int(cfg.get("gate_null_n", 5)), seed=cfg.get("seed", 0))
            null = {"kind": "degree_preserving_rewiring", "Q_mean": float(mu), "Q_sd": float(sd),
                    "Q_excess": float(Q - mu)}
        except Exception as e:
            null = {"kind": "degree_preserving_rewiring", "error": f"{type(e).__name__}: {e}"}
    state["legibility"] = {"Q": float(Q), "tw": tw, "sparsity": sp, "verdict": verdict,
                           "N": int(A.shape[0]), "density": density, "source": source}
    state.set_cell("P0", "certified" if source == "gauge_fixed_weight_graph" else "partial",
                   value=verdict, evidence=dict(state["legibility"], rule=GATE_RULE), null=null,
                   provenance=("legibility.graph.modularity_q + treewidth_bound on the "
                               f"{source} (density {density}); Q/tw are isomorphism invariants of a "
                               "gauge-fixed graph, so the verdict is invariant under G"
                               if source == "gauge_fixed_weight_graph" else
                               "graphs.attribution_graph (EAP); invariance of the EAP scores under G "
                               "was NOT audited here — hence partial"))
    return state


# ----------------------------------------------------------------- 3 extract
@step("step_extract", ("P1", "P2", "P3", "P4", "P5", "P6", "P7"))
def step_extract(state: OpenState) -> OpenState:
    """§3.3 — run every available extraction method and stash candidates.

    Candidates carry an optional ``metric_fn(model) -> array`` so that
    :func:`step_certify` can audit them with ``check_invariance``. Cells whose
    method lives in a module that is not importable become
    ``not reached: <module> unavailable``.
    """
    b, model, cfg = state["behavior"], state["model"], state["config"]
    cand: Dict[str, dict] = state.setdefault("candidates", {})
    X = np.asarray(b.X, dtype=float)

    # ---- P2: path products (exact gauge-invariant per-decision attribution, §2.5)
    try:
        ex = _lazy("whiteboxtopo.invariants.extractors")
        pp = getattr(ex, "path_product")
        A0, c0 = pp(model, X[0])
        cand["P2"] = {"name": "path_product", "value": "A(x) path product",
                      "evidence": {"A_shape": list(np.shape(A0)), "top_contrib":
                                   _jsonable(np.sort(np.abs(np.asarray(A0) @ X[0]).ravel())[::-1][:5])},
                      "metric_fn": (lambda m: np.asarray(pp(m, X[0])[0], dtype=float)),
                      "provenance": "invariants.extractors.path_product (FINAL_SPEC 2.5)"}
    except Exception as e:
        _pending(state, ("P2",), "whiteboxtopo.invariants.extractors", e, "path_product")

    # ---- P1: what it learned
    if b.kind == "group":
        try:
            ex = _lazy("whiteboxtopo.invariants.extractors")
            emb = cfg.get("embedding")
            if emb is None:
                raise ValueError("group extraction needs config['embedding'] (the token embedding matrix)")
            res = ex.fourier_irreps(np.asarray(emb), int(b.p), cfg.get("tau_fourier", 0.05))
            K = tuple(res.K) if hasattr(res, "K") else tuple(res[0])
            kappa = float(res.kappa if hasattr(res, "kappa") else res[1])
            cand["P1"] = {"name": "fourier_irreps", "value": {"K": list(K), "kappa": kappa},
                          "evidence": _jsonable(res), "result": res, "metric_fn": None,
                          "provenance": "invariants.extractors.fourier_irreps (FINAL_SPEC 2.6)"}
            cand["P6"] = {"name": "fourier_concentration", "value": kappa,
                          "evidence": {"kappa": kappa, "pass": bool(kappa > 0.9),
                                       "rule": "generalizer kappa > 0.9, memorizing control < 0.3 (6.7)"},
                          "metric_fn": None, "provenance": "invariants.extractors.fourier_irreps"}
        except Exception as e:
            _pending(state, ("P1",), "whiteboxtopo.invariants.extractors", e, "fourier_irreps")
    elif b.kind == "attractor":
        try:
            ex = _lazy("whiteboxtopo.invariants.extractors")
            cloud = cfg.get("cloud")
            if cloud is None:
                m = _as_mlp(model)
                cloud = m.forward(X)[1][-1]
            theta, ratio = ex.circular_coordinate(np.asarray(cloud))
            cand["P1"] = {"name": "circular_coordinate", "value": "S^1 attractor coordinate",
                          "evidence": {"n": int(np.size(theta)), "lifetime_ratio": float(ratio),
                                       "identified": bool(ratio > 5.0),
                                       "rule": "SPEC 6.2: top H1 bar lifetime > 5x the next"},
                          "theta": theta, "metric_fn": None,
                          "provenance": "invariants.extractors.circular_coordinate (FINAL_SPEC 2.7)"}
        except Exception as e:
            _pending(state, ("P1",), "whiteboxtopo.invariants.extractors", e, "circular_coordinate")
    else:
        try:
            from ..invariants.metrics import invariant_quantities
            q = invariant_quantities(model, X, with_topology=False)
            cand["P1"] = {"name": "invariant_inventory", "value": sorted(q),
                          "evidence": {k: list(np.shape(v)) for k, v in q.items()},
                          "metric_fn": None, "status_hint": "partial",
                          "provenance": ("invariants.metrics.invariant_quantities — a certified "
                                         "inventory of gauge-invariant observables, NOT a feature "
                                         "inventory: no SAE/probe extractor ships for general models")}
        except Exception as e:
            state.not_reached("P1", f"invariant inventory failed: {type(e).__name__}: {e}")

    # ---- P4: what it does not know (spectral geometry)
    try:
        sp = _lazy("whiteboxtopo.spectral")
        m = _as_mlp(model)
        cloud = np.asarray(m.forward(X)[1][-1])
        ood = cfg.get("ood_X")
        h_new = np.asarray(_as_mlp(model).forward(np.asarray(ood, dtype=float))[1][-1]) \
            if ood is not None else cloud
        res = sp.out_of_manifold(cloud, h_new, seed=int(cfg.get("seed", 0)))
        dM = np.asarray(getattr(res, "d_M", res), dtype=float)
        cand["P4"] = {"name": "out_of_manifold", "value": float(np.mean(dM)),
                      "evidence": {"n": int(dM.size), "has_ood_batch": ood is not None,
                                   "mean_d_M": float(np.mean(dM)),
                                   "frac_flagged": float(np.mean(np.asarray(res.flags)))
                                   if hasattr(res, "flags") else None,
                                   "note": None if ood is not None else
                                   "no config['ood_X']: scored the in-distribution cloud against "
                                   "itself, so this is a calibration read-out, not an OOD result"},
                      "metric_fn": None, "provenance": "spectral.out_of_manifold (FINAL_SPEC 2.10, 6.5)"}
    except Exception as e:
        _pending(state, ("P4",), "whiteboxtopo.spectral", e, "out_of_manifold")

    # ---- P5: where it will fail (curvature x Lipschitz)
    try:
        ge = _lazy("whiteboxtopo.geometry")
        fp = ge.failure_predictor(model, X)
        rho = np.asarray(getattr(fp, "rho", fp), dtype=float)
        cand["P5"] = {"name": "failure_predictor", "value": float(np.mean(rho)),
                      "evidence": {"n": int(rho.size), "mean_rho": float(np.mean(rho)),
                                   "note": "SPEC 6.6 needs perturbation_failure_radius + spearman_test "
                                           "against a softmax-margin baseline to become a verdict"},
                      "metric_fn": None,
                      "provenance": "geometry.failure_predictor rho = ||Q|| * Lip (FINAL_SPEC 2.11)"}
    except Exception as e:
        _pending(state, ("P5",), "whiteboxtopo.geometry", e, "failure_predictor")

    # ---- P6: memorizing vs learning (non-group models: the audited M-block signal)
    if "P6" not in cand:
        fam = cfg.get("family")
        if fam is None:
            state.not_reached("P6", "memorization_signal needs config['family'] (per-family calibration, "
                                    "M block); no group task, so the Fourier criterion does not apply")
        else:
            try:
                from ..signals import memorization_signal
                sig = memorization_signal(model=model, probe=cfg.get("probe"), layer=cfg.get("layer"),
                                          activations=cfg.get("activations"), family=fam,
                                          seed=cfg.get("seed", 0))
                cand["P6"] = {"name": "memorization_signal", "value": float(getattr(sig, "score", np.nan)),
                              "evidence": _jsonable(sig), "metric_fn": None,
                              "provenance": "signals.memorization_signal (M block, calibrated per family)"}
            except Exception as e:
                state.not_reached("P6", f"memorization_signal failed: {type(e).__name__}: {e}")

    # ---- P3 / P7: need checkpoints
    ck = state.get("checkpoints")
    if not ck:
        state.not_reached("P3", "no checkpoints provided (P3 needs the training trajectory)")
        state.not_reached("P7", "no checkpoints provided (P7 needs the training trajectory)")
    else:
        try:
            sp = _lazy("whiteboxtopo.spectral")
            clouds = [_as_mlp(c).forward(X)[1][-1] for c in ck]
            flow = sp.eigenflow(clouds)
            cand["P7"] = {"name": "eigenflow", "value": "Laplacian eigenflow over checkpoints",
                          "evidence": {"n_checkpoints": len(ck), "flow": _jsonable(flow)},
                          "metric_fn": None, "provenance": "spectral.eigenflow (FINAL_SPEC 2.10, 6.8)"}
        except Exception as e:
            _pending(state, ("P7",), "whiteboxtopo.spectral", e, "eigenflow")
        state.not_reached("P3", "P3 needs a planted/probe bias target and the zigzag instrument "
                                "(whiteboxtopo.topology.zigzag) wired to it; not run by this pipeline")

    for p, c in cand.items():
        state.set_cell(p, c.get("status_hint", "partial"), value=c.get("value"),
                       evidence=c.get("evidence", {}), provenance=c["provenance"] + " [uncertified]")
    return state


# ----------------------------------------------------------------- 4 certify
@step("step_certify", ())
def step_certify(state: OpenState) -> OpenState:
    """§3.4 — every candidate through ``invariants.check_invariance``. Artifacts are
    discarded (cell ``failed``), covariants are kept with their fitted character,
    invariants become ``certified``. A candidate with no ``metric_fn`` cannot be
    audited and stays ``partial`` with that stated in its provenance."""
    from ..invariants import check_invariance
    model, cfg = state["model"], state["config"]
    n = int(cfg.get("certify_n", 12))
    audits = {}
    for p, c in state.get("candidates", {}).items():
        cell = state.cell(p)
        fn = c.get("metric_fn")
        if fn is None:
            cell.provenance = c["provenance"] + " [not auditable: no metric_fn — status partial]"
            continue
        try:
            res = check_invariance(fn, model, n=n, seed=cfg.get("seed", 0), name=c["name"],
                                   uniform_sweep=False, keep_samples=False)
        except Exception as e:
            cell.provenance = c["provenance"] + f" [certification failed: {type(e).__name__}: {e}]"
            continue
        audits[p] = res.row()
        cell.evidence["audit"] = res.row()
        if res.verdict == "invariant":
            cell.status = "certified"
            cell.provenance = c["provenance"] + f" [certified: check_invariance verdict=invariant, " \
                                                f"median CV={min(res.median_cv.values()):.2g}]"
        elif res.verdict == "covariant":
            cell.status = "certified"
            cell.provenance = c["provenance"] + f" [certified after normalisation: covariant, " \
                                                f"k={res.scaling_exponent}]"
        else:
            cell.status = "failed"
            cell.provenance = c["provenance"] + " [discarded: check_invariance verdict=artifact]"
    state["audits"] = audits
    state.note("certification", {"n_samples": n, "audited": sorted(audits)})
    return state


# ------------------------------------------------------------------- 5 count
@step("step_count", ("P1",))
def step_count(state: OpenState) -> OpenState:
    """§3.5 / §2.9 — cross-fiber circuit alignment over the provided seeds:
    ``K = |pi_0(C(F)/G)|`` and the named algorithms. Attached to P1 as evidence;
    with no seeds the count is simply absent (not a failure)."""
    seeds = state.get("seeds")
    if not seeds:
        state.note("mechanisms", "no seeds provided: cross-fiber count not run "
                                 "(within-network K/N == 1 by Proposition 2.1)")
        return state
    from ..align import count_mechanisms
    cfg = state["config"]
    circuits = cfg.get("circuits") or [None] * len(seeds)
    mc = count_mechanisms(seeds, circuits, tau=float(cfg.get("align_tau", 0.1)),
                          labels=cfg.get("seed_labels"))
    state["mechanisms"] = mc
    state.note("mechanisms", mc.to_dict())
    cell = state.cell("P1")
    cell.evidence["mechanism_count"] = mc.to_dict()
    cell.provenance = (cell.provenance + "; " if cell.provenance else "") + \
        f"align.count_mechanisms: K={mc.K} over N={mc.N} circuits at tau={mc.tau}"
    return state


# -------------------------------------------------------------------- 6 glue
@step("step_glue", ("P2",))
def step_glue(state: OpenState) -> OpenState:
    """§3.6 / §2.12 — cover the behavior's inputs, compute the local explanations'
    Čech H0/H1 and localize the mechanism boundaries. Attached to P2 (the
    per-decision cell) as evidence, with the §2.12 sanity gate recorded."""
    from ..sheaf import cover as make_cover, cech_H0_H1, sanity_gate
    b, model, cfg = state["behavior"], state["model"], state["config"]
    X = np.asarray(b.X, dtype=float)
    E = cfg.get("explanation_scores")
    E = _mlp_explanation_scores(model, X) if E is None else np.asarray(E, dtype=float)
    emb = cfg.get("cover_embedding")
    emb = X if emb is None else np.asarray(emb, dtype=float)
    k = int(cfg.get("cover_k", max(2, min(8, len(X) // 10))))
    n_min = int(cfg.get("cover_n_min", max(2, len(X) // (5 * k))))
    cov = make_cover(emb, k, overlap=float(cfg.get("cover_overlap", 0.5)),
                     seed=int(cfg.get("seed", 0)), n_min=n_min)
    res = cech_H0_H1(cov, E, tau=float(cfg.get("glue_tau", 0.5)),
                     k_e=int(cfg.get("glue_k_e", max(2, E.shape[1] // 4))))
    state["cech"] = res
    state["cover"] = cov
    state.note("glue", {"cover": cov.to_dict(), "cech": res.to_dict(),
                        "sanity_gate_H1_zero": bool(sanity_gate(res))})
    cell = state.cell("P2")
    cell.evidence["cech"] = res.to_dict()
    cell.evidence["cover"] = cov.to_dict()
    cell.evidence["sanity_gate_H1_zero"] = bool(sanity_gate(res))
    cell.provenance = (cell.provenance + "; " if cell.provenance else "") + \
        f"sheaf.cech_H0_H1: H0={res.H0}, H1={res.H1}, {len(res.boundaries)} mechanism boundaries"
    return state


# -------------------------------------------------------------- 7 reconstruct
@step("step_reconstruct", ())
def step_reconstruct(state: OpenState) -> OpenState:
    """§3.7 — rebuild the behavior from certified objects only (no weight copying),
    then the residual and the coverage gap gamma_cov via the Marchenko-Pastur edge
    (``whiteboxtopo.reconstruct``).

    Group tasks rebuild from the certified frequency set (R1), attractors from the
    circular coordinate + fitted tangent dynamics (R2), everything else from a
    certified circuit with its complement mean-ablated (R3). The general path needs
    ``config["circuit_mask"]``: rebuilding from *all* units is the identity, whose
    gamma_cov is zero by construction and means nothing — so without a circuit this
    cell reports why it was not reached rather than a free zero. The residue is
    taken on the output logits unless ``config["residual_acts"]`` supplies
    (h, h_hat) activation matrices.
    """
    b, model, cfg = state["behavior"], state["model"], state["config"]
    try:
        rc = _lazy("whiteboxtopo.reconstruct")
    except Exception as e:
        msg = f"not reached: whiteboxtopo.reconstruct unavailable (module pending): {e}"
        state["errors"].append(msg)
        state.note("reconstruction", msg)
        return state
    X = np.asarray(b.X, dtype=float)
    cand = state.get("candidates", {})
    certified = {p: c for p, c in cand.items() if state.cell(p).status == "certified"}
    if b.kind == "group":
        src = certified.get("P1") or cand.get("P1")
        if src is None or src.get("result") is None:
            state.note("reconstruction", "not reached: no certified irrep inventory to rebuild from (R1)")
            return state
        res = src["result"]
        f_hat = rc.from_irreps(res.K, res.amplitude, b.p)
    elif b.kind == "attractor":
        dyn = cfg.get("dynamics")
        src = certified.get("P1") or cand.get("P1")
        if dyn is None or src is None or src.get("theta") is None:
            state.note("reconstruction", "not reached: R2 needs the circular coordinate and a fitted "
                                         "TangentDynamics in config['dynamics']")
            return state
        f_hat = rc.from_attractor(src["theta"], dyn)
    else:
        mask = cfg.get("circuit_mask")
        if mask is None:
            state.note("reconstruction", "not reached: R3 needs config['circuit_mask'] (a certified "
                                         "circuit); the full mask is the identity and its gamma_cov "
                                         "is zero by construction")
            return state
        f_hat = rc.from_circuits(model, mask, X, ablation=cfg.get("ablation", "mean"))
    pair = cfg.get("residual_acts")
    if pair is not None:
        h, h_hat = (np.asarray(a, dtype=float) for a in pair)
    else:
        h = np.asarray(_as_mlp(model).logits(X), dtype=float)
        h_hat = np.atleast_2d(np.asarray(f_hat(X), dtype=float))
    R = rc.residual(h, h_hat)
    gap = rc.coverage_gap(R, seed=int(cfg.get("seed", 0)))
    state["f_hat"] = f_hat
    state["gamma_cov"] = float(getattr(gap, "gamma_cov", gap))
    state.note("reconstruction", {"kind": b.kind, "coverage_gap": _jsonable(gap),
                                  "residual_shape": list(R.shape),
                                  "residual_on": "logits" if pair is None else "given activations"})
    return state


# ------------------------------------------------------------------ 8 verify
@step("step_verify", ())
def step_verify(state: OpenState) -> OpenState:
    """§3.8 — behavioral equivalence of f_hat vs f_theta (accuracy, logit correlation,
    Gaussian-perturbation curves, OOD agreement) via ``whiteboxtopo.verify.equivalence``.
    Without a reconstruction there is nothing to verify, which is recorded, not raised."""
    if state.get("f_hat") is None:
        state.note("verify", "not reached: no reconstruction to verify")
        return state
    vf = _lazy("whiteboxtopo.verify")
    b, cfg, model = state["behavior"], state["config"], state["model"]
    X = np.asarray(b.X, dtype=float)
    kw = {"y": b.y, "ood": cfg.get("ood_X"), "seed": int(cfg.get("seed", 0))}
    if cfg.get("radii") is not None:
        kw["radii"] = cfg["radii"]
    verdict = vf.equivalence(lambda Z: _as_mlp(model).logits(np.atleast_2d(np.asarray(Z, dtype=float))),
                             state["f_hat"], X, **kw)
    state["reconstruction"] = verdict
    state.note("verify", _jsonable(verdict))
    return state


# ------------------------------------------------------------------ 9 report
@step("step_report", ())
def step_report(state: OpenState) -> OpenState:
    """§3.9 — assemble the Scorecard: eight cells, the reconstruction verdict, the
    coverage gap and the degree of opening ``1 - gamma_cov``. Any cell never
    touched is reported ``not_reached`` with the reason."""
    for p in CELLS:
        if p not in state["cells"]:
            state.not_reached(p, "no step produced a candidate for this cell")
    card = Scorecard(model=state["model_name"], behavior=state["behavior"].to_dict(),
                     cells={p: state["cells"][p] for p in CELLS},
                     reconstruction=state.get("reconstruction"), gamma_cov=state.get("gamma_cov"),
                     errors=list(state["errors"]), provenance=dict(state["provenance"]),
                     config={k: _jsonable(v) for k, v in state["config"].items()
                             if not callable(v) and not isinstance(v, np.ndarray)})
    state["scorecard"] = card
    return state


DEFAULT_STEPS = (step_gauge, step_gate, step_extract, step_certify, step_count,
                 step_glue, step_reconstruct, step_verify, step_report)
_BY_NAME = {f.step_name: f for f in DEFAULT_STEPS}


def open(model, behavior: Behavior, seeds=None, checkpoints=None, steps=None,
         config: Optional[dict] = None) -> Scorecard:
    """Open ``model`` on ``behavior`` to its provable degree (FINAL_SPEC §3).

    Parameters
    ----------
    model : the trained network (ReLU MLP / ``ReluMLP`` / torch ``Sequential`` for
        the paths implemented here; other architectures reach the cells their
        modules support and report ``not_reached`` for the rest).
    behavior : :class:`Behavior` — task name, inputs, targets, metric, kind.
    seeds : optional independently trained parameter objects for the same
        behavior; enables the §2.9 cross-fiber mechanism count.
    checkpoints : optional training checkpoints (P3, P7).
    steps : optional sequence of step callables or step names; defaults to the
        nine of §3. ``step_report`` is always run last.
    config : optional dict; recognised keys include ``seed``, ``certify_n``,
        ``gate_density``, ``gate_null``, ``cover_k``, ``cover_overlap``,
        ``glue_tau``, ``glue_k_e``, ``explanation_scores``, ``cover_embedding``,
        ``align_tau``, ``circuits``, ``seed_labels``, ``embedding`` (group tasks),
        ``family``/``probe``/``layer``/``activations`` (P6), ``ood_X``, ``radii``,
        ``hidden_symmetry``, ``model_name``.

    Returns a :class:`Scorecard`. **Never raises for a failure inside a step**:
    the affected cells come back ``not_reached`` with the error in their
    provenance and in ``Scorecard.errors`` (Operating Constitution — negatives are
    deliverables). Only a malformed call (a bad ``steps`` name) raises.
    """
    cfg = dict(config or {})
    state = OpenState(model=model, behavior=behavior, seeds=list(seeds) if seeds else None,
                      checkpoints=list(checkpoints) if checkpoints else None, config=cfg,
                      cells={}, candidates={}, errors=[], provenance={},
                      gamma_cov=None, reconstruction=None, f_hat=None,
                      model_name=str(cfg.get("model_name", type(model).__name__)))
    if steps is None:
        chosen = list(DEFAULT_STEPS)
    else:
        chosen = []
        for s in steps:
            if isinstance(s, str):
                if s not in _BY_NAME:
                    raise ValueError(f"unknown step {s!r}; known: {sorted(_BY_NAME)}")
                chosen.append(_BY_NAME[s])
            else:
                chosen.append(s)
    if step_report not in chosen:
        chosen.append(step_report)
    for fn in chosen:
        state = _run_step(fn, state)
    if "scorecard" not in state:                         # step_report itself failed
        state["scorecard"] = Scorecard(model=state["model_name"], behavior=behavior.to_dict(),
                                       cells={p: state["cells"].get(p, CellResult(p)) for p in CELLS},
                                       errors=list(state["errors"]))
    return state["scorecard"]


def grid(scorecards: Sequence[Scorecard], values: bool = False) -> str:
    """The §5 headline table: models (rows) x P0-P7 (columns), final column
    ``1 - gamma_cov``. A cell is its status (``values=True`` appends the answer)."""
    head = "| model | " + " | ".join(f"{p}<br>{PROBLEMS[p]}" for p in CELLS) + " | 1 - γ_cov |"
    sep = "|" + "---|" * (len(CELLS) + 2)
    rows = [head, sep]
    for card in scorecards:
        rows.append("| " + " | ".join(_md(str(x)) for x in card.to_grid(values=values)) + " |")
    return "\n".join(rows)
