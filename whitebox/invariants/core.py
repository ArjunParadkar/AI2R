"""check_invariance — empirical gauge-invariance audit of ANY metric by orbit sampling.

Ported from whitebox-v2 E2 (``e2lib.core``); thresholds are E2's pre-registered
ones. The statistic is the relative dispersion of the metric over R orbit
representatives of one network,

    CV = sqrt( mean_r ||q_r - q̄||^2 ) / ||q̄||        (std/|mean| for scalars),

with CV = 0 when every q_r is exactly 0 and CV = inf when the mean vanishes
but the values do not. Verdicts: ``exact`` (CV < 1e-6; the float floor
measured on 687 trained + 840 untrained nets is 1.3e-15 for the sanity
quantities), ``not_invariant`` (CV > 0.05), otherwise ``approximate`` (the
number is reported, no verdict claimed). ``agreement`` is the fraction of
representatives equal to the base value bit-for-bit (meaningful for
integer-valued metrics only).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import numpy as np

from .model import GROUPS, sample_orbit, as_relu_mlp

EXACT_CV = 1e-6       # "exactly invariant" (E2 SPEC)
BROKEN_CV = 0.05      # "not invariant"     (E2 SPEC)
FLOAT_FLOOR = 1e-10   # what the shipped invariants are tested against (E2 measured 1.3e-15)


def cv_of(values) -> float:
    V = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    mean = V.mean(0)
    dev = np.sqrt(np.mean(np.sum((V - mean) ** 2, axis=1)))
    nrm = np.linalg.norm(mean)
    if nrm == 0.0:
        return 0.0 if dev == 0.0 else float("inf")
    return float(dev / nrm)


def agreement_of(values) -> float:
    V = [np.asarray(v) for v in values]
    return float(np.mean([np.array_equal(v, V[0]) for v in V]))


def verdict_for(cv: float, floor: float = EXACT_CV, broken: float = BROKEN_CV) -> str:
    if not np.isfinite(cv):
        return "not_invariant"
    if cv < floor:
        return "exact"
    if cv > broken:
        return "not_invariant"
    return "approximate"


@dataclass
class InvarianceReport:
    """Result of :func:`check_invariance`.

    ``broken_by`` is one of ``"none"``, ``"perm"``, ``"scale"``, ``"perm+scale"``,
    ``"combined-only"`` (neither symmetry alone crosses the broken line but their
    combination does) or ``"n/a"`` (not diagnosed)."""
    metric: str
    group: str
    n_reps: int
    cv: float
    agreement: float
    verdict: str
    max_abs_dev: float
    param_distance_median: float
    param_distance_min: float
    base_value: np.ndarray
    broken_by: str = "n/a"
    per_group: Dict[str, float] = field(default_factory=dict)
    values: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def invariant(self) -> bool:
        return self.verdict == "exact"

    def __str__(self):
        s = (f"InvarianceReport({self.metric}, group={self.group}, n_reps={self.n_reps}: CV={self.cv:.3g}, "
             f"agreement={self.agreement:.2f}, verdict={self.verdict}, broken_by={self.broken_by}, "
             f"param-distance median={self.param_distance_median:.2f})")
        return s


def _is_float32_torch(model) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if isinstance(model, torch.nn.Module):
        return any(p.dtype == torch.float32 for p in model.parameters())
    return False


def _orbit_values(metric_fn, model, n_reps, group, seed, log_scale_sd):
    vals, dists = [], []
    for r, m, perms, scales, d in sample_orbit(model, n_reps, group, seed, log_scale_sd):
        vals.append(np.asarray(metric_fn(m), dtype=np.float64))
        dists.append(d)
    return np.stack(vals), np.asarray(dists)


def _broken_by(cv_perm: float, cv_scale: float, cv_combined: float) -> str:
    p, s = cv_perm > BROKEN_CV or not np.isfinite(cv_perm), cv_scale > BROKEN_CV or not np.isfinite(cv_scale)
    if p and s: return "perm+scale"
    if p: return "perm"
    if s: return "scale"
    if cv_combined > BROKEN_CV or not np.isfinite(cv_combined): return "combined-only"
    return "none"


def check_invariance(metric_fn: Callable, model, n_reps: int = 100, group: str = "combined", seed: int = 0,
                     log_scale_sd: float = 1.0, name: Optional[str] = None, diagnose: bool = True,
                     keep_values: bool = True) -> InvarianceReport:
    """Audit ``metric_fn(model) -> scalar | ndarray`` for invariance under the ReLU-MLP
    gauge group by sampling ``n_reps`` orbit representatives (rep 0 = the model).

    ``group``: ``"perm"`` (unit permutations), ``"scale"`` (per-unit positive
    scalings, log a ~ N(0, log_scale_sd^2)), or ``"combined"``. With
    ``diagnose=True`` and ``group="combined"``, ``perm`` and ``scale`` are also
    sampled (same ``n_reps``) so that ``broken_by`` says which symmetry moves the
    metric; the CVs of all three are in ``per_group``.

    The model may be the E1/E2 numpy MLP (``.Ws``/``.bs``), a torch
    ``nn.Sequential`` of Linear/ReLU, or a :class:`ReluMLP`; ``metric_fn``
    receives representatives of the same type. The metric must return the same
    shape for every representative (a raw per-neuron vector is fine — it will
    be reported as broken by ``perm``, which is the finding).

    Scope: ReLU MLPs only. A verdict of ``exact`` means invariant under the
    permutation x scaling group, not under symmetries this module does not
    sample.
    """
    as_relu_mlp(model)  # type check early
    if _is_float32_torch(model):
        import warnings
        warnings.warn("torch model is float32: the orbit float floor is ~1e-7, so exact invariants will be reported "
                      "as 'approximate'. Call model.double() first for an 'exact' verdict.", stacklevel=2)
    V, dists = _orbit_values(metric_fn, model, n_reps, group, seed, log_scale_sd)
    cv = cv_of(V)
    per_group = {group: cv}
    broken = "n/a"
    if diagnose:
        if group == "combined":
            for g in ("perm", "scale"):
                Vg, _ = _orbit_values(metric_fn, model, n_reps, g, seed + 1, log_scale_sd)
                per_group[g] = cv_of(Vg)
            broken = _broken_by(per_group["perm"], per_group["scale"], cv)
        else:
            broken = group if verdict_for(cv) == "not_invariant" else "none"
    return InvarianceReport(
        metric=name or getattr(metric_fn, "__name__", "metric"), group=group, n_reps=n_reps, cv=cv,
        agreement=agreement_of(V), verdict=verdict_for(cv), max_abs_dev=float(np.max(np.abs(V - V[0]))),
        param_distance_median=float(np.median(dists[1:])) if n_reps > 1 else 0.0,
        param_distance_min=float(np.min(dists[1:])) if n_reps > 1 else 0.0,
        base_value=V[0], broken_by=broken, per_group=per_group, values=V if keep_values else None)


def audit_metrics(quantities_fn: Callable, model, n_reps: int = 100, group: str = "combined", seed: int = 0,
                  log_scale_sd: float = 1.0):
    """Same audit for ``quantities_fn(model) -> {name: array}`` in ONE orbit pass.
    Returns ``({name: {"cv", "agreement", "max_abs_dev", "verdict"}}, param_distances, base_dict)``."""
    per, dists, base = {}, [], None
    for r, m, perms, scales, d in sample_orbit(model, n_reps, group, seed, log_scale_sd):
        q = quantities_fn(m)
        if base is None:
            base = q
        for k, v in q.items():
            per.setdefault(k, []).append(np.asarray(v, dtype=np.float64))
        dists.append(d)
    out = {}
    for k, vals in per.items():
        V = np.stack(vals); cv = cv_of(V)
        out[k] = {"cv": cv, "agreement": agreement_of(V), "max_abs_dev": float(np.max(np.abs(V - V[0]))),
                  "verdict": verdict_for(cv)}
    return out, np.asarray(dists), base
