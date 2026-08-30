"""Orbit sampling and the invariance audit.

The group G (SPEC): hidden-unit permutations within each hidden layer x positive
per-unit scalings through ReLU. Three sampling groups: 'perm', 'scale', 'combined'.
Representative 0 is always the base network itself.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from e1lib.gauge import apply_gauge  # E1's verified implementation

GROUPS = ("perm", "scale", "combined")
EXACT_CV = 1e-6      # SPEC: "exactly invariant" threshold
BROKEN_CV = 0.05     # SPEC: "not invariant" threshold


def param_vector(model):
    return np.concatenate([p.ravel() for p in list(model.Ws) + list(model.bs)])


def param_distance(model_a, model_b):
    """Relative L2 distance in parameter space, ||theta_b - theta_a|| / ||theta_a||."""
    a, b = param_vector(model_a), param_vector(model_b)
    return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-300))


def random_gauge_element(model, rng, group="combined", log_scale_sd=1.0):
    """Sample (perms, scales) for the chosen group. Identity perm / unit scale where the
    group does not act."""
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
    perms = [rng.permutation(w) if group in ("perm", "combined") else np.arange(w)
             for w in model.widths]
    scales = [np.exp(rng.normal(0.0, log_scale_sd, size=w)) if group in ("scale", "combined")
              else np.ones(w) for w in model.widths]
    return perms, scales


def sample_orbit(model, n_reps=100, group="combined", seed=0, log_scale_sd=1.0):
    """Yield (rep_index, transformed_model, perms, scales, rel_param_distance). Rep 0 is the
    base network with identity gauge."""
    rng = np.random.default_rng(seed)
    yield 0, model, [np.arange(w) for w in model.widths], [np.ones(w) for w in model.widths], 0.0
    for r in range(1, n_reps):
        perms, scales = random_gauge_element(model, rng, group, log_scale_sd)
        m = apply_gauge(model, perms, scales)
        yield r, m, perms, scales, param_distance(model, m)


def cv_of(values):
    """Relative dispersion over representatives: sqrt(mean ||q_r - qbar||^2) / ||qbar||.
    Reduces to std/|mean| for scalars. 0 if everything is exactly 0; inf if the mean is 0 but
    the values are not."""
    V = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    mean = V.mean(0)
    dev = np.sqrt(np.mean(np.sum((V - mean) ** 2, axis=1)))
    nrm = np.linalg.norm(mean)
    if nrm == 0.0:
        return 0.0 if dev == 0.0 else float("inf")
    return float(dev / nrm)


def agreement_of(values):
    """Fraction of representatives whose value equals the base (rep 0) value exactly."""
    V = [np.asarray(v) for v in values]
    base = V[0]
    return float(np.mean([np.array_equal(v, base) for v in V]))


def verdict_for(cv, floor=EXACT_CV):
    if not np.isfinite(cv):
        return "not_invariant"
    if cv < floor:
        return "exact"
    if cv > BROKEN_CV:
        return "not_invariant"
    return "approximate"


@dataclass
class InvarianceReport:
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
    values: Optional[np.ndarray] = field(default=None, repr=False)

    def __str__(self):
        return (f"InvarianceReport({self.metric}, group={self.group}, n_reps={self.n_reps}: "
                f"CV={self.cv:.3g}, agreement={self.agreement:.2f}, verdict={self.verdict}, "
                f"param-distance median={self.param_distance_median:.2f})")


def check_invariance(metric_fn: Callable, model, n_reps: int = 100, group: str = "combined",
                     seed: int = 0, log_scale_sd: float = 1.0, name: Optional[str] = None,
                     keep_values: bool = True) -> InvarianceReport:
    """Audit ANY metric for gauge invariance by orbit sampling.

    metric_fn(model) -> scalar or ndarray (same shape for every representative).
    Returns an InvarianceReport whose ``verdict`` is 'exact' (CV < 1e-6), 'approximate',
    or 'not_invariant' (CV > 0.05); see SPEC thresholds. ``agreement`` is the fraction of
    representatives whose value equals the base exactly (useful for discrete metrics).
    """
    vals, dists = [], []
    for r, m, perms, scales, d in sample_orbit(model, n_reps, group, seed, log_scale_sd):
        vals.append(np.asarray(metric_fn(m), dtype=np.float64))
        dists.append(d)
    V = np.stack(vals)
    cv = cv_of(V)
    return InvarianceReport(
        metric=name or getattr(metric_fn, "__name__", "metric"), group=group, n_reps=n_reps,
        cv=cv, agreement=agreement_of(V), verdict=verdict_for(cv),
        max_abs_dev=float(np.max(np.abs(V - V[0]))),
        param_distance_median=float(np.median(dists[1:])) if n_reps > 1 else 0.0,
        param_distance_min=float(np.min(dists[1:])) if n_reps > 1 else 0.0,
        base_value=V[0], values=V if keep_values else None)


def audit_metrics(quantities_fn: Callable, model, n_reps=100, group="combined", seed=0,
                  log_scale_sd=1.0):
    """Same audit for a function returning a dict {name: array} (one forward pass per rep
    instead of one per metric). Returns ({name: {'cv', 'agreement', 'max_abs_dev'}}, dists,
    base_dict)."""
    per = {}
    dists = []
    base = None
    for r, m, perms, scales, d in sample_orbit(model, n_reps, group, seed, log_scale_sd):
        q = quantities_fn(m)
        if base is None:
            base = q
        for k, v in q.items():
            per.setdefault(k, []).append(np.asarray(v, dtype=np.float64))
        dists.append(d)
    out = {}
    for k, vals in per.items():
        V = np.stack(vals)
        out[k] = {"cv": cv_of(V), "agreement": agreement_of(V),
                  "max_abs_dev": float(np.max(np.abs(V - V[0])))}
    return out, np.asarray(dists), base
