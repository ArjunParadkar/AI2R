"""check_invariance — the auditor (handbook S2).

Given ``metric_fn(model) -> scalar | ndarray`` (fixed shape across renamings),
sample ``n`` renamings from each sub-group of G separately — permutation-only,
scaling-only, combined (and rotation-only for attention models) — and report
the per-component coefficient of variation under each, a verdict, and the raw
samples.

Verdict (pre-registered in the handbook, §S2):
* ``invariant``  — CV under the combined group < ``INVARIANT_CV`` (0.01);
* ``covariant``  — permutation CV < 0.01, scaling CV >= 0.01, and the response
  to a *uniform* rescaling alpha of the hidden units is a power law
  q(alpha) ∝ alpha^k with median R² >= ``COVARIANT_R2`` (0.99): the metric is a
  well-defined quantity that carries a known weight, and ``scaling_exponent``
  reports k (Z5 normalises by it);
* ``artifact``   — otherwise (a permutation moves it, or scaling moves it with
  no law).

The overall verdict uses the *median* component; ``frac_invariant`` gives the
fraction of components individually below 0.01, so a mostly-invariant vector
with one broken entry is visible.

``align``: for direction- or unit-valued metrics, samples are aligned to the
base sample before comparison. ``"gauge"`` undoes the *known* renaming exactly
(needs ``unit_key`` — the layer key the metric's ``unit_axis`` indexes, and
``scale_power`` if the metric scales like a^p under unit scaling);
``"procrustes"`` fits an orthogonal map on the last axis (blind alignment for
direction sets, shape (k, d)); ``"hungarian"`` re-matches rows to the base by
minimal L2 cost (blind permutation alignment). ``align=True`` picks ``gauge``
when ``unit_key`` is given, else ``procrustes`` for 2-D outputs, else
``hungarian``. Both raw (unaligned) and aligned CVs are reported.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence
import numpy as np
from .. import symmetry as S

INVARIANT_CV = 0.01
COVARIANT_R2 = 0.99
UNIFORM_ALPHAS = np.exp(np.linspace(np.log(0.1), np.log(10.0), 9))


def per_component_cv(V: np.ndarray) -> np.ndarray:
    """CV_j = std_j / |mean_j| per component (0 where all samples are exactly 0; inf where the
    mean vanishes but the samples do not)."""
    V = np.asarray(V, dtype=np.float64).reshape(V.shape[0], -1)
    mu = V.mean(0); sd = V.std(0)
    out = np.empty_like(mu)
    zero = np.abs(mu) == 0
    out[~zero] = sd[~zero] / np.abs(mu[~zero])
    out[zero] = np.where(sd[zero] == 0, 0.0, np.inf)
    return out


def global_cv(V: np.ndarray) -> float:
    """||q_r - q̄|| RMS / ||q̄|| — the E2 statistic, for continuity with the earlier audit."""
    V = np.asarray(V, dtype=np.float64).reshape(V.shape[0], -1)
    mu = V.mean(0); dev = np.sqrt(np.mean(np.sum((V - mu) ** 2, axis=1))); nrm = np.linalg.norm(mu)
    if nrm == 0:
        return 0.0 if dev == 0 else float("inf")
    return float(dev / nrm)


@dataclass
class AuditResult:
    metric: str
    n: int
    cv: Dict[str, np.ndarray]                # kind -> per-component CV (raw)
    cv_aligned: Dict[str, np.ndarray]        # kind -> per-component CV after alignment ({} if align=False)
    median_cv: Dict[str, float]
    frac_invariant: Dict[str, float]
    verdict: str                             # invariant | covariant | artifact
    verdict_aligned: Optional[str]
    scaling_exponent: Optional[float]        # k in q(alpha) ~ alpha^k (median over components), None if not fitted
    scaling_r2: Optional[float]
    scaling_exponents: Optional[np.ndarray] = field(default=None, repr=False)
    base: Optional[np.ndarray] = field(default=None, repr=False)
    samples: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    samples_aligned: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    uniform_sweep: Optional[np.ndarray] = field(default=None, repr=False)   # (n_alpha, n_components)
    alphas: Optional[np.ndarray] = field(default=None, repr=False)
    align: str = "none"
    notes: str = ""

    def row(self) -> dict:
        """One CSV row (audit_table.csv)."""
        d = {"metric": self.metric, "n": self.n, "verdict": self.verdict, "verdict_aligned": self.verdict_aligned,
             "scaling_exponent": self.scaling_exponent, "scaling_r2": self.scaling_r2, "align": self.align}
        for k, v in self.median_cv.items():
            d[f"cv_{k}"] = v
        for k, v in self.frac_invariant.items():
            d[f"frac_inv_{k}"] = v
        for k, v in self.cv_aligned.items():
            d[f"cv_{k}_aligned"] = float(np.median(v))
        return d

    def __str__(self):
        cvs = ", ".join(f"{k}={v:.3g}" for k, v in self.median_cv.items())
        s = f"AuditResult({self.metric}: verdict={self.verdict}; median CV {cvs}"
        if self.scaling_exponent is not None:
            s += f"; k={self.scaling_exponent:.3g} (R²={self.scaling_r2:.3f})"
        if self.verdict_aligned:
            s += f"; aligned={self.verdict_aligned}"
        return s + ")"


def _procrustes_align(V: np.ndarray, base: np.ndarray) -> np.ndarray:
    out = np.empty_like(V)
    for r in range(V.shape[0]):
        A = V[r].reshape(-1, V.shape[-1]); B = base.reshape(-1, base.shape[-1])
        U, _, Wt = np.linalg.svd(A.T @ B)
        out[r] = (A @ (U @ Wt)).reshape(V[r].shape)
    return out


def _hungarian_align(V: np.ndarray, base: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment
    out = np.empty_like(V)
    B = base.reshape(base.shape[0], -1)
    for r in range(V.shape[0]):
        A = V[r].reshape(V[r].shape[0], -1)
        cost = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        ri, ci = linear_sum_assignment(cost)
        perm = np.empty(len(ri), dtype=int); perm[ci] = ri
        out[r] = V[r][perm]
    return out


def _fit_powerlaw(alphas: np.ndarray, Q: np.ndarray):
    """Per-component least-squares fit log|q| = k log alpha + c. Returns (k, r2) arrays."""
    x = np.log(alphas); x = x - x.mean()
    Q = np.abs(Q.reshape(len(alphas), -1)) + 1e-300
    y = np.log(Q); yc = y - y.mean(0, keepdims=True)
    k = (x[:, None] * yc).sum(0) / (x ** 2).sum()
    resid = yc - x[:, None] * k[None, :]
    ss_tot = (yc ** 2).sum(0); ss_res = (resid ** 2).sum(0)
    r2 = np.where(ss_tot > 1e-20, 1 - ss_res / np.maximum(ss_tot, 1e-300), 1.0)   # constant => "fits" with k=0
    return k, r2


def _verdict(cv_perm, cv_scale, cv_comb, k, r2):
    if np.median(cv_comb) < INVARIANT_CV:
        return "invariant"
    if np.median(cv_perm) < INVARIANT_CV and k is not None and r2 is not None and r2 >= COVARIANT_R2 and abs(k) > 1e-3:
        return "covariant"
    return "artifact"


def check_invariance(metric_fn: Callable, model, n: int = 100, seed: int = 0, kinds: Optional[Sequence[str]] = None,
                     scale_range=(0.1, 10.0), align=False, unit_key=None, unit_axis: int = 0, scale_power: float = 0.0,
                     uniform_sweep: bool = True, name: Optional[str] = None, keep_samples: bool = True,
                     orthogonal: bool = True) -> AuditResult:
    """Audit ``metric_fn`` under the gauge group of ``model`` (see module docstring).

    ``kinds`` defaults to ``("perm", "scale", "combined")`` plus ``"rot"`` for attention models.
    ``uniform_sweep`` additionally evaluates the metric under uniform hidden-unit rescalings
    alpha ∈ [0.1, 10] to fit the covariance law (MLP and toy-transformer MLP units only)."""
    desc = S.describe(model)
    if kinds is None:
        kinds = ("perm", "scale", "combined") + (("rot",) if desc["type"] != "relu_mlp" else ())
    base = np.asarray(metric_fn(model), dtype=np.float64)
    shape = base.shape
    samples, elements = {}, {}
    for ki, kind in enumerate(kinds):
        rng = np.random.default_rng(seed + 1000 * ki)
        vals, gs = [base], [S.GaugeElement(kind=kind)]
        for r in range(1, n):
            g = S.random_element(model, rng, kind=kind, log_range=scale_range, orthogonal=orthogonal)
            v = np.asarray(metric_fn(S.apply(model, g)), dtype=np.float64)
            if v.shape != shape:
                raise ValueError(f"metric returned shape {v.shape} for a renaming, base shape {shape}; "
                                 "metrics must be fixed-shape (summarise diagrams first)")
            vals.append(v); gs.append(g)
        samples[kind] = np.stack(vals); elements[kind] = gs
    cv = {k: per_component_cv(v) for k, v in samples.items()}
    # covariance law
    k_med = r2_med = None; ks = sweep = None; alphas = None
    if uniform_sweep and desc["type"] in ("relu_mlp", "toy_transformer"):
        alphas = UNIFORM_ALPHAS
        sweep = np.stack([np.asarray(metric_fn(S.apply(model, S.uniform_scaling(model, a))), dtype=np.float64)
                          for a in alphas])
        ks, r2s = _fit_powerlaw(alphas, sweep)
        moving = np.abs(ks) > 1e-3
        k_med = float(np.median(ks[moving])) if moving.any() else 0.0
        r2_med = float(np.median(r2s[moving])) if moving.any() else 1.0
    verdict = _verdict(cv.get("perm", cv.get("combined")), cv.get("scale", cv.get("combined")),
                       cv["combined"] if "combined" in cv else np.max(np.stack(list(cv.values())), 0), k_med, r2_med)
    # alignment
    cv_al, samples_al, verdict_al, mode = {}, {}, None, "none"
    if align:
        mode = align if isinstance(align, str) else ("gauge" if unit_key is not None else ("procrustes" if base.ndim >= 2 else "hungarian"))
        for kind, V in samples.items():
            if mode == "gauge":
                A = np.stack([S.inverse_on_units(g, unit_key, v, axis=unit_axis, scale_power=scale_power)
                              for g, v in zip(elements[kind], V)])
            elif mode == "procrustes":
                A = _procrustes_align(V, base)
            elif mode == "hungarian":
                A = _hungarian_align(V, base)
            else:
                raise ValueError(f"align must be one of gauge|procrustes|hungarian|True|False, got {align!r}")
            samples_al[kind] = A; cv_al[kind] = per_component_cv(A)
        verdict_al = _verdict(cv_al.get("perm", cv_al.get("combined")), cv_al.get("scale", cv_al.get("combined")),
                              cv_al["combined"] if "combined" in cv_al else np.max(np.stack(list(cv_al.values())), 0),
                              k_med, r2_med)
    fin = lambda a: a[np.isfinite(a)] if np.isfinite(a).any() else a
    return AuditResult(
        metric=name or getattr(metric_fn, "__name__", "metric"), n=n, cv=cv, cv_aligned=cv_al,
        median_cv={k: float(np.median(v)) for k, v in cv.items()},
        frac_invariant={k: float(np.mean(v < INVARIANT_CV)) for k, v in cv.items()},
        verdict=verdict, verdict_aligned=verdict_al, scaling_exponent=k_med, scaling_r2=r2_med, scaling_exponents=ks,
        base=base, samples=samples if keep_samples else {}, samples_aligned=samples_al if keep_samples else {},
        uniform_sweep=sweep, alphas=alphas, align=mode)


def audit_table(metrics: Dict[str, Callable], model, **kw):
    """Run :func:`check_invariance` on a dict of metrics; returns a list of ``AuditResult``."""
    return [check_invariance(fn, model, name=name, **kw) for name, fn in metrics.items()]
