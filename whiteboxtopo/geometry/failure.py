"""P5, "where it will fail": the geometric predictor rho and its test (SPEC 2.11 / 6.6).

``rho(x) = ||Q_{h(x)}|| * Lip(x)``: curvature of the representation manifold at
the last hidden activation times the exact local Lipschitz constant of the
path product. SPEC 6.6's prediction is that the perturbation radius at which
the argmax flips is monotonically *decreasing* in rho, tested by actually
perturbing: Spearman correlation < -0.5 at p < 0.01, and beating the
softmax-margin baseline.

Nulls / baselines shipped with it: :func:`softmax_margin_baseline` (the top1 -
top2 logit margin, the predictor SPEC 6.6 must beat) and, for a permutation
null, ``spearman_test(rng.permutation(rho), radius)``.

Failure boundary. (1) ``perturbation_failure_radius`` samples ``n_dirs``
Gaussian directions per point, so it is a *lower bound* on robustness that
gets tighter with n_dirs and is coarse-grained to the radius grid; ``inf``
means "no sampled direction flipped it at any radius in the grid", not
"robust". (2) The radii are absolute distances in input space - they are only
comparable across inputs if the inputs are on a common scale. (3) rho is a
product of two locally-estimated quantities and inherits both failure
boundaries (see ``tangent.py`` and ``lipschitz.py``); it is an ordering, never
a probability.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence
import numpy as np
from scipy import stats

from ..invariants.model import as_relu_mlp
from .tangent import curvature
from .lipschitz import lipschitz_field

SPEC_RHO_MAX = -0.5      # SPEC 6.6 pass: Spearman rho < -0.5 ...
SPEC_P_MAX = 0.01        # ... at p < 0.01


@dataclass
class FailurePredictor:
    """SPEC 6.6 predictor rho(x) and the two factors it is built from."""
    rho: np.ndarray             # (n,) ||Q_h(x)|| * Lip(x)
    curvature_norm: np.ndarray  # (n,) ||Q_h(x)||
    lip: np.ndarray             # (n,) ||A(x)||_2
    variation: np.ndarray       # (n,) neighbour variation of A
    dim: int                    # intrinsic dimension of the hidden cloud
    k: int

    def to_dict(self) -> Dict[str, object]:
        return {"n": int(self.rho.size), "k": int(self.k), "hidden_dim": int(self.dim),
                "rho_median": float(np.median(self.rho)), "rho_max": float(self.rho.max()),
                "rho": self.rho.tolist(), "curvature_norm": self.curvature_norm.tolist(),
                "lip": self.lip.tolist(), "variation": self.variation.tolist()}


def failure_predictor(model, X, k: int = 15) -> FailurePredictor:
    """``rho(x) = ||Q_{h(x)}|| * Lip(x)`` (SPEC 2.11), h = last hidden activation cloud.

    Deterministic: no randomness. Returns :class:`FailurePredictor`; pair it
    with :func:`perturbation_failure_radius` and :func:`spearman_test` for the
    SPEC 6.6 verdict.
    """
    m = as_relu_mlp(model)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    H = m.hidden_acts(X)[-1]
    cur = curvature(H, k=k)
    lipr = lipschitz_field(m, X, k=k)
    return FailurePredictor(rho=cur.norm * lipr.lip, curvature_norm=cur.norm, lip=lipr.lip,
                            variation=lipr.variation, dim=cur.dim, k=cur.k)


def _predict(m, X: np.ndarray) -> np.ndarray:
    """argmax class, or the sign bit for a single-output (binary) net."""
    Z = m.logits(X)
    return (Z[:, 0] > 0).astype(np.int64) if Z.shape[1] == 1 else np.argmax(Z, axis=1)


def perturbation_failure_radius(model, X, y=None, radii: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
                                n_dirs: int = 8, seed: int = 0) -> np.ndarray:
    """Smallest radius at which a Gaussian perturbation flips the prediction (SPEC 6.6).

    For each input, ``n_dirs`` directions are drawn once, normalised to the
    unit sphere and reused at every radius (a nested design: the r-ball
    perturbations are the r'-ball ones rescaled), and the input fails at radius
    r if *any* of them changes the predicted class. The reference class is the
    model's own unperturbed prediction, so this measures decision stability;
    ``y`` is optional and used only to drop inputs the model already gets wrong
    (their radius is returned as ``nan``) when it is given.

    Returns ``(n,)`` float array of radii, ``inf`` where nothing flipped.
    Vectorised: one forward pass per radius over all n * n_dirs perturbations.
    Deterministic given ``seed``.
    """
    m = as_relu_mlp(model)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n, d = X.shape
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n, n_dirs, d))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=2, keepdims=True), 1e-300)
    base = _predict(m, X)
    out = np.full(n, np.inf)
    todo = np.ones(n, bool)
    for r in sorted(float(v) for v in radii):
        if not todo.any():
            break
        P = (X[:, None, :] + r * dirs).reshape(n * n_dirs, d)
        flip = (_predict(m, P).reshape(n, n_dirs) != base[:, None]).any(axis=1)
        hit = todo & flip
        out[hit] = r
        todo &= ~flip
    if y is not None:
        out[np.asarray(y).ravel() != base] = np.nan
    return out


def softmax_margin_baseline(model, X) -> np.ndarray:
    """SPEC 6.6 baseline: the logit margin (top1 - top2), ``|logit|`` for one output.

    The margin is a *confidence*: it is monotone in the softmax probability of
    the top class, so ranking by it is ranking by softmax confidence. SPEC 6.6
    requires the geometric predictor to beat this. Deterministic.
    """
    Z = as_relu_mlp(model).logits(np.atleast_2d(np.asarray(X, dtype=np.float64)))
    if Z.shape[1] == 1:
        return np.abs(Z[:, 0])
    S = np.sort(Z, axis=1)
    return S[:, -1] - S[:, -2]


def spearman_test(rho, radius) -> Dict[str, object]:
    """SPEC 6.6 verdict: Spearman correlation of a predictor with the failure radius.

    Sign convention (SPEC 6.6): a *larger* predictor value must mean a
    *smaller* failure radius, so the pass condition is
    ``spearman_rho < -0.5 and p_value < 0.01`` - reported under ``pass_spec``.
    ``nan`` entries (inputs excluded by ``y``) are dropped pairwise; ``inf``
    radii rank highest, which is the intended meaning ("never flipped").

    Returns a plain dict (this is a test result, not an instrument state).
    """
    a = np.asarray(rho, dtype=np.float64).ravel()
    b = np.asarray(radius, dtype=np.float64).ravel()
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    n = int(a.size)
    if n < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return {"spearman_rho": float("nan"), "p_value": float("nan"), "n": n, "pass_spec": False,
                "reason": "degenerate: fewer than 3 usable pairs, or a constant input"}
    res = stats.spearmanr(a, b)
    r, p = float(res.statistic), float(res.pvalue)
    return {"spearman_rho": r, "p_value": p, "n": n,
            "pass_spec": bool(r < SPEC_RHO_MAX and p < SPEC_P_MAX),
            "threshold_rho": SPEC_RHO_MAX, "threshold_p": SPEC_P_MAX}
