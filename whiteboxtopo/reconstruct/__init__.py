"""Block R: rebuild the behavior from certified structure alone, then measure what is left.

FINAL_SPEC Section 7. The discipline of this module is that *no weights are
copied*: a reconstruction is built only from the certified inventory (a
frequency set and its per-frequency weights; a circular coordinate and fitted
tangent dynamics; a circuit support). If the rebuild then matches the model
behaviourally (``whiteboxtopo.verify``), the inventory really was the mechanism;
if it does not, the claimed explanation was incomplete, and the residual
spectrum says by how much.

* :func:`from_irreps`      -- R1, the Z_p group tier (SPEC 2.6, 7 R1).
* :func:`fit_tangent_dynamics` / :func:`from_attractor` -- R2, the ring tier (2.7).
* :func:`from_circuits`    -- R3, certified circuit with the complement mean-ablated.
* :func:`residual` / :func:`coverage_gap` -- R4, the Marchenko-Pastur residue (2.13).

Nulls carried by this module: :func:`coverage_gap` reports, beside the measured
gap, the same statistic on a synthetic Gaussian residue with matched marginals
and on a coordinate-shuffled residue. A gap that is not clear of both is not a
gap. Everything is numpy/scipy; torch is never imported here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..darkmatter.rmt import mp_density, mp_edges, tw_threshold, feature_shuffle, gaussian_matched_diag
from ..invariants.model import as_relu_mlp

__all__ = ["from_irreps", "GrokReconstruction", "fit_tangent_dynamics", "TangentDynamics",
           "from_attractor", "AttractorReconstruction", "from_circuits", "CircuitReconstruction",
           "residual", "coverage_gap", "CoverageGap"]


# --------------------------------------------------------------------- R1: irreps
@dataclass
class GrokReconstruction:
    """Modular addition rebuilt from a certified frequency inventory (SPEC 7 R1).

    ``embed`` rows are ``[cos(w_k a), sin(w_k a)]_{k in K}`` -- constructed, not
    copied. The hidden layer applies the angle-addition identity
    ``cos(w_k(a+b)) = cos cos - sin sin``, ``sin(w_k(a+b)) = sin cos + cos sin``,
    and the readout is ``logits(c) = sum_k w_k cos(w_k (a + b - c))``.
    The only numbers taken from the model are ``K`` and the per-frequency
    weights, both certified gauge-invariant by
    :func:`whiteboxtopo.invariants.fourier_irreps`.
    """
    p: int
    K: Tuple[int, ...]
    w: np.ndarray                 # (|K|,) normalised readout weights, sum to 1
    embed: np.ndarray             # (p, 2|K|) constructed embedding
    readout: np.ndarray           # (2|K|, p) constructed unembedding
    raw_weights: Dict[int, float] = field(default_factory=dict)

    # ---- the rebuilt function
    def features(self, a, b) -> np.ndarray:
        """(N, 2|K|) block features [cos(w_k(a+b)), sin(w_k(a+b))] via trig products only."""
        a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
        ea, eb = self.embed[a % self.p], self.embed[b % self.p]
        out = np.empty_like(ea)
        ca, sa = ea[:, 0::2], ea[:, 1::2]
        cb, sb = eb[:, 0::2], eb[:, 1::2]
        out[:, 0::2] = ca * cb - sa * sb          # cos(w_k(a+b))
        out[:, 1::2] = sa * cb + ca * sb          # sin(w_k(a+b))
        return out

    def forward(self, a, b) -> np.ndarray:
        """Logits over the ``p`` classes; (p,) for scalars, (N, p) for arrays."""
        scalar = np.isscalar(a) and np.isscalar(b)
        out = self.features(a, b) @ self.readout
        return out[0] if scalar else out

    def __call__(self, a, b=None) -> np.ndarray:
        if b is None:                               # token form [a, b, '=']
            t = np.atleast_2d(np.asarray(a))
            return self.forward(t[:, 0], t[:, 1])
        return self.forward(a, b)

    def logits_from_tokens(self, tokens) -> np.ndarray:
        t = np.atleast_2d(np.asarray(tokens))
        return self.forward(t[:, 0], t[:, 1])

    def all_pairs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(a, b, logits) over all p^2 pairs."""
        a, b = np.meshgrid(np.arange(self.p), np.arange(self.p), indexing="ij")
        a, b = a.ravel(), b.ravel()
        return a, b, self.forward(a, b)

    def to_dict(self) -> dict:
        return {"p": self.p, "K": list(self.K), "w": self.w.tolist(),
                "raw_weights": {int(k): float(v) for k, v in self.raw_weights.items()},
                "n_params_constructed": int(self.embed.size + self.readout.size)}


def _as_scalar_weights(K: Sequence[int], weights) -> Dict[int, float]:
    out = {}
    for k in K:
        if isinstance(weights, Mapping):
            if k not in weights and int(k) not in weights:
                raise KeyError(f"no certified weight for frequency k={k}")
            v = weights[k] if k in weights else weights[int(k)]
        else:
            v = np.asarray(weights).ravel()[list(K).index(k)]
        v = np.asarray(v, dtype=np.float64)
        out[int(k)] = float(v) if v.ndim == 0 else float(np.linalg.norm(v))
    return out


def from_irreps(K: Sequence[int], weights, p: int, readout=None) -> GrokReconstruction:
    """R1: rebuild modular addition from the certified irrep inventory (SPEC 7 R1).

    ``K`` is the certified frequency set and ``weights`` the certified
    per-frequency weights -- either scalars (the invariant amplitudes
    ``||E_hat_k||`` from :func:`~whiteboxtopo.invariants.fourier_irreps`) or the
    (2, d) cos/sin blocks, which are reduced to their Frobenius norm (the
    invariant part). ``readout`` optionally supplies separate per-frequency logit
    weights (e.g. the amplitudes of the unembedding); it defaults to ``weights``.

    Weights are normalised to sum to 1: a global scale is not identified by the
    inventory and changes neither argmax nor logit correlation.

    Pass criterion (R1): accuracy >= f_theta - 0.5 %, logit correlation > 0.98,
    verified by :func:`whiteboxtopo.verify.equivalence`.
    Null: rebuilding from artifact frequencies instead must fall to chance
    (:func:`whiteboxtopo.verify.negative_control`, R7).
    Failure boundary: a one-layer Z_p model whose readout is an inverse Fourier
    read-off. A model using a different composition rule will fail R5, which is
    the point of running it.
    """
    K = tuple(int(k) for k in K)
    if len(K) == 0:
        raise ValueError("empty frequency set: nothing certified to rebuild from")
    p = int(p)
    raw = _as_scalar_weights(K, weights if readout is None else readout)
    w = np.array([raw[k] for k in K], dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("per-frequency weights must be non-negative amplitudes")
    s = w.sum()
    w = w / s if s > 0 else np.full(len(K), 1.0 / len(K))
    x = np.arange(p)
    om = 2.0 * np.pi * np.array(K, dtype=np.float64) / p           # (|K|,)
    ang = np.outer(x, om)                                          # (p, |K|)
    embed = np.empty((p, 2 * len(K)))
    embed[:, 0::2] = np.cos(ang)
    embed[:, 1::2] = np.sin(ang)
    read = np.empty((2 * len(K), p))
    read[0::2, :] = (w[:, None] * np.cos(ang).T)                   # cos(w_k c) * w_k
    read[1::2, :] = (w[:, None] * np.sin(ang).T)                   # sin(w_k c) * w_k
    return GrokReconstruction(p, K, w, embed, read, raw)


# --------------------------------------------------------------------- R2: attractor
@dataclass
class TangentDynamics:
    """Fitted tangent dynamics ``dtheta/dt = g(theta, u)`` on a circular attractor (SPEC 2.7)."""
    n_harmonics: int
    u_degree: int
    coef: np.ndarray               # (n_features,)
    r2: float
    rmse: float
    has_input: bool

    def design(self, theta, u=None) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64).ravel()
        cols = [np.ones_like(theta)]
        for h in range(1, self.n_harmonics + 1):
            cols += [np.cos(h * theta), np.sin(h * theta)]
        F = np.stack(cols, 1)
        if not self.has_input:
            return F
        u = np.zeros_like(theta) if u is None else np.asarray(u, dtype=np.float64).ravel()
        P = np.stack([u ** d for d in range(self.u_degree + 1)], 1)
        return (F[:, :, None] * P[:, None, :]).reshape(len(theta), -1)

    def predict(self, theta, u=None) -> np.ndarray:
        return self.design(theta, u) @ self.coef

    def to_dict(self) -> dict:
        return {"n_harmonics": self.n_harmonics, "u_degree": self.u_degree,
                "coef": self.coef.tolist(), "r2": self.r2, "rmse": self.rmse,
                "has_input": self.has_input}


def fit_tangent_dynamics(theta, u, dtheta, n_harmonics: int = 3, u_degree: int = 1,
                         ridge: float = 1e-8) -> TangentDynamics:
    """Least-squares fit of ``dtheta/dt = g(theta, u)`` in a Fourier x polynomial basis.

    Fourier in ``theta`` (the only basis that respects the S^1 topology of the
    attractor) times a small polynomial in the input ``u``; ridge only for
    conditioning. Pass ``u=None`` for autonomous dynamics.
    Null: the fitted ``r2`` is reported; an r2 near 0 means the sampled states do
    not have one-dimensional tangent dynamics and R2 must not be claimed.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    dtheta = np.asarray(dtheta, dtype=np.float64).ravel()
    has_input = u is not None
    proto = TangentDynamics(int(n_harmonics), int(u_degree), np.zeros(1), 0.0, 0.0, has_input)
    A = proto.design(theta, u)
    G = A.T @ A + ridge * np.eye(A.shape[1])
    coef = np.linalg.solve(G, A.T @ dtheta)
    pred = A @ coef
    ss = float(((dtheta - dtheta.mean()) ** 2).sum())
    r2 = 1.0 - float(((dtheta - pred) ** 2).sum()) / ss if ss > 0 else 0.0
    rmse = float(np.sqrt(np.mean((dtheta - pred) ** 2)))
    return TangentDynamics(int(n_harmonics), int(u_degree), coef, r2, rmse, has_input)


@dataclass
class AttractorReconstruction:
    """R2: the ring model rebuilt from its circular coordinate and tangent dynamics only."""
    dynamics: TangentDynamics
    theta_coord: np.ndarray                # the certified circular coordinate of the samples

    def step(self, theta: float, u=None, dt: float = 1e-2) -> float:
        """One RK4 step of dtheta/dt = g(theta, u)."""
        g = self.dynamics.predict
        k1 = g([theta], u)[0]
        k2 = g([theta + 0.5 * dt * k1], u)[0]
        k3 = g([theta + 0.5 * dt * k2], u)[0]
        k4 = g([theta + dt * k3], u)[0]
        return float((theta + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)) % (2 * np.pi))

    def integrate(self, theta0: float, u_seq=None, dt: float = 1e-2, n_steps: Optional[int] = None) -> np.ndarray:
        """Trajectory ``theta_t`` in [0, 2pi) driven by ``u_seq`` (one input per step)."""
        if u_seq is None:
            if n_steps is None:
                raise ValueError("pass u_seq or n_steps")
            u_seq = [None] * int(n_steps)
        u_seq = list(u_seq)
        th = float(theta0) % (2 * np.pi)
        out = np.empty(len(u_seq) + 1)
        out[0] = th
        for i, u in enumerate(u_seq):
            th = self.step(th, None if u is None else float(u), dt)
            out[i + 1] = th
        return out

    def to_dict(self) -> dict:
        return {"dynamics": self.dynamics.to_dict(), "n_states": int(np.size(self.theta_coord))}


def from_attractor(theta_coord, dynamics: TangentDynamics) -> AttractorReconstruction:
    """R2: rebuild a continuous-attractor model from certified structure only (SPEC 7 R2).

    ``theta_coord`` is the circular coordinate of the sampled states
    (:func:`whiteboxtopo.invariants.circular_coordinate`) and ``dynamics`` the
    tangent dynamics fitted on it. No recurrent weights are copied.
    Pass criterion (R2): integration error within 10 % of the original over
    100-step trajectories, same drift statistics.
    Failure boundary: a single circular attractor with one-dimensional tangent
    dynamics; a torus or a line attractor needs a different coordinate.
    """
    if not isinstance(dynamics, TangentDynamics):
        raise TypeError("dynamics must be a TangentDynamics from fit_tangent_dynamics")
    return AttractorReconstruction(dynamics, np.asarray(theta_coord, dtype=np.float64))


# --------------------------------------------------------------------- R3: circuits
def _normalise_mask(circuit_mask, widths) -> list:
    if isinstance(circuit_mask, Mapping):
        masks = [np.zeros(w, dtype=bool) for w in widths]
        for l, idx in circuit_mask.items():
            masks[int(l)][np.asarray(idx, dtype=int)] = True
        return masks
    masks = []
    for l, w in enumerate(widths):
        m = np.asarray(circuit_mask[l])
        if m.dtype == bool:
            if m.shape != (w,):
                raise ValueError(f"circuit_mask[{l}] has shape {m.shape}, expected ({w},)")
            masks.append(m.copy())
        else:
            b = np.zeros(w, dtype=bool); b[m.astype(int)] = True; masks.append(b)
    return masks


@dataclass
class CircuitReconstruction:
    """R3: the certified circuit, complement mean-ablated (SPEC 7 R3). Callable ``f_hat(X)``."""
    masks: list
    means: list
    widths: list
    ablation: str
    model: object = None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        m = as_relu_mlp(self.model)
        h = np.atleast_2d(np.asarray(X, dtype=np.float64))
        for l, (W, b) in enumerate(zip(m.Ws[:-1], m.bs[:-1])):
            h = np.maximum(h @ W + b, 0.0)
            keep = self.masks[l]
            h = np.where(keep[None, :], h, self.means[l][None, :])
        return h @ m.Ws[-1] + m.bs[-1]

    def to_dict(self) -> dict:
        return {"widths": list(self.widths), "ablation": self.ablation,
                "n_kept": [int(m.sum()) for m in self.masks],
                "fraction_kept": float(sum(int(m.sum()) for m in self.masks) / max(sum(self.widths), 1))}


def from_circuits(model, circuit_mask, X: np.ndarray, ablation: str = "mean") -> CircuitReconstruction:
    """R3: rebuild ``f_hat`` as the certified circuit with the complement ablated (SPEC 7 R3).

    ``circuit_mask`` is a per-hidden-layer boolean array (or index array, or a
    ``{layer: indices}`` mapping) of the units in the certified circuit. Units
    outside it are replaced by their mean activation over ``X``
    (``ablation="mean"``) or by zero (``ablation="zero"``). Mean ablation is the
    default because zero ablation moves the model off the data manifold and
    inflates the apparent importance of everything.

    Pass criterion (R3): faithfulness >= 0.9; report gamma_cov.
    Null: the full mask must reproduce ``f`` exactly (checked in tests), and a
    random mask of the same size is the control for any faithfulness claim.
    Failure boundary: ReLU MLPs (uses ``whiteboxtopo.invariants.model``).
    """
    m = as_relu_mlp(model)
    widths = m.widths
    masks = _normalise_mask(circuit_mask, widths)
    acts = m.hidden_acts(np.atleast_2d(np.asarray(X, dtype=np.float64)))
    if ablation == "mean":
        means = [a.mean(0) for a in acts]
    elif ablation == "zero":
        means = [np.zeros(w) for w in widths]
    else:
        raise ValueError(f"ablation must be 'mean' or 'zero', got {ablation!r}")
    return CircuitReconstruction(masks, means, widths, ablation, model)


# --------------------------------------------------------------------- R4: residue
def residual(acts: np.ndarray, recon_acts: np.ndarray) -> np.ndarray:
    """``R = h - h_hat`` (SPEC 2.13): what the reconstruction did not account for."""
    A = np.asarray(acts, dtype=np.float64)
    B = np.asarray(recon_acts, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError(f"activations {A.shape} and reconstruction {B.shape} must have the same shape")
    if A.ndim != 2:
        raise ValueError(f"expected (n_samples, d) arrays, got {A.shape}")
    return A - B


def _mp_quantile(gamma: float, q: float = 0.5, n_grid: int = 20000) -> float:
    """Quantile of the Marchenko-Pastur law (sigma^2 = 1), atom at 0 included when gamma > 1."""
    lo, hi = mp_edges(gamma, 1.0)
    atom = max(0.0, 1.0 - 1.0 / gamma)
    if atom >= q:
        return 0.0
    x = np.linspace(max(lo, 0.0), hi, n_grid)
    d = mp_density(x, gamma, 1.0)
    cdf = atom + np.cumsum(d) * (x[1] - x[0])   # mp_density integrates to min(1, 1/gamma)
    i = int(np.searchsorted(cdf, q))
    return float(x[min(i, n_grid - 1)])


@dataclass
class CoverageGap:
    """The structured fraction of the residue, with its two nulls (SPEC 2.13 / 7 R4).

    ``gamma_cov`` uses the Tracy-Widom-corrected edge at level ``q`` as the
    operational lambda_+ (the bare MP edge is reported as ``gamma_cov_mp_edge``);
    at finite n the bare edge is crossed by ordinary bulk fluctuation, which
    would manufacture a gap out of pure noise.

    Nulls: ``gamma_null_gaussian`` (i.i.d. Gaussian residue with the same
    per-coordinate marginals) and ``gamma_null_shuffle`` (each coordinate
    permuted independently across samples -- marginals kept, cross-coordinate
    structure destroyed). A gamma that is not clearly above both is not a gap:
    an isotropic-sigma^2 MP fit to a residue with a *heterogeneous diagonal*
    produces outliers from the diagonal alone (the E6 kill (i) that the
    correlation form of ``darkmatter.rmt.mp_outliers`` exists to avoid). The
    honest headline is therefore ``gamma_excess = gamma_cov - max(nulls)``, which
    is also reported; ``gamma_cov`` itself is the literal SPEC 2.13 quantity.
    Pass criterion on the provable tier (R4): gamma_cov < 0.05 with the
    synthetic-noise null < 0.02.
    """
    n: int
    d: int
    aspect: float
    sigma2: float
    mp_edge: float
    tw_edge: float
    n_outliers: int
    gamma_cov: float
    gamma_cov_mp_edge: float
    gamma_null_gaussian: float
    gamma_null_shuffle: float
    gamma_excess: float
    q: float
    seed: int
    eigenvalues: np.ndarray = field(repr=False, default=None)

    def to_dict(self) -> dict:
        return {"n": self.n, "d": self.d, "aspect": self.aspect, "sigma2": self.sigma2,
                "mp_edge": self.mp_edge, "tw_edge": self.tw_edge, "n_outliers": self.n_outliers,
                "gamma_cov": self.gamma_cov, "gamma_cov_mp_edge": self.gamma_cov_mp_edge,
                "gamma_null_gaussian": self.gamma_null_gaussian,
                "gamma_null_shuffle": self.gamma_null_shuffle,
                "gamma_excess": self.gamma_excess, "q": self.q, "seed": self.seed}


def _gap_from(R: np.ndarray, q: float) -> Tuple[float, float, float, float, float, int, np.ndarray]:
    R = np.asarray(R, dtype=np.float64)
    n, d = R.shape
    S = R.T @ R / n
    ev = np.sort(np.linalg.eigvalsh(S))[::-1]
    aspect = d / n
    med_mp = _mp_quantile(aspect, 0.5)
    med = float(np.median(ev))
    sigma2 = med / med_mp if (med_mp > 0 and med > 0) else float(np.trace(S) / d)
    hi = mp_edges(aspect, sigma2)[1]
    thr = tw_threshold(n, d, sigma2, q)
    tot = float(ev.sum())
    g = float(ev[ev > thr].sum() / tot) if tot > 0 else 0.0
    g_mp = float(ev[ev > hi].sum() / tot) if tot > 0 else 0.0
    return g, g_mp, float(sigma2), float(hi), float(thr), int((ev > thr).sum()), ev


def coverage_gap(R: np.ndarray, q: float = 0.99, seed: int = 0) -> CoverageGap:
    """gamma_cov = sum_{lambda_i > lambda_+} lambda_i / tr Sigma for the residue R (SPEC 2.13).

    Sigma = R^T R / n. sigma^2 is estimated from the *median* eigenvalue matched
    to the Marchenko-Pastur median at this aspect ratio (a median is insensitive
    to the outliers we are trying to count, whereas tr Sigma / d is not).

    Returns the gap together with both required nulls. On pure Gaussian residue
    gamma_cov ~ 0; three planted strong directions in 60 coordinates put it above
    0.3 (``tests/test_reconstruct.py``).
    Failure boundary: R must be (n_samples, d) with n > d for the MP fit to have
    a bulk; heavy-tailed residue violates the MP assumption and the nulls are
    what detects that.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError(f"residual must be (n_samples, d), got {R.shape}")
    n, d = R.shape
    g, g_mp, sigma2, hi, thr, nout, ev = _gap_from(R, q)
    g_gauss = _gap_from(gaussian_matched_diag(R, seed=seed), q)[0]
    g_shuf = _gap_from(feature_shuffle(R, seed=seed), q)[0]
    return CoverageGap(n, d, d / n, sigma2, hi, thr, nout, g, g_mp, g_gauss, g_shuf,
                       float(g - max(g_gauss, g_shuf)), float(q), int(seed), ev)
