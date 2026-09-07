"""whiteboxtopo.nulls — the null models every experiment is killed against (handbook S3).

* ``feature_shuffle(X)``  — independently permute each column: destroys all joint
  structure, keeps every marginal (the null that killed v1's TC).
* ``label_permute(y)``    — permute labels.
* ``marchenko_pastur(spectrum, aspect, var=None)`` — MP bulk edges and the Tracy–Widom
  threshold for a covariance spectrum with aspect ratio p/n; ``var`` estimated from the
  bulk median when not given. Returns outlier count and structured fraction.
* ``planted(kind, size)`` — data with structure of known size for calibration:
  ``"gaussian"`` (none), ``"spikes"`` (k spiked covariance directions), ``"clusters"``
  (k Gaussian blobs), ``"circle"`` (one H1 loop), ``"torus"`` (two H1 loops),
  ``"blocks"`` (block-structured graph adjacency), ``"lowrank"`` (rank-k signal + noise).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import optimize, integrate
from ..darkmatter.rmt import feature_shuffle, mp_edges, tw_threshold, mp_density


def label_permute(y, seed: int = 0):
    rng = np.random.default_rng(seed)
    return np.asarray(y)[rng.permutation(len(y))]


def _mp_median(gamma: float, sigma2: float = 1.0) -> float:
    """Median of the MP law (for sigma^2 estimation from the bulk median)."""
    lo, hi = mp_edges(gamma, sigma2)
    if gamma > 1:  # point mass at 0 of weight 1-1/gamma; the continuous part carries 1/gamma
        pass
    cdf = lambda t: integrate.quad(lambda x: mp_density(x, gamma, sigma2), lo, t, limit=200)[0]
    total = cdf(hi)
    target = 0.5 * total if gamma <= 1 else max(0.5 - (1 - 1 / gamma), 1e-9) / (1 / gamma) * total
    return optimize.brentq(lambda t: cdf(t) - target, lo + 1e-12, hi - 1e-12)


@dataclass
class MPNull:
    n_outliers: int
    structured_fraction: float
    bulk_lo: float
    bulk_hi: float
    tw_threshold: float
    sigma2: float
    gamma: float
    outlier_eigenvalues: np.ndarray


def marchenko_pastur(spectrum, aspect: float, var: Optional[float] = None, q: float = 0.99, n: Optional[int] = None) -> MPNull:
    """MP null for a covariance ``spectrum`` (eigenvalues, any order) at aspect ratio
    ``aspect = p/n``. ``var`` (sigma^2) defaults to the bulk-median estimate: sigma^2 such that
    the MP median matches the sample median. Outliers are eigenvalues above the Tracy–Widom
    threshold at level ``q`` (needs ``n``; if None, n = p/aspect)."""
    ev = np.sort(np.asarray(spectrum, dtype=np.float64))[::-1]
    p = len(ev); n = int(round(p / aspect)) if n is None else n
    gamma = p / n
    if var is None:
        med = float(np.median(ev[ev > 1e-12])) if (ev > 1e-12).any() else 1.0
        var = med / _mp_median(gamma, 1.0)
    lo, hi = mp_edges(gamma, var)
    thr = tw_threshold(n, p, var, q)
    out = ev > thr
    tot = ev.sum()
    return MPNull(int(out.sum()), float(ev[out].sum() / tot) if tot > 0 else 0.0, float(lo), float(hi), float(thr),
                  float(var), float(gamma), ev[out])


def planted(kind: str, size, seed: int = 0, noise: float = 1.0, **kw):
    """Calibration data with known structure. ``size`` is (n, p) for matrix kinds, n for clouds.
    Returns ``(data, truth)`` where ``truth`` states what was planted."""
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        n, p = size
        return rng.normal(0, noise, size=(n, p)), {"n_outliers": 0, "rank": 0, "b1": 0}
    if kind == "spikes":
        n, p = size; k = kw.get("k", 5); strength = kw.get("strength", 10.0)
        U = np.linalg.qr(rng.normal(size=(p, k)))[0]
        Z = rng.normal(size=(n, k)) * np.sqrt(strength)
        return rng.normal(0, noise, size=(n, p)) + Z @ U.T, {"n_outliers": k, "rank": k, "strength": strength}
    if kind == "lowrank":
        n, p = size; k = kw.get("k", 5)
        return rng.normal(size=(n, k)) @ rng.normal(size=(k, p)) + noise * rng.normal(size=(n, p)), {"rank": k}
    if kind == "clusters":
        n, p = size; k = kw.get("k", 4); sep = kw.get("sep", 8.0)
        centers = rng.normal(size=(k, p)) * sep / np.sqrt(p)
        lab = rng.integers(0, k, size=n)
        return centers[lab] + noise * rng.normal(size=(n, p)), {"k": k, "labels": lab, "b0": k}
    if kind == "circle":
        n = size if np.isscalar(size) else size[0]; p = kw.get("p", 2); r = kw.get("radius", 1.0)
        t = rng.uniform(0, 2 * np.pi, n)
        X = np.zeros((n, p)); X[:, 0] = r * np.cos(t); X[:, 1] = r * np.sin(t)
        return X + noise * rng.normal(size=(n, p)), {"b1": 1, "radius": r, "death_scale": np.sqrt(3) * r}
    if kind == "torus":
        n = size if np.isscalar(size) else size[0]; p = kw.get("p", 3); R, r = kw.get("R", 2.0), kw.get("r", 0.7)
        u, v = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
        X = np.zeros((n, p)); X[:, 0] = (R + r * np.cos(v)) * np.cos(u); X[:, 1] = (R + r * np.cos(v)) * np.sin(u); X[:, 2] = r * np.sin(v)
        return X + noise * rng.normal(size=(n, p)), {"b1": 2, "b2": 1}
    if kind == "blocks":
        n = size if np.isscalar(size) else size[0]; k = kw.get("k", 4); p_in, p_out = kw.get("p_in", 0.3), kw.get("p_out", 0.02)
        lab = np.sort(rng.integers(0, k, size=n))
        P = np.where(lab[:, None] == lab[None, :], p_in, p_out)
        A = (rng.uniform(size=(n, n)) < P).astype(float); A = np.triu(A, 1); A = A + A.T
        return A, {"k": k, "labels": lab}
    raise ValueError(f"unknown planted kind {kind!r}")


__all__ = ["feature_shuffle", "label_permute", "marchenko_pastur", "MPNull", "planted", "mp_edges", "tw_threshold"]
