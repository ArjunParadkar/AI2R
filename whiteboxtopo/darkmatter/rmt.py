"""Marchenko-Pastur bulk + Tracy-Widom edge for a sample covariance/correlation.

Ported from whitebox-v2 E6 (``e6lib/rmt.py``), unchanged except for the
addition of ``subsample_outliers`` (the without-replacement CI form, SPEC
Amendment 2). References: Marchenko & Pastur 1967; Johnstone 2001 (Ann.
Stat. 29(2)) for the Wishart largest-eigenvalue centring/scaling; Tracy &
Widom 1994; Bejan 2005 for TW1 quantiles; Baik, Ben Arous & Peche 2005 (BBP)
for when a planted spike is detectable at all.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import numpy as np

# TW1 (beta=1) quantiles, Bejan 2005 / Bornemann 2010 tables.
TW1_QUANTILES = {0.90: 0.4501, 0.95: 0.9793, 0.99: 2.0234, 0.999: 3.2724}


def mp_edges(gamma: float, sigma2: float = 1.0):
    """Marchenko-Pastur support [lambda-, lambda+] for aspect ratio gamma = p/n."""
    return sigma2 * (1 - np.sqrt(gamma)) ** 2, sigma2 * (1 + np.sqrt(gamma)) ** 2


def mp_density(lam, gamma: float, sigma2: float = 1.0):
    lam = np.asarray(lam, dtype=float)
    lo, hi = mp_edges(gamma, sigma2)
    out = np.zeros_like(lam)
    m = (lam > lo) & (lam < hi)
    out[m] = np.sqrt((hi - lam[m]) * (lam[m] - lo)) / (2 * np.pi * gamma * sigma2 * lam[m])
    return out


def tw_threshold(n: int, p: int, sigma2: float = 1.0, q: float = 0.99) -> float:
    """Outlier threshold for eigenvalues of S = X^T X / n (X n x p, entries var sigma2).

    Johnstone 2001: n*lambda_max ~ mu + sigma*TW1, mu = (sqrt(n-1)+sqrt(p))^2,
    sigma = (sqrt(n-1)+sqrt(p))*(1/sqrt(n-1)+1/sqrt(p))^(1/3).
    """
    a, b = np.sqrt(n - 1), np.sqrt(p)
    mu = (a + b) ** 2
    sig = (a + b) * (1 / a + 1 / b) ** (1.0 / 3.0)
    return sigma2 * (mu + sig * TW1_QUANTILES[q]) / n


@dataclass
class MPResult:
    n: int
    p: int
    gamma: float
    sigma2: float
    mp_edge: float          # lambda+ (bulk edge)
    tw_threshold: float     # lambda+ plus TW fluctuation at the chosen quantile
    n_outliers: int
    structured_fraction: float   # sum outlier lambda / sum lambda
    excess_fraction: float       # sum (outlier lambda - lambda+) / sum lambda
    eigenvalues: np.ndarray
    form: str

    def as_row(self) -> dict:
        d = asdict(self); d.pop("eigenvalues"); return d


def _eigs_sym(M: np.ndarray) -> np.ndarray:
    ev = np.linalg.eigvalsh(M)
    return np.sort(ev)[::-1]


def mp_outliers(E: np.ndarray, form: str = "correlation", q: float = 0.99, eps: float = 1e-12) -> MPResult:
    """Count eigenvalues of the sample covariance/correlation of E (n x p) above the TW edge.

    form='correlation': columns standardised, sigma^2=1 (primary; E6 SPEC Amendment/JC:
      an isotropic-sigma^2 MP fit to a heterogeneous-diagonal covariance produces spurious
      outliers from the diagonal alone -- see kill (i) in the docstring of ``residual_structure``).
    form='covariance' : columns centred only, sigma^2 = trace(S)/p (secondary; over-counts).
    """
    E = np.asarray(E, dtype=np.float64)
    n, p = E.shape
    Ec = E - E.mean(0, keepdims=True)
    if form == "correlation":
        sd = Ec.std(0, ddof=0); sd = np.where(sd < eps, 1.0, sd)
        Ec = Ec / sd
        S = Ec.T @ Ec / n
        sigma2 = 1.0
    elif form == "covariance":
        S = Ec.T @ Ec / n
        sigma2 = float(np.trace(S) / p)
    else:
        raise ValueError(form)
    gamma = p / n
    ev = _eigs_sym(S)
    _, hi = mp_edges(gamma, sigma2)
    thr = tw_threshold(n, p, sigma2, q)
    out = ev > thr
    tot = float(ev.sum())
    sf = float(ev[out].sum() / tot) if tot > 0 else 0.0
    ef = float((ev[out] - hi).sum() / tot) if tot > 0 else 0.0
    return MPResult(n, p, gamma, sigma2, float(hi), float(thr), int(out.sum()), sf, ef, ev, form)


def feature_shuffle(E: np.ndarray, seed: int = 0) -> np.ndarray:
    """Kill (iii): permute each column independently across rows (keeps marginals, kills
    cross-dimension correlation). If this produces > 1 outlier the instrument is broken
    on this data (see ``shuffle_validity`` / Test 1's ``spectral_test_valid`` field)."""
    rng = np.random.default_rng(seed)
    E = np.asarray(E)
    return np.stack([rng.permutation(E[:, j]) for j in range(E.shape[1])], 1)


def gaussian_matched_diag(E: np.ndarray, seed: int = 0) -> np.ndarray:
    """Kill (i): i.i.d. Gaussian rows with the same per-dimension variance as E (tests
    whether the correlation-form statistic is just picking up a heterogeneous diagonal)."""
    rng = np.random.default_rng(seed)
    E = np.asarray(E, dtype=np.float64)
    return rng.normal(size=E.shape) * E.std(0, keepdims=True) + E.mean(0, keepdims=True)


def gaussian_matched_cov(E: np.ndarray, seed: int = 0, n: Optional[int] = None) -> np.ndarray:
    """Linear-Gaussian reference N(mean(E), Cov(E)): 'covariance but no extra topology'
    control used by Test 2 (the covariance-matched Gaussian control)."""
    rng = np.random.default_rng(seed)
    E = np.asarray(E, dtype=np.float64)
    n = n or E.shape[0]
    mu = E.mean(0); C = np.cov(E, rowvar=False)
    method = "cholesky" if np.all(np.linalg.eigvalsh(C) > 0) else "svd"
    return rng.multivariate_normal(mu, C, size=n, method=method)


def project_out_top_pcs(E: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """Remove the top-k principal directions of X from E (kill (ii) refinement: is the
    spectral structure in E just X's leading directions leaking through the SAE?)."""
    Xc = X - X.mean(0, keepdims=True)
    C = Xc.T @ Xc / len(Xc)
    w, V = np.linalg.eigh(C)
    Vk = V[:, np.argsort(w)[::-1][:k]]
    Ec = E - E.mean(0, keepdims=True)
    return Ec - (Ec @ Vk) @ Vk.T


def _percentile_ci(a) -> tuple:
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def block_bootstrap_outliers(E: np.ndarray, groups: np.ndarray, B: int = 50, seed: int = 0,
                             form: str = "correlation", q: float = 0.99) -> Dict:
    """With-replacement block bootstrap over ``groups`` (e.g. sequences).

    WARNING (E6 SPEC Amendment 2 / same trap as E1 Amendment 5): resampling
    WITH replacement duplicates whole groups; applied to a feature-shuffled
    matrix a duplicated row is identical in every column, which re-introduces
    cross-column dependence the shuffle was meant to destroy, and produces a
    spuriously wide/high CI. Do not use this for the shuffle null or for the
    H-E6-style ">10 with CI lower bound >10" criterion -- use
    ``subsample_outliers`` (without replacement) for that; this function is
    kept only because some callers want the (biased) with-replacement form
    for comparison.
    """
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    n_out, sf, ef = [], [], []
    for _ in range(B):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        r = mp_outliers(E[idx], form, q)
        n_out.append(r.n_outliers); sf.append(r.structured_fraction); ef.append(r.excess_fraction)
    return dict(n_outliers_ci=_percentile_ci(n_out), structured_fraction_ci=_percentile_ci(sf),
               excess_fraction_ci=_percentile_ci(ef), B=B)


def subsample_outliers(E: np.ndarray, groups: Optional[np.ndarray] = None, n_draws: int = 20, frac: float = 0.8,
                       seed: int = 1, form: str = "correlation", q: float = 0.99) -> Dict:
    """Without-replacement subsample CI (E6 SPEC Amendment 2): ``n_draws`` draws of
    ``frac`` of the groups (rows are their own group by default -- pass ``groups``
    for sequence-level structure, e.g. token positions grouped by document), no
    row is ever duplicated within a draw, so the shuffle-independence trap of
    ``block_bootstrap_outliers`` cannot occur. This is the CI used by
    ``residual_structure`` for the spectral-structure verdict.
    """
    rng = np.random.default_rng(seed)
    E = np.asarray(E, dtype=np.float64)
    if groups is None:
        groups = np.arange(len(E))
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    m = max(2, int(round(frac * len(ug))))
    n_out, sf, ef = [], [], []
    for _ in range(n_draws):
        pick = rng.choice(ug, size=m, replace=False)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        r = mp_outliers(E[idx], form, q)
        n_out.append(r.n_outliers); sf.append(r.structured_fraction); ef.append(r.excess_fraction)
    return dict(n_outliers_ci=_percentile_ci(n_out), structured_fraction_ci=_percentile_ci(sf),
               excess_fraction_ci=_percentile_ci(ef), n_draws=n_draws, frac=frac, m=m,
               n_outliers_samples=np.array(n_out))
