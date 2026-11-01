"""Test 1 — Marchenko–Pastur bulk + Tracy–Widom edge for a sample covariance.

References: Marchenko & Pastur 1967; Johnstone 2001 (Ann. Stat. 29(2)) for the
Wishart largest-eigenvalue centring/scaling; Tracy & Widom 1994; Bejan 2005 for
TW1 quantiles; Baik, Ben Arous & Péché 2005 (BBP) for when a planted spike is
detectable at all.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

# TW1 (beta=1) quantiles, Bejan 2005 / Bornemann 2010 tables.
TW1_QUANTILES = {0.90: 0.4501, 0.95: 0.9793, 0.99: 2.0234, 0.999: 3.2724}


def mp_edges(gamma: float, sigma2: float = 1.0):
    """Marchenko–Pastur support [λ-, λ+] for aspect ratio gamma = p/n."""
    return sigma2 * (1 - np.sqrt(gamma)) ** 2, sigma2 * (1 + np.sqrt(gamma)) ** 2


def mp_density(lam, gamma: float, sigma2: float = 1.0):
    lam = np.asarray(lam, dtype=float)
    lo, hi = mp_edges(gamma, sigma2)
    out = np.zeros_like(lam)
    m = (lam > lo) & (lam < hi)
    out[m] = np.sqrt((hi - lam[m]) * (lam[m] - lo)) / (2 * np.pi * gamma * sigma2 * lam[m])
    return out


def tw_threshold(n: int, p: int, sigma2: float = 1.0, q: float = 0.99) -> float:
    """Outlier threshold for eigenvalues of S = X^T X / n (X n×p, entries var sigma2).

    Johnstone 2001: n*λ_max ≈ μ + σ·TW1, μ = (√(n−1)+√p)², σ = (√(n−1)+√p)(1/√(n−1)+1/√p)^{1/3}.
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
    mp_edge: float          # λ+ (bulk edge)
    tw_threshold: float     # λ+ plus TW fluctuation at the chosen quantile
    n_outliers: int
    structured_fraction: float   # Σ outlier λ / Σ λ
    excess_fraction: float       # Σ (outlier λ − λ+) / Σ λ
    eigenvalues: np.ndarray
    form: str

    def as_row(self) -> dict:
        d = asdict(self); d.pop("eigenvalues"); return d


def _eigs_sym(M: np.ndarray) -> np.ndarray:
    ev = np.linalg.eigvalsh(M)
    return np.sort(ev)[::-1]


def mp_outliers(E: np.ndarray, form: str = "correlation", q: float = 0.99, eps: float = 1e-12) -> MPResult:
    """Count eigenvalues of the sample covariance/correlation of E (n×p) above the TW edge.

    form='correlation': columns standardised, σ²=1 (primary in SPEC).
    form='covariance' : columns centred, σ² = trace(S)/p (secondary).
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
    """Kill (iii): permute each column independently across rows (keeps marginals, kills correlations)."""
    rng = np.random.default_rng(seed)
    E = np.asarray(E)
    return np.stack([rng.permutation(E[:, j]) for j in range(E.shape[1])], 1)


def gaussian_matched_diag(E: np.ndarray, seed: int = 0) -> np.ndarray:
    """Kill (i): i.i.d. Gaussian rows with the same per-dimension variance as E."""
    rng = np.random.default_rng(seed)
    E = np.asarray(E, dtype=np.float64)
    return rng.normal(size=E.shape) * E.std(0, keepdims=True) + E.mean(0, keepdims=True)


def gaussian_matched_cov(E: np.ndarray, seed: int = 0, n: int | None = None) -> np.ndarray:
    """Linear-Gaussian reference N(mean(E), Cov(E)) (used in Test 2)."""
    rng = np.random.default_rng(seed)
    E = np.asarray(E, dtype=np.float64)
    n = n or E.shape[0]
    mu = E.mean(0); C = np.cov(E, rowvar=False)
    return rng.multivariate_normal(mu, C, size=n, method="cholesky" if np.all(np.linalg.eigvalsh(C) > 0) else "svd")


def project_out_top_pcs(E: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """Remove the top-k principal directions of X from E (kill (ii) refinement)."""
    Xc = X - X.mean(0, keepdims=True)
    # top-k right singular vectors of Xc via eig of the d×d covariance
    C = Xc.T @ Xc / len(Xc)
    w, V = np.linalg.eigh(C)
    Vk = V[:, np.argsort(w)[::-1][:k]]
    Ec = E - E.mean(0, keepdims=True)
    return Ec - (Ec @ Vk) @ Vk.T


def block_bootstrap_outliers(E: np.ndarray, groups: np.ndarray, B: int = 50, seed: int = 0, form: str = "correlation",
                             q: float = 0.99) -> dict:
    """Resample whole groups (sequences) with replacement; percentile CIs for the MP statistics."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    n_out, sf, ef = [], [], []
    for _ in range(B):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        r = mp_outliers(E[idx], form, q)
        n_out.append(r.n_outliers); sf.append(r.structured_fraction); ef.append(r.excess_fraction)
    pct = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return dict(n_outliers_ci=pct(n_out), structured_fraction_ci=pct(sf), excess_fraction_ci=pct(ef), B=B)
