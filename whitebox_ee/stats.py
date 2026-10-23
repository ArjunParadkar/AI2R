"""Statistics: bootstrap CIs, permutation tests, Holm correction, AUROC."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    bs = np.array([stat(x[i]) for i in idx])
    return (float(stat(x)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2)))


def bootstrap_ratio_ci(num, den, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    n = len(num)
    idx = rng.integers(0, n, size=(n_boot, n))
    r = num[idx].mean(1) / np.maximum(den[idx].mean(1), 1e-12)
    return (float(num.mean() / max(den.mean(), 1e-12)), float(np.quantile(r, alpha / 2)), float(np.quantile(r, 1 - alpha / 2)))


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def auroc(score, label):
    """P(score of positive > score of negative), ties count 1/2. label in {0,1}."""
    from scipy.stats import rankdata
    s = np.asarray(score, float)
    y = np.asarray(label).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_ci(score, label, n_boot: int = 2000, seed: int = 0, groups=None):
    """Bootstrap CI of AUROC; if `groups` is given, resample groups (unit of analysis)."""
    s, y = np.asarray(score, float), np.asarray(label).astype(int)
    rng = np.random.default_rng(seed)
    pt = auroc(s, y)
    bs = []
    if groups is None:
        for _ in range(n_boot):
            i = rng.integers(0, len(s), len(s))
            bs.append(auroc(s[i], y[i]))
    else:
        g = np.asarray(groups)
        ug = np.unique(g)
        idx = {u: np.where(g == u)[0] for u in ug}
        for _ in range(n_boot):
            pick = rng.choice(ug, len(ug))
            i = np.concatenate([idx[u] for u in pick])
            bs.append(auroc(s[i], y[i]))
    bs = np.array(bs)
    bs = bs[~np.isnan(bs)]
    return pt, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def paired_auroc_diff_ci(s1, s2, label, n_boot: int = 2000, seed: int = 0, groups=None):
    """CI for AUROC(s1) - AUROC(s2) on the same items (paired bootstrap)."""
    s1, s2, y = map(np.asarray, (s1, s2, label))
    rng = np.random.default_rng(seed)
    pt = auroc(s1, y) - auroc(s2, y)
    bs = []
    if groups is None:
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            bs.append(auroc(s1[i], y[i]) - auroc(s2[i], y[i]))
    else:
        g = np.asarray(groups)
        ug = np.unique(g)
        idx = {u: np.where(g == u)[0] for u in ug}
        for _ in range(n_boot):
            pick = rng.choice(ug, len(ug))
            i = np.concatenate([idx[u] for u in pick])
            bs.append(auroc(s1[i], y[i]) - auroc(s2[i], y[i]))
    bs = np.array(bs)
    bs = bs[~np.isnan(bs)]
    return float(pt), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def perm_pvalue_less(obs, null_samples):
    """One-sided p for obs being smaller than the null."""
    ns = np.asarray(null_samples)
    return float((1 + (ns <= obs).sum()) / (1 + len(ns)))
