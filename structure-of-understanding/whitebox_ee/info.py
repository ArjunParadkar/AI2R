"""Discrete information estimators (bits)."""
from __future__ import annotations

import numpy as np


def _codes(cols):
    """Joint integer code for a list of int arrays (row-wise tuple)."""
    if len(cols) == 0:
        return None
    M = np.stack([np.asarray(c, dtype=np.int64) for c in cols], 1)
    _, inv = np.unique(M, axis=0, return_inverse=True)
    return inv.reshape(-1)


def entropy(v) -> float:
    v = np.asarray(v)
    _, c = np.unique(v, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def joint_code(cols):
    return _codes(cols)


def cond_entropy_plugin(c, cols) -> float:
    if not cols:
        return entropy(c)
    z = _codes(cols)
    return entropy(_codes([z, c])) - entropy(z)


def heldout_cond_xent(c, cols, folds: int = 5, alpha: float = 0.5, seed: int = 0) -> float:
    """Held-out conditional cross-entropy H(C|S) in bits by K-fold lookup tables
    with additive smoothing; unseen parent tuples fall back to the training
    marginal of C (so over-fine parent sets are penalised)."""
    c = np.asarray(c, dtype=np.int64)
    n = len(c)
    _, c = np.unique(c, return_inverse=True)
    K = c.max() + 1
    z = _codes(cols) if cols else np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold = np.empty(n, dtype=np.int64)
    fold[perm] = np.arange(n) % folds
    tot = 0.0
    nz = z.max() + 1
    for f in range(folds):
        tr, te = fold != f, fold == f
        marg = np.bincount(c[tr], minlength=K) + alpha
        marg = marg / marg.sum()
        tab = np.zeros((nz, K))
        np.add.at(tab, (z[tr], c[tr]), 1.0)
        seen = tab.sum(1) > 0
        probs = np.where(seen[:, None], (tab + alpha / K) / (tab.sum(1, keepdims=True) + alpha), marg[None, :])
        p = probs[z[te], c[te]]
        tot += -np.log2(np.clip(p, 1e-12, 1)).sum()
    return float(tot / n)


def mutual_info(a, b) -> float:
    return entropy(a) + entropy(b) - entropy(_codes([a, b]))


def variation_of_information(a, b) -> float:
    return 2 * entropy(_codes([a, b])) - entropy(a) - entropy(b)


def normalised_vi(a, b, n_eff=None) -> float:
    """VI / log2 N_eff (theory Def. 2.12)."""
    n = len(a) if n_eff is None else n_eff
    return variation_of_information(a, b) / max(np.log2(n), 1e-9)


def adjusted_mi(a, b) -> float:
    from sklearn.metrics import adjusted_mutual_info_score
    return float(adjusted_mutual_info_score(a, b))
