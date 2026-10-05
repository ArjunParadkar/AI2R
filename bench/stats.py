"""Statistics for the pre-registered analyses.  Unit of analysis = independent training run (seed).  Authors: Arjun Paradkar and Reva Paradkar."""
import itertools, math
import numpy as np
from scipy.stats import rankdata, spearmanr


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 ties; pos/neg are arrays of per-seed scores."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    r = rankdata(np.r_[pos, neg]); n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def perm_p_auroc(pos, neg, max_exact=2_000_000, n_mc=200_000, seed=0):
    """One-sided permutation p for AUROC >= observed (exact enumeration when C(n,k) <= max_exact, else Monte Carlo with +1 correction)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float); allv = np.r_[pos, neg]; n1 = len(pos); n = len(allv)
    obs = auroc(pos, neg)
    if math.comb(n, n1) <= max_exact:
        cnt = tot = 0
        for idx in itertools.combinations(range(n), n1):
            m = np.zeros(n, bool); m[list(idx)] = True; a = auroc(allv[m], allv[~m]); tot += 1; cnt += a >= obs - 1e-12
        return obs, cnt / tot, 'exact'
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n_mc):
        m = np.zeros(n, bool); m[rng.choice(n, n1, replace=False)] = True; cnt += auroc(allv[m], allv[~m]) >= obs - 1e-12
    return obs, (cnt + 1) / (n_mc + 1), 'mc'


def boot_ci(x, fn=np.mean, n=5000, seed=0, alpha=0.05):
    """percentile bootstrap CI over seeds."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (float('nan'),) * 3
    rng = np.random.default_rng(seed); bs = np.array([fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    return float(fn(x)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def boot_ci_auroc(pos, neg, n=3000, seed=0):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float); rng = np.random.default_rng(seed)
    bs = [auroc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(n)]
    return auroc(pos, neg), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (dict key -> p)."""
    ks = sorted(pvals, key=lambda k: pvals[k]); m = len(ks); adj = {}; run = 0.0
    for i, k in enumerate(ks): run = max(run, min(1.0, (m - i) * pvals[k])); adj[k] = run
    return adj


def spearman_boot(x, y, n=3000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float); ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    r = float(spearmanr(x, y).statistic); rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x)); bs.append(spearmanr(x[i], y[i]).statistic)
    return r, float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))


def paired_abs_spearman_diff(x1, x2, y, n=3000, seed=0):
    """bootstrap CI of |rho(x1,y)| - |rho(x2,y)| over models (resampling models jointly)."""
    x1, x2, y = (np.asarray(v, float) for v in (x1, x2, y)); ok = ~(np.isnan(x1) | np.isnan(x2) | np.isnan(y)); x1, x2, y = x1[ok], x2[ok], y[ok]
    f = lambda i: abs(spearmanr(x1[i], y[i]).statistic) - abs(spearmanr(x2[i], y[i]).statistic)
    d = f(np.arange(len(y))); rng = np.random.default_rng(seed); bs = [f(rng.integers(0, len(y), len(y))) for _ in range(n)]
    return float(d), float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))
