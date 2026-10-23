"""Condition 1 (quotient) and Condition 3 (causal use) of theory §2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .info import entropy
from .probe import Member, Node, Probe
from .stats import bootstrap_ratio_ci, perm_pvalue_less
from .systems import softmax


@dataclass
class Thresholds:
    eps_Q: float = 0.10
    kappa_max: float = 0.6           # Phase 1 calibration (proposed 0.25 rejects every node of a 2-input system)
    h_min: float = 0.5
    eps_D: float = 0.05
    k_max: int = 8
    J_min: float = 0.8
    tau_E: float = 0.9
    tau_IIA: float = 0.90
    delta_IIA: float = 0.20
    eps_merge: float = 0.05
    alpha: float = 0.5
    taus: tuple = (0.25, 0.5)
    eps_M: float = 0.10
    n_min: int = 5
    eps_cov: float = 0.2
    theta_dir: float = 0.05          # direct-effect parent threshold (Phase 1 rule)
    n_pairs: int = 512
    n_partners: int = 8
    n_random_realisers: int = 20
    min_effective_pairs: int = 20


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


def swap_effect(probe: Probe, node: Node, rows, srcs, extra_patches=()):
    """Mean over behaviours (weights w) of TV between clean and patched outputs, per pair."""
    lg, _ = probe.run_patched(rows, [(node, srcs)], extra_patches=extra_patches)
    eff = []
    for b in probe.sys.behaviours:
        eff.append(tv(softmax(probe.logits[b][rows]), softmax(lg[b])))
    return np.stack(eff, 1)        # (n_pairs, n_behaviours)


def random_member(site, d, k, rng, match_U=None):
    """Random realiser of matched dimension; if match_U is given, each patch delta
    is rescaled to the norm the candidate's own patch would have (matched norm)."""
    U = np.linalg.qr(rng.standard_normal((d, k)))[0]
    return Member(site, U, np.zeros((1, k)), np.array([0]), "random", match_U)


def random_realiser_effects(probe: Probe, site: str, k: int, n: int, n_pairs: int, rng, match_U=None):
    """Unconditioned swap effect of random realisers of matched dimension and norm."""
    d = probe.sys.site_dim[site]
    out = []
    for _ in range(n):
        node = Node("rand", "concept", [random_member(site, d, k, rng, match_U)])
        rows = rng.integers(0, probe.N, n_pairs)
        srcs = rng.integers(0, probe.N, n_pairs)
        out.append(swap_effect(probe, node, rows, srcs).mean(0))
    return np.stack(out)            # (n, n_behaviours)


@dataclass
class QuotientResult:
    passed: bool
    A: float
    A_eq: float
    delta_Q: float
    delta_ci: tuple
    eta: float
    p_perm: float
    H: float
    kappa: float
    reasons: List[str] = field(default_factory=list)


def quotient_test(probe: Probe, node: Node, H_X: float, th: Thresholds, eta_cache: Dict, rng,
                  weights: Optional[np.ndarray] = None) -> QuotientResult:
    vals = node.values
    nb = len(probe.sys.behaviours)
    w = np.ones(nb) / nb if weights is None else np.asarray(weights)
    m = node.primary
    re = random_realiser_effects(probe, m.site, m.U.shape[1], th.n_random_realisers, th.n_pairs // 4, rng,
                                 match_U=m.U)
    eta_b = np.quantile(re, 0.95, axis=0)
    eta = float(np.quantile(re @ w, 0.95))
    # unconditioned pairs
    rows = rng.integers(0, probe.N, th.n_pairs)
    srcs = rng.integers(0, probe.N, th.n_pairs)
    E_all = swap_effect(probe, node, rows, srcs)
    # within-fibre pairs
    rows2 = rng.integers(0, probe.N, th.n_pairs)
    srcs2 = probe.partners(vals, 1, differ=False, rows=rows2, rng=rng)[:, 0]
    E_eq = swap_effect(probe, node, rows2, srcs2)
    A_b = E_all.mean(0)
    use = A_b > eta_b
    reasons = []
    A = float(A_b @ w)
    H = entropy(vals)
    kappa = H / max(H_X, 1e-9)
    # (i) Phase 1 calibration: the realiser must matter more than norm-matched random
    # realisers for at least one behaviour (A_b > eta_b); the pooled form A >= eta
    # is recorded but not used (it fails on hard-threshold planted nets, phase1.md §2).
    if not use.any():
        return QuotientResult(False, A, float(E_eq.mean(0) @ w), np.nan, (np.nan, np.nan), eta, 1.0, H, kappa,
                              ["realiser does not matter (A < eta)"])
    num = E_eq[:, use] @ w[use]
    den = E_all[:, use] @ w[use]
    dq, lo, hi = bootstrap_ratio_ci(num, den, n_boot=500, seed=int(rng.integers(1 << 30)))
    # permutation null: within-fibre effects vs effects of random partitions (pooled labels)
    pooled = np.concatenate([num, den])
    nn = len(num)
    null = []
    for _ in range(500):
        pi = rng.permutation(len(pooled))
        null.append(pooled[pi[:nn]].mean() / max(pooled[pi[nn:]].mean(), 1e-12))
    p = perm_pvalue_less(dq, null)
    ok = True
    if dq > th.eps_Q:
        ok = False; reasons.append(f"delta_Q {dq:.3f} > eps_Q")
    if hi > 2 * th.eps_Q:
        ok = False; reasons.append(f"delta_Q CI upper {hi:.3f} > 2 eps_Q")
    if kappa > th.kappa_max:
        ok = False; reasons.append(f"kappa {kappa:.3f} > kappa_max")
    if H < th.h_min:
        ok = False; reasons.append(f"H {H:.3f} < h_min")
    return QuotientResult(ok, A, float(E_eq.mean(0) @ w), dq, (lo, hi), eta, p, H, kappa, reasons)


def decodability(states: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    """Held-out linear-probe accuracy (for 'represented, not used')."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    if len(np.unique(labels)) < 2:
        return 1.0
    Xtr, Xte, ytr, yte = train_test_split(states, labels, test_size=0.3, random_state=seed)
    clf = LogisticRegression(max_iter=500)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))
