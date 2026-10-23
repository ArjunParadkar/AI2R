"""Candidate proposers (master prompt §3.1 step 1). Proposal confers nothing."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .probe import Member, Probe, member_from_labels


def member_from_means(H: np.ndarray, labels: np.ndarray, site: str, var_keep: float = 0.99,
                      source: str = "probe-means") -> Optional[Member]:
    """Realiser = span of the centred class means of the labelled variable."""
    vals = np.unique(labels)
    if len(vals) < 2:
        return None
    mu = H.mean(0)
    M = np.stack([H[labels == v].mean(0) - mu for v in vals])
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    if S[0] <= 1e-9:
        return None
    e = np.cumsum(S ** 2) / (S ** 2).sum()
    k = int(min(np.searchsorted(e, var_keep) + 1, len(vals) - 1))
    U = Vt[:k].T
    return Member(site, U, np.zeros((len(vals), k)), vals, source)


def propose_probes(probe: Probe, labels: Dict[str, np.ndarray], sites: Optional[Sequence[str]] = None,
                   mode: str = "delta", estimators=("ridge",), decode_min: Optional[float] = None,
                   **kw) -> List[Member]:
    """Probe directions for known (or hypothesised) variables.
    mode='plain': decode from the state at the site.
    mode='delta': decode from what the block wrote (h_l - h_{l-1}); the canonical
                  realiser in residual architectures (phase1.md §2). Site 0 uses plain."""
    sites = list(sites or probe.sites)
    out = []
    for s in sites:
        i = probe.site_idx(s)
        ps = probe.prev_site(s)
        if mode == "delta" and ps is not None:
            H = probe.states[s] - probe.states[ps]
        else:
            H = probe.states[s]
        for name, lab in labels.items():
          for est in estimators:
            m = (member_from_labels(H, lab, s, source=f"probe:{name}", **kw) if est == "ridge"
                 else member_from_means(H, lab, s, source=f"probe-means:{name}"))
            if m is None:
                continue
            # minimal decoding prefix of the (singular-value ordered) basis: the smallest k
            # whose nearest-centroid decoding accuracy reaches decode_min (else the full basis)
            vals = np.unique(lab)
            Z = probe.states[s] @ m.U
            k_use = m.U.shape[1]
            for kk in (range(1, m.U.shape[1] + 1) if decode_min else []):
                cz = np.stack([Z[lab == v, :kk].mean(0) for v in vals])
                d2 = ((Z[:, None, :kk] - cz[None]) ** 2).sum(-1)
                if (vals[d2.argmin(1)] == lab).mean() >= decode_min:
                    k_use = kk
                    break
            m.U = m.U[:, :k_use]
            z = probe.states[s] @ m.U
            m.centroids = np.stack([z[lab == v].mean(0) for v in vals])
            m.value_ids = vals
            if mode == "delta" and ps is not None and np.abs(H @ m.U).max() < 1e-6:
                continue
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Sparse dictionary proposer: top-k SAE per site, then exclusive groups.
# ---------------------------------------------------------------------------

def train_topk_sae(H: np.ndarray, n_feat: int, k: int, steps: int = 3000, lr: float = 3e-3, seed: int = 0,
                   batch: int = 512, device: str = "cpu", center: bool = False):
    g = torch.Generator().manual_seed(seed)
    Ht = torch.as_tensor(H, dtype=torch.float32, device=device)
    mu = Ht.mean(0) if center else torch.zeros(Ht.shape[1], device=device)
    d = Ht.shape[1]
    Wd = torch.randn(n_feat, d, generator=g).to(device)
    Wd = Wd / Wd.norm(dim=1, keepdim=True)
    Wd.requires_grad_(True)
    We = Wd.detach().clone().T.contiguous().requires_grad_(True)
    be = torch.zeros(n_feat, device=device, requires_grad=True)
    opt = torch.optim.Adam([Wd, We, be], lr=lr)
    n = len(Ht)
    for t in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=g).to(device)
        x = Ht[idx] - mu
        pre = x @ We + be
        top = torch.topk(pre, k, dim=1)
        a = torch.zeros_like(pre).scatter(1, top.indices, torch.relu(top.values))
        xh = a @ Wd
        loss = ((xh - x) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            Wd /= Wd.norm(dim=1, keepdim=True).clamp_min(1e-8)
    with torch.no_grad():
        x = Ht - mu
        pre = x @ We + be
        top = torch.topk(pre, k, dim=1)
        a = torch.zeros_like(pre).scatter(1, top.indices, torch.relu(top.values))
        xc = Ht - Ht.mean(0)
        fvu = (((a @ Wd - x) ** 2).sum() / (xc ** 2).sum().clamp_min(1e-12)).item()
    return a.cpu().numpy(), Wd.detach().cpu().numpy(), fvu


def exclusive_groups(A: np.ndarray, min_freq: float = 0.01, max_freq: float = 0.95, min_cover: float = 0.95,
                     max_overlap: float = 0.01, max_groups: int = 200):
    """Groups of mutually exclusive binary features whose union covers the inputs
    (each group is a categorical variable: value = which feature is on)."""
    B = A > 1e-6
    f = B.mean(0)
    feats = np.where((f >= min_freq) & (f <= max_freq))[0]
    if len(feats) == 0:
        return []
    Bf = B[:, feats].astype(np.float32)
    co = (Bf.T @ Bf) / len(B)           # co-activation probability
    excl = co <= max_overlap * np.minimum.outer(f[feats], f[feats])
    np.fill_diagonal(excl, False)
    groups, seen = [], set()
    order = np.argsort(-f[feats])
    for start in order:
        grp = [start]
        cover = f[feats[start]]
        cand = set(np.where(excl[start])[0])
        while cover < min_cover and cand:
            best = max(cand, key=lambda j: f[feats[j]])
            grp.append(best)
            cover += f[feats[best]]
            cand &= set(np.where(excl[best])[0])
        if cover >= min_cover and len(grp) >= 2:
            key = tuple(sorted(feats[grp]))
            if key not in seen:
                seen.add(key)
                groups.append(list(key))
        if len(groups) >= max_groups:
            break
    return groups


def propose_dictionary(probe: Probe, sites: Optional[Sequence[str]] = None, mode: str = "delta",
                       expansion: int = 4, k: Optional[int] = None, seed: int = 0, steps: int = 3000,
                       binary_features: bool = False, device: str = "cpu", fvu_max: float = 0.02) -> List[Member]:
    """Unsupervised: SAE features -> exclusive groups -> categorical candidates.
    Realiser = span of the group's decoder directions; value = argmax group activation
    (nearest centroid in realiser coordinates)."""
    sites = list(sites or probe.sites)
    out = []
    for s in sites:
        i = probe.site_idx(s)
        ps = probe.prev_site(s)
        H = probe.states[s] - probe.states[ps] if (mode == "delta" and ps is not None) else probe.states[s]
        if np.abs(H).max() < 1e-8:
            continue
        d = H.shape[1]
        # effective rank
        sv = np.linalg.svd(H - H.mean(0), compute_uv=False)
        r = int((sv > 1e-6 * sv[0]).sum())
        # smallest k (1, 2, 3, ...) whose top-k SAE reconstructs the site (FVU <= fvu_max)
        ks = [k] if k else list(range(1, min(32, r) + 1))
        for kk in ks:
            A, Wd, fvu = train_topk_sae(H, expansion * max(r, 1), kk, steps=steps, seed=seed + i, device=device)
            if fvu <= fvu_max:
                break
        groups = exclusive_groups(A)
        for gi, grp in enumerate(groups):
            lab = np.argmax(A[:, grp], 1)
            dirs = Wd[grp]
            U = np.linalg.qr(dirs.T - dirs.T.mean(1, keepdims=True))[0]
            # keep only the rank of the centred directions
            sv2 = np.linalg.svd(dirs - dirs.mean(0), compute_uv=False)
            kr = max(1, int((sv2 > 1e-3 * sv2[0]).sum()))
            U = U[:, :kr]
            z = probe.states[s] @ U
            vals = np.unique(lab)
            cents = np.stack([z[lab == v].mean(0) for v in vals])
            out.append(Member(s, U, cents, vals, f"sae:{s}:g{gi}"))
        if binary_features:
            for j in range(A.shape[1]):
                on = A[:, j] > 1e-6
                if 0.05 < on.mean() < 0.95:
                    U = (Wd[j] / np.linalg.norm(Wd[j]))[:, None]
                    z = probe.states[s] @ U
                    cents = np.stack([z[~on].mean(0), z[on].mean(0)])
                    out.append(Member(s, U, cents, np.array([0, 1]), f"saef:{s}:f{j}"))
    return out


# ---------------------------------------------------------------------------
# Pullback proposer: used subspace of g = E[J^T J], then k-means quotients.
# ---------------------------------------------------------------------------

def propose_pullback(probe: Probe, sites: Optional[Sequence[str]] = None, n_dirs: int = 32,
                     ks: Sequence[int] = (2, 3, 5, 10), n_jac: int = 256, seed: int = 0,
                     per_behaviour: bool = True) -> List[Member]:
    """Requires system.downstream_from(site, h) (differentiable)."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    sites = list(sites or probe.sites)
    out = []
    for s in sites:
        rows = rng.choice(probe.N, min(n_jac, probe.N), replace=False)
        h = torch.as_tensor(probe.states[s][rows], dtype=torch.float64).requires_grad_(True)
        lg = probe.sys.downstream_from(s, h)
        mats = {}
        for b, L in lg.items():
            Gm = np.zeros((h.shape[1], h.shape[1]))
            p = torch.softmax(L, -1)
            # Jacobian of log-probs projected on a few random output directions
            for _ in range(min(L.shape[1], 8)):
                v = torch.as_tensor(rng.standard_normal(L.shape), dtype=L.dtype)
                gr, = torch.autograd.grad((torch.log_softmax(L, -1) * v).sum(), h, retain_graph=True)
                gnp = gr.numpy()
                Gm += gnp.T @ gnp / len(rows)
            mats[b] = Gm
        groups = list(mats.items()) if per_behaviour else [("all", sum(mats.values()))]
        for b, Gm in groups:
            w, V = np.linalg.eigh(Gm)
            w, V = w[::-1], V[:, ::-1]
            if w[0] <= 1e-10:
                continue
            kd = int(min(n_dirs, (w > 1e-3 * w[0]).sum()))
            U = V[:, :kd]
            z = probe.states[s] @ U
            for K in ks:
                km = KMeans(K, n_init=4, random_state=seed).fit(z)
                lab = km.labels_
                cents = np.stack([z[lab == v].mean(0) for v in range(K)])
                out.append(Member(s, U, cents, np.arange(K), f"pullback:{s}:{b}:K{K}"))
    return out


# ---------------------------------------------------------------------------
# Independent subspace analysis proposer: FastICA, then group the components by
# dependence (MI of quantile-binned components), one candidate per group, values
# by k-means in the group's subspace (K by silhouette).
# ---------------------------------------------------------------------------

def _discretise(a, bins=8):
    """Exact values if the component is discrete (<= 3*bins distinct values after rounding),
    else equal-mass bins on ranks (tie-safe)."""
    r = np.round(a, 5)
    u, inv = np.unique(r, return_inverse=True)
    if len(u) <= 3 * bins:
        return inv
    from scipy.stats import rankdata
    return np.minimum((rankdata(a, method="ordinal") - 1) * bins // len(a), bins - 1).astype(int)


def _binned_mi(a, b, bins=8):
    """Plug-in MI minus a permutation-null mean (bias-corrected), in bits."""
    from .info import mutual_info
    da, db = _discretise(a, bins), _discretise(b, bins)
    mi = mutual_info(da, db)
    rng = np.random.default_rng(0)
    null = np.mean([mutual_info(da, rng.permutation(db)) for _ in range(3)])
    return max(0.0, mi - null)


def propose_isa(probe: Probe, sites: Optional[Sequence[str]] = None, mode: str = "delta", seed: int = 0,
                mi_thresh: float = 0.15, k_max: int = 20, max_rank: int = 64) -> List[Member]:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import FastICA
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import fcluster, linkage
    sites = list(sites or probe.sites)
    out = []
    rng = np.random.default_rng(seed)
    for s in sites:
        i = probe.site_idx(s)
        ps = probe.prev_site(s)
        H = probe.states[s] - probe.states[ps] if (mode == "delta" and ps is not None) else probe.states[s]
        Hc = H - H.mean(0)
        if np.abs(Hc).max() < 1e-8:
            continue
        U_, S, Vt = np.linalg.svd(Hc, full_matrices=False)
        r = int(min(max_rank, (S > 1e-6 * S[0]).sum()))
        if r == 0:
            continue
        Z = Hc @ Vt[:r].T
        if r == 1:
            groups = [[0]]
            Sx = Z
            Mix = Vt[:1].T
        else:
            ica = FastICA(r, random_state=seed, max_iter=1000, whiten="unit-variance")
            Sx = ica.fit_transform(Z)
            Mix = Vt[:r].T @ ica.mixing_            # (d, r): component directions in state space
            sub = rng.choice(len(Sx), min(len(Sx), 2000), replace=False)
            M = np.zeros((r, r))
            for a in range(r):
                for b in range(a + 1, r):
                    M[a, b] = M[b, a] = _binned_mi(Sx[sub, a], Sx[sub, b])
            D = np.maximum(0, mi_thresh * 4 - M)
            np.fill_diagonal(D, 0)
            from scipy.spatial.distance import squareform
            Lk = linkage(squareform(D, checks=False), "average")
            lab = fcluster(Lk, t=mi_thresh * 3, criterion="distance")
            groups = [list(np.where(lab == g)[0]) for g in np.unique(lab)]
        for gi, grp in enumerate(groups):
            U = np.linalg.qr(Mix[:, grp])[0]
            z = probe.states[s] @ U
            zs = z[rng.choice(len(z), min(len(z), 1500), replace=False)]
            best, bestK = None, None
            n_distinct = len(np.unique(np.round(zs, 6), axis=0))
            for K in range(2, min(k_max, n_distinct) + 1):
                km = KMeans(K, n_init=3, random_state=seed).fit(zs)
                if len(np.unique(km.labels_)) < 2:
                    continue
                sc = silhouette_score(zs, km.labels_)
                if best is None or sc > best + 1e-3:
                    best, bestK = sc, K
            if bestK is None:
                continue
            km = KMeans(bestK, n_init=4, random_state=seed).fit(z)
            out.append(Member(s, U, km.cluster_centers_, np.arange(bestK), f"isa:{s}:g{gi}:K{bestK}"))
    return out
