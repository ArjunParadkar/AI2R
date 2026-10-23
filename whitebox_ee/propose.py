"""Candidate proposers (master prompt §3.1 step 1). Proposal confers nothing."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .probe import Member, Probe, member_from_labels


def propose_probes(probe: Probe, labels: Dict[str, np.ndarray], sites: Optional[Sequence[str]] = None,
                   mode: str = "delta", **kw) -> List[Member]:
    """Probe directions for known (or hypothesised) variables.
    mode='plain': decode from the state at the site.
    mode='delta': decode from what the block wrote (h_l - h_{l-1}); the canonical
                  realiser in residual architectures (phase1.md §2). Site 0 uses plain."""
    sites = list(sites or probe.sites)
    out = []
    for s in sites:
        i = probe.site_idx(s)
        if mode == "delta" and i > 0:
            H = probe.states[s] - probe.states[probe.sites[i - 1]]
        else:
            H = probe.states[s]
        for name, lab in labels.items():
            m = member_from_labels(H, lab, s, source=f"probe:{name}", **kw)
            if m is None:
                continue
            # value map acts on the full state
            z = probe.states[s] @ m.U
            vals = np.unique(lab)
            m.centroids = np.stack([z[lab == v].mean(0) for v in vals])
            m.value_ids = vals
            if mode == "delta" and i > 0 and np.abs(H @ m.U).max() < 1e-6:
                continue
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Sparse dictionary proposer: top-k SAE per site, then exclusive groups.
# ---------------------------------------------------------------------------

def train_topk_sae(H: np.ndarray, n_feat: int, k: int, steps: int = 3000, lr: float = 3e-3, seed: int = 0,
                   batch: int = 512, device: str = "cpu"):
    g = torch.Generator().manual_seed(seed)
    Ht = torch.as_tensor(H, dtype=torch.float32, device=device)
    mu = Ht.mean(0)
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
        fvu = (((a @ Wd - x) ** 2).sum() / (x ** 2).sum()).item()
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
                       binary_features: bool = False, device: str = "cpu") -> List[Member]:
    """Unsupervised: SAE features -> exclusive groups -> categorical candidates.
    Realiser = span of the group's decoder directions; value = argmax group activation
    (nearest centroid in realiser coordinates)."""
    sites = list(sites or probe.sites)
    out = []
    for s in sites:
        i = probe.site_idx(s)
        H = probe.states[s] - probe.states[probe.sites[i - 1]] if (mode == "delta" and i > 0) else probe.states[s]
        if np.abs(H).max() < 1e-8:
            continue
        d = H.shape[1]
        # effective rank
        sv = np.linalg.svd(H - H.mean(0), compute_uv=False)
        r = int((sv > 1e-6 * sv[0]).sum())
        kk = k or max(1, min(32, r // 3))
        A, Wd, fvu = train_topk_sae(H, expansion * max(r, 1), kk, steps=steps, seed=seed + i, device=device)
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
