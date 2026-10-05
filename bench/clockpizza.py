"""Offline descriptors of the modular-addition logit surface L[a,b,c] (clock vs pizza, H2.1).  Authors: Arjun Paradkar and Reva Paradkar.
Access level B (input->output map; no internals).  All descriptors are functions of the model's own outputs."""
import numpy as np, torch


def logit_tensor(model, n, DEV='cuda'):
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    X = torch.as_tensor(np.stack([a.ravel(), b.ravel()], 1), device=DEV); out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096): out.append(model(X[i:i + 4096]).float()[:, 0])
    return torch.cat(out).cpu().numpy().reshape(n, n, -1)


def spectral_split(L):
    """3D DFT of the (a,b,c) logit tensor (c-mean removed per (a,b)).  clock ~ joint (k,k,-k); pizza-like additive ~ marginal (k,0,-k),(0,k,-k).
    Returns M/(M+J) over all non-zero k, plus the same restricted to the top-3 key frequencies (blind: chosen by total energy)."""
    n = L.shape[0]; Lc = L - L.mean(-1, keepdims=True); F = np.fft.fftn(Lc, axes=(0, 1, 2)); E = np.abs(F) ** 2
    ks = np.arange(1, n); J = np.array([E[k, k, (-k) % n] for k in ks]); M = np.array([E[k, 0, (-k) % n] + E[0, k, (-k) % n] for k in ks])
    tot = np.array([E[k, :, :].sum() + E[:, k, :].sum() for k in ks]); key = np.argsort(tot)[::-1][:3]
    return dict(spec_marginal_frac=float(M.sum() / max(M.sum() + J.sum(), 1e-30)), spec_marginal_frac_key=float(M[key].sum() / max((M[key] + J[key]).sum(), 1e-30)),
                spec_joint=float(J.sum() / E.sum()), spec_marg=float(M.sum() / E.sum()))


def amp_modulation(L):
    """max-logit surface over (a,b): CV across a-b at fixed a+b, averaged (pizza modulates amplitude with cos(w(a-b)/2))."""
    n = L.shape[0]; amp = L.max(-1); a, b = np.meshgrid(np.arange(n), np.arange(n), indexing='ij'); s = (a + b) % n; cv = 0.0
    for m in range(n):
        v = amp[s == m]; cv += v.std() / max(abs(v.mean()), 1e-12)
    return float(cv / n)


def margin_modulation(L):
    """margin (top1-top2) CV across a-b at fixed a+b."""
    n = L.shape[0]; t = np.sort(L, -1); mar = t[..., -1] - t[..., -2]; a, b = np.meshgrid(np.arange(n), np.arange(n), indexing='ij'); s = (a + b) % n; cv = 0.0
    for m in range(n):
        v = mar[s == m]; cv += v.std() / max(abs(v.mean()), 1e-12)
    return float(cv / n)


def swap_asymmetry(L):
    """||L[a,b]-L[b,a]|| / ||L|| : (a,b)->(b,a) symmetry of the learned map (commutativity), the cheapest group-structure descriptor."""
    return float(np.linalg.norm(L - L.transpose(1, 0, 2)) / np.linalg.norm(L))


def all_descriptors(model, n):
    L = logit_tensor(model, n); d = spectral_split(L); d.update(amp_cv=amp_modulation(L), margin_cv=margin_modulation(L), swap_asym=swap_asymmetry(L)); return d
