"""Level-D (biological access) routines for grid-cell populations.  Authors: Arjun Paradkar and Reva Paradkar.
 Response route  : recorded rates r_i(x) -> per-neuron translation-symmetry profile (autocorrelogram) -> module clustering -> lattice (a1,a2) per module -> characters.
 Connectivity route: recurrent matrix W -> spectral module clustering -> eigenvector embedding -> persistent H1 (torus rank 2).
 Nothing here takes the period, orientation or number of modules as input.  Validated on planted populations with exact ground truth (N7 gate)."""
import numpy as np
from scipy.signal import correlate2d
from scipy.cluster.vq import kmeans2
from scipy.ndimage import maximum_filter
from scipy.sparse.linalg import eigsh


# ------------------------------------------------------------------ planted ground truth
def planted_population(modules, arena=2.0, seed=0, bins=48):
    """modules: [(spacing, orientation_deg, n_neurons)].  Returns ratemaps (N,bins,bins), truth dict."""
    rng = np.random.default_rng(seed); g = (np.arange(bins) + 0.5) / bins * arena; X, Y = np.meshgrid(g, g, indexing='ij'); maps, mod, phases = [], [], []
    truth = dict(spacing=[], orientation=[], n=[], lattice=[])
    for m, (lam, th, n) in enumerate(modules):
        th = np.deg2rad(th); k = 4 * np.pi / (np.sqrt(3) * lam); ks = [k * np.array([np.cos(th + j * np.pi / 3), np.sin(th + j * np.pi / 3)]) for j in range(3)]
        a1 = lam * np.array([np.cos(th + np.pi / 6 + np.pi / 3), np.sin(th + np.pi / 6 + np.pi / 3)])   # lattice vectors of this wave set (|a|=lam)
        a1 = lam * np.array([np.cos(th + np.pi / 6), np.sin(th + np.pi / 6)]); a2 = lam * np.array([np.cos(th + np.pi / 6 + np.pi / 3), np.sin(th + np.pi / 6 + np.pi / 3)])
        truth['spacing'].append(lam); truth['orientation'].append(float((np.rad2deg(th) + 30) % 60))   # orientation of a lattice vector (waves are 30 deg off); truth['n'].append(n); truth['lattice'].append((a1, a2))
        for _ in range(n):
            ph = rng.uniform(0, 1, 2); p = ph[0] * a1 + ph[1] * a2   # phase on the lattice torus
            r = sum(np.cos(kk[0] * (X - p[0]) + kk[1] * (Y - p[1])) for kk in ks) / 3
            maps.append(np.maximum(r, 0) * rng.uniform(0.7, 1.3)); mod.append(m); phases.append(ph)
    truth['module'] = np.array(mod); truth['phase'] = np.array(phases); truth['arena'] = arena
    return np.array(maps), truth


# ------------------------------------------------------------------ response route
def autocorr(rm, maxlag=24):
    z = rm - rm.mean(); ac = correlate2d(z, z, mode='full') / max((z ** 2).sum(), 1e-12); c = np.array(ac.shape) // 2
    return ac[c[0] - maxlag:c[0] + maxlag + 1, c[1] - maxlag:c[1] + maxlag + 1]


def profiles(ratemaps, maxlag=24):
    return np.array([autocorr(r, maxlag).ravel() for r in ratemaps])


def cluster_modules(P, kmax=6, seed=0):
    """k-means on PCA of the symmetry profiles; k by silhouette (k=1 if best silhouette < 0.5)."""
    U, S, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False); Z = U[:, :6] * S[:6]
    best, bs = (1, np.zeros(len(P), int)), 0.5
    for k in range(2, kmax + 1):
        np.random.seed(seed); cen, lab = kmeans2(Z, k, minit='++', seed=seed)
        if len(set(lab)) < k: continue
        D = np.linalg.norm(Z[:, None] - Z[None], axis=2); sil = []
        for i in range(len(Z)):
            same = lab == lab[i]
            if same.sum() < 2: sil.append(0); continue
            a = D[i, same].sum() / (same.sum() - 1); b = min(D[i, lab == j].mean() for j in set(lab) if j != lab[i]); sil.append((b - a) / max(a, b))
        s = float(np.mean(sil))
        if s > bs: bs, best = s, (k, lab)
    return best[0], best[1], bs


def lattice_from_profile(ac, bin_size, maxlag=24, thresh=0.25):
    """peaks of the mean autocorrelogram (sub-bin centroids) -> two shortest independent vectors -> least-squares lattice fit over all peaks
    -> spacing, orientation (mod 60 deg)."""
    mx = maximum_filter(ac, size=5); pk = np.argwhere((ac == mx) & (ac > thresh)); pts = []
    for (i, j) in pk:
        if 1 <= i < ac.shape[0] - 1 and 1 <= j < ac.shape[1] - 1:
            w = np.maximum(ac[i - 1:i + 2, j - 1:j + 2], 0); ii, jj = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing='ij'); pts.append([i + (w * ii).sum() / w.sum(), j + (w * jj).sum() / w.sum()])
    if not pts: return None
    V = (np.array(pts) - np.array([maxlag, maxlag])) * bin_size; V = V[np.linalg.norm(V, axis=1) > 2 * bin_size]
    if len(V) < 2: return None
    V = V[np.argsort(np.linalg.norm(V, axis=1))]; a1 = V[0]; a2 = None
    for v in V[1:]:
        if abs(a1[0] * v[1] - a1[1] * v[0]) > 0.5 * np.linalg.norm(a1) * np.linalg.norm(v) * np.sin(np.deg2rad(30)): a2 = v; break
    if a2 is None: return None
    for _ in range(3):                                             # LSQ refinement: integer coordinates of every peak in the current basis
        B = np.stack([a1, a2], 1); C = np.linalg.solve(B, V.T).T; Ci = np.round(C); ok = (np.abs(C - Ci).max(1) < 0.25) & (np.abs(Ci).max(1) <= 4) & (np.abs(Ci).sum(1) > 0)
        if ok.sum() < 2: break
        Bn, *_ = np.linalg.lstsq(Ci[ok], V[ok], rcond=None); a1, a2 = Bn[0], Bn[1]
    ang = np.rad2deg(np.arctan2(a1[1], a1[0])) % 60
    return dict(a1=a1, a2=a2, spacing=float((np.linalg.norm(a1) + np.linalg.norm(a2)) / 2), orientation=float(ang), n_peaks=int(len(V)))


def hex_template_r2(rm, lam, arena):
    """characters: variance of the ratemap explained by the best 3 hexagonal plane waves of spacing lam (free orientation, phase)."""
    bins = rm.shape[0]; g = (np.arange(bins) + 0.5) / bins * arena; X, Y = np.meshgrid(g, g, indexing='ij'); k = 4 * np.pi / (np.sqrt(3) * lam); z = rm - rm.mean(); best = 0
    for th in np.linspace(0, np.pi / 3, 31)[:-1]:
        cols = []
        for j in range(3):
            ang = th + j * np.pi / 3; ph = k * (np.cos(ang) * X + np.sin(ang) * Y); cols += [np.cos(ph).ravel(), np.sin(ph).ravel()]
        A = np.stack(cols, 1); coef, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None); r2 = 1 - ((z.ravel() - A @ coef) ** 2).sum() / max((z ** 2).sum(), 1e-12); best = max(best, r2)
    return best


def response_route(ratemaps, arena, maxlag=24):
    bins = ratemaps.shape[1]; bs = arena / bins; P = profiles(ratemaps, maxlag); k, lab, sil = cluster_modules(P); out = []; W = 2 * maxlag + 1
    for m in range(k):
        idx = np.where(lab == m)[0]; ac = P[idx].mean(0).reshape(W, W); L = lattice_from_profile(ac, bs, maxlag)
        if L is None: out.append(dict(n=len(idx), lattice=None)); continue
        r2 = float(np.mean([hex_template_r2(ratemaps[i], L['spacing'], arena) for i in idx[:20]])); out.append(dict(n=int(len(idx)), r2_characters=r2, **L))
    return dict(n_modules=k, silhouette=sil, labels=lab, modules=out)


def score_response_route(res, truth, tol_spacing=0.05, tol_orient=5.0):
    """match recovered modules to truth by spacing; report errors."""
    rec = [m for m in res['modules'] if m.get('lattice', 1) is not None and 'spacing' in m]; ts = truth['spacing']; used = set(); errs = []
    for t, (lam, ori) in enumerate(zip(ts, truth['orientation'])):
        cand = [(abs(m['spacing'] - lam) / lam, j) for j, m in enumerate(rec) if j not in used]
        if not cand: errs.append(None); continue
        e, j = min(cand); used.add(j); do = abs((rec[j]['orientation'] - ori + 30) % 60 - 30); errs.append(dict(spacing_rel_err=float(e), orient_err=float(do), r2=rec[j]['r2_characters']))
    ok = (res['n_modules'] == len(ts)) and all(e is not None and e['spacing_rel_err'] <= tol_spacing and e['orient_err'] <= tol_orient for e in errs)
    return dict(n_modules_ok=res['n_modules'] == len(ts), per_module=errs, pass_=bool(ok))


# ------------------------------------------------------------------ connectivity route
def planted_connectivity(n_per_module, sigma=0.35, coupling=0.02, seed=0):
    """Translation-invariant W on a square torus per module (phases uniform), weak random inter-module coupling."""
    rng = np.random.default_rng(seed); N = sum(n_per_module); W = np.zeros((N, N)); off = 0; ph = []
    for n in n_per_module:
        p = rng.uniform(0, 1, (n, 2)); d = np.abs(p[:, None] - p[None]); d = np.minimum(d, 1 - d); W[off:off + n, off:off + n] = np.exp(-(d ** 2).sum(-1) / (2 * (sigma / 2) ** 2)); ph.append(p); off += n
    W += coupling * rng.random((N, N)) * (W.max()); perm = rng.permutation(N)
    return W[np.ix_(perm, perm)], perm, np.concatenate(ph)


def spectral_modules(W, kmax=6, min_gap=0.25, min_level=0.7):
    """normalised-adjacency spectrum.  k modules <=> k eigenvalues near the Perron value then a gap >= min_gap (modules are almost disconnected);
    a smooth manifold (ring/torus) has no such gap, so k=1."""
    A = (W + W.T) / 2; A = A - np.diag(np.diag(A)); d = A.sum(1); L = A / np.sqrt(np.outer(d, d)); ev, V = np.linalg.eigh(L); ev, V = ev[::-1], V[:, ::-1]
    k = 1
    for i in range(kmax):
        if ev[i] >= min_level and ev[i] - ev[i + 1] >= min_gap: k = i + 1; break
    return ev, V, k


def embed_dim(ev_block, maxd=8):
    """number of oscillatory modes after the constant one: position of the largest eigenvalue gap among ev[1:maxd+1], rounded up to even."""
    e = ev_block[1:maxd + 2]; g = e[:-1] - e[1:]; d = int(np.argmax(g)) + 1
    return d + (d % 2)


def torus_from_block(Wb, dim=None):
    """Connectivity -> metric on neurons: d_ij = sqrt(-2 log(W_ij / max W)) (exact toroidal phase distance for a Gaussian kernel, monotone-equivalent
    otherwise), then persistent H1 of the Rips filtration.  No coordinates, periods or dimensions are supplied."""
    from ripser import ripser
    A = (Wb + Wb.T) / 2; A = np.maximum(A, 0); np.fill_diagonal(A, A.max()); Wn = np.clip(A / A.max(), 1e-4, 1.0); D = np.sqrt(-2 * np.log(Wn)); np.fill_diagonal(D, 0)
    dg = ripser(D, maxdim=1, distance_matrix=True)['dgms'][1]; life = np.sort(dg[:, 1] - dg[:, 0])[::-1] if len(dg) else np.array([0.0])
    return D, life, None


def betti_from_life(life, min_ratio=1.5):
    """number of 'long' bars = position of the largest drop between consecutive sorted lifetimes (ratio >= min_ratio); one bar alone counts if >0."""
    l = np.sort(np.asarray(life, float))[::-1][:6]; l = l[l > 1e-9]
    if len(l) == 0: return 0
    if len(l) == 1: return 1
    r = l[:-1] / l[1:]; i = int(np.argmax(r)); return i + 1 if r[i] >= min_ratio else 0


def null_lifetime(Wb, n_null=6, seed=0):
    """max H1 lifetime over symmetric entry-shuffles of the block (same weight distribution, no geometry)."""
    rng = np.random.default_rng(seed + 991); n = len(Wb); iu = np.triu_indices(n, 1); vals = ((Wb + Wb.T) / 2)[iu]; mx = 0.0
    for _ in range(n_null):
        Wn = np.zeros((n, n)); Wn[iu] = rng.permutation(vals); Wn = Wn + Wn.T; np.fill_diagonal(Wn, vals.max()); _, life, _ = torus_from_block(Wn); mx = max(mx, float(np.max(life)))
    return mx


def connectivity_route(W, kmax=6, seed=0):
    """No periods, no module count, no coordinates given: eigengap -> k modules -> per-module eigenvector embedding -> persistent H1 count."""
    from scipy.cluster.vq import kmeans2
    ev, V, k = spectral_modules(W, kmax); np.random.seed(seed); _, lab = kmeans2(V[:, :max(k, 1)] if k > 1 else V[:, :1], k, minit='++', seed=seed) if k > 1 else (None, np.zeros(len(W), int))
    mods = []
    for m in range(k):
        idx = np.where(lab == m)[0]
        if len(idx) < 20: mods.append(dict(n=len(idx), b1=None)); continue
        Wb = W[np.ix_(idx, idx)]; D, life, _ = torus_from_block(Wb)
        mods.append(dict(n=int(len(idx)), b1=betti_from_life(life), life=np.sort(life)[::-1][:5].tolist(), idx=idx))   # (shuffle-null threshold tried and rejected: see numerics/test_n7_bio.py header)
    return dict(k=k, eigs=ev[:kmax + 2].tolist(), modules=mods, labels=lab)
