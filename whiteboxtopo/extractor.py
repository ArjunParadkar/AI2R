"""The learned-symmetry extractor (Run 7, SPEC7.md).

A transformation family T is a LEARNED symmetry of a model at layer l iff training removed at least half of the layer's sensitivity to it:
    lambda_l(T) = (q_l^untrained(T) - q_l^trained(T)) / q_l^untrained(T) >= 0.5,
q_l(T) = mean ||h_l(Tx) - h_l(x)|| / mean ||h_l(x) - h_l(x')||.  Built-in symmetries (q_untrained ~ 0) are excluded by construction.
Neighbours: LieGG (Moskalev et al. 2022); Bouchacourt et al. 2021; Immer et al. 2022."""
import math, io
import numpy as np, torch, torch.nn.functional as F

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1); STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


def _img(x):   # normalised -> [0,1]
    return (x * STD.to(x.device) + MEAN.to(x.device))
def _nrm(im):
    return (im - MEAN.to(im.device)) / STD.to(im.device)


def _affine(x, theta):
    """theta: (2,3) affine on normalised coordinates, applied to the [0,1] image with zero fill (differentiable)."""
    im = _img(x); grid = F.affine_grid(theta.to(x.device).expand(len(x), 2, 3), im.shape, align_corners=False)
    return _nrm(F.grid_sample(im, grid, padding_mode='zeros', align_corners=False))


def _photometric(x, brightness=0.0, log_contrast=0.0, log_gamma=0.0, saturation=0.0, hue=0.0):
    im = _img(x).clamp(0, 1)
    if hue:
        M = YIQ.to(x.device); c, s = math.cos(hue), math.sin(hue); R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], device=x.device, dtype=im.dtype)
        im = torch.einsum('ij,bjhw->bihw', M.inverse() @ R @ M, im).clamp(0, 1)
    if saturation:
        lum = (0.299 * im[:, 0] + 0.587 * im[:, 1] + 0.114 * im[:, 2])[:, None]; im = (lum + (im - lum) * math.exp(saturation) if not torch.is_tensor(saturation) else lum + (im - lum) * torch.exp(saturation)).clamp(0, 1)
    if torch.is_tensor(log_contrast) or log_contrast:
        m = im.mean((1, 2, 3), keepdim=True); im = (m + (im - m) * (torch.exp(log_contrast) if torch.is_tensor(log_contrast) else math.exp(log_contrast))).clamp(0, 1)
    if torch.is_tensor(brightness) or brightness:
        im = (im + brightness).clamp(0, 1)
    if torch.is_tensor(log_gamma) or log_gamma:
        im = im.clamp_min(1e-4) ** (torch.exp(log_gamma) if torch.is_tensor(log_gamma) else math.exp(log_gamma))
    return _nrm(im)


def _blur(x, sigma):
    if sigma <= 0: return x
    k = int(2 * math.ceil(2 * sigma) + 1); t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2; g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    im = _img(x); im = F.conv2d(F.pad(im, (k // 2,) * 4, mode='reflect'), g.view(1, 1, 1, k).expand(3, 1, 1, k), groups=3); im = F.conv2d(im, g.view(1, 1, k, 1).expand(3, 1, k, 1), groups=3); return _nrm(im)


def _jpeg(x, quality):
    from PIL import Image
    im = (_img(x).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1); out = []
    for a in im:
        b = io.BytesIO(); Image.fromarray(a).save(b, 'JPEG', quality=int(quality)); out.append(np.asarray(Image.open(io.BytesIO(b.getvalue()))))
    return _nrm(torch.as_tensor(np.stack(out).transpose(0, 3, 1, 2).astype(np.float32) / 255, device=x.device))


def _cutout(x, size, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed); n = len(x); r = torch.randint(0, 32 - size + 1, (n, 2), generator=g); im = _img(x).clone()
    for i in range(n): im[i, :, r[i, 0]:r[i, 0] + size, r[i, 1]:r[i, 1] + size] = 0.5
    return _nrm(im)


def _elastic(x, alpha, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed); n = len(x); d = torch.randn(n, 2, 4, 4, generator=g); d = F.interpolate(d, size=(32, 32), mode='bicubic', align_corners=False).to(x.device) * alpha
    base = F.affine_grid(torch.eye(2, 3, device=x.device).expand(n, 2, 3), x.shape, align_corners=False); grid = base + d.permute(0, 2, 3, 1)
    return _nrm(F.grid_sample(_img(x), grid, padding_mode='zeros', align_corners=False))


def _noise(x, sigma, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed); return _nrm((_img(x) + sigma * torch.randn(x.shape, generator=g).to(x.device)).clamp(0, 1))


def _rot(deg):
    a = math.radians(deg); return torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0]], dtype=torch.float32)


# family -> (list of 5 magnitudes, callable(x, magnitude))
LIBRARY = {
    'rotation': ([3, 6, 10, 15, 20], lambda x, m: _affine(x, _rot(m))),
    'iso_scale': ([1.05, 1.1, 1.15, 1.2, 1.3], lambda x, m: _affine(x, torch.tensor([[1 / m, 0, 0], [0, 1 / m, 0]]))),
    'aniso_scale': ([1.05, 1.1, 1.15, 1.2, 1.3], lambda x, m: _affine(x, torch.tensor([[1 / m, 0, 0], [0, m ** 0.5, 0]]))),
    'shear': ([0.05, 0.1, 0.15, 0.2, 0.3], lambda x, m: _affine(x, torch.tensor([[1, m, 0], [0, 1, 0]]))),
    'translation_beyond_padding': ([5, 6, 7, 8, 10], lambda x, m: _affine(x, torch.tensor([[1, 0, 2 * m / 32], [0, 1, 0]]))),
    'brightness': ([0.05, 0.1, 0.15, 0.2, 0.3], lambda x, m: _photometric(x, brightness=m)),
    'contrast': ([0.1, 0.2, 0.3, 0.45, 0.6], lambda x, m: _photometric(x, log_contrast=-m)),
    'gamma': ([0.1, 0.2, 0.3, 0.45, 0.6], lambda x, m: _photometric(x, log_gamma=m)),
    'hue': ([0.1, 0.2, 0.35, 0.5, 0.7], lambda x, m: _photometric(x, hue=m)),
    'saturation': ([0.15, 0.3, 0.5, 0.7, 1.0], lambda x, m: _photometric(x, saturation=-m)),
    'gaussian_blur': ([0.3, 0.5, 0.7, 0.9, 1.2], lambda x, m: _blur(x, m)),
    'gaussian_noise': ([0.02, 0.04, 0.06, 0.08, 0.10], lambda x, m: _noise(x, m)),
    'jpeg': ([90, 75, 60, 45, 30], lambda x, m: _jpeg(x, m)),
    'cutout': ([4, 6, 8, 10, 12], lambda x, m: _cutout(x, int(m))),
    'elastic': ([0.02, 0.04, 0.06, 0.08, 0.10], lambda x, m: _elastic(x, m)),
    # built-in (for the exclusion check): circular shift and horizontal flip
    'shift_circular': ([1, 2, 3, 4, 5], lambda x, m: torch.roll(x, int(m), 3)),
    'hflip': ([1, 1, 1, 1, 1], lambda x, m: x.flip(3)),
}


def q_layers(feat_fn, x, Tx, pairs_idx):
    """feat_fn: x -> dict layer -> (N, d) features.  Returns dict layer -> q."""
    A, B = feat_fn(x), feat_fn(Tx); out = {}
    for L in A:
        a, b = A[L], B[L]; num = np.linalg.norm(b - a, axis=1).mean(); den = np.linalg.norm(a[pairs_idx[:, 0]] - a[pairs_idx[:, 1]], axis=1).mean(); out[L] = float(num / den)
    return out


def inventory(feat_fn, x, families=LIBRARY, seed=0):
    """q per family, magnitude and layer for one model."""
    rng = np.random.default_rng(seed); pairs = rng.integers(0, len(x), (len(x), 2)); out = {}
    for fam, (mags, T) in families.items():
        out[fam] = [q_layers(feat_fn, x, T(x, m), pairs) for m in mags]
    return out


def lambdas(q_trained, q_untrained_list, layer, thresh_builtin=0.05):
    """lambda per family at a layer: mean over magnitudes of (q_u - q_t)/q_u with q_u = mean over untrained seeds.  Built-in flagged."""
    out = {}
    for fam in q_trained:
        lam, qu_all = [], []
        for i in range(len(q_trained[fam])):
            qu = float(np.mean([Q[fam][i][layer] for Q in q_untrained_list])); qt = q_trained[fam][i][layer]; qu_all.append(qu); lam.append((qu - qt) / max(qu, 1e-9))
        out[fam] = dict(lambda_=float(np.mean(lam)), per_magnitude=lam, q_untrained=qu_all, q_trained=[q_trained[fam][i][layer] for i in range(len(lam))], builtin=bool(np.mean(qu_all) < thresh_builtin))
    return out


# ---- contrastive Lie-algebra search (X.3) ----
PHOTO = ('brightness', 'log_contrast', 'log_gamma', 'saturation')
LIB_DIRECTIONS = {   # representable library generators as 10-vectors [affine 2x3 flattened (6), brightness, log_contrast, log_gamma, saturation]
    'rotation': [0, -1, 0, 1, 0, 0, 0, 0, 0, 0], 'iso_scale': [-1, 0, 0, 0, -1, 0, 0, 0, 0, 0], 'aniso_scale': [-1, 0, 0, 0, 0.5, 0, 0, 0, 0, 0], 'shear': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'translation_beyond_padding': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'brightness': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'contrast': [0, 0, 0, 0, 0, 0, 0, -1, 0, 0], 'gamma': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'saturation': [0, 0, 0, 0, 0, 0, 0, 0, 0, -1]}


def apply_algebra(x, A, t):
    """exp(t * affine part) as a 2x3 matrix exponential, then the photometric part scaled by t."""
    M = torch.zeros(3, 3, device=x.device, dtype=x.dtype); M[:2] = (t * A[:6]).view(2, 3); E = torch.matrix_exp(M); theta = E[:2]
    y = _affine(x, theta); return _photometric(y, brightness=t * A[6], log_contrast=t * A[7], log_gamma=t * A[8], saturation=t * A[9])


def contrastive_search(feat_trained, feat_untrained, x, restarts=10, steps=60, lr=0.05, seed=0, scale=0.3):
    """Minimise q_trained - q_untrained at the penultimate layer over unit-norm A.  feat_*: x -> (N, d) penultimate features (torch, differentiable)."""
    g = torch.Generator().manual_seed(seed); res = []; n = len(x); idx = torch.randperm(n, generator=g)
    with torch.no_grad():
        den_t = (feat_trained(x) - feat_trained(x[idx])).norm(dim=1).mean(); den_u = (feat_untrained(x) - feat_untrained(x[idx])).norm(dim=1).mean(); base_t = feat_trained(x); base_u = feat_untrained(x)
    for r in range(restarts):
        A = torch.randn(10, generator=g).to(x.device); A = (A / A.norm()).requires_grad_(True); opt = torch.optim.Adam([A], lr=lr)
        for s in range(steps):
            An = scale * A / A.norm(); obj = 0
            for t in (0.25, 0.5, 0.75, 1.0):
                y = apply_algebra(x, An, t); obj = obj + ((feat_trained(y) - base_t).norm(dim=1).mean() / den_t - (feat_untrained(y) - base_u).norm(dim=1).mean() / den_u) / 4
            opt.zero_grad(); obj.backward(); opt.step()
        with torch.no_grad():
            An = scale * A / A.norm(); qt = float(np.mean([((feat_trained(apply_algebra(x, An, t)) - base_t).norm(dim=1).mean() / den_t).item() for t in (0.25, 0.5, 0.75, 1.0)])); qu = float(np.mean([((feat_untrained(apply_algebra(x, An, t)) - base_u).norm(dim=1).mean() / den_u).item() for t in (0.25, 0.5, 0.75, 1.0)]))
            a = (A / A.norm()).cpu().numpy(); near = max(LIB_DIRECTIONS, key=lambda k: abs(float(np.dot(a, np.array(LIB_DIRECTIONS[k]) / np.linalg.norm(LIB_DIRECTIONS[k])))))
            cos = float(np.dot(a, np.array(LIB_DIRECTIONS[near]) / np.linalg.norm(LIB_DIRECTIONS[near])))
        res.append(dict(A=a.tolist(), q_trained=qt, q_untrained=qu, lambda_=(qu - qt) / max(qu, 1e-9), nearest=near, cosine=cos))
    return sorted(res, key=lambda d: -d['lambda_'])
