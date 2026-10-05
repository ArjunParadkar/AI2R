"""ResNet-18 / CIFAR-10: P6 (clean vs scrambled), P4 (SVHN, Gaussian noise), P5 (perturbation
failure radius), P1 fibers -- all at the penultimate 512-d layer.

psi from this layer is the final linear map, so J is one fixed matrix: the pullback metric is a
single quadratic form g = W^T W and the Lipschitz field is constant.  rho therefore reduces to
curvature alone here; that is stated with the P5 number.
"""
import sys, os, json, glob, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.resnet_cifar import make_model, features, load_cifar, MEAN, STD
from whiteboxtopo.problems import LayerData, p6_geodesic_generalization, fiber_beta0, p5_failure
from whiteboxtopo.problems.core import _softmax_conf
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.geometry import reach, ignorance
from whiteboxtopo.nulls import feature_shuffle, shuffle_predictor
from whiteboxtopo.topology import persistence, significant_classes
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

DEV = 'cuda'


def feats(m, x, bs=500):
    out = []
    with torch.no_grad():
        for s in range(0, len(x), bs):
            out.append(features(m, torch.as_tensor(x[s:s + bs]).to(DEV)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def layer(m, x, y):
    X = feats(m, x)
    W = m.fc.weight.detach().float().cpu().numpy(); b = m.fc.bias.detach().float().cpu().numpy()
    J = np.broadcast_to(W[None], (len(X),) + W.shape).astype(np.float32)
    return LayerData(name='penultimate', X=X, J=np.ascontiguousarray(J), logits=X @ W.T + b, labels=y), W, b


def _chunked(f, m, x, size=100, **kw):
    """The GPU is shared: run perturbation searches in small batches and release memory between."""
    out = []
    for i in range(0, len(x), size):
        out.append(f(m, x[i:i + size], **kw)); torch.cuda.empty_cache()
    return np.concatenate(out)


def failure_radius_input(m, x, **kw):
    return _chunked(_failure_radius_input, m, x, **kw)


def gaussian_failure_radius(m, x, **kw):
    return _chunked(_gaussian_failure_radius, m, x, **kw)


def _failure_radius_input(m, x, n_iter=14, hi=8.0):
    """Smallest L2 input perturbation along the margin-gradient direction that flips the
    prediction (one-step PGD direction + bisection), in normalised-pixel units."""
    xt = torch.as_tensor(x).to(DEV).requires_grad_(True)
    m.zero_grad()
    lg = m(xt).float()
    t2 = lg.topk(2, 1).values
    g, = torch.autograd.grad((t2[:, 0] - t2[:, 1]).sum(), xt)
    d = -(g / g.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None, None]).detach()
    base = lg.argmax(1).detach(); xd = xt.detach()
    lo = torch.zeros(len(x), device=DEV); hi_t = torch.full((len(x),), hi, device=DEV)
    with torch.no_grad():
        ok = m(xd + hi_t[:, None, None, None] * d).argmax(1) != base
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi_t)
            f = m(xd + mid[:, None, None, None] * d).argmax(1) != base
            hi_t = torch.where(f, mid, hi_t); lo = torch.where(f, lo, mid)
    r = (0.5 * (lo + hi_t)).cpu().numpy(); r[~ok.cpu().numpy()] = np.inf
    return r


def _gaussian_failure_radius(m, x, radii=(0.05, 0.1, 0.2, 0.4, 0.8, 1.6), n_rep=4, seed=0):
    """The spec's Gaussian test: smallest of six noise scales at which the prediction flips."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    xt = torch.as_tensor(x).to(DEV)
    out = np.full(len(x), np.inf)
    with torch.no_grad():
        base = m(xt).argmax(1)
        for r in radii:
            fl = torch.zeros(len(x), dtype=torch.bool, device=DEV)
            for _ in range(n_rep):
                fl |= m(xt + r * torch.randn(xt.shape, device=DEV, generator=g)).argmax(1) != base
            f = fl.cpu().numpy()
            out[f & ~np.isfinite(out)] = r
    return out


def analyse(tag, run, xtr, ytr_used, xte, yte, svhn, n_probe=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = make_model().to(DEV); m.load_state_dict(torch.load(sorted(glob.glob(f'{run}/ckpt_*.pt'))[-1])); m.eval()
    res = dict(run=run); t0 = time.time()
    # probe: training points, labelled with what the model was trained on
    sel = np.concatenate([rng.choice(np.where(ytr_used == c)[0], n_probe // 10, replace=False) for c in range(10)])
    ld, W, b = layer(m, xtr[sel], ytr_used[sel])
    res['train_probe_acc'] = float((ld.pred == ld.labels).mean())
    tf = time.time(); fit = fit_manifold(ld.X, ld.J, metric='g'); res['t_fit'] = time.time() - tf
    res['m'] = fit.m; res['d_intrinsic'] = float(np.nanmedian(fit.d_local))
    # P6
    tp = time.time(); g = p6_geodesic_generalization(ld, fit, pairs_per_class=200); b0 = fiber_beta0(ld, fit)
    res['P6'] = dict(G=g['G'], per_class=g['per_class'], beta0_fiber=b0,
                     beta0_fiber_mean=float(np.mean(list(b0.values()))), t=time.time() - tp)
    Xs = feature_shuffle(ld.X, 0)
    lds = LayerData(name='null', X=Xs, J=ld.J, logits=Xs @ W.T + b, labels=ld.labels)
    fs = fit_manifold(Xs, ld.J, metric='g')
    res['P6_null_feature_shuffle'] = dict(G=p6_geodesic_generalization(lds, fs)['G'], m=fs.m)
    # held-out generalisation (what the method does NOT use)
    Lte = feats(m, xte) @ W.T + b
    res['heldout_acc'] = float((Lte.argmax(1) == yte).mean())
    # P1 (fibers): H1 of each class fiber vs the null band
    rn = persistence(Xs[:600], do_cocycles=False); ln = rn['dgms'][1][:, 1] - rn['dgms'][1][:, 0]
    res['P1'] = dict(noise_band=float(ln.max()) if len(ln) else 0.0)
    # P4
    hold = rng.choice(len(xte), 1500, replace=False)
    Xh = feats(m, xte[hold]); Lh = Xh @ W.T + b
    eps, _, nnb = reach(fit.X, fit.bases, fit.J, Xh, 99)
    _, dg_in = ignorance(Xh, fit.X, fit.bases, fit.J, eps, nnb)
    nn_e = NearestNeighbors(n_neighbors=1).fit(fit.X); e_in = nn_e.kneighbors(Xh)[0][:, 0]
    noise = rng.standard_normal((1500, 3, 32, 32)).astype(np.float32)
    p4 = dict(reach=float(eps), sets={})
    for nm, xo in dict(svhn=svhn[:1500], gaussian_noise=noise).items():
        Xo = feats(m, xo); Lo = Xo @ W.T + b
        _, dg_o = ignorance(Xo, fit.X, fit.bases, fit.J, eps, nnb)
        y = np.r_[np.zeros(len(dg_in)), np.ones(len(dg_o))]
        p4['sets'][nm] = dict(auc_ignorance=float(roc_auc_score(y, np.r_[dg_in, dg_o])),
                              auc_softmax=float(roc_auc_score(y, np.r_[-_softmax_conf(Lh), -_softmax_conf(Lo)])),
                              auc_euclid_knn=float(roc_auc_score(y, np.r_[e_in, nn_e.kneighbors(Xo)[0][:, 0]])),
                              frac_outside_reach=float((dg_o > eps).mean()))
    res['P4'] = p4
    # P5 (on held-out inputs: where will it fail?)
    hp = hold[:1000]
    ldh, _, _ = layer(m, xte[hp], yte[hp])
    fith = fit_manifold(ldh.X, ldh.J, metric='g')
    fr = failure_radius_input(m, xte[hp]); frg = gaussian_failure_radius(m, xte[hp])
    p5 = p5_failure(ldh, fith, fr)
    ok = np.isfinite(fr)
    okg = np.isfinite(frg)
    res['P5'] = dict(spearman_rho=p5['spearman_rho'], p_rho=p5['p_rho'],
                     spearman_softmax_margin=p5['spearman_softmax_margin'],
                     null_shuffled=float(spearmanr(shuffle_predictor(p5['rho'], 0)[ok], fr[ok]).statistic),
                     gaussian_spearman_rho=float(spearmanr(p5['rho'][okg], frg[okg]).statistic) if okg.sum() > 20 else None,
                     n_saddles=int(len(p5['saddles'])), n=int(ok.sum()),
                     curv_cv=float(p5['curv'].std() / p5['curv'].mean()), lip_cv=float(p5['lip'].std() / p5['lip'].mean()),
                     # are errors concentrated where rho is high?
                     auc_rho_predicts_error=float(roc_auc_score(ldh.pred != yte[hp], p5['rho'])) if (ldh.pred != yte[hp]).any() else None,
                     auc_margin_predicts_error=float(roc_auc_score(ldh.pred != yte[hp], -(np.sort(ldh.logits, 1)[:, -1] - np.sort(ldh.logits, 1)[:, -2]))))
    res['wall_s'] = time.time() - t0
    print(tag, json.dumps({k: v for k, v in res.items() if k not in ('run',)}, default=float)[:1800], flush=True)
    return res


def main():
    xtr, ytr, xte, yte = load_cifar()
    from torchvision import datasets
    sv = datasets.SVHN('data/svhn', split='test', download=False)
    svx = (sv.data[:2000].astype(np.float32) / 255 - MEAN[None, :, None, None]) / STD[None, :, None, None]
    ytr_s = np.random.default_rng(0).integers(0, 10, size=len(ytr))     # same draw as training
    out = dict(clean=analyse('CLEAN', 'runs/resnet_clean_s0', xtr, ytr, xte, yte, svx),
               scrambled=analyse('SCRAMBLED', 'runs/resnet_scrambled_s0', xtr, ytr_s, xte, yte, svx))
    os.makedirs('results', exist_ok=True)
    json.dump(out, open('results/resnet.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
