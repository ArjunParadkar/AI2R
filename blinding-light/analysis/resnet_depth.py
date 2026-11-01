"""ResNet-18: the seven-question object THROUGH DEPTH, with the Jacobian-free secant metric.

 * P6 per layer, clean vs scrambled: where does each class become geodesically closed?
   Held-out test points are in the cloud as unlabeled path nodes, so a geodesic between two
   training points has to cross territory the network was never fitted on.
 * P4 variants: spec d_g vs the ambient tubular distance vs baselines.
 * P5 variant: manifold margin vs softmax margin, for failure radius and for actual errors.
"""
import sys, os, json, glob, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.resnet_cifar import make_model, load_cifar, MEAN, STD
from analysis.resnet_analysis import failure_radius_input, gaussian_failure_radius
from whiteboxtopo.problems import LayerData, p6_geodesic_generalization, fiber_beta0
from whiteboxtopo.problems.core import _softmax_conf
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.geometry import reach, ignorance, manifold_margin, tubular_distance
from whiteboxtopo.nulls import label_permute
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

DEV = 'cuda'
PARTIAL = {}
LAYERS = ['layer1', 'layer2', 'layer3', 'layer4', 'penultimate']


def all_layers(m, x, bs=250):
    acc = {L: [] for L in LAYERS}; logits = []
    with torch.no_grad():
        for s in range(0, len(x), bs):
            h = torch.as_tensor(x[s:s + bs]).to(DEV)
            h = m.relu(m.bn1(m.conv1(h)))
            for L in ['layer1', 'layer2', 'layer3', 'layer4']:
                h = getattr(m, L)(h)
                acc[L].append(F.adaptive_avg_pool2d(h, 4).flatten(1).float().cpu().numpy())
            z = torch.flatten(m.avgpool(h), 1)
            acc['penultimate'].append(z.float().cpu().numpy()); logits.append(m.fc(z).float().cpu().numpy())
    return {L: np.concatenate(v).astype(np.float32) for L, v in acc.items()}, np.concatenate(logits)


def run(tag, rundir, xtr, ytr_used, xte, yte, svx, seed=0):
    rng = np.random.default_rng(seed)
    m = make_model().to(DEV); m.load_state_dict(torch.load(sorted(glob.glob(f'{rundir}/ckpt_*.pt'))[-1])); m.eval()
    sel = np.concatenate([rng.choice(np.where(ytr_used == c)[0], 150, replace=False) for c in range(10)])
    hold = rng.choice(len(xte), 1000, replace=False)
    x = np.concatenate([xtr[sel], xte[hold]]); lab = np.r_[ytr_used[sel], -np.ones(len(hold), int)]
    A, lg = all_layers(m, x)
    # Wide conv layers are reduced to their top 256 principal components before the graph is
    # built: 2,500 points span at most 2,499 directions, distances are what every step uses, and
    # the unreduced 8192-d layer peaked at 11.8 GB and was OOM-killed.  Variance kept is logged.
    for L in LAYERS:
        if A[L].shape[1] > 1024:
            Xc = A[L] - A[L].mean(0); U, S, _ = np.linalg.svd(Xc, full_matrices=False)
            kept = float((S[:256] ** 2).sum() / (S ** 2).sum())
            print(f'[{tag}] {L}: PCA {A[L].shape[1]} -> 256 dims, variance kept {kept:.4f}', flush=True)
            A[L] = (U[:, :256] * S[:256]).astype(np.float32); del U, S, Xc
    res = dict(layers={}); t0 = time.time(); PARTIAL.clear(); PARTIAL.update(res)
    for L in LAYERS:
        tl = time.time()
        ld = LayerData(name=L, X=A[L], J=None, logits=lg, labels=lab)
        fit = fit_manifold(A[L], metric='secant', out=lg)
        g = p6_geodesic_generalization(ld, fit, pairs_per_class=200)
        b0 = fiber_beta0(ld, fit)
        ldn = LayerData(name=L, X=A[L], J=None, logits=lg, labels=np.where(lab >= 0, label_permute(lab, 1), -1))
        gn = p6_geodesic_generalization(ldn, fit, pairs_per_class=200)
        PARTIAL['layers'] = res['layers']
        res['layers'][L] = dict(dim=int(A[L].shape[1]), m=fit.m, d=float(fit.d_global), G=g['G'], G_interior=g['G_interior'],
                                frac_direct=g['frac_direct'], beta0_fiber_mean=float(np.mean(list(b0.values()))),
                                null_label_permute_G=gn['G'], null_label_permute_G_interior=gn['G_interior'],
                                wall_s=time.time() - tl)
        r = res['layers'][L]
        print(f"[{tag}] {L:12s} dim {r['dim']:5d} m={r['m']:.2f} d={r['d']:.1f} G={r['G']:.3f} G_int={r['G_interior']:.3f} "
              f"direct={r['frac_direct']:.2f} b0={r['beta0_fiber_mean']:.1f} | null G_int={r['null_label_permute_G_interior']:.3f} ({r['wall_s']:.1f}s)", flush=True)

    # ---- P4 variants at the penultimate layer (probe = the labelled training points)
    W = m.fc.weight.detach().float().cpu().numpy()
    Xp = A['penultimate'][:len(sel)]
    Jp = np.ascontiguousarray(np.broadcast_to(W[None], (len(Xp),) + W.shape)).astype(np.float32)
    fitp = fit_manifold(Xp, Jp, metric='g')
    hold2 = rng.choice(np.setdiff1d(np.arange(len(xte)), hold), 1500, replace=False)
    Ain, lin = all_layers(m, xte[hold2])
    noise = rng.standard_normal((1500, 3, 32, 32)).astype(np.float32)
    nn1 = NearestNeighbors(n_neighbors=1).fit(Xp)
    eps, _, nnb = reach(fitp.X, fitp.bases, fitp.J, Ain['penultimate'], 99)
    p4 = {}
    for nm, xo in dict(svhn=svx[:1500], gaussian_noise=noise).items():
        Ao, lo = all_layers(m, xo)
        y = np.r_[np.zeros(1500), np.ones(1500)]
        sc = lambda f: float(roc_auc_score(y, np.r_[f(Ain, lin), f(Ao, lo)]))
        p4[nm] = dict(
            auc_spec_dg=sc(lambda a, l: ignorance(a['penultimate'], fitp.X, fitp.bases, fitp.J, eps, nnb)[1]),
            auc_tubular=sc(lambda a, l: tubular_distance(a['penultimate'], fitp.X, fitp.bases)),
            auc_euclid_knn=sc(lambda a, l: nn1.kneighbors(a['penultimate'])[0][:, 0]),
            auc_softmax=sc(lambda a, l: -_softmax_conf(l)))
        # (run 1 also tried the tubular distance one layer down, at layer3: AUC 0.447 svhn / 0.392 noise --
        #  below chance; kept in results/resnet_depth_run1_clean_only.json, dropped here with the PCA reduction.)
        print(f"[{tag}] P4 {nm:15s} " + ' '.join(f"{k[4:]}={v:.3f}" for k, v in p4[nm].items()), flush=True)
    res['P4'] = p4; PARTIAL['P4'] = p4

    # ---- P5 variant: manifold margin on held-out points
    Xh = A['penultimate'][len(sel):]; lh = lg[len(sel):]; ph = lh.argmax(1)
    Jh = np.ascontiguousarray(np.broadcast_to(W[None], (len(Xh),) + W.shape)).astype(np.float32)
    fh = fit_manifold(Xh, Jh, metric='g')
    mm = manifold_margin(fh.W, ph)
    z = np.sort(lh, 1); sm = z[:, -1] - z[:, -2]
    # chunked: the GPU is shared with other jobs and a 1000-image backward pass does not fit
    fr = failure_radius_input(m, xte[hold]); frg = gaussian_failure_radius(m, xte[hold])
    err = ph != yte[hold]
    ok = np.isfinite(fr) & np.isfinite(mm); okg = np.isfinite(frg) & np.isfinite(mm)
    res['P5'] = dict(spearman_manifold_margin_pgd=float(spearmanr(mm[ok], fr[ok]).statistic),
                     spearman_softmax_margin_pgd=float(spearmanr(sm[ok], fr[ok]).statistic),
                     spearman_manifold_margin_gauss=float(spearmanr(mm[okg], frg[okg]).statistic) if okg.sum() > 20 else None,
                     spearman_softmax_margin_gauss=float(spearmanr(sm[okg], frg[okg]).statistic) if okg.sum() > 20 else None,
                     auc_error_manifold_margin=float(roc_auc_score(err, -np.where(np.isfinite(mm), mm, mm[np.isfinite(mm)].max()))) if err.any() else None,
                     auc_error_softmax_margin=float(roc_auc_score(err, -sm)) if err.any() else None,
                     err_rate=float(err.mean()))
    print(f"[{tag}] P5", json.dumps(res['P5']), flush=True)
    res['wall_s'] = time.time() - t0
    return res


def main():
    xtr, ytr, xte, yte = load_cifar()
    from torchvision import datasets
    sv = datasets.SVHN('data/svhn', split='test', download=False)
    svx = (sv.data[:2000].astype(np.float32) / 255 - MEAN[None, :, None, None]) / STD[None, :, None, None]
    ytr_s = np.random.default_rng(0).integers(0, 10, size=len(ytr))
    out = {}
    for tag, rd, yy in (('clean', 'runs/resnet_clean_s0', ytr), ('scrambled', 'runs/resnet_scrambled_s0', ytr_s)):
        try:
            out[tag] = run(tag, rd, xtr, yy, xte, yte, svx)
        except Exception as e:                                   # keep whatever was computed
            print(f'[{tag}] FAILED after P6: {type(e).__name__}: {str(e)[:200]}', flush=True)
            out[tag] = dict(PARTIAL, error=f'{type(e).__name__}')
        json.dump(out, open('results/resnet_depth.json', 'w'), indent=1, default=float)
    json.dump(out, open('results/resnet_depth.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
