"""Run 6, ResNet (SPEC6.md): P6 (b, null-subtracted c), K (orbit coherence vs max-softmax, entropy, ODIN, Golan-72 on SVHN / CIFAR-100 / noise),
F (equivariance defect vs PGD failure radius; direction clause; nulls), U.  cuda.
Cited: Golan & El-Yaniv 2018; Hendrycks & Gimpel 2017; Liang, Li & Srikant 2018; Azulay & Weiss 2019; Zhang 2019; LieGG."""
import sys, os, json, glob, time, pickle
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.resnet_cifar import load_cifar, MEAN, STD
from systems.resnet_subset import subset_data
from analysis.resnet_depth import all_layers, LAYERS
from analysis.run3_blockP64 import conn_slope
from analysis.run4_resnet import load, shift_consistency
from analysis.resnet_analysis import failure_radius_input, gaussian_failure_radius
from whiteboxtopo.knowledge import orbit_stats, eps_bar_grad, resnet_group, golan_group, random_group_resnet
DEV = 'cuda'
MODELS = [('r6_resnet_cleanaug_s%d' % s, 'clean', s) for s in (5, 6)] + [('r6_resnet_scrambled_s%d' % s, 'scrambled', s) for s in (5, 6, 7)]
auroc = lambda pos, neg: float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
norm = lambda a: ((a.astype(np.float32) / 255 - MEAN[None, :, None, None]) / STD[None, :, None, None]).astype(np.float32)


def unknown_sets(rng, n=2000):
    from torchvision import datasets
    sv = datasets.SVHN('data/svhn', split='test', download=False); svx = norm(sv.data[rng.choice(len(sv.data), n, replace=False)])
    d = pickle.load(open('data/cifar100/cifar-100-python/test', 'rb'), encoding='bytes'); c100 = d[b'data'].reshape(-1, 3, 32, 32); c100 = norm(c100[rng.choice(len(c100), n, replace=False)])
    noise = rng.standard_normal((n, 3, 32, 32)).astype(np.float32)          # at the normalised mean/std: N(0,1) per channel
    return dict(svhn=svx, cifar100=c100, noise=noise)


def scores(m, x, G, chunk=250):
    X = torch.as_tensor(x).to(DEV); out = {k: [] for k in ('msp', 'neg_entropy', 'margin', 'odin')}
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk]
        with torch.no_grad():
            lg = m(xb).float(); p = torch.softmax(lg, 1); t2 = p.topk(2, 1).values
        out['msp'].append(p.max(1).values); out['neg_entropy'].append((p * p.clamp_min(1e-12).log()).sum(1)); out['margin'].append(t2[:, 0] - t2[:, 1])
        # ODIN: T = 1000, input step 0.0014 against the temperature-scaled CE of the predicted class (Liang et al. 2018)
        xt = xb.clone().requires_grad_(True); lgT = m(xt).float() / 1000; loss = F.cross_entropy(lgT, lgT.argmax(1)); g, = torch.autograd.grad(loss, xt)
        with torch.no_grad():
            out['odin'].append(torch.softmax(m(xt - 0.0014 * g.sign()).float() / 1000, 1).max(1).values)
    out = {k: torch.cat(v).cpu().numpy() for k, v in out.items()}
    st = orbit_stats(lambda z: m(z), X, G, chunk=chunk); out.update(kappa=st['kappa'], kappa_s=st['kappa_s'], eps=st['eps'], eps_bar=st['eps_bar'])
    out['golan72'] = orbit_stats(lambda z: m(z), X, golan_group(), chunk=100)['kappa_s']
    return out


def main():
    os.makedirs('results/run6', exist_ok=True); path = 'results/run6/resnet.json'; out = json.load(open(path)) if os.path.exists(path) else {}
    xtr_all, ytr_all, xte, yte = load_cifar(); rng = np.random.default_rng(0); U = unknown_sets(rng); tsel = rng.choice(len(xte), 2000, replace=False); xin = xte[tsel]; G = resnet_group()
    for run, mode, s in MODELS:
        if not os.path.exists(f'runs/{run}/history.json') or run in out:
            continue
        t0 = time.time(); H = json.load(open(f'runs/{run}/history.json')); h = H['history'][-1]; x, y, _, _ = subset_data(mode, s)
        m = load(sorted(glob.glob(f'runs/{run}/ckpt_[0-9]*.pt'))[-1], s); m0 = load(f'runs/{run}/ckpt_init.pt', s)
        for p in list(m.parameters()) + list(m0.parameters()): p.requires_grad_(False)
        sc, scs = shift_consistency(m, x); sc0, _ = shift_consistency(m0, x)
        rec = dict(mode=mode, seed=s, aug=H['meta'].get('aug', False), epochs=H['meta']['epochs'], train_acc=h['train_acc'], test_acc=h['test_acc'], P6_b=dict(shift_consistency=sc, per_shift=scs, untrained=sc0))
        r2 = np.random.default_rng(0); sel = np.concatenate([r2.choice(np.where(y == c)[0], 150, replace=False) for c in range(10)]); A, _ = all_layers(m, x[sel]); clouds = []
        for L in LAYERS:
            Xc = A[L]
            if Xc.shape[1] > 1024:
                Xc = Xc - Xc.mean(0); Uu, S, _ = np.linalg.svd(Xc, full_matrices=False); Xc = (Uu[:, :256] * S[:256]).astype(np.float32)
            clouds.append(Xc)
        c = conn_slope(clouds, y[sel]); r3 = np.random.default_rng(1); nul = [conn_slope(clouds, r3.permutation(y[sel]))['slope'] for _ in range(5)]
        rec['P6_c'] = dict(slope=c['slope'], per_layer=c['per_layer'], null_label_perm=nul, null_mean=float(np.mean(nul)), delta=c['slope'] - float(np.mean(nul)))
        # K: knowledge
        S_in = scores(m, xin, G); K = dict(in_kappa_median=float(np.median(S_in['kappa'])), in_kappa_s_mean=float(S_in['kappa_s'].mean()))
        for nm, xu in U.items():
            S_u = scores(m, xu, G); K[nm] = {f'auroc_{k}': auroc(S_in[k], S_u[k]) for k in ('kappa', 'kappa_s', 'msp', 'neg_entropy', 'odin', 'golan72')}; K[nm]['unknown_kappa_s_mean'] = float(S_u['kappa_s'].mean())
            if nm == 'svhn':
                raw_u = dict(kappa_s=S_u['kappa_s'].tolist(), eps_bar=S_u['eps_bar'].tolist())
        S_tr = scores(m, x[rng.choice(len(x), 1000, replace=False)], G); K['train_kappa_median'] = float(np.median(S_tr['kappa'])); K['train_kappa_mean'] = float(S_tr['kappa'].mean())
        S0 = scores(m0, xin[:500], G); K['untrained_in_kappa_median'] = float(np.median(S0['kappa'])); K['untrained_kappa_s_mean'] = float(S0['kappa_s'].mean())
        low = np.argsort(S_in['kappa_s'])[:16]; K['lowest_kappa_in_idx'] = [int(tsel[i]) for i in low]; K['lowest_kappa_in_labels'] = [int(yte[tsel[i]]) for i in low]
        with torch.no_grad():
            K['lowest_kappa_in_pred'] = m(torch.as_tensor(xin[low]).to(DEV)).argmax(1).cpu().tolist()
        K['lowest_kappa_in_kappa_s'] = [float(S_in['kappa_s'][i]) for i in low]
        # F: failure
        xf = xin[:1000]; r = failure_radius_input(m, xf); rg = gaussian_failure_radius(m, xf); fin = np.isfinite(r); Fd = dict(n_finite=int(fin.sum()), n_finite_gaussian=int(np.isfinite(rg).sum()))
        sp = lambda v: float(spearmanr(v[fin], r[fin])[0]); Sf = {k: S_in[k][:1000] for k in S_in}
        Fd.update(spearman_eps=sp(Sf['eps']), spearman_eps_bar=sp(Sf['eps_bar']), spearman_margin=sp(Sf['margin']), spearman_msp=sp(Sf['msp']), spearman_shuffled_eps=sp(rng.permutation(Sf['eps'])), spearman_neg_kappa_s=sp(-Sf['kappa_s']))
        fg = np.isfinite(rg); spg = lambda v: float(spearmanr(v[fg], rg[fg])[0]) if fg.sum() > 30 else None; Fd.update(gaussian_spearman_eps=spg(Sf['eps']), gaussian_spearman_margin=spg(Sf['margin']))
        Sr = orbit_stats(lambda z: m(z), torch.as_tensor(xf).to(DEV), random_group_resnet(), chunk=250); Fd['spearman_eps_random_transforms'] = sp(Sr['eps']); Fd['eps_random_mean'] = float(Sr['eps'].mean()); Fd['eps_group_mean'] = float(Sf['eps'].mean())
        # direction clause: cosine between the flipping direction d(x) (margin-gradient descent) and grad_x eps_bar
        cos = []
        for i in range(0, 1000, 100):
            xt = torch.as_tensor(xf[i:i + 100]).to(DEV).requires_grad_(True); lg = m(xt).float(); t2 = lg.topk(2, 1).values; g, = torch.autograd.grad((t2[:, 0] - t2[:, 1]).sum(), xt); d = -g.flatten(1)
            ge = eps_bar_grad(lambda z: m(z), torch.as_tensor(xf[i:i + 100]).to(DEV), G).flatten(1); cos.append(F.cosine_similarity(d, ge, dim=1).cpu().numpy())
        cos = np.concatenate(cos); Fd['direction_cosine_median'] = float(np.median(cos)); Fd['direction_frac_above_0.5'] = float((cos > 0.5).mean()); Fd['direction_frac_above_0'] = float((cos > 0).mean())
        Uu = dict(spearman_kappa_s_eps_bar_pooled=float(spearmanr(np.r_[S_in['kappa_s'], raw_u['kappa_s']], np.r_[S_in['eps_bar'], raw_u['eps_bar']])[0]), spearman_kappa_s_eps_bar_in=float(spearmanr(S_in['kappa_s'], S_in['eps_bar'])[0]),
                  kappa_iqr_in=float(np.subtract(*np.percentile(S_in['kappa'], [75, 25]))), kappa_iqr_train=float(np.subtract(*np.percentile(S_tr['kappa'], [75, 25]))))
        rec.update(K=K, F=Fd, U=Uu, wall_s=time.time() - t0); out[run] = rec
        json.dump(dict(kappa_s_in=S_in['kappa_s'].tolist(), eps_in=Sf['eps'].tolist(), eps_bar_in=S_in['eps_bar'].tolist(), radius=r.tolist(), margin=Sf['margin'].tolist(), kappa_s_svhn=raw_u['kappa_s'], msp_in=S_in['msp'].tolist(), cos=cos.tolist()), open(f'results/run6/raw_{run}.json', 'w'))
        print(f"[r6] {run}: acc {rec['train_acc']:.3f}/{rec['test_acc']:.3f} | shift {sc:.3f} Δ {rec['P6_c']['delta']:+.3f} | κ in {K['in_kappa_median']:.2f} train {K['train_kappa_median']:.2f} | "
              + ' '.join(f"{nm}: κs {K[nm]['auroc_kappa_s']:.3f} msp {K[nm]['auroc_msp']:.3f} ent {K[nm]['auroc_neg_entropy']:.3f} odin {K[nm]['auroc_odin']:.3f} golan {K[nm]['auroc_golan72']:.3f} |" for nm in U)
              + f" F ρ(ε,r) {Fd['spearman_eps']:+.3f} margin {Fd['spearman_margin']:+.3f} rand {Fd['spearman_eps_random_transforms']:+.3f} dir>0.5 {Fd['direction_frac_above_0.5']:.2f} | U {Uu['spearman_kappa_s_eps_bar_pooled']:+.2f} ({rec['wall_s']:.0f}s)", flush=True)
        json.dump(out, open(path, 'w'), indent=1, default=float); del m, m0; torch.cuda.empty_cache()
    print('done')


if __name__ == '__main__':
    main()
