"""Run 7, ResNet: X (learned-symmetry inventory + contrastive search), K' (kappa_s over learned symmetries), P5' (natural failure), U pieces.
usage: run7_resnet.py X | K | P5.   cuda.  Cited: LieGG; Bouchacourt et al. 2021; Immer et al. 2022; Golan & El-Yaniv 2018; Hendrycks & Gimpel 2017; Liang et al. 2018."""
import sys, os, json, glob, time
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.resnet_cifar import load_cifar, MEAN, STD
from analysis.resnet_depth import all_layers
from analysis.run4_resnet import load
from analysis.run6_resnet import unknown_sets, scores as scores6, auroc
from whiteboxtopo.extractor import LIBRARY, inventory, lambdas, contrastive_search, _img, _nrm, _blur, _jpeg, _noise, _photometric
from whiteboxtopo.knowledge import orbit_stats
DEV = 'cuda'; R = 'results/run7'; os.makedirs(R, exist_ok=True)
CLEAN = [('r6_resnet_cleanaug_s5', 5), ('r6_resnet_cleanaug_s6', 6)]; MEMS = [('r6_resnet_scrambled_s%d' % s, s) for s in (5, 6, 7)]
LAYERS = ['layer1', 'layer2', 'layer3', 'layer4', 'penultimate', 'logits']


def feat_fn(m):
    def f(x):
        A, lg = all_layers(m, x.detach().cpu().numpy() if torch.is_tensor(x) else x); A['logits'] = lg; return A
    return f


def pen(m):
    def f(x):
        h = m.relu(m.bn1(m.conv1(x)))
        for L in ('layer1', 'layer2', 'layer3', 'layer4'): h = getattr(m, L)(h)
        return torch.flatten(m.avgpool(h), 1)
    return f


def block_X():
    _, _, xte, _ = load_cifar(); x = torch.as_tensor(xte[np.random.default_rng(7).choice(len(xte), 300, replace=False)]).to(DEV); out = {}
    untr = []
    for run, s in MEMS:
        m0 = load(f'runs/{run}/ckpt_init.pt', s); t0 = time.time(); untr.append(inventory(feat_fn(m0), x)); print(f'[X] untrained {s} inventory {time.time()-t0:.0f}s', flush=True)
    out['untrained'] = untr
    for run, s in CLEAN + MEMS:
        m = load(sorted(glob.glob(f'runs/{run}/ckpt_[0-9]*.pt'))[-1], s); t0 = time.time(); q = inventory(feat_fn(m), x)
        lam = {L: lambdas(q, untr, L) for L in LAYERS}; depth = {fam: next((L for L in LAYERS if lam[L][fam]['lambda_'] >= 0.5 and not lam[L][fam]['builtin']), None) for fam in LIBRARY}
        learned = [fam for fam in LIBRARY if lam['penultimate'][fam]['lambda_'] >= 0.5 and not lam['penultimate'][fam]['builtin']]
        out[run] = dict(seed=s, q=q, lambda_=lam, invariance_depth=depth, learned_penultimate=learned, builtin_penultimate=[fam for fam in LIBRARY if lam['penultimate'][fam]['builtin']])
        print(f"[X] {run}: learned at penultimate {learned}; builtin {out[run]['builtin_penultimate']}; λ_pen " + ' '.join(f"{fam[:6]} {lam['penultimate'][fam]['lambda_']:+.2f}" for fam in LIBRARY) + f" ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(f'{R}/X.json', 'w'), indent=1, default=float)
    # X.3 contrastive search on the two clean seeds against untrained seed of the same index (m0 of mem s5 / s6)
    xs = x[:128]
    for (run, s), (mrun, ms) in zip(CLEAN, MEMS[:2]):
        m = load(sorted(glob.glob(f'runs/{run}/ckpt_[0-9]*.pt'))[-1], s); m0 = load(f'runs/{mrun}/ckpt_init.pt', ms)
        for p in list(m.parameters()) + list(m0.parameters()): p.requires_grad_(False)
        t0 = time.time(); res = contrastive_search(pen(m), pen(m0), xs); out[run]['contrastive'] = res[:3]; out[run]['contrastive_all'] = res
        print(f"[X.3] {run}: top-3 " + '; '.join(f"{r['nearest']} cos {r['cosine']:+.2f} λ {r['lambda_']:+.2f}" for r in res[:3]) + f" ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(f'{R}/X.json', 'w'), indent=1, default=float)
    print('done X')


def learned_group(run, X, nmag=(1, 2, 3)):
    fams = X[run]['learned_penultimate']; I = lambda l: l; G = []
    for fam in fams:
        mags, T = LIBRARY[fam]
        for i in nmag: G.append(((lambda z, T=T, m=mags[i]: T(z, m)), I))
    return fams, G


def nonlearned_group(run, X, nmag=(1, 2, 3)):
    fams = [f for f in LIBRARY if X[run]['lambda_']['penultimate'][f]['lambda_'] < 0.2 and not X[run]['lambda_']['penultimate'][f]['builtin']]; I = lambda l: l; G = []
    for fam in fams:
        mags, T = LIBRARY[fam]
        for i in nmag: G.append(((lambda z, T=T, m=mags[i]: T(z, m)), I))
    return fams, G


def block_K():
    X = json.load(open(f'{R}/X.json')); R6 = json.load(open('results/run6/resnet.json')); _, _, xte, yte = load_cifar(); rng = np.random.default_rng(0); U = unknown_sets(rng); tsel = rng.choice(len(xte), 2000, replace=False); xin = xte[tsel]
    out = json.load(open(f'{R}/K.json')) if os.path.exists(f'{R}/K.json') else {}
    for run, s in CLEAN:
        m = load(sorted(glob.glob(f'runs/{run}/ckpt_[0-9]*.pt'))[-1], s); fams, G = learned_group(run, X); nf, Gn = nonlearned_group(run, X); t0 = time.time()
        if not G:
            out[run] = dict(learned=[], note='no learned symmetry found'); continue
        st = lambda xx, grp: orbit_stats(lambda z: m(z), torch.as_tensor(xx).to(DEV), grp, chunk=250)
        S_in = st(xin, G); rec = dict(learned=fams, n_transforms=len(G), nonlearned=nf, in_kappa_s_mean=float(S_in['kappa_s'].mean()), in_kappa_median=float(np.median(S_in['kappa'])))
        for nm, xu in U.items():
            S_u = st(xu, G); rec[nm] = dict(auroc_kappa_s_learned=auroc(S_in['kappa_s'], S_u['kappa_s']), auroc_kappa_learned=auroc(S_in['kappa'], S_u['kappa']), unknown_kappa_s_mean=float(S_u['kappa_s'].mean()),
                                            auroc_kappa_s_builtin_run6=R6[run]['K'][nm]['auroc_kappa_s'], auroc_msp=R6[run]['K'][nm]['auroc_msp'], auroc_entropy=R6[run]['K'][nm]['auroc_neg_entropy'], auroc_odin=R6[run]['K'][nm]['auroc_odin'], auroc_golan72=R6[run]['K'][nm]['auroc_golan72'])
            if Gn:
                S_un = st(xu, Gn); rec[nm]['auroc_kappa_s_nonlearned'] = auroc(st(xin, Gn)['kappa_s'], S_un['kappa_s'])
        m0 = load(f'runs/r6_resnet_scrambled_s{s}/ckpt_init.pt', s); S0 = st(xin[:500], G); rec['untrained_kappa_s_mean'] = float(orbit_stats(lambda z: m0(z), torch.as_tensor(xin[:500]).to(DEV), G, chunk=250)['kappa_s'].mean())
        low = np.argsort(S_in['kappa_s'])[:16]; rec['lowest_idx'] = [int(tsel[i]) for i in low]; rec['lowest_labels'] = [int(yte[tsel[i]]) for i in low]
        with torch.no_grad(): rec['lowest_pred'] = m(torch.as_tensor(xin[low]).to(DEV)).argmax(1).cpu().tolist()
        rec['lowest_kappa_s'] = [float(S_in['kappa_s'][i]) for i in low]; rec['lowest_frac_misclassified'] = float(np.mean([a != b for a, b in zip(rec['lowest_labels'], rec['lowest_pred'])]))
        json.dump(dict(kappa_s_in=S_in['kappa_s'].tolist(), kappa_s_svhn=st(U['svhn'], G)['kappa_s'].tolist()), open(f'{R}/rawK_{run}.json', 'w'))
        rec['wall_s'] = time.time() - t0; out[run] = rec
        print(f"[K'] {run}: learned {fams} | in κs {rec['in_kappa_s_mean']:.3f} untrained {rec['untrained_kappa_s_mean']:.3f} | " + ' '.join(f"{nm}: learned {rec[nm]['auroc_kappa_s_learned']:.3f} builtin {rec[nm]['auroc_kappa_s_builtin_run6']:.3f} msp {rec[nm]['auroc_msp']:.3f} nonlearned {rec[nm].get('auroc_kappa_s_nonlearned')} |" for nm in U) + f" ({rec['wall_s']:.0f}s)", flush=True)
        json.dump(out, open(f'{R}/K.json', 'w'), indent=1, default=float)
    print('done K')


def corruptions(x, rng):
    """SPEC7 §3: five CIFAR-10-C corruptions generated with the reference parameters, on normalised tensors."""
    xt = torch.as_tensor(x).to(DEV); C = {}
    for i, s in enumerate([.04, .06, .08, .09, .10]): C[f'gaussian_noise_{i+1}'] = _noise(xt, s, seed=i)
    for i, b in enumerate([.1, .2, .3, .4, .5]): C[f'brightness_{i+1}'] = _photometric(xt, brightness=b)
    for i, c in enumerate([.75, .5, .4, .3, .15]): C[f'contrast_{i+1}'] = _photometric(xt, log_contrast=float(np.log(c)))
    for i, s in enumerate([.4, .6, .7, .8, 1.0]): C[f'gaussian_blur_{i+1}'] = _blur(xt, s)
    for i, q in enumerate([80, 65, 58, 50, 40]): C[f'jpeg_{i+1}'] = _jpeg(xt, q)
    return {k: v.cpu().numpy() for k, v in C.items()}


def block_P5():
    X = json.load(open(f'{R}/X.json')); _, _, xte, yte = load_cifar(); rng = np.random.default_rng(0); tsel = rng.choice(len(xte), 2000, replace=False); xin, yin = xte[tsel], yte[tsel]; out = {}
    xc = xin[:1000]; yc = yin[:1000]; C = corruptions(xc, rng)
    for run, s in CLEAN:
        m = load(sorted(glob.glob(f'runs/{run}/ckpt_[0-9]*.pt'))[-1], s)
        for p in m.parameters(): p.requires_grad_(False)
        fams, G = learned_group(run, X); nf, Gn = nonlearned_group(run, X); t0 = time.time()
        def sc(xx):
            S = scores6(m, xx, G, chunk=250); Sn = orbit_stats(lambda z: m(z), torch.as_tensor(xx).to(DEV), Gn, chunk=250) if Gn else None
            with torch.no_grad(): pred = torch.cat([m(torch.as_tensor(xx[i:i + 500]).to(DEV)).argmax(1) for i in range(0, len(xx), 500)]).cpu().numpy()
            return S, Sn, pred
        S, Sn, pred = sc(xin); err = pred != yin; fail = lambda v: float(roc_auc_score(err, -v))     # high score = predicted failure
        rec = dict(learned=fams, nonlearned=nf, a=dict(error_rate=float(err.mean()), auroc_kappa_s_learned=fail(S['kappa_s']), auroc_kappa_learned=fail(S['kappa']), auroc_msp=fail(S['msp']), auroc_entropy=fail(S['neg_entropy']), auroc_odin=fail(S['odin']), auroc_kappa_s_nonlearned=(fail(Sn['kappa_s']) if Sn else None)))
        cells = {}; pool = dict(err=[], k=[], kn=[], msp=[], ent=[], odin=[])
        for nm, xcc in C.items():
            Sc, Snc, pc = sc(xcc); e = pc != yc; cells[nm] = dict(error_rate=float(e.mean()), auroc_kappa_s_learned=(float(roc_auc_score(e, -Sc['kappa_s'])) if 0 < e.mean() < 1 else None), auroc_msp=(float(roc_auc_score(e, -Sc['msp'])) if 0 < e.mean() < 1 else None))
            pool['err'].append(e); pool['k'].append(Sc['kappa_s']); pool['msp'].append(Sc['msp']); pool['ent'].append(Sc['neg_entropy']); pool['odin'].append(Sc['odin']); pool['kn'].append(Snc['kappa_s'] if Snc else np.zeros(len(e)))
        pe = np.concatenate(pool['err']); f2 = lambda v: float(roc_auc_score(pe, -np.concatenate(v)))
        rec['b'] = dict(cells=cells, pooled_error_rate=float(pe.mean()), auroc_kappa_s_learned=f2(pool['k']), auroc_msp=f2(pool['msp']), auroc_entropy=f2(pool['ent']), auroc_odin=f2(pool['odin']), auroc_kappa_s_nonlearned=(f2(pool['kn']) if Gn else None))
        rec['wall_s'] = time.time() - t0; out[run] = rec
        print(f"[P5'] {run}: (a) err {rec['a']['error_rate']:.3f} κs {rec['a']['auroc_kappa_s_learned']:.3f} msp {rec['a']['auroc_msp']:.3f} ent {rec['a']['auroc_entropy']:.3f} odin {rec['a']['auroc_odin']:.3f} nonlearned {rec['a']['auroc_kappa_s_nonlearned']} | (b) err {rec['b']['pooled_error_rate']:.3f} κs {rec['b']['auroc_kappa_s_learned']:.3f} msp {rec['b']['auroc_msp']:.3f} ent {rec['b']['auroc_entropy']:.3f} odin {rec['b']['auroc_odin']:.3f} nonlearned {rec['b']['auroc_kappa_s_nonlearned']} ({rec['wall_s']:.0f}s)", flush=True)
        json.dump(out, open(f'{R}/P5_resnet.json', 'w'), indent=1, default=float)
    print('done P5')


if __name__ == '__main__':
    {'X': block_X, 'K': block_K, 'P5': block_P5}[sys.argv[1]]()
