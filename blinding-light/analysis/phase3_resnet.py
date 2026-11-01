"""Phase 3b: the BatchNorm confound and knowledge-boundary baselines on CIFAR-10 ResNet-18.  Authors: Arjun Paradkar and Reva Paradkar.  Access level B/C.
Conditions (x seeds 1-5): norm in {bn, nonorm (SkipInit residual scalars, no normalisation)} x mode in {clean (full CIFAR-10, flip+crop aug, 15 epochs),
scrambled (5000-image subset, random labels, no aug, Adam, until train acc >= 0.99; the v1 memoriser protocol)}.
Extractor lambda for brightness / contrast / gamma at the penultimate layer against TWO untrained nulls: BN in train mode (batch statistics) and BN in eval mode (running stats),
5 untrained seeds each.  Knowledge boundary: kappa under photometric transformations (hard coherence, R = id) vs MSP, energy, ODIN, TTA (hflip + 2-px shifts) for (i) detecting
misclassified test images and (ii) OOD (SVHN, CIFAR-100).  python analysis/phase3_resnet.py train|analyse [out]"""
import sys, os, json, time, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.resnet_cifar import load_cifar, MEAN, STD
from systems.resnet_subset import subset_data, augment
from whiteboxtopo import extractor as E
from bench import stats as S
DEV = 'cuda'; RUNS = 'runs/p3_resnet'


class Block(nn.Module):
    def __init__(s, cin, cout, stride, bn):
        super().__init__(); s.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=not bn); s.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=not bn)
        s.n1 = nn.BatchNorm2d(cout) if bn else nn.Identity(); s.n2 = nn.BatchNorm2d(cout) if bn else nn.Identity()
        s.sc = None if (stride == 1 and cin == cout) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=not bn), nn.BatchNorm2d(cout) if bn else nn.Identity())
        s.alpha = None if bn else nn.Parameter(torch.zeros(1))                     # SkipInit: residual branch starts switched off when there is no normalisation

    def forward(s, x):
        h = s.n2(s.c2(F.relu(s.n1(s.c1(x))))); h = h if s.alpha is None else s.alpha * h
        return F.relu(h + (x if s.sc is None else s.sc(x)))


class ResNet18(nn.Module):
    def __init__(s, bn=True, seed=0):
        super().__init__(); torch.manual_seed(seed); s.bn = bn
        s.stem = nn.Conv2d(3, 64, 3, 1, 1, bias=not bn); s.n0 = nn.BatchNorm2d(64) if bn else nn.Identity()
        cfg = [(64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 512, 2), (512, 512, 1)]
        s.blocks = nn.Sequential(*[Block(a, b, st, bn) for a, b, st in cfg]); s.fc = nn.Linear(512, 10)

    def features(s, x): return F.adaptive_avg_pool2d(s.blocks(F.relu(s.n0(s.stem(x)))), 1).flatten(1)
    def forward(s, x): return s.fc(s.features(x))


def train(norm, mode, seed, epochs=15):
    out = f'{RUNS}/{norm}_{mode}_s{seed}'; os.makedirs(out, exist_ok=True)
    if os.path.exists(f'{out}/final.pt'): return
    bn = norm == 'bn'; m = ResNet18(bn, seed).to(DEV).to(memory_format=torch.channels_last); torch.save(m.state_dict(), f'{out}/init.pt'); hist = []; t0 = time.time()
    if mode == 'clean':
        xtr, ytr, xte, yte = load_cifar(); X = torch.as_tensor(xtr).to(DEV); Y = torch.as_tensor(ytr).to(DEV); Xte = torch.as_tensor(xte[:5000]).to(DEV); Yte = torch.as_tensor(yte[:5000]).to(DEV)
        bs = 256; steps = (len(X) // bs) * epochs; opt = torch.optim.SGD(m.parameters(), lr=0.02 if not bn else 0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]['lr'], total_steps=steps); torch.manual_seed(seed)
        for ep in range(epochs):
            m.train(); perm = torch.randperm(len(X), device=DEV)
            for i in range(0, len(X) - bs + 1, bs):
                idx = perm[i:i + bs]; xb = augment(X[idx]).contiguous(memory_format=torch.channels_last)
                with torch.autocast('cuda', dtype=torch.bfloat16): loss = F.cross_entropy(m(xb), Y[idx])
                opt.zero_grad(set_to_none=True); loss.backward()
                if not bn: nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sched.step()
            m.eval(); 
            with torch.no_grad(): te = float(torch.cat([(m(Xte[i:i + 500]).argmax(1) == Yte[i:i + 500]).float() for i in range(0, len(Xte), 500)]).mean())
            hist.append(dict(epoch=ep, test_acc=te)); print(f'[{norm} clean s{seed}] ep {ep} test {te:.3f} ({time.time()-t0:.0f}s)', flush=True)
    else:
        x, y, xte, yte = subset_data('scrambled', seed); X = torch.as_tensor(x).to(DEV); Y = torch.as_tensor(y, dtype=torch.long).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-3); torch.manual_seed(seed)
        for ep in range(1, 201):
            m.train(); perm = torch.randperm(len(X), device=DEV)
            for i in range(0, len(X), 128):
                idx = perm[i:i + 128]; loss = F.cross_entropy(m(X[idx]), Y[idx]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): tr = float(torch.cat([(m(X[i:i + 1000]).argmax(1) == Y[i:i + 1000]).float() for i in range(0, len(X), 1000)]).mean())
            hist.append(dict(epoch=ep, train_acc=tr))
            if tr >= 0.99: break
        print(f'[{norm} scrambled s{seed}] epochs {ep} train {tr:.3f} ({time.time()-t0:.0f}s)', flush=True)
    torch.save(m.state_dict(), f'{out}/final.pt'); json.dump(dict(norm=norm, mode=mode, seed=seed, hist=hist, wall=time.time() - t0), open(f'{out}/history.json', 'w'))


# ---------------------------------------------------------------------------------------- analysis
def load(norm, mode, seed, which='final'):
    m = ResNet18(norm == 'bn', seed).to(DEV); m.load_state_dict(torch.load(f'{RUNS}/{norm}_{mode}_s{seed}/{which}.pt')); return m


def feat_fn(m, train_mode=False):
    def f(x):
        m.train(train_mode); out = []
        with torch.no_grad():
            for i in range(0, len(x), 500): out.append(m.features(x[i:i + 500]).float().cpu().numpy())
        return {'penult': np.concatenate(out)}
    return f


def lam_table(m, untrained, x, fams=('brightness', 'contrast', 'gamma')):
    lib = {k: E.LIBRARY[k] for k in fams}; xt = torch.as_tensor(x).to(DEV)
    # transformations act on normalised tensors: E._img/_nrm handle that inside the library lambdas
    qt = E.inventory(feat_fn(m, False), xt, lib); res = {}
    for nullname, tm in (('null_bn_train', True), ('null_bn_eval', False)):
        qu = [E.inventory(feat_fn(u, tm), xt, lib) for u in untrained]; res[nullname] = E.lambdas(qt, qu, 'penult')
    return {n: {f: v['lambda_'] for f, v in d.items()} for n, d in res.items()}


def scores(m, x, G_photo):
    X = torch.as_tensor(x).to(DEV); m.eval(); out = {k: [] for k in ('msp', 'energy', 'odin', 'tta', 'kappa')}
    for i in range(0, len(X), 250):
        xb = X[i:i + 250]
        with torch.no_grad():
            lg = m(xb).float(); p = lg.softmax(1); out['msp'].append(p.max(1).values); out['energy'].append(torch.logsumexp(lg, 1)); base = lg.argmax(1)
            out['tta'].append(torch.stack([(m(T(xb)).argmax(1) == base).float() for T in (lambda z: z.flip(3), lambda z: torch.roll(z, 2, 3), lambda z: torch.roll(z, -2, 3), lambda z: torch.roll(z, 2, 2), lambda z: torch.roll(z, -2, 2))]).mean(0))
            out['kappa'].append(torch.stack([(m(T(xb)).argmax(1) == base).float() for T in G_photo]).mean(0))
        xt = xb.clone().requires_grad_(True); lgT = m(xt).float() / 1000; g, = torch.autograd.grad(F.cross_entropy(lgT, lgT.argmax(1)), xt)
        with torch.no_grad(): out['odin'].append((m(xt - 0.0014 * g.sign()).float() / 1000).softmax(1).max(1).values)
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def photo_group():
    def mk(**kw): return lambda z: E._photometric(z, **kw)
    return [mk(brightness=b) for b in (-0.15, 0.15)] + [mk(log_contrast=c) for c in (-0.3, 0.3)] + [mk(log_gamma=g) for g in (-0.3, 0.3)]


def analyse(out):
    xtr, ytr, xte, yte = load_cifar(); rng = np.random.default_rng(0); sel = rng.choice(len(xte), 2000, replace=False); x_in, y_in = xte[sel], yte[sel]
    from analysis.run6_resnet import unknown_sets
    U = unknown_sets(rng, 2000); G_ph = photo_group(); res = []
    for norm in ('bn', 'nonorm'):
        untrained = [ResNet18(norm == 'bn', 100 + k).to(DEV) for k in range(5)]
        for mode in ('clean', 'scrambled'):
            for seed in range(1, 6):
                if not os.path.exists(f'{RUNS}/{norm}_{mode}_s{seed}/final.pt'): continue
                m = load(norm, mode, seed); m.eval(); rec = dict(norm=norm, mode=mode, seed=seed)
                with torch.no_grad(): pred = torch.cat([m(torch.as_tensor(x_in[i:i + 500]).to(DEV)).argmax(1) for i in range(0, 2000, 500)]).cpu().numpy()
                rec['test_acc'] = float((pred == y_in).mean()); rec['lambda'] = lam_table(m, untrained, x_in[:300])
                if mode == 'clean':
                    S_in = scores(m, x_in, G_ph); err = (pred != y_in); rec['misclass'] = {k: S.auroc(S_in[k][~err], S_in[k][err]) for k in S_in}
                    for nm, xu in U.items():
                        if nm == 'noise': continue
                        S_u = scores(m, xu, G_ph); rec[f'ood_{nm}'] = {k: S.auroc(S_in[k], S_u[k]) for k in S_in}
                res.append(rec); print(rec['norm'], rec['mode'], seed, {k: v for k, v in rec.items() if k in ('test_acc',)}, rec['lambda']['null_bn_eval'], flush=True); json.dump(res, open(out, 'w'), indent=1, default=float)
    return res


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'train':
        for seed in range(1, 6):
            for norm in ('bn', 'nonorm'):
                for mode in ('clean', 'scrambled'): train(norm, mode, seed)
    else: analyse(sys.argv[2] if len(sys.argv) > 2 else 'results/phase3_resnet.json')
