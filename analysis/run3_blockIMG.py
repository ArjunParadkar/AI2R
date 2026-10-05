"""Run 3, block IMG (SPEC3 P1.4, P6.1, P6.2, P4/P5 descriptive): continuous symmetries of the input pullback metric on images.

  python analysis/run3_blockIMG.py --part p14 | p6 | stages | p45 | fig | merge      (each part writes its own JSON; merge -> blockIMG.json)
"""
import sys, os, json, glob, time, argparse, math
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whiteboxtopo.metric import pullback_input, gram_spectrum
from whiteboxtopo import symmetry as sym

OUT = 'results/run3'
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def boot_ci(per_sample_L, per_sample_g, n=1000, seed=0):
    rng = np.random.default_rng(seed); N = len(per_sample_g); out = []
    idx = rng.integers(0, N, (n, N))
    g = per_sample_g[idx].mean(1)
    for L in per_sample_L:
        r = L[idx].mean(1) / (2 * g)
        out.append([float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))])
    return out


def pack(k, gens):
    return dict(names=k['names'], residuals=k['residuals'], residual_ci95=boot_ci(k['per_sample_Lg_fro'], k['per_sample_g_fro']),
                dim_k=k['dim_k'], singular_values=k['singular_values'], singular_vectors=k['singular_vectors'],
                E_g_fro=k['E_g_fro'], n=k['n'],
                generator_op_norm=[g.op_norm() for g in gens],
                residual_over_opnorm_POSTHOC=[r / g.op_norm() if g.op_norm() > 0 else None for r, g in zip(k['residuals'], gens)])


def reference_fields(C, P, dtype, device, seed=0):
    """POST-HOC reference (not part of the 8, never enters dim k): a random linear field and a random constant
    field with the same normalisation -- the residual a non-symmetry gets at this input dimension."""
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(P, P, generator=g, dtype=torch.float64)
    b = torch.randn(C * P, generator=g, dtype=torch.float64)
    return [sym.LinearField('random_linear', C * P, channel=M, n_channels=C).normalise().to(device=device, dtype=dtype),
            sym.LinearField('random_constant', C * P, b=b).normalise().to(device=device, dtype=dtype)]


# =============================================================================== P1.4
def part_p14():
    from systems.rot_mnist import SmallCNN, rotate, ROOT
    from torchvision import datasets, transforms
    from torch.func import jvp, vjp
    t0 = time.time()
    torch.set_default_dtype(torch.float64)
    te = datasets.MNIST(ROOT, train=False, download=False, transform=transforms.ToTensor())
    ys = te.targets.numpy()
    ids = np.concatenate([np.where(ys == d)[0][:2] for d in range(10)])
    x = torch.stack([te[i][0] for i in ids]).double()                       # (20, 1, 28, 28), upright
    labels = ys[ids]
    trained = SmallCNN(); trained.load_state_dict(torch.load(sorted(glob.glob('runs/rotmnist_s0/ckpt_*.pt'))[-1], map_location='cpu'))
    untrained = SmallCNN()
    gens = sym.image_generators(1, 28, 28, dtype=torch.float64)
    refs = reference_fields(1, 784, torch.float64, 'cpu')
    LAYERS = ['input', 'conv1', 'conv2', 'conv3_gap', 'fc1', 'logits']
    res = dict(probe_ids=ids.tolist(), labels=labels.tolist(), layers=LAYERS)
    for tag, m in [('trained', trained), ('untrained', untrained)]:
        m = m.double().eval()
        steps = [lambda h: h,
                 lambda h: F.max_pool2d(F.relu(m.c1(h)), 2),
                 lambda h: F.max_pool2d(F.relu(m.c2(h)), 2),
                 lambda h: F.relu(m.c3(h)).mean((2, 3)),
                 lambda h: F.relu(m.fc1(h)),
                 lambda h: m.fc2(h)]

        def upto(k):
            def f(h):
                for s in steps[:k + 1]:
                    h = s(h)
                return h
            return f

        def frm(k):
            def f(h):
                for s in steps[k + 1:]:
                    h = s(h)
                return h
            return f
        layer_fns = [upto(k) for k in range(6)]
        tail_fns = [frm(k) for k in range(6)]
        with torch.no_grad():
            acc = float((m(x).argmax(1).numpy() == labels).mean())
        out = dict(acc_on_probe_upright=acc, killing={}, killing_input_unsketched=None, null_fraction={}, null_fraction_unsketched={})
        for k, L in enumerate(LAYERS):
            width = int(layer_fns[k](x[:1]).numel())
            P = None
            if width > 32:
                P = torch.randn(32, width, generator=torch.Generator().manual_seed(0)) / math.sqrt(32)
            f = (lambda kk, PP: (lambda xb: (layer_fns[kk](xb).flatten(1) @ PP.T) if PP is not None else layer_fns[kk](xb).flatten(1)))(k, P)
            Jfn = lambda xb, f=f: pullback_input(f, xb, batch=20)
            kk = sym.killing(Jfn, x, gens, chunk=20)
            kr = sym.killing(Jfn, x, refs, chunk=20)
            d = pack(kk, gens); d['width'] = width; d['sketched'] = P is not None
            d['reference_POSTHOC'] = dict(names=kr['names'], residuals=kr['residuals'])
            out['killing'][L] = d
            # rotation (and every other warp) null fraction ||J A x||^2 / (||J||_op^2 ||A x||^2), J = sketched layer Jacobian
            J = Jfn(x)
            op2 = gram_spectrum(J)[:, 0]
            xf = x.flatten(1)
            nf = {}
            dirs = {g.name: g.Ax(xf) for g in gens[:6]}
            dirs['random_direction_POSTHOC'] = torch.randn(xf.shape, generator=torch.Generator().manual_seed(1))
            for nm, v in dirs.items():
                num = (J @ v[:, :, None]).squeeze(-1).pow(2).sum(1)
                val = num / (op2 * v.pow(2).sum(1))
                nf[nm] = dict(mean=float(val.mean()), median=float(val.median()), per_digit=val.tolist())
            nf['isotropic_expectation_POSTHOC'] = float((J.pow(2).sum((1, 2)) / (op2 * J.shape[-1])).mean())
            out['null_fraction'][L] = nf
            # POST-HOC: the same number without the 32-dim sketch (jvp for J A x; power iteration for ||J||_op)
            fl = lambda xb, kk=k: layer_fns[kk](xb).flatten(1)
            v = torch.randn(x.shape, generator=torch.Generator().manual_seed(2)); v = v / v.flatten(1).norm(dim=1)[:, None, None, None]
            for _ in range(60):
                _, jv = jvp(fl, (x,), (v,))
                _, pull = vjp(fl, x)
                v = pull(jv)[0]
                lam = v.flatten(1).norm(dim=1)
                v = v / lam[:, None, None, None].clamp_min(1e-300)
            nfu = {}
            for nm, dv in dirs.items():
                _, jv = jvp(fl, (x,), (dv.reshape(x.shape),))
                val = jv.pow(2).sum(1) / (lam * dv.pow(2).sum(1))
                nfu[nm] = dict(mean=float(val.mean()), median=float(val.median()))
            out['null_fraction_unsketched'][L] = nfu
            print(f"[p14 {tag}] {L:10s} w={width:5d} rot r={d['residuals'][2]:.4f} dim_k={d['dim_k']} "
                  f"nullfrac(rot)={nf['rotation']['mean']:.4f} unsketched={nfu['rotation']['mean']:.4f} "
                  f"rand={nf['random_direction_POSTHOC']['mean']:.4f} ref={kr['residuals']} [{time.time()-t0:.0f}s]", flush=True)
        # exact identity metric at the input (unsketched): g = I
        ki = sym.killing(lambda xb: torch.eye(784)[None].expand(len(xb), 784, 784), x, gens, chunk=20)
        out['killing_input_unsketched'] = pack(ki, gens)
        Tx = rotate(x, torch.full((len(x),), math.pi / 6))
        q = sym.quotient_depth(layer_fns, tail_fns, x, Tx, thresh=0.05, groups=labels)
        out['quotient_depth_rot30'] = q
        print(f"[p14 {tag}] q_pullback={np.round(q['q_pullback'],4).tolist()} depth={q['depth_pullback']} | "
              f"q_euclid={np.round(q['q_euclid'],4).tolist()} depth={q['depth_euclid']}", flush=True)
        res[tag] = out
    # ---- score by the letter
    ri = {t: [res[t]['killing'][L]['residuals'][2] for L in ['input', 'conv1', 'conv2']] for t in ['trained', 'untrained']}
    crit_i = all(v < 0.1 for t in ri for v in ri[t])
    nft = [res['trained']['null_fraction'][L]['rotation']['mean'] for L in LAYERS]
    nfu = [res['untrained']['null_fraction'][L]['rotation']['mean'] for L in LAYERS]
    crit_ii = (min(nft) < 0.05) and not (min(nfu) < 0.05)
    res['score'] = dict(i_rotation_residuals_input_conv1_conv2=ri, i_MET=bool(crit_i),
                        ii_null_fraction_trained=nft, ii_null_fraction_untrained=nfu, ii_MET=bool(crit_ii),
                        P1_4_MET=bool(crit_i and crit_ii))
    res['wall_s'] = time.time() - t0
    print('[p14] score', json.dumps(res['score']), flush=True)
    json.dump(res, open(f'{OUT}/blockIMG_p14.json', 'w'), indent=1)


# =============================================================================== ResNet helpers
def resnet_setup():
    from systems.resnet_cifar import make_model, load_cifar
    xtr, ytr, _, _ = load_cifar()
    yscr = np.random.default_rng(0).integers(0, 10, size=len(ytr))

    def probe(lab, n_per):
        rng = np.random.default_rng(0)
        return np.concatenate([rng.choice(np.where(lab == c)[0], 50, replace=False)[:n_per] for c in range(10)])

    def load(tag):
        m = make_model(0)
        if tag != 'untrained':
            m.load_state_dict(torch.load(sorted(glob.glob(f'runs/resnet_{tag}_s0/ckpt_*.pt'))[-1], map_location='cpu'))
        return m.to(DEV).eval()
    labs = dict(clean=ytr, scrambled=yscr, untrained=ytr)
    return xtr, labs, probe, load


def part_p6(n_per=50, batch=32, tags=None):
    t0 = time.time()
    xtr, labs, probe, load = resnet_setup()
    gens = sym.image_generators(3, 32, 32, device=DEV)
    refs = reference_fields(3, 1024, torch.float32, DEV)
    res = dict(n_probe=10 * n_per, generators=[g.name for g in gens])
    fn_out = f'{OUT}/blockIMG_p6.json' if tags is None else f'{OUT}/blockIMG_p6_{tags}.json'
    for tag in (['clean', 'scrambled', 'untrained'] if tags is None else tags.split(',')):
        tt = time.time()
        m = load(tag)
        idx = probe(labs[tag], n_per)
        x = torch.as_tensor(xtr[idx]).to(DEV)
        Jfn = lambda xb: pullback_input(m, xb, batch=batch)
        # P6.2 first (cheap)
        J = Jfn(x)
        ns = sym.null_space(J.cpu().double(), tol=1e-3)
        lam = ns['eigenvalues']
        pr = (lam.sum(1) ** 2 / (lam ** 2).sum(1)).cpu().numpy()
        with torch.no_grad():
            pred = torch.cat([m(x[s:s + 25]).argmax(1) for s in range(0, len(x), 25)]).cpu().numpy()
        p62 = dict(nu_mean=float(ns['nu'].mean()), nu_min=float(ns['nu'].min()), nu_max=float(ns['nu'].max()),
                   rank_mean=float(ns['rank'].mean()), rank_hist=np.bincount(ns['rank'], minlength=11).tolist(),
                   participation_ratio_mean=float(pr.mean()), participation_ratio_sd=float(pr.std()),
                   participation_ratio_median=float(np.median(pr)),
                   mean_spectrum_normalised=(lam / lam[:, :1]).mean(0).tolist(),
                   acc_vs_training_labels=float((pred == labs[tag][idx]).mean()))
        np.save(f'{OUT}/blockIMG_J_{tag}.npy', J.cpu().numpy().astype(np.float16)) if tag == 'clean' else None
        del J; torch.cuda.empty_cache()
        print(f"[p6 {tag}] P6.2 nu={p62['nu_mean']:.6f} rank={p62['rank_mean']:.2f} PR={p62['participation_ratio_mean']:.3f} "
              f"acc={p62['acc_vs_training_labels']:.3f} [{time.time()-tt:.0f}s]", flush=True)
        k = sym.killing(Jfn, x, gens, chunk=batch)
        d = pack(k, gens)
        sub = torch.arange(0, len(x), 5, device=DEV)
        kr = sym.killing(Jfn, x[sub], refs, chunk=batch)
        d['reference_POSTHOC'] = dict(names=kr['names'], residuals=kr['residuals'], n=int(len(sub)))
        res[tag] = dict(killing=d, p62=p62, probe_idx=idx.tolist(), wall_s=time.time() - tt)
        print(f"[p6 {tag}] residuals " + ' '.join(f"{n}={r:.4f}" for n, r in zip(d['names'], d['residuals'])) +
              f" | dim_k={d['dim_k']} sv/s1={np.round(np.array(d['singular_values'])/d['singular_values'][0],3).tolist()} "
              f"ref={kr['residuals']} [{time.time()-tt:.0f}s]", flush=True)
        json.dump(res, open(fn_out, 'w'), indent=1)
        del m; torch.cuda.empty_cache()
    if tags is not None:
        return
    dk = {t: res[t]['killing']['dim_k'] for t in ['clean', 'scrambled', 'untrained']}
    nu = {t: res[t]['p62']['nu_mean'] for t in dk}
    res['score'] = dict(P6_1=dict(dim_k=dk, MET=bool(dk['clean'] >= dk['scrambled'] + 2)),
                        P6_2=dict(nu_mean=nu, gap=nu['clean'] - nu['scrambled'], MET=bool(nu['clean'] > nu['scrambled'] + 0.1),
                                  participation_ratio={t: res[t]['p62']['participation_ratio_mean'] for t in dk}))
    res['wall_s'] = time.time() - t0
    print('[p6] score', json.dumps(res['score']), flush=True)
    json.dump(res, open(f'{OUT}/blockIMG_p6.json', 'w'), indent=1)


def part_stages(n_per=10, batch=16, tags=None):
    t0 = time.time()
    xtr, labs, probe, load = resnet_setup()
    gens = sym.image_generators(3, 32, 32, device=DEV)
    widths = dict(layer1=64 * 32 * 32, layer2=128 * 16 * 16, layer3=256 * 8 * 8, layer4=512 * 4 * 4)
    Ps = {L: (torch.randn(32, w, generator=torch.Generator().manual_seed(0)) / math.sqrt(32)).to(DEV) for L, w in widths.items()}
    res = dict(n_probe=10 * n_per, sketch_dim=32)
    fn_out = f'{OUT}/blockIMG_stages.json' if tags is None else f'{OUT}/blockIMG_stages_{tags}.json'
    for tag in (['clean', 'scrambled', 'untrained'] if tags is None else tags.split(',')):
        m = load(tag)
        x = torch.as_tensor(xtr[probe(labs[tag], n_per)]).to(DEV)
        res[tag] = {}
        for L in ['layer1', 'layer2', 'layer3', 'layer4']:
            tt = time.time()

            def f(xb, L=L):
                h = m.relu(m.bn1(m.conv1(xb)))
                for name in ['layer1', 'layer2', 'layer3', 'layer4']:
                    h = getattr(m, name)(h)
                    if name == L:
                        break
                return h.flatten(1) @ Ps[L].T
            k = sym.killing(lambda xb: pullback_input(f, xb, batch=batch), x, gens, chunk=batch)
            res[tag][L] = pack(k, gens)
            print(f"[stages {tag}] {L} residuals {np.round(k['residuals'],4).tolist()} dim_k={k['dim_k']} [{time.time()-tt:.0f}s]", flush=True)
            json.dump(res, open(fn_out, 'w'), indent=1)
            torch.cuda.empty_cache()
        del m; torch.cuda.empty_cache()
    res['wall_s'] = time.time() - t0
    json.dump(res, open(fn_out, 'w'), indent=1)


# =============================================================================== P4/P5 descriptive
def part_p45():
    from analysis import figstyle; figstyle.setup()
    import matplotlib.pyplot as plt
    t0 = time.time()
    xtr, labs, probe, load = resnet_setup()
    m = load('clean')
    idx = probe(labs['clean'], 50)
    x = torch.as_tensor(xtr[idx]).to(DEV)
    J = pullback_input(m, x, batch=16).cpu().double()                         # (500, 10, 3072)
    S = J.reshape(-1, J.shape[-1])
    _, sv, Vt = torch.linalg.svd(S, full_matrices=True)                        # E[g] = S^T S / N
    top, null = Vt[:3], Vt[-3:]
    ev = (sv ** 2 / len(J))
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.6))
    for j in range(3):
        for r, (V, nm, lamv) in enumerate([(top, 'top', ev[:3]), (null, 'null', None)]):
            im = V[j].reshape(3, 32, 32).abs().sum(0).numpy() if False else V[j].reshape(3, 32, 32).mean(0).numpy()
            a = axs[r, j]; a.imshow(im, cmap='gray', vmin=-np.abs(im).max(), vmax=np.abs(im).max()); a.set_xticks([]); a.set_yticks([]); a.grid(False)
            a.set_title(f'{nm} {j+1}' + (f'  (λ/λ₁ = {float(lamv[j]/ev[0]):.2f})' if lamv is not None else '  (λ ≈ 0)'), fontsize=9)
    fig.suptitle('ResNet-18 clean: eigen-directions of the probe-averaged pullback metric E[g]\n(channel mean; top = what moves the logits, bottom = what it is blind to)', fontsize=9, x=0.02, ha='left')
    fig.tight_layout(); fig.savefig('figures/run3_g_directions.png', dpi=300); plt.close(fig)
    # PGD direction vs top-10 eigenspace of g(x)
    sel = np.arange(500).reshape(10, 50)[:, :20].ravel()
    grs = []
    for s0 in range(0, len(sel), 20):
        xs = x[sel[s0:s0 + 20]].clone().requires_grad_(True)
        lg = m(xs).float(); t2 = lg.topk(2, 1).values
        grs.append(torch.autograd.grad((t2[:, 0] - t2[:, 1]).sum(), xs)[0].detach())
    gr = torch.cat(grs)
    d = -(gr.flatten(1) / gr.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)).double().cpu()
    cos_local, cos_avg, cos_top3 = [], [], []
    V10 = Vt[:10].T
    for i, s in enumerate(sel):
        ns = sym.null_space(J[s], tol=0.0)
        V = ns['V'][:, :10]
        cos_local.append(float((V.T @ d[i]).norm()))
        cos_top3.append(float((ns['V'][:, :3].T @ d[i]).norm()))
        cos_avg.append(float((V10.T @ d[i]).norm()))
    res = dict(n=len(sel), cos_pgd_top10_local_g_mean=float(np.mean(cos_local)), cos_pgd_top10_local_g_min=float(np.min(cos_local)),
               criterion_0p5_MET=bool(np.mean(cos_local) > 0.5),
               note='g(x) has rank <= 10 and the margin gradient is J^T(e_a - e_b), so the top-10 eigenspace of g(x) is the whole row space and the cosine is 1 by construction',
               cos_pgd_top3_local_g_mean_POSTHOC=float(np.mean(cos_top3)),
               cos_pgd_top10_of_probe_averaged_g_mean_POSTHOC=float(np.mean(cos_avg)),
               random_direction_expectation_top10_avg=float(math.sqrt(10 / 3072)),
               Eg_spectrum_top20=(ev[:20] / ev[0]).tolist(),
               Eg_participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()),
               Eg_numerical_rank_tol1em3=int((ev >= 1e-3 * ev[0]).sum()), wall_s=time.time() - t0)
    print('[p45]', json.dumps(res), flush=True)
    json.dump(res, open(f'{OUT}/blockIMG_p45.json', 'w'), indent=1)


def load_stages():
    res = {}
    for t in ['clean', 'scrambled', 'untrained']:
        fn = f'{OUT}/blockIMG_stages_{t}.json'
        if os.path.exists(fn):
            d = json.load(open(fn)); res[t] = d[t]; res['n_probe'] = d['n_probe']; res['sketch_dim'] = 32
            res.setdefault('wall_s', {})[t] = d.get('wall_s')
    return res or None


def load_p6():
    """P6 may have been run per network in separate processes (GPU shared; each process is kernel-launch bound)."""
    res = json.load(open(f'{OUT}/blockIMG_p6.json')) if os.path.exists(f'{OUT}/blockIMG_p6.json') else {}
    for t in ['clean', 'scrambled', 'untrained']:
        fn = f'{OUT}/blockIMG_p6_{t}.json'
        if t not in res and os.path.exists(fn):
            res[t] = json.load(open(fn))[t]; res.setdefault('n_probe', json.load(open(fn))['n_probe'])
    if all(t in res for t in ['clean', 'scrambled', 'untrained']):
        dk = {t: res[t]['killing']['dim_k'] for t in ['clean', 'scrambled', 'untrained']}
        nu = {t: res[t]['p62']['nu_mean'] for t in dk}
        res['score'] = dict(P6_1=dict(dim_k=dk, MET=bool(dk['clean'] >= dk['scrambled'] + 2)),
                            P6_2=dict(nu_mean=nu, gap=nu['clean'] - nu['scrambled'], MET=bool(nu['clean'] > nu['scrambled'] + 0.1),
                                      participation_ratio={t: res[t]['p62']['participation_ratio_mean'] for t in dk}))
    return res


# =============================================================================== figures
def part_fig():
    from analysis import figstyle; figstyle.setup()
    import matplotlib.pyplot as plt
    INK = figstyle.INK
    p6 = load_p6()
    st = load_stages()
    tags = ['clean', 'scrambled', 'untrained']
    style = dict(clean=dict(color=figstyle.BLUE, hatch='', ls='-', marker='o'),
                 scrambled=dict(color=figstyle.ORANGE, hatch='///', ls='--', marker='s'),
                 untrained=dict(color='#9a9a94', hatch='...', ls=':', marker='^'))
    if all(t in p6 for t in tags):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.7, 1]))
        a = axs[0]; names = p6['clean']['killing']['names']; w = 0.27
        for j, t in enumerate(tags):
            r = np.array(p6[t]['killing']['residuals']); ci = np.array(p6[t]['killing']['residual_ci95'])
            a.bar(np.arange(len(names)) + (j - 1) * w, r, w, color=style[t]['color'], hatch=style[t]['hatch'], edgecolor=INK, linewidth=0.6,
                  yerr=[r - ci[:, 0], ci[:, 1] - r], error_kw=dict(lw=0.8, ecolor=INK), label=f"{t}  (dim k = {p6[t]['killing']['dim_k']}; random-field reference r = {p6[t]['killing']['reference_POSTHOC']['residuals'][0]:.2f})")
        a.axhline(0.1, color=INK, lw=1, ls='--'); a.text(len(names) - 0.4, 0.1, 'r = 0.1', va='bottom', ha='right', fontsize=8)
        a.set_yscale('log'); a.set_xticks(np.arange(len(names))); a.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
        a.set_ylabel('Killing residual  r = E‖L_V g‖ / 2E‖g‖'); a.legend(fontsize=8, loc='upper left')
        a.set_title(f"ResNet-18 input pullback, {p6['n_probe']} images: Killing residual")
        b = axs[1]
        if st is not None:
            Ls = ['layer1', 'layer2', 'layer3', 'layer4']
            for t in tags:
                if t in st:
                    have = [i for i, L in enumerate(Ls) if L in st[t]]
                    xs_ = have + [4]; y = [st[t][Ls[i]]['dim_k'] for i in have] + [p6[t]['killing']['dim_k']]
                    off = {'clean': -0.06, 'scrambled': 0.0, 'untrained': 0.06}[t]
                    b.plot(xs_, np.array(y) + off, color=style[t]['color'], ls=style[t]['ls'], marker=style[t]['marker'], mfc='white' if t != 'clean' else None, label=t)
            b.set_xticks(range(5)); b.set_xticklabels(Ls + ['logits'], fontsize=8); b.set_ylim(-0.3, 8.3)
            b.set_ylabel('dim k  (of 8 generators)'); b.legend(fontsize=8); b.set_title(f"dim k by depth ({st['n_probe']} images)")
        fig.tight_layout(); fig.savefig('figures/run3_killing_dim.png', dpi=300); plt.close(fig)

    p14 = json.load(open(f'{OUT}/blockIMG_p14.json')); Ls = p14['layers']
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    sty = dict(trained=dict(color=figstyle.BLUE, ls='-', marker='o', mfc=None), untrained=dict(color=figstyle.ORANGE, ls='--', marker='s', mfc='white'))
    for t in ['trained', 'untrained']:
        s = sty[t]
        axs[0].plot(range(6), [p14[t]['null_fraction'][L]['rotation']['mean'] for L in Ls], color=s['color'], ls=s['ls'], marker=s['marker'], mfc=s['mfc'], label=f'{t}: rotation direction')
        axs[0].plot(range(6), [p14[t]['null_fraction'][L]['isotropic_expectation_POSTHOC'] for L in Ls], color=s['color'], ls=':', lw=1.2, marker='x', label=f'{t}: isotropic expectation (post hoc)')
        axs[1].plot(range(6), [p14[t]['null_fraction_unsketched'][L]['rotation']['mean'] for L in Ls], color=s['color'], ls=s['ls'], marker=s['marker'], mfc=s['mfc'], label=f'{t}: rotation')
        axs[1].plot(range(6), [p14[t]['null_fraction_unsketched'][L]['random_direction_POSTHOC']['mean'] for L in Ls], color=s['color'], ls=':', lw=1.2, marker='x', label=f'{t}: random direction')
        q = p14[t]['quotient_depth_rot30']
        axs[2].plot(range(6), q['q_pullback'], color=s['color'], ls=s['ls'], marker=s['marker'], mfc=s['mfc'], label=f'{t}: linearised pullback')
        axs[2].plot(range(6), q['q_euclid'], color=s['color'], ls=':', lw=1.2, marker='x', label=f'{t}: Euclidean')
    for a, ttl in zip(axs, ['Rotation null fraction (32-d sketch)', 'Same, no sketch (post hoc)', 'q_ℓ(rotate 30°)']):
        a.axhline(0.05, color=INK, lw=1, ls='--'); a.set_yscale('log'); a.set_xticks(range(6)); a.set_xticklabels(Ls, fontsize=8, rotation=20)
        a.set_title(ttl); a.legend(fontsize=7)
    axs[0].set_ylabel('‖J A x‖² / (‖J‖²op ‖A x‖²)'); axs[2].set_ylabel('E d(x, Tx) / E d(x, x′)')
    fig.tight_layout(); fig.savefig('figures/run3_rotation_invariance_depth.png', dpi=300); plt.close(fig)
    print('[fig] written', flush=True)


def _posthoc(d):
    """POST-HOC additions to every Killing record: the algebra restricted to the 6 image warps (brightness and contrast
    are normalised differently -- ||b|| = 1 vs ||A||_F = 1 -- and set s_1), and the count of residuals below 0.1."""
    if isinstance(d, dict):
        if 'singular_vectors' in d and 'residuals' in d and len(d['residuals']) == 8:
            W = np.array(d['singular_vectors']); s = np.array(d['singular_values'])
            G = (W * s[None] ** 2) @ W.T
            sw = np.sqrt(np.clip(np.linalg.eigvalsh(G[:6, :6])[::-1], 0, None))
            d['POSTHOC_warps_only'] = dict(singular_values=sw.tolist(), dim_k=int((sw < 0.1 * sw[0]).sum()))
            d['POSTHOC_per_generator_sketch_norm'] = np.sqrt(np.clip(np.diag(G), 0, None)).tolist()
            d['POSTHOC_n_residuals_below_0p1'] = int((np.array(d['residuals']) < 0.1).sum())
        for v in d.values():
            _posthoc(v)


def part_merge():
    out = {}
    out['stages'] = load_stages()
    for p in ['p14', 'p45']:
        fn = f'{OUT}/blockIMG_{p}.json'
        if os.path.exists(fn):
            out[p] = json.load(open(fn))
    out['p6'] = load_p6()
    _posthoc(out)
    json.dump(out, open(f'{OUT}/blockIMG.json', 'w'), indent=1)
    print('[merge]', list(out), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True)
    ap.add_argument('--n_per', type=int, default=None); ap.add_argument('--batch', type=int, default=None); ap.add_argument('--tags', default=None)
    a = ap.parse_args()
    kw = {k: v for k, v in dict(n_per=a.n_per, batch=a.batch, tags=a.tags).items() if v is not None}
    dict(p14=part_p14, p6=part_p6, stages=part_stages, p45=part_p45, fig=part_fig, merge=part_merge)[a.part](**kw)
