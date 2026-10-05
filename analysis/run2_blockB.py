"""Run 2, Block B (SPEC2.md section B): invisible displacement iota, neighbourhood persistence
kappa, and iota * (1 - kappa), against softmax confidence / entropy / Euclidean 1-NN / run-1 tubular
distance, on ResNet-18, GPT-2 small L6, Z113 (3 seeds) and the ring RNN (3 seeds, iota only).

    python analysis/run2_blockB.py                 # every system, in the order resnet gpt2 z113 ring, then merge
    python analysis/run2_blockB.py --only gpt2     # one system (writes its part), then merge what exists
    python analysis/run2_blockB.py --only merge    # rebuild blockB.json / blockB_scores.npz / the figure from the parts

Each system writes results/run2/blockB_parts/<system>.json + .npz so a crash loses one system only.
Every threshold is SPEC2's: rcond = 1e-3 * sigma_max, k = 15, nearest training activation Euclidean.
"""
import sys, os, json, glob, time, argparse
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whiteboxtopo.kernel import invisible_displacement, iota, knn_sets, kappa, combined, auroc, summary
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.geometry import tubular_distance
from whiteboxtopo.extract import jacobians, logits_of

DEV = 'cuda'
OUT = 'results/run2'; PARTS = f'{OUT}/blockB_parts'
SCORES = ['iota', 'par', 'ratio', 'one_minus_kappa', 'combined']
RCOND, K = 1e-3, 15


# ------------------------------------------------------------------------------ shared
def softmax_stats(logits):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return -p.max(1), -(p * np.log(p + 1e-12)).sum(1)               # (-confidence, entropy): higher = more OOD


def evaluate(raw_in, raw_ood):
    """raw_*: dict(perp, par, ratio, kappa or None, base=dict(name -> score, higher = OOD)).
    Returns (cells: set -> score -> AUROC, scores: set -> name -> per-point array, kappa summary)."""
    def table(r):
        t = dict(iota=iota(r['perp'], raw_in['perp']), par=r['par'], ratio=r['ratio'])
        if r.get('kappa') is not None:
            t['one_minus_kappa'] = 1.0 - r['kappa']
            t['combined'] = combined(t['iota'], r['kappa'])
        t.update(r['base'])
        return t
    tin = table(raw_in)
    cells, scores, kap = {}, {'in': tin}, {}
    if raw_in.get('kappa') is not None:
        kap['in'] = dict(summary(raw_in['kappa']), per_pair_mean=[float(v) for v in raw_in['kappa_pairs'].mean(1)])
    for nm, r in raw_ood.items():
        to = table(r); scores[nm] = to
        cells[nm] = {k: auroc(tin[k], to[k]) for k in tin}
        if r.get('kappa') is not None:
            kap[nm] = dict(summary(r['kappa']), per_pair_mean=[float(v) for v in r['kappa_pairs'].mean(1)])
    return cells, scores, kap


def save_part(name, res, scores):
    os.makedirs(PARTS, exist_ok=True)
    json.dump(res, open(f'{PARTS}/{name}.json', 'w'), indent=1, default=float)
    flat = {f'{name}/{s}/{k}': np.asarray(v, np.float32) for s, t in scores.items() for k, v in t.items()}
    np.savez_compressed(f'{PARTS}/{name}.npz', **flat)


def show(tag, cells):
    for nm, c in cells.items():
        print(f'[{tag}] {nm:16s} ' + ' '.join(f'{k}={v:.3f}' for k, v in c.items()), flush=True)


def rank_info(*ranks, d):
    r = np.concatenate(ranks)
    return dict(rank_median=float(np.median(r)), rank_min=int(r.min()), rank_max=int(r.max()),
                frac_points_below_max_rank=float((r < r.max()).mean()), frac_points_rank_below_half_max=float((r < 0.5 * r.max()).mean()),
                kernel_dim_median=float(d - np.median(r)), ambient_dim=int(d))


# ------------------------------------------------------------------------------ ResNet-18
def resnet():
    from systems.resnet_cifar import make_model, load_cifar, MEAN, STD
    from analysis.resnet_depth import all_layers, LAYERS
    from torchvision import datasets
    T0 = time.time(); t = {}
    xtr, ytr, xte, yte = load_cifar()
    norm = lambda a: ((a.astype(np.float32) / 255).transpose(0, 3, 1, 2) - MEAN[None, :, None, None]) / STD[None, :, None, None]
    sv = datasets.SVHN('data/svhn', split='test', download=False)
    svx = (sv.data[:2000].astype(np.float32) / 255 - MEAN[None, :, None, None]) / STD[None, :, None, None]
    c100 = datasets.CIFAR100('data/cifar100', train=False, download=True)
    c100x = norm(c100.data[:1500])
    m = make_model().to(DEV); m.load_state_dict(torch.load(sorted(glob.glob('runs/resnet_clean_s0/ckpt_*.pt'))[-1])); m.eval()
    # same RNG sequence as run 1 (analysis/resnet_depth.py): probe, run-1 depth hold-out (unused here), P4 hold-out, noise
    rng = np.random.default_rng(0)
    sel = np.concatenate([rng.choice(np.where(ytr == c)[0], 150, replace=False) for c in range(10)])
    hold = rng.choice(len(xte), 1000, replace=False)
    hold2 = rng.choice(np.setdiff1d(np.arange(len(xte)), hold), 1500, replace=False)
    noise = rng.standard_normal((1500, 3, 32, 32)).astype(np.float32)
    t0 = time.time()
    Atr, ltr = all_layers(m, xtr[sel]); del xtr
    sets = dict(svhn=svx[:1500], cifar100=c100x, gaussian_noise=noise)
    acts = {'in': all_layers(m, xte[hold2])}
    for nm, x in sets.items():
        acts[nm] = all_layers(m, x); torch.cuda.empty_cache()
    t['activations_s'] = time.time() - t0
    # PCA of the wide layers: fitted on the TRAINING PROBE only, the same projection applied to everything
    t0 = time.time(); pca = {}
    for L in LAYERS:
        if Atr[L].shape[1] > 1024:
            mu = Atr[L].mean(0); _, S, Vt = np.linalg.svd(Atr[L] - mu, full_matrices=False)
            pca[L] = dict(dim_in=int(Atr[L].shape[1]), var_kept_train=float((S[:256] ** 2).sum() / (S ** 2).sum()))
            P = Vt[:256].T.astype(np.float32)
            Atr[L] = (Atr[L] - mu) @ P
            for nm in acts:
                acts[nm][0][L] = (acts[nm][0][L] - mu) @ P
            del S, Vt, P
    t['pca_s'] = time.time() - t0
    W = m.fc.weight.detach().float().cpu().numpy()
    Xp = Atr['penultimate']
    t0 = time.time()
    Jp = np.ascontiguousarray(np.broadcast_to(W[None], (len(Xp),) + W.shape)).astype(np.float32)
    fitp = fit_manifold(Xp, Jp, metric='g'); del Jp                       # run-1 tangent bases, for the tubular baseline
    t['fit_manifold_s'] = time.time() - t0

    def raw(A, lg):
        t1 = time.time(); r = invisible_displacement(A['penultimate'], Xp, W, rcond=RCOND); t['iota_s'] = t.get('iota_s', 0) + time.time() - t1
        t1 = time.time(); kp, pairs = kappa([knn_sets(Atr[L], A[L], K) for L in LAYERS]); t['kappa_s'] = t.get('kappa_s', 0) + time.time() - t1
        conf, ent = softmax_stats(lg)
        r.update(kappa=kp, kappa_pairs=pairs, base=dict(softmax_conf=conf, entropy=ent, euclid_1nn=r['total'],
                                                        tubular=tubular_distance(A['penultimate'], fitp.X, fitp.bases)))
        return r
    rin = raw(*acts['in']); rood = {nm: raw(*acts[nm]) for nm in sets}
    cells, scores, kap = evaluate(rin, rood)
    sv_ = np.linalg.svd(W, compute_uv=False)
    res = dict(system='resnet18_clean_s0', layer='penultimate', n_train=int(len(Xp)), n_in=1500, n_ood=1500,
               layers_kappa=LAYERS, layer_dims={L: int(Atr[L].shape[1]) for L in LAYERS}, pca=pca,
               J=dict(kind='final linear layer W (one matrix)', shape=list(W.shape), sigma_max=float(sv_[0]), sigma_min=float(sv_[-1]),
                      **rank_info(rin['rank'], d=W.shape[1])),
               heldout_acc=float((acts['in'][1].argmax(1) == yte[hold2]).mean()),
               in_dist=dict(perp=summary(rin['perp']), par=summary(rin['par']), ratio=summary(rin['ratio'])),
               ood_ratio={nm: summary(r['ratio']) for nm, r in rood.items()},
               auroc=cells, kappa=kap, timings=t, wall_s=time.time() - T0)
    show('resnet', cells); print('[resnet] kappa', json.dumps(kap), flush=True)
    save_part('resnet', res, scores)
    return res


# ------------------------------------------------------------------------------ GPT-2 small, layer 6
KLAYERS = [2, 4, 6, 8, 10, 12]


def gpt2(n_probe=2000, n_hold=1000):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from analysis.gpt2_explore import Tap, natural_windows, LAYER, CTX
    T0 = time.time(); t = {}
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    # eager attention: the SDPA efficient-attention BACKWARD is wrong on this GPU (run-1 night log), and B1 needs gradients
    model = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation='eager').to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tap = Tap(model)
    ids, src = natural_windows(tok, n_probe + n_hold, 0)                 # same windows as run 1
    rng = np.random.default_rng(3)                                       # same OOD sets as run 1
    ood = dict(random_tokens=rng.integers(0, 50257, size=(n_hold, CTX)).astype(np.int64),
               shuffled_order=np.stack([rng.permutation(r) for r in ids[n_probe:]]))

    @torch.no_grad()
    def fwd(idsx, bs=125):
        Hs = {l: [] for l in KLAYERS}; Fo, conf, ent = [], [], []
        for s in range(0, len(idsx), bs):
            o = model.transformer(torch.as_tensor(idsx[s:s + bs], device=DEV), output_hidden_states=True)
            for l in KLAYERS:
                Hs[l].append(o.hidden_states[l][:, -1].float().cpu().numpy())
            f = o.last_hidden_state[:, -1]
            Fo.append(f.float().cpu().numpy())
            lp = torch.log_softmax(model.lm_head(f).float(), -1)
            conf.append((-lp.max(-1).values.exp()).cpu().numpy()); ent.append((-(lp.exp() * lp).sum(-1)).cpu().numpy())
            del o, lp; torch.cuda.empty_cache()
        return {l: np.concatenate(v).astype(np.float32) for l, v in Hs.items()}, np.concatenate(Fo), np.concatenate(conf), np.concatenate(ent)

    t0 = time.time()
    Htr, Ftr, _, _ = fwd(ids[:n_probe])
    A = {'in': fwd(ids[n_probe:])}
    for nm, io in ood.items():
        A[nm] = fwd(io)
    t['activations_s'] = time.time() - t0
    WU = model.lm_head.weight.detach().float().cpu().numpy()
    _, S, Vt = np.linalg.svd(WU, full_matrices=False); del WU
    Otr = (Ftr @ Vt.T) * S                                               # exact isometric image of the logits (run 1)
    mu = Otr.mean(0); _, So, Vo = np.linalg.svd(Otr - mu, full_matrices=False)
    P64 = Vo[:64]; var64 = float((So[:64] ** 2).sum() / (So ** 2).sum())
    Vt_t, S_t, P_t, mu_t = (torch.as_tensor(a, device=DEV, dtype=torch.float32) for a in (Vt, S, P64, mu))

    def z_of(f):                                                         # top-64 principal components of the logit image
        return ((f.float() @ Vt_t.T) * S_t - mu_t) @ P_t.T

    def jac(idsx, bs=32):
        """(n, 64, 768) Jacobian of z w.r.t. the layer-6 residual at the last position, at the window's own activation:
        64 vector-Jacobian products per batch through blocks 6-11 + ln_f."""
        out, s = [], 0
        while s < len(idsx):
            try:
                x = torch.as_tensor(idsx[s:s + bs], device=DEV)
                delta = torch.zeros(len(x), 768, device=DEV, requires_grad=True)
                tap.delta = delta
                with torch.enable_grad():
                    z = z_of(model.transformer(x).last_hidden_state[:, -1])
                    rows = [torch.autograd.grad(z[:, c].sum(), delta, retain_graph=c < 63)[0] for c in range(64)]
                out.append(torch.stack(rows, 1).cpu().numpy()); s += bs
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                bs = max(1, bs // 2); print(f'[gpt2] CUDA OOM -> batch {bs}', flush=True)
            finally:
                tap.delta = None; torch.cuda.empty_cache()
        return np.concatenate(out)

    Xp = Htr[LAYER]
    from sklearn.neighbors import NearestNeighbors
    nn1 = NearestNeighbors(n_neighbors=1).fit(Xp)
    need = np.unique(np.concatenate([nn1.kneighbors(A[nm][0][LAYER])[1][:, 0] for nm in A]))
    t0 = time.time()
    Jn = jac(ids[:n_probe][need])
    t['jacobian_s'] = time.time() - t0; t['n_jacobians'] = int(len(need))
    # finite-difference check of the autograd Jacobian (central difference through the same Tap)
    chk = need[:8]; g = np.random.default_rng(0)
    v = g.standard_normal((len(chk), 768)).astype(np.float32); v /= np.linalg.norm(v, axis=1, keepdims=True)
    eps = 1e-2 * np.linalg.norm(Xp[chk], axis=1, keepdims=True).astype(np.float32)
    with torch.no_grad():
        x = torch.as_tensor(ids[:n_probe][chk], device=DEV)
        tap.delta = torch.as_tensor(eps * v, device=DEV); zp = z_of(model.transformer(x).last_hidden_state[:, -1]).cpu().numpy()
        tap.delta = torch.as_tensor(-eps * v, device=DEV); zm = z_of(model.transformer(x).last_hidden_state[:, -1]).cpu().numpy()
        tap.delta = None
    fd = (zp - zm) / (2 * eps); an = np.einsum('nod,nd->no', Jn[:len(chk)], v)
    fd_cos = [float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in zip(fd, an)]
    fd_rel = [float(np.linalg.norm(a - b) / np.linalg.norm(b)) for a, b in zip(fd, an)]
    print(f'[gpt2] Jacobian FD check: cos {np.min(fd_cos):.5f}..{np.max(fd_cos):.5f} rel err max {np.max(fd_rel):.4f}', flush=True)
    pos = {int(i): j for j, i in enumerate(need)}
    Jfn = lambda u: Jn[[pos[int(i)] for i in u]]
    t0 = time.time()
    fit = fit_manifold(Xp, metric='secant', out=Otr)                     # run-1 tangent bases, for the tubular baseline
    t['fit_manifold_s'] = time.time() - t0
    svals = np.linalg.svd(Jn[: min(200, len(Jn))].astype(np.float64), compute_uv=False)

    def raw(a):
        Hs, _, conf, ent = a
        t1 = time.time(); r = invisible_displacement(Hs[LAYER], Xp, Jfn, rcond=RCOND); t['iota_s'] = t.get('iota_s', 0) + time.time() - t1
        t1 = time.time(); kp, pairs = kappa([knn_sets(Htr[l], Hs[l], K) for l in KLAYERS]); t['kappa_s'] = t.get('kappa_s', 0) + time.time() - t1
        r.update(kappa=kp, kappa_pairs=pairs, base=dict(softmax_conf=conf, entropy=ent, euclid_1nn=r['total'],
                                                        tubular=tubular_distance(Hs[LAYER], fit.X, fit.bases)))
        return r
    rin = raw(A['in']); rood = {nm: raw(A[nm]) for nm in ood}
    cells, scores, kap = evaluate(rin, rood)
    res = dict(system='gpt2_small', layer=LAYER, ctx=CTX, text_source=src, attn_implementation='eager',
               n_train=n_probe, n_in=n_hold, n_ood=n_hold, layers_kappa=KLAYERS,
               J=dict(kind='Jacobian of the top-64 PCs of the logit image w.r.t. the L6 last-position residual, at hbar',
                      shape=[64, 768], logit_image_var_in_64_pcs=var64, n_computed=int(len(need)),
                      sigma_ratio_min_over_max_median=float(np.median(svals[:, -1] / svals[:, 0])),
                      fd_check_cos=fd_cos, fd_check_rel_err=fd_rel,
                      **rank_info(rin['rank'], *[r['rank'] for r in rood.values()], d=768)),
               in_dist=dict(perp=summary(rin['perp']), par=summary(rin['par']), ratio=summary(rin['ratio'])),
               ood_ratio={nm: summary(r['ratio']) for nm, r in rood.items()},
               ood_perp={nm: summary(r['perp']) for nm, r in rood.items()},
               auroc=cells, kappa=kap, timings=t, wall_s=time.time() - T0)
    show('gpt2', cells); print('[gpt2] kappa', json.dumps(kap), flush=True)
    save_part('gpt2', res, scores)
    return res


# ------------------------------------------------------------------------------ Z113
Z_RUNS = [('z113_s0', 'runs/z113_clean_s0', 0, 0.3), ('z113_s1', 'runs/z113_clean_s1', 1, 0.3),
          ('z113_s2ctrl', 'runs/z113_cleanctrl_s2', 2, 0.5)]
Z_LAYERS = ['embed', 'attn', 'resid_mid', 'resid_post']


def z113():
    from analysis.z113_common import checkpoints
    from analysis.z113_static import ood_sets
    from systems.z113 import P, NULL, EQ, build_data
    from systems.z113_probe import load, psi_of
    out = {}
    for name, run, seed, frac in Z_RUNS:
        T0 = time.time(); t = {}
        cks, steps = checkpoints(run)
        m = load(cks[-1], seed=seed)
        d = build_data(seed, 'clean', frac=frac)
        split_ok = bool(np.array_equal(d['train_idx'], np.load(f'{run}/data.npz')['train_idx']))

        def layers(toks):
            tk = torch.as_tensor(toks, device=DEV)
            with torch.no_grad():
                x = m.embed(tk); a = m.attn(x); mid = (x + a)[:, -1]; post = mid + m.mlp(mid)
                A = dict(embed=x[:, :2].flatten(1), attn=a[:, -1], resid_mid=mid, resid_post=post)
                return {k: v.float().cpu().numpy() for k, v in A.items()}, (post @ m.W_U).float().cpu().numpy()
        Atr, ltr = layers(d['train_toks'])                               # ALL training pairs of the model's own split
        te = d['test_idx'][:1000]; a, b = d['a'], d['b']
        hold = np.stack([a[te], b[te], np.full(len(te), NULL), np.full(len(te), EQ)], 1)
        rng = np.random.default_rng(0)
        osets = ood_sets(d, te, rng)
        psi = psi_of(m, 'resid_mid')
        Xp = Atr['resid_mid'].astype(np.float32)
        t0 = time.time(); Jtr = jacobians(psi, Xp, device=DEV).astype(np.float32); t['jacobian_s'] = time.time() - t0
        t0 = time.time(); fit = fit_manifold(Xp, Jtr, metric='g'); t['fit_manifold_s'] = time.time() - t0

        def raw(toks):
            A, lg = layers(toks)
            t1 = time.time(); r = invisible_displacement(A['resid_mid'], Xp, Jtr, rcond=RCOND); t['iota_s'] = t.get('iota_s', 0) + time.time() - t1
            t1 = time.time(); kp, pairs = kappa([knn_sets(Atr[L], A[L], K) for L in Z_LAYERS]); t['kappa_s'] = t.get('kappa_s', 0) + time.time() - t1
            conf, ent = softmax_stats(lg)
            r.update(kappa=kp, kappa_pairs=pairs, base=dict(softmax_conf=conf, entropy=ent, euclid_1nn=r['total'],
                                                            tubular=tubular_distance(A['resid_mid'], fit.X, fit.bases)), logits=lg)
            return r
        rin = raw(hold); rood = {nm: raw(tk) for nm, tk in osets.items()}
        cells, scores, kap = evaluate(rin, rood)
        res = dict(system=name, run=run, seed=seed, frac=frac, split_matches_run_data=split_ok, step=int(steps[-1]), layer='resid_mid',
                   n_train=int(len(Xp)), n_in=int(len(hold)), n_ood=int(len(hold)), layers_kappa=Z_LAYERS,
                   heldout_acc=float((rin['logits'].argmax(1) == d['c'][te]).mean()),
                   J=dict(kind='Jacobian of psi(h) = (h + MLP(h)) W_U at hbar', shape=[P, 128],
                          **rank_info(rin['rank'], *[r['rank'] for r in rood.values()], d=128)),
                   in_dist=dict(perp=summary(rin['perp']), par=summary(rin['par']), ratio=summary(rin['ratio'])),
                   ood_ratio={nm: summary(r['ratio']) for nm, r in rood.items()},
                   auroc=cells, kappa=kap, timings=t, wall_s=time.time() - T0)
        show(name, cells); print(f'[{name}] rank', json.dumps(res['J']), 'kappa', json.dumps(kap), flush=True)
        save_part(name, res, scores); out[name] = res
        del Jtr, fit; torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------ ring RNN (iota only)
def ring():
    from systems.ring_rnn import RingRNN
    from analysis.ring_analysis import cloud
    out = {}
    for seed in (0, 1, 2):
        T0 = time.time(); t = {}
        m = RingRNN(seed=seed).to(DEV); m.load_state_dict(torch.load(sorted(glob.glob(f'runs/ring_s{seed}/ckpt_*.pt'))[-1])); m.eval()
        X, _ = cloud(m, 1500, seed=7)                                    # run-1 training cloud, hold-out and off-ring sets
        t0 = time.time(); J = jacobians(m.psi, X).astype(np.float32); t['jacobian_s'] = time.time() - t0
        t0 = time.time(); fit = fit_manifold(X, J, metric='g'); t['fit_manifold_s'] = time.time() - t0
        Xh, _ = cloud(m, 800, seed=99)
        rng = np.random.default_rng(5)
        sets = {'radial_shrink': Xh * 0.6,
                'gaussian_state': (rng.standard_normal(Xh.shape) * X.std()).astype(np.float32),
                'big_velocity': cloud(m, 800, seed=42, sigma=3.0)[0]}

        def raw(A):
            t1 = time.time(); r = invisible_displacement(A, X, J, rcond=RCOND); t['iota_s'] = t.get('iota_s', 0) + time.time() - t1
            conf = np.abs(np.linalg.norm(logits_of(m.psi, A), axis=1) - 1.0)      # run-1 output-norm confidence (no softmax exists)
            r.update(kappa=None, base=dict(output_norm_conf=conf, euclid_1nn=r['total'], tubular=tubular_distance(A, fit.X, fit.bases)))
            return r
        rin = raw(Xh); rood = {nm: raw(A) for nm, A in sets.items()}
        cells, scores, _ = evaluate(rin, rood)
        name = f'ring_s{seed}'
        res = dict(system=name, layer='hidden state', n_train=1500, n_in=800, n_ood=800, kappa='undefined (single layer)',
                   J=dict(kind='Jacobian of psi (one zero-velocity step + readout) at hbar', shape=[2, 128],
                          **rank_info(rin['rank'], *[r['rank'] for r in rood.values()], d=128)),
                   in_dist=dict(perp=summary(rin['perp']), par=summary(rin['par']), ratio=summary(rin['ratio'])),
                   ood_ratio={nm: summary(r['ratio']) for nm, r in rood.items()},
                   auroc=cells, timings=t, wall_s=time.time() - T0)
        show(name, cells); save_part(name, res, scores); out[name] = res
    return out


# ------------------------------------------------------------------------------ merge, criterion, figure
def merge():
    parts = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f'{PARTS}/*.json'))}
    allsc = {}
    for f in sorted(glob.glob(f'{PARTS}/*.npz')):
        z = np.load(f); allsc.update({k: z[k] for k in z.files})
    np.savez_compressed(f'{OUT}/blockB_scores.npz', **allsc)
    from scipy.stats import spearmanr
    for name, p in parts.items():                                    # how different is iota from plain Euclidean 1-NN?
        p['spearman_iota_vs_euclid_1nn'] = {k.split('/')[1]: float(spearmanr(allsc[k], allsc[k.replace('/iota', '/euclid_1nn')]).statistic)
                                            for k in allsc if k.startswith(name + '/') and k.endswith('/iota')}
    crit = {}
    if 'resnet' in parts:
        a = parts['resnet']['auroc']; per = {}
        for s in ('iota', 'one_minus_kappa', 'combined'):
            per[s] = {nm: dict(score=a[nm][s], softmax_conf=a[nm]['softmax_conf'], entropy=a[nm]['entropy'],
                               beats_both=bool(a[nm][s] > a[nm]['softmax_conf'] and a[nm][s] > a[nm]['entropy']))
                      for nm in ('svhn', 'cifar100')}
            per[s]['wins_on_both_sets'] = bool(per[s]['svhn']['beats_both'] and per[s]['cifar100']['beats_both'])
        crit['resnet'] = dict(per_score=per, met=bool(any(v['wins_on_both_sets'] for v in per.values())),
                              winning_scores=[s for s, v in per.items() if v['wins_on_both_sets']])
    if 'gpt2' in parts:
        a = parts['gpt2']['auroc']
        crit['gpt2'] = dict(iota={nm: a[nm]['iota'] for nm in a}, met=bool(all(a[nm]['iota'] > 0.5 for nm in a)),
                            family={s: {nm: a[nm][s] for nm in a} for s in ('iota', 'one_minus_kappa', 'combined')})
    if 'resnet' in crit and 'gpt2' in crit:
        crit['pass_B'] = bool(crit['resnet']['met'] and crit['gpt2']['met'])
    out = dict(block='B', spec='SPEC2.md section B', rcond=RCOND, k=K, score_orientation='higher = out-of-distribution',
               criterion=crit, systems=parts, wall_s_total=float(sum(p.get('wall_s', 0) for p in parts.values())))
    json.dump(out, open(f'{OUT}/blockB.json', 'w'), indent=1, default=float)
    print('[merge] criterion', json.dumps(crit), flush=True)
    figure(allsc)


def figure(sc):
    from sklearn.metrics import roc_curve
    from analysis import figstyle as fs
    import matplotlib.pyplot as plt
    fs.setup()
    panels = [('resnet', 'svhn', 'ResNet-18 · SVHN'), ('resnet', 'cifar100', 'ResNet-18 · CIFAR-100'),
              ('resnet', 'gaussian_noise', 'ResNet-18 · Gaussian noise'),
              ('gpt2', 'random_tokens', 'GPT-2 L6 · random tokens'), ('gpt2', 'shuffled_order', 'GPT-2 L6 · shuffled order')]
    panels = [p for p in panels if f'{p[0]}/{p[1]}/iota' in sc]
    if not panels:
        return
    series = [('iota', 'ι  invisible displacement', fs.BLUE, '-', 'o'), ('one_minus_kappa', '1 − κ  neighbourhood turnover', fs.ORANGE, '--', 's'),
              ('combined', 'ι·(1 − κ)', fs.AQUA, '-.', '^'), ('softmax_conf', 'softmax confidence', fs.INK, ':', 'x')]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2)); axes = axes.ravel()
    for ax, (sysn, st, title) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], color=fs.GRID, lw=1.0, zorder=0)
        for key, lab, col, ls, mk in series:
            a, b = sc[f'{sysn}/in/{key}'], sc[f'{sysn}/{st}/{key}']
            fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
            grid = np.linspace(0, 1, 201); tg = np.interp(grid, fpr, tpr)
            ax.plot(grid, tg, color=col, ls=ls, marker=mk, markevery=(10 + 5 * series.index((key, lab, col, ls, mk)), 25), ms=5.5,
                    mfc=fs.SURFACE, mew=1.3, label=f'{lab}   {auroc(a, b):.3f}')
        ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1.005); ax.set_aspect('equal')
        ax.set_xlabel('false-positive rate (in-distribution flagged)'); ax.set_ylabel('true-positive rate')
        lg = ax.legend(loc='lower right', fontsize=8, title='score   AUROC', title_fontsize=8, frameon=True, framealpha=0.92,
                       facecolor=fs.SURFACE, edgecolor=fs.GRID, handlelength=3.2)
    for ax in axes[len(panels):]:
        ax.axis('off')
    if len(panels) < len(axes):
        axes[-1].text(0.0, 0.95, 'Run 2 · Block B\nOut-of-distribution detection from\nthe part of the displacement the head\ncannot see (ι), the turnover of nearest\n'
                                 'training neighbours across depth (1 − κ),\nand their product, against softmax\nconfidence. Diagonal = chance.',
                      va='top', fontsize=10, color=fs.INK2, transform=axes[-1].transAxes)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/run2_kernel_roc.png', dpi=300); plt.close(fig)
    print('[merge] figures/run2_kernel_roc.png', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--only', default='all')
    a = ap.parse_args()
    os.makedirs(PARTS, exist_ok=True)
    todo = ['resnet', 'gpt2', 'z113', 'ring'] if a.only == 'all' else [s for s in a.only.split(',') if s != 'merge']
    for s in todo:
        try:
            dict(resnet=resnet, gpt2=gpt2, z113=z113, ring=ring)[s]()
        except Exception as e:                                           # keep the other systems
            import traceback; traceback.print_exc()
            print(f'[{s}] FAILED: {type(e).__name__}: {str(e)[:300]}', flush=True)
        torch.cuda.empty_cache()
    merge()
