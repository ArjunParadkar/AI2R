"""GPT-2-small, layer 6 -- EXPLORATORY (spec 6.1: "reported as such", with its manifold score).

Nothing here has a pre-registered pass criterion.  The questions are
  m     does the residual stream at layer 6 look like a manifold at all, by our own instrument?
  P1    is there any persistent H^1 above the feature-shuffle noise band on natural text, and
        on a structured probe (weekdays / months, the cyclic variables GPT-2 is known to carry)
        does the symmetry-informed isotypic route certify a circle?
  P4    natural held-out tokens vs (a) uniformly random token sequences, (b) the same texts
        with token order shuffled: tubular distance, spec d_g, Euclidean kNN, softmax, entropy.
  P5    failure radius of the next-token argmax under Gaussian perturbation of the layer-6
        residual vs rho = ||II|| * Lip, vs softmax margin.

The output metric is exact and Jacobian-free: logits = W_U f, so ||dlogits|| = ||S V^T df|| with
W_U = U S V^T, and the secant metric runs on the 768-d vector S V^T f.
"""
import sys, os, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.topology import (persistence, circular_coordinates, circular_correlation,
                                   isotypic_decomposition, normalized_persistence)
from whiteboxtopo.geometry import second_fundamental_form, tubular_distance
from whiteboxtopo.nulls import feature_shuffle, shuffle_predictor
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

DEV, LAYER, CTX = 'cuda', 6, 32


class Tap:
    """Add a perturbation to the last position's residual entering block LAYER."""
    def __init__(self, model):
        self.delta = None
        model.transformer.h[LAYER].register_forward_pre_hook(self.hook, with_kwargs=True)

    def hook(self, mod, args, kwargs):
        if self.delta is None:
            return None
        if args:
            h = args[0].clone(); h[:, -1, :] += self.delta; return (h,) + tuple(args[1:]), kwargs
        h = kwargs['hidden_states'].clone(); h[:, -1, :] += self.delta
        kwargs = dict(kwargs); kwargs['hidden_states'] = h
        return args, kwargs


@torch.no_grad()
def run(model, tap, ids, delta=None, bs=125):
    H, Fo, L = [], [], []
    for s in range(0, len(ids), bs):
        x = torch.as_tensor(ids[s:s + bs], device=DEV)
        tap.delta = None if delta is None else torch.as_tensor(delta[s:s + bs], device=DEV)
        o = model.transformer(x, output_hidden_states=True)                # unembed the last position only
        tap.delta = None
        H.append(o.hidden_states[LAYER][:, -1].float().cpu().numpy())
        f = o.last_hidden_state[:, -1]                                     # after ln_f
        Fo.append(f.float().cpu().numpy())
        L.append(model.lm_head(f).float().cpu())
    return np.concatenate(H), np.concatenate(Fo), torch.cat(L)


def natural_windows(tok, n, seed):
    """Windows of OpenWebText (GPT-2's own training distribution), read from the local HF cache."""
    import glob, pyarrow as pa
    f = sorted(glob.glob(os.path.expanduser('~/.cache/huggingface/datasets/stas___openwebtext-10k/default/0.0.0/*/openwebtext-10k-train.arrow')))[0]
    with pa.memory_map(f) as src:
        try:
            tb = pa.ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            tb = pa.ipc.open_file(src).read_all()
    text, tot = [], 0
    for t in tb.column('text').to_pylist():
        text.append(t); tot += len(t)
        if tot > 3_000_000:
            break
    ids = tok('\n\n'.join(text))['input_ids']
    rng = np.random.default_rng(seed)
    st = rng.choice(len(ids) - CTX - 1, size=n, replace=False)
    return np.stack([ids[s:s + CTX] for s in st]).astype(np.int64), f


def cyclic_probe(tok, names, templates):
    rows, k = [], []
    for ti, t in enumerate(templates):
        for i, nm in enumerate(names):
            rows.append(tok(t.format(nm))['input_ids']); k.append(i)
    return rows, np.array(k)


@torch.no_grad()
def last_hidden_varlen(model, rows):
    out = []
    for r in rows:
        o = model.transformer(torch.as_tensor([r], device=DEV), output_hidden_states=True)
        out.append(o.hidden_states[LAYER][0, -1].float().cpu().numpy())
    return np.stack(out).astype(np.float32)


def top_h1(X, band=None):
    res = persistence(X)
    h1 = res['dgms'][1]
    life = np.sort(h1[:, 1] - h1[:, 0])[::-1] if len(h1) else np.zeros(1)
    return dict(n_h1=int(len(h1)), top_life=float(life[0]), top5=[float(v) for v in life[:5]],
                n_above_band=int((life > band).sum()) if band is not None else None,
                rel_top_life=float(life[0] / res['median_dist']),
                tildeC=normalized_persistence(h1, res['median_dist']) if len(h1) else 0.0)


def main(n_probe=2000, n_hold=1000):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained('gpt2'); model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEV).eval()
    tap = Tap(model)
    out = dict(layer=LAYER, ctx=CTX, exploratory=True); T0 = time.time()
    ids, src = natural_windows(tok, n_probe + n_hold, 0); out['text_source'] = src
    H, Fo, Lg = run(model, tap, ids)
    WU = model.lm_head.weight.detach().float().cpu().numpy()
    _, S, Vt = np.linalg.svd(WU, full_matrices=False)
    O = (Fo @ Vt.T) * S                                   # exact isometric image of the logits
    Xp, Op, Xh = H[:n_probe], O[:n_probe], H[n_probe:]

    # ---- the manifold hypothesis, by our own instrument
    t0 = time.time()
    fit = fit_manifold(Xp, metric='secant', out=Op)
    fit15 = fit_manifold(Xp, metric='secant', out=Op, score_k=15)
    Xs = feature_shuffle(Xp, 0); fs = fit_manifold(Xs)
    out['manifold'] = dict(m=fit.m, score_k=fit.score['score_k'], m_k15=fit15.m, d_global=float(fit.d_global),
                           d_local_median=float(np.nanmedian(fit.d_local)), D=int(Xp.shape[1]),
                           n_components=int(fit.n_components()),
                           null_feature_shuffle=dict(m=fs.m, d_global=float(fs.d_global)), wall_s=time.time() - t0)
    print('[gpt2] manifold', json.dumps(out['manifold']), flush=True)
    # GPT-2's residual stream has a few enormous-variance coordinates; report what they do to geometry
    var = Xp.var(0); o3 = np.argsort(var)[::-1][:3]
    out['outlier_dims'] = dict(dims=[int(i) for i in o3], var_share=float(var[o3].sum() / var.sum()))
    keep = np.setdiff1d(np.arange(Xp.shape[1]), o3)
    fo = fit_manifold(Xp[:, keep], metric='secant', out=Op)
    out['manifold_no_outlier_dims'] = dict(m=fo.m, d_global=float(fo.d_global), score_k=fo.score['score_k'])
    print('[gpt2] outliers', json.dumps(out['outlier_dims']), json.dumps(out['manifold_no_outlier_dims']), flush=True)

    # ---- P1 on natural text: anything above the noise band?
    t0 = time.time()
    sub = np.random.default_rng(1).choice(n_probe, 1200, replace=False)
    nb = top_h1(Xs[sub]); band = nb['top_life']
    out['P1_natural'] = dict(real=top_h1(Xp[sub], band), null_feature_shuffle=nb, wall_s=time.time() - t0)
    print('[gpt2] P1 natural', json.dumps(out['P1_natural']), flush=True)

    # ---- P1 on a structured probe: weekdays (period 7), months (period 12)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
              'October', 'November', 'December']
    subj = ['We', 'They', 'I', 'The team', 'My friends', 'The students', 'Our neighbours', 'The committee', 'She', 'He',
            'The club', 'The band', 'Her family', 'The council', 'The crew', 'Both of them']
    verb = ['will meet on', 'arrived on', 'are leaving on', 'usually rest on', 'had a party on', 'go shopping on',
            'were married on', 'planned the trip for', 'start work on', 'always call on']
    tmpl_d = [f'{s} {v} {{}}' for s in subj for v in verb]
    verb_m = ['will meet in', 'arrived in', 'are leaving in', 'were married in', 'planned the trip for', 'start work in',
              'moved here in', 'always travel in', 'graduated in', 'opened the shop in']
    tmpl_m = [f'{s} {v} {{}}' for s in subj for v in verb_m]
    out['P1_cyclic'] = {}
    for nm, names, tm, per in (('weekday', days, tmpl_d, 7), ('month', months, tmpl_m, 12)):
        rows, k = cyclic_probe(tok, names, tm)
        X = last_hidden_varlen(model, rows)
        half = np.arange(len(X)) % 2 == 0                 # fit planes on even templates, read on odd
        res = {}
        for tag, XX in (('real', X), ('null_feature_shuffle', feature_shuffle(X, 0))):
            dec, fr = isotypic_decomposition(XX[half], k[half], per, n_freq=2, detect='dft', var_floor=0.0, X_eval=XX[~half])
            rr = []
            for d in dec:
                Y = np.ascontiguousarray(d['proj']).astype(np.float32)
                ang = np.arctan2(Y[:, 1], Y[:, 0]) if Y.shape[1] >= 2 else np.zeros(len(Y))
                r_plane = circular_correlation(ang, 2 * np.pi * d['freq'] * k[~half] / per)
                pr = persistence(Y); h1 = pr['dgms'][1]
                if len(h1):
                    life = np.sort(h1[:, 1] - h1[:, 0]); ratio = float(life[-1] / max(life[-2], 1e-9)) if len(life) > 1 else 99.0
                    cc = circular_coordinates(Y, h1, pr['cocycles'][1], D=pr['D'], class_idx=int(np.argmax(h1[:, 1] - h1[:, 0])))
                    r_h1 = float(circular_correlation(cc['theta'], 2 * np.pi * d['freq'] * k[~half] / per)) if cc else 0.0
                else:
                    ratio, r_h1 = 0.0, 0.0
                rr.append(dict(freq=int(d['freq']), var=float(d['var']), r_plane_angle=float(r_plane), r_h1_coordinate=r_h1, life_ratio=ratio))
            res[tag] = rr
        # unsupervised: raw Rips on the whole structured cloud
        pr = persistence(X); h1 = pr['dgms'][1]
        if len(h1):
            j = int(np.argmax(h1[:, 1] - h1[:, 0])); cc = circular_coordinates(X, h1, pr['cocycles'][1], D=pr['D'], class_idx=j)
            res['unsupervised_rips_r'] = float(circular_correlation(cc['theta'], 2 * np.pi * k / per)) if cc else 0.0
        res['n'] = int(len(X))
        out['P1_cyclic'][nm] = res
        print(f'[gpt2] P1 cyclic {nm}', json.dumps(res), flush=True)

    # ---- P4
    t0 = time.time()
    rng = np.random.default_rng(3)
    ood = dict(random_tokens=rng.integers(0, 50257, size=(n_hold, CTX)).astype(np.int64),
               shuffled_order=np.stack([rng.permutation(r) for r in ids[n_probe:]]))
    nnE = NearestNeighbors(n_neighbors=1).fit(Xp)
    Lh = Lg[n_probe:]

    def scores(Hx, Lx, idsx):
        p = torch.softmax(Lx, -1)
        tub = tubular_distance(Hx, fit.X, fit.bases)
        # spec d_g: push the normal residual through the Jacobian (one finite-difference JVP each)
        _, nid = nnE.kneighbors(Hx); nid = nid[:, 0]
        dl = Hx - fit.X[nid]; B = fit.bases[nid]
        nrm = dl - np.einsum('nDd,nd->nD', B, np.einsum('nDd,nD->nd', B, dl))
        _, F1, _ = run(model, tap, idsx, delta=(-dl + 1e-2 * nrm).astype(np.float32))   # at hbar + eps*normal
        _, F0, _ = run(model, tap, idsx, delta=(-dl).astype(np.float32))               # at hbar (same context)
        dg = np.linalg.norm(((F1 - F0) @ Vt.T) * S, axis=1) / 1e-2
        return dict(tubular=tub, spec_dg=dg, euclid_knn=nnE.kneighbors(Hx)[0][:, 0],
                    softmax=-p.max(-1).values.numpy(), entropy=(-(p * torch.log(p + 1e-12)).sum(-1)).numpy())
    s_in = scores(Xh, Lh, ids[n_probe:])
    out['P4'] = {}
    for nm, io in ood.items():
        Ho, _, Lo = run(model, tap, io); so = scores(Ho, Lo, io)
        y = np.r_[np.zeros(n_hold), np.ones(n_hold)]
        out['P4'][nm] = {f'auc_{k}': float(roc_auc_score(y, np.r_[s_in[k], so[k]])) for k in s_in}
        print(f'[gpt2] P4 {nm}', json.dumps(out['P4'][nm]), flush=True)
    out['P4']['wall_s'] = time.time() - t0

    # ---- P5
    t0 = time.time()
    n5 = 600; sel = np.arange(n_probe, n_probe + n5); base_pred = Lg[sel].argmax(-1)
    scale = float(np.median(np.linalg.norm(H, axis=1))); radii = scale * np.geomspace(0.01, 1.0, 14)
    fr = np.full(n5, np.inf); g = np.random.default_rng(11)
    for r in radii:
        flips = np.zeros(n5)
        for _ in range(6):
            z = g.standard_normal((n5, H.shape[1])).astype(np.float32); z *= r / np.linalg.norm(z, axis=1, keepdims=True)
            _, _, Lz = run(model, tap, ids[sel], delta=z)
            flips += (Lz.argmax(-1) != base_pred).numpy()
        hit = (flips >= 3) & ~np.isfinite(fr); fr[hit] = r
    fin = np.isfinite(fr)
    # rho: curvature at the held-out point's nearest probe point x a Lipschitz estimate (max ||J v|| over 6 directions)
    curv = second_fundamental_form(fit.X, k=40, dim_cap=4)
    _, nid = nnE.kneighbors(H[sel]); nid = nid[:, 0]
    lip = np.zeros(n5)
    _, F0, _ = run(model, tap, ids[sel], delta=np.zeros((n5, H.shape[1]), np.float32))
    for _ in range(6):
        v = g.standard_normal((n5, H.shape[1])).astype(np.float32); v /= np.linalg.norm(v, axis=1, keepdims=True)
        _, F1, _ = run(model, tap, ids[sel], delta=0.05 * scale * v)
        lip = np.maximum(lip, np.linalg.norm(((F1 - F0) @ Vt.T) * S, axis=1) / (0.05 * scale))
    rho = curv[nid] * lip
    z = torch.sort(Lg[sel], -1).values; sm = (z[:, -1] - z[:, -2]).numpy()
    pm = torch.softmax(Lg[sel], -1).max(-1).values.numpy()
    sp = lambda a: [float(v) for v in spearmanr(a[fin], fr[fin])]
    out['P5'] = dict(n=int(fin.sum()), n_never_flipped=int((~fin).sum()), n_distinct_radii=int(len(np.unique(fr[fin]))),
                     spearman_rho=sp(rho), spearman_curv=sp(curv[nid]), spearman_lip=sp(lip),
                     spearman_softmax_margin=sp(sm), spearman_softmax_conf=sp(pm),
                     null_shuffled_rho=sp(shuffle_predictor(rho, 0)), wall_s=time.time() - t0)
    print('[gpt2] P5', json.dumps(out['P5']), flush=True)
    out['wall_s'] = time.time() - T0
    json.dump(out, open('results/gpt2_explore.json', 'w'), indent=1, default=float)
    print(f"[gpt2] done {out['wall_s']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
