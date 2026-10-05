"""Run 6, Z113 (SPEC6.md): P6 (Run-5 instruments + re-registered P6.d + null-subtracted P6.c), K (orbit coherence, unknown sets), F (equivariance
defect vs failure radius), U.  usage: run6_z113.py finals | sweep <run> <seed> <mode> <frac>.  CPU except the activation-space failure radius.
Cited: Golan & El-Yaniv 2018; Hendrycks & Gimpel 2017; Liang et al. 2018; LieGG (Moskalev et al. 2022)."""
import sys, os, json, glob, time
os.environ.setdefault('WB_DEV', 'cpu')
import numpy as np, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.z113 import build_data, P, NULL, EQ, all_pairs
from analysis.run4_z113 import load, fresh, metric_stats, group
from analysis.run5_z113 import clouds_of, finite_tower, sweep as sweep5
from analysis.run3_blockZ import swap_q
from analysis.run3_blockP64 import conn_slope
from analysis.run2_blockD12 import probe
from whiteboxtopo.knowledge import orbit_stats, z113_group

CLEAN = [('r6_z113_clean_s%d' % s, s, 'clean', .3) for s in (12, 13, 14)]
MEM = [('r6_z113_scrambled_decoy_s%d' % s, s, 'scrambled_decoy', .3) for s in (15, 16, 17)]
DECOY = [('r6_z113_decoy_s%d' % s, s, 'decoy', .5) for s in (12, 13)]
auroc = lambda pos, neg: float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))   # score high = "known"


def softmax_scores(m, tk):
    with torch.no_grad():
        p = torch.softmax(m(torch.as_tensor(tk)).float(), 1)
    return dict(msp=p.max(1).values.numpy(), neg_entropy=(p * p.clamp_min(1e-12).log()).sum(1).numpy(), margin=(p.topk(2, 1).values[:, 0] - p.topk(2, 1).values[:, 1]).numpy())


def failure_radius_mid(m, tk):
    """Run 1's activation-space bisection at resid_mid (analysis/z113_static.failure_radius_activation), run on cuda if available."""
    from analysis.z113_static import failure_radius_activation
    from systems.z113_probe import acts
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = acts(m, tk, 'resid_mid', device='cpu')
    if dev == 'cuda':
        mg = type(m)(seed=m._seed).to(dev); mg.load_state_dict({k: v.to(dev) for k, v in m.state_dict().items()}); mg.eval()
        for p in mg.parameters(): p.requires_grad_(False)
        r, scale = failure_radius_activation(mg, X); del mg; torch.cuda.empty_cache()
    else:
        raise RuntimeError('failure radius needs cuda')
    return r


def knowledge_and_failure(m, seed, mode, frac, u, rng):
    d = build_data(seed, mode, frac=frac); te = d['test_toks'].copy(); te[:, 2] = NULL; sel = rng.choice(len(te), 1000, replace=False); tin = te[sel]
    a, b, _ = all_pairs(); G = z113_group(u); Gv = z113_group(u, valid_slot_only=True)
    st = lambda tk, grp: orbit_stats(lambda t: m(t), torch.as_tensor(tk), grp)
    K_in = st(tin, G); K_in_v = st(tin, Gv); soft_in = softmax_scores(m, tin)
    # unknown sets
    oov = {}
    for nm, tok in (('NULL', NULL), ('EQ', EQ)):
        t = tin.copy(); t[:, 0] = tok; oov[nm] = t
    oov_tk = np.concatenate([oov['NULL'], oov['EQ']]); K_oov = st(oov_tk, Gv); soft_oov = softmax_scores(m, oov_tk)
    # continuous OOV: random vector at the embedding norm in slot a (needs an embedding-level forward)
    Wn = float(m.W_E[:P].norm(dim=1).mean()); tcont = tin[:1000]
    def logits_with_random_slot(tk, vecs, grp_on_b):
        # generic: build embeddings, replace slot 0 by vecs, run attn+mlp+head as the model does
        with torch.no_grad():
            x = m.embed(torch.as_tensor(tk)); x = x.clone(); x[:, 0] = vecs + m.W_pos[0]; x = x + m.attn(x); h = x[:, -1]; h = h + m.mlp(h); return m.head(h).float()
    vecs = torch.randn(len(tcont), m.W_E.shape[1]); vecs = vecs / vecs.norm(dim=1, keepdim=True) * Wn
    base = logits_with_random_slot(tcont, vecs, None); hard, soft, dev = [], [], []
    for T, R in Gv:
        tk2 = T(torch.as_tensor(tcont)).numpy(); swapped = (tk2[:, 1] == tcont[:, 0]).all()
        if swapped:      # swap moves the random vector to slot 1
            with torch.no_grad():
                x = m.embed(torch.as_tensor(tk2)).clone(); x[:, 1] = vecs + m.W_pos[1]; x[:, 0] = m.W_E[torch.as_tensor(tcont[:, 1])] + m.W_pos[0]; x = x + m.attn(x); h = x[:, -1]; h = h + m.mlp(h); fg = m.head(h).float()
        else:
            fg = logits_with_random_slot(tk2, vecs, None)
        ref = R(base); hard.append((fg.argmax(1) == ref.argmax(1)).float()); soft.append(torch.nn.functional.cosine_similarity(fg, ref, dim=1)); dev.append((fg - ref).norm(dim=1) / base.norm(dim=1))
    K_cont = dict(kappa=torch.stack(hard).mean(0).numpy(), kappa_s=torch.stack(soft).mean(0).numpy(), eps=torch.stack(dev).max(0).values.numpy())
    pc = torch.softmax(base, 1); soft_cont = dict(msp=pc.max(1).values.numpy(), neg_entropy=(pc * pc.clamp_min(1e-12).log()).sum(1).numpy())
    # p' = 97: pairs with a, b < 97, hint NULL, not in the training set
    trn = set(map(tuple, d['train_toks'][:, :2].tolist())); cand = [(i, j) for i in range(97) for j in range(97) if (i, j) not in trn]; c97 = np.array(cand)[rng.choice(len(cand), 2000, replace=False)]
    t97 = np.c_[c97, np.full(2000, NULL), np.full(2000, EQ)]; K_97 = st(t97, G); soft_97 = softmax_scores(m, t97)
    K = dict(n_in=len(tin), in_kappa_median=float(np.median(K_in['kappa'])), in_kappa_mean=float(K_in['kappa'].mean()), in_kappa_s_mean=float(K_in['kappa_s'].mean()),
             oov=dict(auroc_kappa=auroc(K_in_v['kappa'], K_oov['kappa']), auroc_kappa_s=auroc(K_in_v['kappa_s'], K_oov['kappa_s']), auroc_msp=auroc(soft_in['msp'], soft_oov['msp']), auroc_entropy=auroc(soft_in['neg_entropy'], soft_oov['neg_entropy']),
                      unknown_kappa_median=float(np.median(K_oov['kappa'])), unknown_kappa_mean=float(K_oov['kappa'].mean()), n=len(oov_tk), note='slot a = NULL or EQ token; kappa on the valid-slot orbit (|G| = 3) for both in and unknown'),
             oov_continuous=dict(auroc_kappa=auroc(K_in_v['kappa'], K_cont['kappa']), auroc_kappa_s=auroc(K_in_v['kappa_s'], K_cont['kappa_s']), auroc_msp=auroc(soft_in['msp'], soft_cont['msp']), auroc_entropy=auroc(soft_in['neg_entropy'], soft_cont['neg_entropy']),
                                 unknown_kappa_mean=float(K_cont['kappa'].mean()), n=len(tcont), note='slot a = random vector at the embedding norm'),
             p97=dict(auroc_kappa=auroc(K_in['kappa'], K_97['kappa']), auroc_kappa_s=auroc(K_in['kappa_s'], K_97['kappa_s']), auroc_msp=auroc(soft_in['msp'], soft_97['msp']), auroc_entropy=auroc(soft_in['neg_entropy'], soft_97['neg_entropy']),
                      unknown_kappa_mean=float(K_97['kappa'].mean()), n=len(t97), note='a, b < 97, unseen pairs: a subset of valid inputs (SPEC6 §2)'))
    # lowest-kappa in-distribution pairs
    order = np.argsort(K_in['kappa_s'])[:12]; K['lowest_kappa_in'] = [dict(a=int(tin[i, 0]), b=int(tin[i, 1]), kappa=float(K_in['kappa'][i]), kappa_s=float(K_in['kappa_s'][i]), correct=bool(m(torch.as_tensor(tin[i:i + 1])).argmax(1).item() == (tin[i, 0] + tin[i, 1]) % P)) for i in order]
    # memoriser control / clean: kappa on own training pairs
    trk = d['train_toks'].copy(); trk[:, 2] = NULL; trs = trk[rng.choice(len(trk), min(1000, len(trk)), replace=False)]; K_tr = st(trs, G)
    K['train_kappa_median'] = float(np.median(K_tr['kappa'])); K['train_kappa_mean'] = float(K_tr['kappa'].mean())
    # F: failure radius at resid_mid vs eps, softmax margin, nulls
    r = failure_radius_mid(m, tin); fin = np.isfinite(r); Fd = dict(n_finite=int(fin.sum()))
    if fin.sum() > 30:
        sp = lambda v: float(spearmanr(v[fin], r[fin])[0])
        Fd.update(spearman_eps=sp(K_in['eps']), spearman_eps_bar=sp(K_in['eps_bar']), spearman_margin=sp(soft_in['margin']), spearman_msp=sp(soft_in['msp']),
                  spearman_shuffled_eps=sp(rng.permutation(K_in['eps'])), spearman_neg_kappa_s=sp(-K_in['kappa_s']))
        # non-group null: 7 random single-slot token substitutions, rho = id
        Gr = []
        for k in range(7):
            sl, tok = int(rng.integers(0, 2)), int(rng.integers(0, P))
            def T(x, sl=sl, tok=tok):
                y = x.clone(); y[:, sl] = (y[:, sl] + tok) % P; return y
            Gr.append((T, lambda l: l))
        Kr = st(tin, Gr); Fd['spearman_eps_random_transforms'] = sp(Kr['eps']); Fd['eps_random_mean'] = float(Kr['eps'].mean())
    U = dict(spearman_kappa_s_eps_bar_pooled=float(spearmanr(np.r_[K_in['kappa_s'], K_oov['kappa_s'][:1000], K_97['kappa_s'][:1000]], np.r_[K_in['eps_bar'], K_oov['eps_bar'][:1000], K_97['eps_bar'][:1000]])[0]),
             spearman_kappa_s_eps_bar_in=float(spearmanr(K_in['kappa_s'], K_in['eps_bar'])[0]) if K_in['kappa_s'].std() > 0 else None, kappa_iqr_in=float(np.subtract(*np.percentile(K_in['kappa'], [75, 25]))), kappa_iqr_train=float(np.subtract(*np.percentile(K_tr['kappa'], [75, 25]))))
    raw = dict(kappa_in=K_in['kappa'].tolist(), kappa_s_in=K_in['kappa_s'].tolist(), eps_in=K_in['eps'].tolist(), eps_bar_in=K_in['eps_bar'].tolist(), radius=r.tolist(), margin=soft_in['margin'].tolist(),
               kappa_oov=K_oov['kappa'].tolist(), kappa_s_oov=K_oov['kappa_s'].tolist(), kappa_97=K_97['kappa'].tolist(), kappa_s_97=K_97['kappa_s'].tolist(), kappa_cont=K_cont['kappa'].tolist(), eps_bar_oov=K_oov['eps_bar'].tolist(), eps_bar_97=K_97['eps_bar'].tolist())
    return K, Fd, U, raw


def finals():
    out = {}; t00 = time.time()
    for run, seed, mode, frac in CLEAN + MEM + DECOY:
        cks = sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))
        if not cks or not os.path.exists(f'runs/{run}/history.json'):
            print('[r6] missing', run); continue
        t0 = time.time(); H = json.load(open(f'runs/{run}/history.json'))['history']; h = H[-1]; m = load(cks[-1], seed); m._seed = seed; m0 = load(cks[0], seed); m0._seed = seed
        tk = probe(seed, mode, frac)[0][:500]; rec = dict(mode=mode, seed=seed, train_acc=h['train_acc'], test_acc=h['test_acc'])
        ms, ms0 = metric_stats(m, tk), metric_stats(m0, tk); rec['slot_share'] = ms['slot_share']; rec['slot_share_untrained'] = ms0['slot_share']
        rec['q_swap'] = swap_q(m)['resid_mid']['q_pullback']; rec['q_swap_untrained'] = swap_q(m0)['resid_mid']['q_pullback']
        L, r, sig = group(m); rec['group'] = dict(full_cycle=bool(sig is not None), generator_step=(int(sig[0]) if sig is not None else None), R_star=r['R_star']); _, r0, sig0 = group(m0); rec['group_untrained'] = dict(full_cycle=bool(sig0 is not None))
        u = int(sig[0]) if sig is not None else 1; rec['u'] = u
        cl, y = clouds_of(run, seed, mode, frac); c = conn_slope(cl, y); rng = np.random.default_rng(0); nul = [conn_slope(cl, rng.permutation(y))['slope'] for _ in range(5)]
        rec['connection'] = dict(slope=c['slope'], per_layer=c['per_layer'], null_label_perm=nul, null_mean=float(np.mean(nul)), delta=c['slope'] - float(np.mean(nul)))
        rng = np.random.default_rng(1); K, Fd, U, raw = knowledge_and_failure(m, seed, mode, frac, u, rng); rec.update(K=K, F=Fd, U=U)
        rec['wall_s'] = time.time() - t0; out[run] = rec; json.dump(raw, open(f'results/run6/raw_{run}.json', 'w'))
        print(f"[r6] {run}: acc {h['train_acc']:.3f}/{h['test_acc']:.3f} | q {rec['q_swap']:.3f} slots {[round(v, 3) for v in ms['slot_share']]} cycle {rec['group']['full_cycle']} Δ {rec['connection']['delta']:+.3f} | "
              f"κ in {K['in_kappa_median']:.2f} train {K['train_kappa_median']:.2f} | AUROC κ: oov {K['oov']['auroc_kappa']:.3f} cont {K['oov_continuous']['auroc_kappa']:.3f} p97 {K['p97']['auroc_kappa']:.3f} (msp {K['oov']['auroc_msp']:.3f}/{K['p97']['auroc_msp']:.3f}) | "
              f"F ρ(ε,r) {Fd.get('spearman_eps')} margin {Fd.get('spearman_margin')} rand {Fd.get('spearman_eps_random_transforms')} | U {U['spearman_kappa_s_eps_bar_pooled']:+.2f} ({rec['wall_s']:.0f}s)", flush=True)
        json.dump(out, open('results/run6/z113_finals.json', 'w'), indent=1, default=float)
    print(f'done {time.time()-t00:.0f}s')


if __name__ == '__main__':
    os.makedirs('results/run6', exist_ok=True)
    if sys.argv[1] == 'finals':
        finals()
    else:
        # every-checkpoint sweep, Run-5 code, Run-6 results dir
        run, seed, mode, frac = sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5])
        src = 'results/run5/sweep_' + run + '.json'
        sweep5(run, seed, mode, frac)      # writes results/run5/sweep_<run>.json
        os.replace(src, f'results/run6/sweep_{run}.json')
