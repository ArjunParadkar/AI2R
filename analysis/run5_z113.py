"""Run 5, Z113 blocks (SPEC5.md).  usage: run5_z113.py finals | sweep <run> <seed> <mode> <frac>
Instruments are Run 4's, imported unchanged; new here: the label-permutation null for the depth slope and the FINITE tower (block T).
CPU only.  Neighbouring work: LieGG (Moskalev, Sepliarskaia, Sosnovik, Smeulders, NeurIPS 2022)."""
import sys, os, json, glob, re, time
os.environ.setdefault('WB_DEV', 'cpu')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.z113 import build_data, P, NULL, EQ
from systems.z113_probe import psi_of
from analysis.run4_z113 import load, fresh, metric_stats, group, tower as tower_linearised
from analysis.run3_blockZ import swap_q
from analysis.run3_blockP3 import slot_fn, slots_of
from analysis.run3_blockP64 import conn_slope
from analysis.run2_blockD12 import probe
from whiteboxtopo.finite_tower import finite_q

CLEAN = [('r5_z113_clean_s%d' % s, s, 'clean', .3) for s in (9, 10, 11)]
MEM = [('r5_z113_scrambled_decoy_s%d' % s, s, 'scrambled_decoy', .3) for s in (9, 10)]
DECOY = [('r5_z113_decoy_s%d' % s, s, 'decoy', .5) for s in (9, 10)]
LAYERS = ('embed', 'resid_mid', 'resid_post', 'logits')


def finite_tower(m, u, n=500, seed=0, K=16):
    rng = np.random.default_rng(seed); ab = rng.integers(0, P, (n, 2)); ab2 = rng.integers(0, P, (n, 2)); hint = rng.integers(0, P, n)
    mk = lambda x, h: np.c_[x, h, np.full(len(x), EQ)]; nul = np.full(n, NULL)
    T = dict(swap=mk(ab[:, ::-1], nul), translation=mk(np.c_[(ab[:, 0] + u) % P, (ab[:, 1] - u) % P], nul), hint_change=mk(ab, hint),
             orbit_translation=mk(np.c_[(ab[:, 0] + u) % P, ab[:, 1]], nul))
    f3 = torch.func.vmap(slot_fn(m, layers=True)); f1 = torch.func.vmap(slot_fn(m))
    def acts(tk):
        with torch.no_grad():
            s = slots_of(m, tk); mid, post, lg = f3(s)
        return dict(embed=s, resid_mid=mid, resid_post=post, logits=lg)
    psi = dict(embed=f1, resid_mid=psi_of(m, 'resid_mid'), resid_post=psi_of(m, 'resid_post'), logits=lambda h: h)
    base, other = acts(mk(ab, nul)), acts(mk(ab2, nul)); out = {}
    for name, tk in T.items():
        a = acts(tk); out[name] = {L: finite_q(psi[L], base[L], a[L], other[L], K) for L in LAYERS}
        out[name]['secant_logits'] = float((a['logits'] - base['logits']).norm(dim=1).mean() / (other['logits'] - base['logits']).norm(dim=1).mean())
    return out


def clouds_of(run, seed, mode, frac):
    d = build_data(seed, mode, frac=frac); m = load(sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))[-1], seed); tk = torch.as_tensor(d['train_toks'])
    with torch.no_grad():
        x = m.embed(tk); e = x[:, :2].reshape(len(tk), -1); mid = (x + m.attn(x))[:, -1]; post = m.resid_final(tk)
    return [t.float().numpy() for t in (e, mid, post)], d['train_y']


def finals():
    out = {}; t00 = time.time()
    for run, seed, mode, frac in CLEAN + MEM + DECOY:
        cks = sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))
        if not cks or not os.path.exists(f'runs/{run}/history.json'):
            print('[r5] missing', run); continue
        t0 = time.time(); H = json.load(open(f'runs/{run}/history.json'))['history']; h = H[-1]; m = load(cks[-1], seed); m0 = load(cks[0], seed)
        tk = probe(seed, mode, frac)[0][:500]; rec = dict(mode=mode, seed=seed, train_acc=h['train_acc'], test_acc=h['test_acc'], max_test_acc=max(x['test_acc'] for x in H))
        ms, ms0, ms1 = metric_stats(m, tk), metric_stats(m0, tk), metric_stats(fresh(seed + 100), tk)
        rec['concentration'] = dict(PR=ms['PR'], PR_untrained=ms0['PR'], C=1 - ms['PR'] / ms0['PR'], C_second_untrained=1 - ms1['PR'] / ms0['PR'])
        rec['slot_share'] = ms['slot_share']; rec['slot_share_untrained'] = ms0['slot_share']
        rec['q_swap'] = swap_q(m)['resid_mid']['q_pullback']; rec['q_swap_untrained'] = swap_q(m0)['resid_mid']['q_pullback']
        L, r, sig = group(m); rec['group'] = dict(full_cycle=bool(sig is not None), generator_step=(int(sig[0]) if sig is not None else None), letter_order=r['order'], n_accepted=r['order_rel'], R_star=r['R_star'])
        _, r0, sig0 = group(m0); rec['group_untrained'] = dict(full_cycle=bool(sig0 is not None), R_star=r0['R_star'])
        u = int(sig[0]) if sig is not None else 1
        rec['tower_finite'] = finite_tower(m, u); rec['tower_linearised'] = tower_linearised(m, u); rec['tower_u'] = u
        if mode in ('clean', 'scrambled_decoy'):
            cl, y = clouds_of(run, seed, mode, frac); c = conn_slope(cl, y); rng = np.random.default_rng(0)
            nul = [conn_slope(cl, rng.permutation(y))['slope'] for _ in range(5)]
            rec['connection'] = dict(slope=c['slope'], per_layer=c['per_layer'], null_label_perm=nul, null_mean=float(np.mean(nul)))
        rec['wall_s'] = time.time() - t0; out[run] = rec
        tf = rec['tower_finite']
        print(f"[r5] {run}: acc {h['train_acc']:.3f}/{h['test_acc']:.3f} | q_swap {rec['q_swap']:.3f} (init {rec['q_swap_untrained']:.3f}) | slots {[round(v, 3) for v in ms['slot_share']]} | cycle {rec['group']['full_cycle']} (init {rec['group_untrained']['full_cycle']}) | "
              f"slope {rec.get('connection', {}).get('slope')} null {rec.get('connection', {}).get('null_mean')} | C {rec['concentration']['C']:+.2f} | finite tower hint {[round(tf['hint_change'][L], 3) for L in LAYERS]} transl {[round(tf['translation'][L], 3) for L in LAYERS]} swap {[round(tf['swap'][L], 3) for L in LAYERS]} ({rec['wall_s']:.0f}s)", flush=True)
        json.dump(out, open('results/run5/z113_finals.json', 'w'), indent=1, default=float)
    print(f'done {time.time()-t00:.0f}s')


def sweep(run, seed, mode, frac):
    """EVERY checkpoint (SPEC5 P7).  Resumable: rows already on disk are kept."""
    hist = {h['step']: h for h in json.load(open(f'runs/{run}/history.json'))['history']}; cks = sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))
    steps = [int(re.search(r'ckpt_(\d+)', c)[1]) for c in cks]; te = np.array([hist.get(s, {}).get('test_acc') or 0 for s in steps])
    g = int(np.argmax(np.diff(te))) + 1 if te.max() > 0.5 else None
    tk = probe(seed, mode, frac)[0][:300]; path = f'results/run5/sweep_{run}.json'; rows = json.load(open(path)) if os.path.exists(path) else []; done = {r['ckpt'] for r in rows}
    for i in range(len(cks)):
        if i in done:
            continue
        t0 = time.time(); m = load(cks[i], seed); _, r, sig = group(m); ms = metric_stats(m, tk)
        rows.append(dict(ckpt=i, step=steps[i], jump_ckpt=g, test_acc=hist.get(steps[i], {}).get('test_acc'), train_acc=hist.get(steps[i], {}).get('train_acc'), full_cycle=bool(sig is not None),
                         generator_step=(int(sig[0]) if sig is not None else None), R_star=r['R_star'], q_swap=swap_q(m, n=200)['resid_mid']['q_pullback'], hint_share=ms['hint_share'], slot_share=ms['slot_share']))
        print(f"[sweep {run}] ckpt {i} step {steps[i]} test {rows[-1]['test_acc']} cycle {rows[-1]['full_cycle']} q {rows[-1]['q_swap']:.3f} hint {ms['hint_share']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        rows.sort(key=lambda r: r['ckpt']); json.dump(rows, open(path, 'w'), indent=1, default=float)
    print('done')


if __name__ == '__main__':
    os.makedirs('results/run5', exist_ok=True)
    finals() if sys.argv[1] == 'finals' else sweep(sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5]))
