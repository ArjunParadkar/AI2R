"""Run 4, Z113 blocks (SPEC4.md).  usage: run4_z113.py finals | sweep <run> <seed> <mode> <frac>
Everything on CPU (the model is tiny; the GPU is reserved for specimen training and the interrupt).
Neighbouring work: LieGG (Moskalev, Sepliarskaia, Sosnovik, Smeulders, NeurIPS 2022)."""
import sys, os, json, glob, re, time
os.environ.setdefault('WB_DEV', 'cpu')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.z113 import Z113Transformer, build_data, P, NULL, EQ
from systems.z113_probe import psi_of
from analysis.run3_blockZ import all_logits, D_g, fold, swap_q
from analysis.run3_blockP3 import slot_fn, slots_of
from analysis.run2_blockD12 import probe
from whiteboxtopo.symmetry_breaking import breaking_score
from whiteboxtopo.symmetry_finite import automorphisms, full_cycle, cycle_coordinate, characters
from whiteboxtopo.topology import isotypic_decomposition, circular_correlation
from whiteboxtopo.extract import jacobians

CLEAN = [('r4_z113_clean_s%d' % s, s, 'clean', .3) for s in (5, 6, 7, 8)]
MEM = [('r4_z113_scrambled_decoy_s%d' % s, s, 'scrambled_decoy', .3) for s in (5, 6)]
DECOY = [('r4_z113_decoy_s%d' % s, s, 'decoy', .5) for s in (5, 6)]
SHORT = [('r4_z113_shortcut_s%d' % s, s, 'shortcut', .5) for s in (5, 6)]
AONLY = [('r4_z113_aonly_s5', 5, 'aonly', .3)]


def load(ck, seed):
    m = Z113Transformer(seed=seed); m.load_state_dict(torch.load(ck, map_location='cpu')); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def fresh(seed):
    m = Z113Transformer(seed=seed).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def metric_stats(m, tk):
    r = breaking_score(slot_fn(m), slots_of(m, tk), nuisance=2, batch=50, return_metric=True)
    lam = np.linalg.eigvalsh(r['metric'].detach().double().numpy()); lam = np.clip(lam, 0, None)
    return dict(slot_share=[float(v) for v in r['slot_share']], PR=float(lam.sum() ** 2 / (lam ** 2).sum()), hint_share=float(r['slot_share'][2]))


def group(m):
    L = all_logits(m); r = automorphisms(D_g(L)); sig = full_cycle(r, 'perms_rel') if r['has_full_cycle_rel'] else None
    return L, r, sig


def tower(m, u, n=500, seed=0):
    rng = np.random.default_rng(seed); ab = rng.integers(0, P, (n, 2)); ab2 = rng.integers(0, P, (n, 2)); hint = rng.integers(0, P, n)
    mk = lambda x, h: np.c_[x, h, np.full(len(x), EQ)]; nul = np.full(n, NULL)
    T = dict(swap=mk(ab[:, ::-1], nul), translation=mk(np.c_[(ab[:, 0] + u) % P, (ab[:, 1] - u) % P], nul), hint_change=mk(ab, hint))
    f = slot_fn(m, layers=True)
    def acts(tk):
        with torch.no_grad():
            s = slots_of(m, tk); mid, post, lg = torch.func.vmap(f)(s)
        return dict(embed=s.reshape(len(tk), -1), resid_mid=mid, resid_post=post, logits=lg)
    base, other = acts(mk(ab, nul)), acts(mk(ab2, nul))
    Je = torch.cat([torch.func.vmap(torch.func.jacrev(slot_fn(m)))(slots_of(m, mk(ab, nul))[i:i + 100]).reshape(-1, P, 3 * m.d_model) for i in range(0, n, 100)])
    J = dict(embed=Je, resid_mid=torch.as_tensor(jacobians(psi_of(m, 'resid_mid'), base['resid_mid'].numpy(), device='cpu')), resid_post=m.W_U.T[None].expand(n, -1, -1))
    out = {}
    for name, tk in T.items():
        a = acts(tk); out[name] = {}
        for L in ('embed', 'resid_mid', 'resid_post', 'logits'):
            if L == 'logits':
                q = float((a[L] - base[L]).norm(dim=1).mean() / (other[L] - base[L]).norm(dim=1).mean())
            else:
                lin = lambda d: torch.einsum('nod,nd->no', J[L].float(), d.float()).norm(dim=1)
                q = float(lin(a[L] - base[L]).mean() / lin(other[L] - base[L]).mean())
            out[name][L] = q
        out[name]['first_null_layer'] = next((i for i, L in enumerate(('embed', 'resid_mid', 'resid_post', 'logits')) if out[name][L] < 0.05), None)
    return out


def finals():
    from analysis.run3_blockP64 import z113 as conn_z113
    out = {}; t00 = time.time()
    for run, seed, mode, frac in CLEAN + MEM + DECOY + SHORT + AONLY:
        cks = sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))
        if not cks or not os.path.exists(f'runs/{run}/history.json'):
            print('[r4] missing', run); continue
        t0 = time.time(); h = json.load(open(f'runs/{run}/history.json'))['history'][-1]; m = load(cks[-1], seed); m0 = load(cks[0], seed)
        tk = probe(seed, mode, frac)[0][:500]; rec = dict(mode=mode, train_acc=h['train_acc'], test_acc=h['test_acc'])
        ms, ms0 = metric_stats(m, tk), metric_stats(m0, tk); ms1 = metric_stats(fresh(seed + 100), tk)
        rec['P6_1'] = dict(PR=ms['PR'], PR_untrained=ms0['PR'], C=1 - ms['PR'] / ms0['PR'], C_second_untrained=1 - ms1['PR'] / ms0['PR'])
        rec['slot_share'] = ms['slot_share']; rec['slot_share_untrained'] = ms0['slot_share']
        rec['q_swap'] = swap_q(m)['resid_mid']['q_pullback']
        L, r, sig = group(m); rec['group'] = dict(full_cycle=bool(sig is not None), letter_order=r['order'], letter_cycle=r['has_full_cycle'], n_accepted=r['order_rel'], R_star=r['R_star'])
        pred = L.argmax(-1)
        if sig is not None and mode in ('clean', 'decoy'):
            t = cycle_coordinate(sig); ch = characters(m.W_E[:P].numpy(), t, top=5); kp = [c['freq'] for c in ch]; a = np.arange(P)
            with torch.no_grad():
                x = m.embed(torch.as_tensor(np.stack([a, np.zeros(P, int), np.full(P, NULL), np.full(P, EQ)], 1))); Xs = (x + m.attn(x))[:, -1, :].numpy()
            K = [fold(k) for k in isotypic_decomposition(Xs, a, P, n_freq=5, detect='dft', var_floor=0.0)[1]]
            per_u = {u: sum(fold(u * k) in K for k in kp) for u in range(1, 57)}; bu = max(per_u, key=per_u.get); rs = []
            for c in ch:
                k = fold(bu * c['freq'])
                if k in K:
                    rs.append(float(max(circular_correlation(c['angle'], 2 * np.pi * k * a / P), circular_correlation(c['angle'], -2 * np.pi * k * a / P))))
            rec['characters'] = dict(generator_step=int(sig[0]), discovered=kp, inventory=K, unit=int(bu), unit_implied=fold(pow(int(sig[0]), -1, P)), n_matched=int(per_u[bu]),
                                     n_units_4plus=int(sum(v >= 4 for v in per_u.values())), r=rs, n_r09=int(sum(v > 0.9 for v in rs)))
        sg = sig if sig is not None else (np.arange(P) + 1) % P
        pw, pi, inv = [np.arange(P)], [np.arange(P)], np.argsort(sg)
        for _ in range(P - 1):
            pw.append(sg[pw[-1]]); pi.append(inv[pi[-1]])
        pw, pi = np.array(pw), np.array(pi); ab = np.random.default_rng(0).integers(0, P, (2000, 2)); orb = pred[pw[:, ab[:, 0]], pi[:, ab[:, 1]]]
        rec['orbit'] = dict(frac_both=float(((orb == orb[0]).all(0) & (pred[ab[:, 0], ab[:, 1]] == pred[ab[:, 1], ab[:, 0]])).mean()), generator_supplied=bool(sig is None))
        rec['tower'] = tower(m, int(sg[0]) if mode != 'aonly' else 1)
        if mode in ('clean', 'scrambled_decoy'):
            c = conn_z113(run, seed, mode); rec['connection'] = c['connection']
        rec['wall_s'] = time.time() - t0; out[run] = rec
        print(f"[r4] {run}: acc {h['train_acc']:.3f}/{h['test_acc']:.3f} | C {rec['P6_1']['C']:+.3f} (PR {ms['PR']:.1f} vs untrained {ms0['PR']:.1f}) | slots {[round(v, 3) for v in ms['slot_share']]} | q_swap {rec['q_swap']:.3f} | "
              f"cycle {rec['group']['full_cycle']} chars {rec.get('characters', {}).get('n_r09')} | orbit {rec['orbit']['frac_both']:.3f} | conn {rec.get('connection', {}).get('slope')} | "
              f"tower first-null {{{', '.join(k + ':' + str(v['first_null_layer']) for k, v in rec['tower'].items())}}} ({rec['wall_s']:.0f}s)", flush=True)
        json.dump(out, open('results/run4/z113_finals.json', 'w'), indent=1, default=float)
    print(f'wall {time.time()-t00:.0f}s')


def sweep(run, seed, mode, frac):
    hist = {h['step']: h for h in json.load(open(f'runs/{run}/history.json'))['history']}; cks = sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))
    steps = [int(re.search(r'ckpt_(\d+)', c)[1]) for c in cks]; te = np.array([hist.get(s, {}).get('test_acc') or 0 for s in steps])
    g = int(np.argmax(np.diff(te))) + 1 if te.max() > 0.5 else None
    keep = [i for i in range(len(cks)) if i % 3 == 0 or i == len(cks) - 1 or (g is not None and abs(i - g) <= 4)]; tk = probe(seed, mode, frac)[0][:300]; rows = []
    for i in keep:
        t0 = time.time(); m = load(cks[i], seed); _, r, sig = group(m); ms = metric_stats(m, tk)
        rows.append(dict(ckpt=i, step=steps[i], grok_ckpt=g, test_acc=hist.get(steps[i], {}).get('test_acc'), train_acc=hist.get(steps[i], {}).get('train_acc'), full_cycle=bool(sig is not None),
                         letter_order=r['order'], q_swap=swap_q(m, n=200)['resid_mid']['q_pullback'], hint_share=ms['hint_share'], slot_share=ms['slot_share']))
        print(f"[sweep {run}] ckpt {i} step {steps[i]} test {rows[-1]['test_acc']} cycle {rows[-1]['full_cycle']} q {rows[-1]['q_swap']:.3f} hint {ms['hint_share']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(rows, open(f'results/run4/sweep_{run}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    os.makedirs('results/run4', exist_ok=True)
    finals() if sys.argv[1] == 'finals' else sweep(sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5]))
