"""Run 3, Z113 blocks (SPEC3.md): P1.1 group discovery + characters, P1.2 memoriser, P1.5 / P7.2 swap quotient depth, P2.1 orbit
explanation, and the checkpoint sweeps behind P6.3 / P7.3.   usage:  run3_blockZ.py finals | sweep <run> <seed> <mode> <frac>
Neighbouring work: LieGG (Moskalev et al., NeurIPS 2022)."""
import sys, os, json, time, glob, re
import numpy as np, torch
from scipy.spatial.distance import pdist, squareform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.z113 import Z113Transformer, P, NULL, EQ
from systems.z113_probe import psi_of
from whiteboxtopo.symmetry_finite import automorphisms, full_cycle, cycle_coordinate, characters, element_order
from whiteboxtopo.topology import isotypic_decomposition, circular_correlation
from whiteboxtopo.extract import jacobians
DEV = os.environ.get('WB_DEV', 'cpu')          # the GPU is shared with training jobs; this model is tiny
fold = lambda k: int(min(k % P, P - k % P))


def load(ck, seed):
    m = Z113Transformer(seed=seed).to(DEV); m.load_state_dict(torch.load(ck, map_location=DEV)); m.eval(); return m


@torch.no_grad()
def all_logits(m):
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing='ij')
    tk = np.stack([a.ravel(), b.ravel(), np.full(P * P, NULL), np.full(P * P, EQ)], 1)
    return m(torch.as_tensor(tk, device=DEV)).float().cpu().numpy().reshape(P, P, P)             # [a, b, :]


def D_g(L):
    return squareform(pdist(L.reshape(P, -1))) / np.sqrt(P)                                        # sqrt(mean_b ||logits(i,b)-logits(j,b)||^2)


def summarise(r):
    return {k: r[k] for k in ('order', 'has_full_cycle', 'has_involution', 'max_element_order', 'degenerate', 'truncated', 'closure', 'spread', 'T', 'median', 'nn_median', 'order_mean', 'has_full_cycle_mean', 'delta_star', 'order_rel', 'has_full_cycle_rel', 'R_star', 'delta_rand') if k in r}


def swap_q(m, n=1000, seed=0):
    """q_l(swap): linearised pullback (primary) and Euclidean, at embed, resid_mid, resid_post, logits."""
    rng = np.random.default_rng(seed); ab = rng.integers(0, P, (n, 2)); ab2 = rng.integers(0, P, (n, 2))
    mk = lambda x: torch.as_tensor(np.c_[x, np.full(len(x), NULL), np.full(len(x), EQ)], device=DEV)
    rest = lambda: (m.W_E[torch.tensor([NULL, EQ], device=DEV)] + m.W_pos[2:4]).detach()
    def from_slots(s):                                         # s: (2, d) embedded a,b slots -> logits
        x = torch.cat([s, rest()], 0)[None]; x = x + m.attn(x); x = x + m.mlp(x); return (x[:, -1, :] @ m.W_U)[0]
    out = {}
    with torch.no_grad():
        E = lambda t: m.embed(t)[:, :2, :]
        mid = lambda t: (lambda x: (x + m.attn(x))[:, -1, :])(m.embed(t))
        post = lambda t: m.resid_final(t)
        H = {'embed': (E(mk(ab)).reshape(n, -1), E(mk(ab[:, ::-1].copy())).reshape(n, -1), E(mk(ab2)).reshape(n, -1)),
             'resid_mid': (mid(mk(ab)), mid(mk(ab[:, ::-1].copy())), mid(mk(ab2))),
             'resid_post': (post(mk(ab)), post(mk(ab[:, ::-1].copy())), post(mk(ab2)))}
        lg = [m(mk(x)) for x in (ab, ab[:, ::-1].copy(), ab2)]
    J = {'embed': torch.cat([torch.func.vmap(torch.func.jacrev(from_slots))(H['embed'][0][i:i + 250].reshape(-1, 2, m.d_model)).reshape(-1, P, 2 * m.d_model) for i in range(0, n, 250)]).detach(),
         'resid_mid': torch.as_tensor(jacobians(psi_of(m, 'resid_mid'), H['resid_mid'][0].cpu().numpy(), device=DEV), device=DEV),
         'resid_post': m.W_U.T.detach()[None].expand(n, -1, -1)}
    for L, (h, hT, h2) in H.items():
        lin = lambda d: torch.einsum('nod,nd->no', J[L].float(), d.float()).norm(dim=1)
        out[L] = dict(q_pullback=float(lin(hT - h).mean() / lin(h2 - h).mean()), q_euclid=float((hT - h).norm(dim=1).mean() / (h2 - h).norm(dim=1).mean()))
    out['logits'] = dict(q_pullback=float((lg[1] - lg[0]).norm(dim=1).mean() / (lg[2] - lg[0]).norm(dim=1).mean()))
    out['logits']['q_euclid'] = out['logits']['q_pullback']
    order = ['embed', 'resid_mid', 'resid_post', 'logits']
    out['invariance_depth_pullback'] = next((L for L in order if out[L]['q_pullback'] < 0.05), None)
    out['invariance_depth_euclid'] = next((L for L in order if out[L]['q_euclid'] < 0.05), None)
    return out


def finals():
    out = {}; t00 = time.time()
    models = [('z113_clean_s0', 0, 'grokked'), ('z113_clean_s1', 1, 'grokked'), ('z113_cleanctrl_s2', 2, 'grokked'), ('z113_decoy_s0', 0, 'grokked'),
              ('z113_decoy_s1', 1, 'grokked'), ('z113_scrambled_s0', 0, 'memoriser')]
    for run, seed, kind in models:
        t0 = time.time(); m = load(sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))[-1], seed); L = all_logits(m); pred = L.argmax(-1)
        rg = automorphisms(D_g(L)); WE = m.W_E[:P].detach().float().cpu().numpy(); re_ = automorphisms(squareform(pdist(WE)))
        rec = dict(kind=kind, acc_all_pairs=float((pred == (np.arange(P)[:, None] + np.arange(P)[None]) % P).mean()), group_g=summarise(rg), group_euclid=summarise(re_))
        sig = full_cycle(rg); rec['generator_from'] = 'letter (max-norm)'
        if sig is None and rg.get('has_full_cycle_rel'):
            sig = full_cycle(rg, 'perms_rel'); rec['generator_from'] = 'POST-HOC relative mean-norm (A-10)'
        if sig is not None:
            t = cycle_coordinate(sig); ch = characters(WE, t, top=5); kp = [c['freq'] for c in ch]
            a = np.arange(P); sl = np.stack([a, np.zeros(P, int), np.full(P, NULL), np.full(P, EQ)], 1)
            with torch.no_grad():
                x = m.embed(torch.as_tensor(sl, device=DEV)); Xs = (x + m.attn(x))[:, -1, :].cpu().numpy()
            K = [fold(k) for k in isotypic_decomposition(Xs, a, P, n_freq=5, detect='dft', var_floor=0.0)[1]]
            per_u = {u: sum(fold(u * k) in K for k in kp) for u in range(1, 57)}; best_u = max(per_u, key=per_u.get)
            matched = []
            for c in ch:
                k = fold(best_u * c['freq'])
                if k in K:
                    matched.append(dict(discovered=c['freq'], natural=k, r=float(max(circular_correlation(c['angle'], 2 * np.pi * k * a / P), circular_correlation(c['angle'], -2 * np.pi * k * a / P)))))
            s = int(sig[0]); u_implied = fold(pow(s, -1, P))
            rec['characters'] = dict(generator_moves_0_to=s, discovered_freqs=kp, power_share=[c['power_share'] for c in ch], inventory_K=K, best_unit=int(best_u),
                                     n_matched=int(per_u[best_u]), n_units_with_4plus=int(sum(v >= 4 for v in per_u.values())), unit_implied_by_generator=u_implied,
                                     matched=matched, n_matched_r09=int(sum(x['r'] > 0.9 for x in matched)))
            sg = sig
        else:
            sg = (np.arange(P) + 1) % P                       # nothing discovered: the true +1 translation is SUPPLIED (memoriser only, per SPEC3 P2.1)
        rec['generator_supplied'] = bool(sig is None)
        # P2.1 orbit explanation
        pw = [np.arange(P)]; inv = np.argsort(sg); pi = [np.arange(P)]
        for _ in range(P - 1):
            pw.append(sg[pw[-1]]); pi.append(inv[pi[-1]])
        pw, pi = np.array(pw), np.array(pi); rng = np.random.default_rng(0); ab = rng.integers(0, P, (2000, 2))
        orb = pred[pw[:, ab[:, 0]], pi[:, ab[:, 1]]]                                  # (113, 2000)
        const = (orb == orb[0]).all(0); coset = pred[ab[:, 0], ab[:, 1]] == pred[ab[:, 1], ab[:, 0]]
        rec['P2_1'] = dict(frac_orbit_constant=float(const.mean()), frac_coset_constant=float(coset.mean()), frac_both=float((const & coset).mean()))
        rec['swap_q'] = swap_q(m); rec['wall_s'] = time.time() - t0; out[run] = rec
        print(f"[Z] {run}: g-group order {rg['order']} cycle {rg['has_full_cycle']} | POST-HOC rel order {rg['order_rel']} cycle {rg['has_full_cycle_rel']} R* {rg['R_star']} [{rec['generator_from']}] (euclid {re_['order']}, cycle {re_['has_full_cycle']}) | chars "
              f"{rec.get('characters', {}).get('n_matched')}/5 matched, {rec.get('characters', {}).get('n_matched_r09')} at r>0.9, units>=4: {rec.get('characters', {}).get('n_units_with_4plus')} | "
              f"P2.1 both {rec['P2_1']['frac_both']:.3f} | q_mid {rec['swap_q']['resid_mid']['q_pullback']:.4f} (E {rec['swap_q']['resid_mid']['q_euclid']:.4f}) depth {rec['swap_q']['invariance_depth_pullback']} ({rec['wall_s']:.0f}s)", flush=True)
    g5 = [r for r in out if out[r]['kind'] == 'grokked']
    v = dict(P1_1_group=sum(out[r]['group_g']['order'] >= P and out[r]['group_g']['has_full_cycle'] for r in g5),
             P1_1_freqs=sum(out[r].get('characters', {}).get('n_matched_r09', 0) >= 4 for r in g5),
             P1_2_memoriser_order=out['z113_scrambled_s0']['group_g']['order'],
             P1_5=sum(out[r]['swap_q']['resid_mid']['q_pullback'] < 0.05 for r in g5),
             P2_1_grokked=sum(out[r]['P2_1']['frac_both'] >= 0.99 for r in g5), P2_1_memoriser=out['z113_scrambled_s0']['P2_1']['frac_both'])
    v['PASS_P1_1_group'] = bool(v['P1_1_group'] >= 4); v['PASS_P1_1_freqs'] = bool(v['P1_1_freqs'] >= 4); v['PASS_P1_2'] = bool(v['P1_2_memoriser_order'] <= 2)
    v['PASS_P1_5'] = bool(v['P1_5'] == 5); v['PASS_P2_1'] = bool(v['P2_1_grokked'] == 5 and v['P2_1_memoriser'] < 0.60)
    out['_verdict'] = v; print(json.dumps(v)); print(f'wall {time.time()-t00:.0f}s')
    json.dump(out, open('results/run3/blockZ_finals.json', 'w'), indent=1, default=float)


def sweep(run, seed):
    hist = {h['step']: h for h in json.load(open(f'runs/{run}/history.json'))['history']}; rows = []
    cks = sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))
    if os.environ.get('WB_SPARSE'):                          # A-11 (time budget): all checkpoints within +-4 of grokking, every 5th elsewhere
        steps = [int(re.search(r'ckpt_(\d+)', c)[1]) for c in cks]; te = np.array([hist.get(st, {}).get('test_acc') or 0 for st in steps])
        g = int(np.argmax(np.diff(te))) + 1 if te.max() > 0.5 else None
        keep = [i for i in range(len(cks)) if i % 5 == 0 or i == len(cks) - 1 or (g is not None and abs(i - g) <= 4)]; cks = [cks[i] for i in keep]
    for ck in cks:
        t0 = time.time(); step = int(re.search(r'ckpt_(\d+)', ck)[1]); m = load(ck, seed); L = all_logits(m); r = automorphisms(D_g(L))
        q = swap_q(m, n=200)['resid_mid']
        rows.append(dict(step=step, test_acc=hist.get(step, {}).get('test_acc'), train_acc=hist.get(step, {}).get('train_acc'), q_mid_pullback=q['q_pullback'], q_mid_euclid=q['q_euclid'], **summarise(r)))
        print(f"[sweep {run}] {step:6d} test {rows[-1]['test_acc']} order {r['order']} cycle {r['has_full_cycle']} | rel {r['order_rel']} cyc {r['has_full_cycle_rel']} R* {r['R_star']} | degen {r['degenerate']} q {q['q_pullback']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(rows, open(f'results/run3/sweep_{run}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    os.makedirs('results/run3', exist_ok=True)
    finals() if sys.argv[1] == 'finals' else sweep(sys.argv[2], int(sys.argv[3]))
