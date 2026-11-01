"""Run 2, Block E1/E2 (SPEC2.md).  E1: rank the 30 longest-lived H^1 classes of the pullback-geodesic cloud by output
sensitivity s_c = E||J v_c||^2 and name the top 5 WITHOUT the group parameter.  E2: classes above the noise band under the
Euclidean vs the pullback-geodesic metric."""
import sys, os, json, glob, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.problems.core import geodesic_distance_matrix, _coordinate_tangent
from whiteboxtopo.topology import persistence, circular_coordinates, name_coordinate, isotypic_decomposition
from whiteboxtopo.extract import jacobians
from whiteboxtopo.nulls import feature_shuffle


def lifetimes(h1):
    l = h1[:, 1] - h1[:, 0]
    return np.where(np.isfinite(l), l, np.nanmax(l[np.isfinite(l)], initial=0.0)) if len(h1) else np.zeros(0)


def e1(X, J, variables, period, key_freqs, tag):
    t0 = time.time()
    fit = fit_manifold(X, J, metric='g'); res = persistence(geodesic_distance_matrix(fit), distance_matrix=True)
    h1 = res['dgms'][1]; life = lifetimes(h1); cand = np.argsort(-life)[:30]; rows = []
    for ci in cand:
        cc = circular_coordinates(None, h1, res['cocycles'][1], D=res['D'], class_idx=int(ci))
        if cc is None:
            continue
        v = _coordinate_tangent(fit, cc['theta'])
        s = float((np.einsum('nod,nd->no', J, v) ** 2).sum(1).mean())
        b = name_coordinate(cc['theta'], variables, modular_period=period, threshold=0.9)['best']
        rows.append(dict(class_idx=int(ci), lifetime=float(life[ci]), s_c=s, best_var=b['name'], best_freq=b['freq'], best_r=float(b['r'])))
    rows.sort(key=lambda r: -r['s_c']); top = rows[:5]
    hits = [r for r in top if r['best_r'] > 0.9 and r['best_freq'] in key_freqs]
    by_life = sorted(rows, key=lambda r: -r['lifetime'])[:5]
    out = dict(n_h1=int(len(h1)), n_candidates=len(rows), key_freqs=key_freqs, top5_by_sc=top, n_named_top5=len(hits),
               max_r_any_candidate=max((r['best_r'] for r in rows), default=0.0),
               n_candidates_r_above_0_9=sum(r['best_r'] > 0.9 for r in rows), top5_by_lifetime=by_life, wall_s=time.time() - t0)
    print(f"[E1] {tag}: H1 {len(h1)}, top-5 by s_c -> named {len(hits)}/5 | best r among all 30: {out['max_r_any_candidate']:.3f} | "
          + ' '.join(f"({r['best_var']} k={r['best_freq']} r={r['best_r']:.2f} s={r['s_c']:.2g})" for r in top), flush=True)
    return out


def count_above_band(X, dist_fn, tag):
    """classes above 1.5 x the longest H^1 bar over two feature shuffles, shuffles pushed through the same metric"""
    tops = []
    for sd in (0, 1):
        ls = lifetimes(persistence(dist_fn(feature_shuffle(X, sd), True), distance_matrix=True, do_cocycles=False)['dgms'][1])
        tops.append(float(ls.max()) if len(ls) else 0.0)
    l = lifetimes(persistence(dist_fn(X, False), distance_matrix=True, do_cocycles=False)['dgms'][1])
    band = 1.5 * max(tops)
    r = dict(n_h1=int(len(l)), band=band, n_above=int((l > band).sum()), top_life=float(l.max()) if len(l) else 0.0,
             top_over_band=float(l.max() / band) if len(l) and band > 0 else None)
    print(f'[E2] {tag}: {r}', flush=True); return r


def main():
    from scipy.spatial.distance import pdist, squareform
    from analysis.z113_common import checkpoints, load, acts, slice_probe
    from systems.z113 import build_data, P
    from systems.z113_probe import psi_of
    out = dict(E1={}, E2={})
    eu = lambda X, shuffled: squareform(pdist(X))
    for run, seed, frac in (('z113_clean_s1', 1, .3), ('z113_clean_s0', 0, .3), ('z113_cleanctrl_s2', 2, .5)):
        m = load(checkpoints(f'runs/{run}')[0][-1], seed=seed); d = build_data(seed, 'clean', frac=frac)
        tk = d['train_toks'][np.random.default_rng(0).choice(len(d['train_toks']), 1200, replace=False)]
        X = acts(m, tk, 'resid_mid').astype(np.float32); psi = psi_of(m, 'resid_mid'); J = jacobians(psi, X).astype(np.float32)
        a, b = tk[:, 0].astype(int), tk[:, 1].astype(int)
        var = dict(a=a, b=b, a_plus_b=(a + b) % P, a_minus_b=(a - b) % P)
        sl = slice_probe(); key = [int(k) for k in isotypic_decomposition(acts(m, sl['toks'], 'resid_mid'), sl['a'], P, n_freq=5, detect='dft', var_floor=0.0)[1]]
        out['E1'][run] = e1(X, J, var, P, key, run)
        Xs = feature_shuffle(X, 0); out['E1'][run + '|null_feature_shuffle'] = e1(Xs, jacobians(psi, Xs).astype(np.float32), var, P, key, run + ' NULL')
        if run == 'z113_clean_s1':
            gd = lambda Y, shuffled: geodesic_distance_matrix(fit_manifold(Y, jacobians(psi, Y).astype(np.float32), metric='g'))
            out['E2']['z113_clean_s1'] = dict(euclidean=count_above_band(X, eu, 'z113 euclid'), pullback=count_above_band(X, gd, 'z113 pullback'))
    # ring s1
    from systems.ring_rnn import RingRNN
    from analysis.ring_pullback import cloud
    r = RingRNN(seed=1).cuda(); r.load_state_dict(torch.load(sorted(glob.glob('runs/ring_s1/ckpt_*.pt'))[-1])); r.eval()
    Xr = cloud(r, 1200, seed=7)[0]
    gdr = lambda Y, shuffled: geodesic_distance_matrix(fit_manifold(Y, jacobians(r.psi, Y).astype(np.float32), metric='g'))
    out['E2']['ring_s1'] = dict(euclidean=count_above_band(Xr, eu, 'ring euclid'), pullback=count_above_band(Xr, gdr, 'ring pullback'))
    json.dump(out, open('results/run2/blockE12.json', 'w'), indent=1, default=float)
    # GPT-2 L6 (secant metric; shuffled activations are pushed through blocks 6-11 to get their outputs)
    from analysis import gpt2_explore as g
    from transformers import GPT2LMHeadModel
    c = np.load('results/run2/cache_gpt2_L6.npz'); sub = np.random.default_rng(1).choice(2000, 1200, replace=False)
    H, O, ids = c['H'][sub], c['O'][sub], c['ids'][sub]
    gm = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda').eval(); tap = g.Tap(gm)
    WU = gm.lm_head.weight.detach().float().cpu().numpy(); _, S, Vt = np.linalg.svd(WU, full_matrices=False)
    def sec(Y, shuffled):
        Oy = O if not shuffled else ((g.run(gm, tap, ids, delta=(Y - H).astype(np.float32), bs=64)[1] @ Vt.T) * S).astype(np.float32)
        return geodesic_distance_matrix(fit_manifold(Y, metric='secant', out=Oy))
    out['E2']['gpt2_L6'] = dict(euclidean=count_above_band(H, eu, 'gpt2 euclid'), pullback_secant=count_above_band(H, sec, 'gpt2 secant'))
    v = out['E1']['z113_clean_s1']
    out['_verdict'] = dict(E1_primary_named_top5=v['n_named_top5'], E1_PASS=bool(v['n_named_top5'] >= 4),
                           E1_null_named_top5=out['E1']['z113_clean_s1|null_feature_shuffle']['n_named_top5'])
    print(json.dumps(out['_verdict'])); json.dump(out, open('results/run2/blockE12.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
