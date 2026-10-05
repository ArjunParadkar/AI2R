"""Run 3, P1.3 + P7.1 (SPEC3.md): automorphisms of 72 equally spaced headings of the ring RNN, final models and across checkpoints.
Neighbouring work: LieGG (Moskalev et al., NeurIPS 2022)."""
import sys, os, json, glob
import numpy as np, torch
from scipy.spatial.distance import pdist, squareform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.ring_rnn import RingRNN
from whiteboxtopo.symmetry_finite import automorphisms
from whiteboxtopo.nulls import feature_shuffle
N = 72
keys = ('order', 'has_full_cycle', 'max_element_order', 'degenerate', 'order_rel', 'has_full_cycle_rel', 'R_star', 'T', 'median', 'nn_median')


def states(m):
    th = torch.arange(N, dtype=torch.float32) * (2 * np.pi / N)
    with torch.no_grad():
        h = m(torch.zeros(N, 25), th); y = m.psi(h)
    return h.numpy(), y.numpy(), m


def grp(D):
    r = automorphisms(D); o = {k: r.get(k) for k in keys}
    o['cyclic_order_ge_36'] = bool(r.get('max_element_order', 1) >= 36)
    o['cyclic_order_ge_36_rel'] = bool(max([1] + [__import__('whiteboxtopo.symmetry_finite', fromlist=['element_order']).element_order(p) for p in r.get('perms_rel', [])]) >= 36)
    return o


def main():
    out = {}
    for s in (0, 1, 2):
        cks = sorted(glob.glob(f'runs/ring_s{s}/ckpt_*.pt')); hist = json.load(open(f'runs/ring_s{s}/history.json'))['history']; rows = []
        for ck, hh in zip(cks, hist):
            m = RingRNN(seed=s); m.load_state_dict(torch.load(ck, map_location='cpu')); m.eval(); h, y, _ = states(m)
            g = grp(squareform(pdist(y))); rows.append(dict(step=hh['step'], ang_err=hh['ang_err_rad'], **g))
        h, y, m = states(m); hs = feature_shuffle(h.astype(np.float32), 0)
        with torch.no_grad():
            ys = m.psi(torch.as_tensor(hs)).numpy()
        t_err = next((i for i, r in enumerate(rows) if r['ang_err'] < 0.0873), None); t_grp = next((i for i, r in enumerate(rows) if r['cyclic_order_ge_36']), None)
        t_rel = next((i for i, r in enumerate(rows) if r['cyclic_order_ge_36_rel']), None)
        out[f's{s}'] = dict(final_g=rows[-1], final_euclid=grp(squareform(pdist(h))), null_feature_shuffle=grp(squareform(pdist(ys))), sweep=rows,
                            t_err=t_err, t_grp=t_grp, t_grp_rel_posthoc=t_rel, min_ang_err=min(r['ang_err'] for r in rows))
        print(f"[ring s{s}] final order {rows[-1]['order']} maxelem {rows[-1]['max_element_order']} | POST-HOC rel order {rows[-1]['order_rel']} R* {rows[-1]['R_star']} | euclid order {out[f's{s}']['final_euclid']['order']} | "
              f"null order {out[f's{s}']['null_feature_shuffle']['order']} rel {out[f's{s}']['null_feature_shuffle']['order_rel']} | t_err {t_err} t_grp {t_grp} (rel {t_rel}) min err {out[f's{s}']['min_ang_err']:.3f}", flush=True)
    v = dict(P1_3=sum(out[f's{s}']['final_g']['cyclic_order_ge_36'] for s in range(3)), P1_3_null_ok=sum(out[f's{s}']['null_feature_shuffle']['order'] <= 2 for s in range(3)),
             P7_1=sum(out[f's{s}']['t_err'] is not None and out[f's{s}']['t_grp'] is not None and abs(out[f's{s}']['t_grp'] - out[f's{s}']['t_err']) <= 2 for s in range(3)))
    v['PASS_P1_3'] = bool(v['P1_3'] == 3 and v['P1_3_null_ok'] == 3); v['PASS_P7_1'] = bool(v['P7_1'] == 3); out['_verdict'] = v; print(json.dumps(v))
    json.dump(out, open('results/run3/blockRing.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
