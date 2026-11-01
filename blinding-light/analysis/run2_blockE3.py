"""Run 2, Block E3 (SPEC2.md): invariance as quotient.  Z113: the Z2 quotient witness on seen (RE-SCORE) and UNSEEN (decoy)
models, plus H_*(F_y) by Rips with the two-shuffle band.  rot-MNIST: invariance depth by the run-1 certification rule."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.z113_common import checkpoints, load, acts, slice_probe
from analysis.z113_extras import fiber, axis_ratio
from systems.z113 import P
from whiteboxtopo.topology import isotypic_decomposition, persistence, significant_classes
from whiteboxtopo.nulls import feature_shuffle

MODELS = [('z113_clean_s0', 0, 'RE-SCORE'), ('z113_clean_s1', 1, 'RE-SCORE'), ('z113_cleanctrl_s2', 2, 'RE-SCORE'),
          ('z113_decoy_s0', 0, 'UNSEEN'), ('z113_decoy_s1', 1, 'UNSEEN')]


def main():
    out, a = dict(z113={}), np.arange(P)
    for run, seed, status in MODELS:
        m = load(checkpoints(f'runs/{run}')[0][-1], seed=seed); sw, ecc, b0, b1 = [], [], [], []
        for c in range(0, P, 12):
            X = fiber(m, c).astype(np.float64); Xc = X - X.mean(0)
            sw.append(float(np.linalg.norm(X - X[(c - a) % P], axis=1).mean() / np.sqrt((Xc ** 2).sum(1).mean())))
            ecc += [axis_ratio(d['proj']) for d in isotypic_decomposition(X, a, P, n_freq=4, detect='dft', var_floor=0.0)[0]]
            Xf = X.astype(np.float32); tops = []
            for sd in (0, 1):
                l = persistence(feature_shuffle(Xf, sd), do_cocycles=False)['dgms'][1]; l = l[:, 1] - l[:, 0]; tops.append(l[np.isfinite(l)].max() if len(l) else 0.0)
            r = persistence(Xf, do_cocycles=False)
            b1.append(int(len(significant_classes(r['dgms'][1], 1.5 * max(tops))[0]))); b0.append(int(len(significant_classes(r['dgms'][0], None, essential_always=True)[0])))
        S = acts(m, slice_probe()['toks'], 'resid_mid').astype(np.float64)
        ref = [axis_ratio(d['proj']) for d in isotypic_decomposition(S, a, P, n_freq=4, detect='dft', var_floor=0.0)[0]]
        r = dict(status=status, swap_distance_rel=float(np.mean(sw)), fiber_axis_ratio_median=float(np.median(ecc)), slice_axis_ratio_median=float(np.median(ref)),
                 fiber_beta0=b0, fiber_beta1_above_band=b1)
        r['met'] = bool(r['fiber_axis_ratio_median'] < 0.1 and r['slice_axis_ratio_median'] > 0.9 and r['swap_distance_rel'] < 0.02)
        out['z113'][run] = r; print(f'[E3] {run} ({status}):', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    c = json.load(open('results/rotmnist_controls.json')); L = ['conv1', 'conv2', 'conv3_gap', 'fc1']
    tr = {l: c['trained']['r_median'][l] for l in L}; un = {l: c['untrained_s0']['r_median'][l] for l in L}
    depth = next((l for l in L if tr[l] <= 0.9 and un[l] > 0.9), None)
    out['rotmnist'] = dict(trained_r=tr, untrained_r=un, invariance_depth=depth, met=bool(depth is not None), status='RE-SCORE',
                           first_layer_without_circle=dict(trained=next((l for l in L if tr[l] <= 0.9), None), untrained=next((l for l in L if un[l] <= 0.9), None)),
                           graded_gap={l: un[l] - tr[l] for l in L})
    print('[E3] rot-MNIST', out['rotmnist'])
    out['_verdict'] = dict(z113_all_met=bool(all(v['met'] for v in out['z113'].values())), z113_unseen_met=bool(all(v['met'] for v in out['z113'].values() if v['status'] == 'UNSEEN')),
                           rotmnist_met=out['rotmnist']['met'])
    out['_verdict']['PASS_E3'] = bool(out['_verdict']['z113_all_met'] and out['_verdict']['rotmnist_met'])
    print(json.dumps(out['_verdict'])); json.dump(out, open('results/run2/blockE3.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
