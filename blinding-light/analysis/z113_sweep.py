"""The bundle over training for Z113: P1, P6 and P7 at every checkpoint.

For each checkpoint we compute
  * on the group-orbit probe (113 points, b=0): the isotypic decomposition, and for each
    frequency whether it carries a certified circle -- one dominant H^1 class whose DSMVJ
    coordinate tracks k*a.  That is the birth record P7 needs.
  * on the fiber probe (training points grouped by label): the manifold score, the geodesic
    generalization G, and beta_0 of each fiber under the pullback metric.  That is P6.
"""
import sys, os, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.z113_common import *
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.problems import p6_geodesic_generalization, fiber_beta0
from whiteboxtopo.topology import (isotypic_decomposition, persistence, circular_coordinates,
                                   circular_correlation, normalized_persistence)
from whiteboxtopo.nulls import feature_shuffle
from systems.z113 import P


def certify_circles(X, a, n_freq=6, r_pass=0.9, life_ratio=3.0):
    """Which characters the cloud actually carries a circle for."""
    out = []
    dec, freqs = isotypic_decomposition(X, a, P, n_freq=n_freq, detect='dft', var_floor=0.0)
    for d in dec:
        k = d['freq']
        Y = np.ascontiguousarray(d['proj']).astype(np.float32)
        res = persistence(Y)
        h1 = res['dgms'][1]
        if len(h1) == 0:
            out.append(dict(freq=k, var=d['var'], present=False, r=0.0, life_ratio=0.0))
            continue
        life = h1[:, 1] - h1[:, 0]
        j = int(np.argmax(life))
        srt = np.sort(life)
        ratio = float(srt[-1] / max(srt[-2], 1e-9)) if len(srt) > 1 else np.inf
        cc = circular_coordinates(Y, h1, res['cocycles'][1], D=res['D'], class_idx=j)
        r = circular_correlation(cc['theta'], 2 * np.pi * k * a / P) if cc else 0.0
        out.append(dict(freq=int(k), var=float(d['var']), r=float(r), life_ratio=ratio,
                        lifetime=float(srt[-1]),
                        present=bool(r > r_pass and ratio > life_ratio and d['var'] > 0.02)))
    return out


def main(run, mode, seed, every=1, n_classes=25, per_class=40, out=None):
    cks, steps = checkpoints(run)
    hist = {h['step']: h for h in json.load(open(f'{run}/history.json'))['history']}
    sl = slice_probe()
    fp = fiber_probe(mode, seed, n_classes, per_class)
    recs = []
    t00 = time.time()
    for i in range(0, len(cks), every):
        t0 = time.time()
        m = load(cks[i], seed=seed)
        Xs = acts(m, sl['toks'], 'resid_mid')
        circles = certify_circles(Xs, sl['a'])
        ld, psi = layer_from(m, fp)
        fit = fit_manifold(ld.X, ld.J, metric='g')
        g = p6_geodesic_generalization(ld, fit, pairs_per_class=40)
        b0 = fiber_beta0(ld, fit)
        res = persistence(ld.X)
        h = hist.get(int(steps[i]), {})
        recs.append(dict(step=int(steps[i]), train_acc=h.get('train_acc'), test_acc=h.get('test_acc'),
                         abl_acc=h.get('abl_acc'), dec_acc=h.get('dec_acc'),
                         shortcut_reliance=h.get('shortcut_reliance'),
                         G=g['G'], G_interior=g['G_interior'], frac_direct=g['frac_direct'], m=fit.m, beta0_fiber_mean=float(np.mean(list(b0.values()))),
                         beta0_fiber_max=int(np.max(list(b0.values()))),
                         d_intrinsic=float(np.nanmedian(fit.d_local)),
                         tildeC=normalized_persistence(res['dgms'][1], res['median_dist']),
                         circles=circles, wall=time.time() - t0))
        print(f"  [{os.path.basename(run)}] step {steps[i]:6d} test {str(h.get('test_acc'))[:5]:>5} "
              f"G {g['G']:.3f} m {fit.m:.2f} b0 {recs[-1]['beta0_fiber_mean']:.2f} "
              f"circles {sum(c['present'] for c in circles)} ({time.time()-t0:.0f}s)", flush=True)
    out = out or f'{run}/sweep.json'
    json.dump(dict(run=run, mode=mode, seed=seed, n_classes=n_classes, per_class=per_class,
                   total_wall=time.time() - t00, records=recs), open(out, 'w'), indent=1)
    print(f'-> {out}  ({time.time()-t00:.0f}s)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--mode', required=True)
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--every', type=int, default=1)
    a = ap.parse_args()
    main(a.run, a.mode, a.seed, a.every)
