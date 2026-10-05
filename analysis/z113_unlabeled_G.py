"""POST-HOC, motivated by the ResNet failure of P6 (scrambled G = 0.89): a memorising network
ALSO makes its training points class-contiguous at a late layer, so geodesics between training
points that pass only through training points prove nothing.

Amendment: put UNLABELED held-out inputs into the cloud as path nodes.  A geodesic between two
training points of class y must now cross territory the network was never fitted on, and G_int is
the fraction of those interior nodes the network still calls y.  No held-out LABEL is used.

Test (fair, with a baseline): across all checkpoints of all runs, how well does each label-free
quantity track the true test accuracy?
   G_int_unl     this statistic
   conf          mean max-softmax on the same unlabeled points (the standard label-free proxy)
   G_int_train   the interior-only statistic without unlabeled nodes (what the sweep computed)
"""
import sys, os, json, time
import numpy as np, torch
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.z113_common import checkpoints, load, layer_from
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.problems import p6_geodesic_generalization
from systems.z113 import build_data, P

RUNS = [('z113_clean_s0', 'clean', 0, 0.3), ('z113_clean_s1', 'clean', 1, 0.3), ('z113_cleanctrl_s2', 'clean', 2, 0.5),
        ('z113_scrambled_s0', 'scrambled', 0, 0.3)]


def probe(d, per_class=15, n_unl=1700, seed=0):
    rng = np.random.default_rng(seed)
    y = d['train_y']; sel = []
    for c in np.unique(y):
        m = np.where(y == c)[0]; sel.append(rng.choice(m, size=min(per_class, len(m)), replace=False))
    sel = np.concatenate(sel)
    te = rng.choice(len(d['test_toks']), size=n_unl, replace=False)
    toks = np.concatenate([d['train_toks'][sel], d['test_toks'][te]])
    labels = np.r_[y[sel], -np.ones(len(te), int)]
    return dict(toks=toks, labels=labels, a=toks[:, 0], b=toks[:, 1], c=(toks[:, 0] + toks[:, 1]) % P), len(sel), d['test_y'][te]


def main(every=3):
    rows = []
    for run, mode, seed, frac in RUNS:
        d = build_data(seed, mode, frac=frac); pr, ntr, yte = probe(d)
        cks, steps = checkpoints(f'runs/{run}')
        for ck, st in list(zip(cks, steps))[::every]:
            t0 = time.time(); m = load(ck, seed=seed)
            ld, _ = layer_from(m, pr)
            fit = fit_manifold(ld.X, ld.J, metric='g')
            g = p6_geodesic_generalization(ld, fit, pairs_per_class=20)
            z = ld.logits[ntr:]; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
            acc = float((ld.logits[ntr:].argmax(1) == yte).mean())          # ground truth, never shown to the statistic
            rows.append(dict(run=run, step=int(st), test_acc_on_unlabeled=acc, G_int_unl=g['G_interior'], G_unl=g['G'],
                             conf=float(p.max(1).mean()), frac_direct=g['frac_direct']))
            del ld, fit, m; import gc; gc.collect(); torch.cuda.empty_cache()
            print(f"[{run}] {st:6d} acc {acc:.3f} G_int_unl {g['G_interior']:.3f} conf {rows[-1]['conf']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    res = dict(rows=rows, summary={})
    A = np.array([r['test_acc_on_unlabeled'] for r in rows])
    for key in ('G_int_unl', 'G_unl', 'conf'):
        v = np.array([r[key] for r in rows], float); ok = np.isfinite(v)
        res['summary'][key] = dict(spearman=float(spearmanr(v[ok], A[ok]).statistic), pearson=float(pearsonr(v[ok], A[ok])[0]),
                                   mae_as_estimator=float(np.abs(v[ok] - A[ok]).mean()))
    # the regime that matters: memorised (train acc = 1) but not yet generalising vs generalising
    print(json.dumps(res['summary'], indent=1))
    json.dump(res, open('results/z113_unlabeled_G.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
