"""Four follow-ups on the grokked Z113 models, each testing a ground truth the spec asserted.

  fibers       are the answer-fibers {(a, c-a)} circles?            -> results/z113_fibers.json
  symmetry     why not: is the network commutative at resid_mid?    -> results/z113_fiber_symmetry.json
  boundary     which residue does the boundary geodesic reach, and is it predicted by the P1 spectrum?
                                                                     -> results/z113_p2_boundary.json
  unsup_g      does g-geodesic Rips name circles with no symmetry supplied?  -> results/z113_unsupervised_g.json

usage: python analysis/z113_extras.py [fibers symmetry boundary unsup_g]
"""
import sys, os, json
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.z113_common import checkpoints, load, acts, layer_from, slice_probe
from systems.z113 import build_data, P, NULL, EQ
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.problems import p1_symmetry_inventory, p1_inventory, p2_explain
from whiteboxtopo.topology import persistence, significant_classes, isotypic_decomposition
from whiteboxtopo.nulls import feature_shuffle

RUNS = (('runs/z113_clean_s1', 1, .3), ('runs/z113_clean_s0', 0, .3), ('runs/z113_cleanctrl_s2', 2, .5))
final = lambda run, seed: load(checkpoints(run)[0][-1], seed=seed)
fiber = lambda m, c: acts(m, np.stack([np.arange(P), (c - np.arange(P)) % P, np.full(P, NULL), np.full(P, EQ)], 1), 'resid_mid')
axis_ratio = lambda Y: (lambda sv: float(sv[1] / sv[0]))(np.linalg.svd(Y - Y.mean(0), compute_uv=False))


def fibers():
    out, a = {}, np.arange(P)
    for run, seed, _ in RUNS[:2]:
        m, rows = final(run, seed), []
        for c in range(0, P, 12):
            X = fiber(m, c).astype(np.float32)
            tops = []
            for sd in (0, 1):
                d1 = persistence(feature_shuffle(X, sd), do_cocycles=False)['dgms'][1]; lf = d1[:, 1] - d1[:, 0]
                tops.append(lf[np.isfinite(lf)].max() if len(lf) else 0)
            sig, _ = significant_classes(persistence(X, do_cocycles=False)['dgms'][1], 1.5 * max(tops))
            rows.append(dict(c=c, n_circles=sum(i['present'] for i in p1_symmetry_inventory(X, a, P, n_freq=5)), raw_beta1=int(len(sig)),
                             null_circles=sum(i['present'] for i in p1_symmetry_inventory(feature_shuffle(X, 0), a, P, n_freq=5))))
        out[run] = rows; print(run, [r['n_circles'] for r in rows], [r['raw_beta1'] for r in rows])
    json.dump(out, open('results/z113_fibers.json', 'w'), indent=1)


def symmetry():
    out, a = {}, np.arange(P)
    for run, seed, _ in RUNS:
        m, sw, ecc = final(run, seed), [], []
        for c in range(0, P, 12):
            X = fiber(m, c).astype(np.float64); Xc = X - X.mean(0)
            sw.append(float(np.linalg.norm(X - X[(c - a) % P], axis=1).mean() / np.sqrt((Xc ** 2).sum(1).mean())))
            ecc += [axis_ratio(d['proj']) for d in isotypic_decomposition(X, a, P, n_freq=4, detect='dft', var_floor=0.0)[0]]
        S = acts(m, slice_probe()['toks'], 'resid_mid').astype(np.float64)
        ref = [axis_ratio(d['proj']) for d in isotypic_decomposition(S, a, P, n_freq=4, detect='dft', var_floor=0.0)[0]]
        out[run] = dict(swap_distance_rel=float(np.mean(sw)), fiber_plane_axis_ratio_median=float(np.median(ecc)),
                        slice_plane_axis_ratio_median=float(np.median(ref))); print(run, out[run])
    json.dump(out, open('results/z113_fiber_symmetry.json', 'w'), indent=1)


def rep_probe(m, d, n):
    sub = np.random.default_rng(0).choice(len(d['train_toks']), n, replace=False); tk = d['train_toks'][sub]
    ld, _ = layer_from(m, dict(toks=tk, a=tk[:, 0], b=tk[:, 1], c=(tk[:, 0] + tk[:, 1]) % P, labels=d['train_y'][sub]))
    return ld, fit_manifold(ld.X, ld.J, metric='g')


def boundary():
    out = {}
    for run, seed, frac in RUNS:
        m = final(run, seed); ld, fit = rep_probe(m, build_data(seed, 'clean', frac=frac), 1500)
        ex = p2_explain(ld, fit, list(range(300)))
        fold = lambda v: np.minimum(v % P, P - v % P)
        dgeo = fold(np.array([e['flips_to'] - e['pred'] for e in ex])); srt = np.argsort(-ld.logits[:300], 1); dlog = fold(srt[:, 1] - srt[:, 0])
        sl = slice_probe(); ks = [(i['freq'], i['var']) for i in p1_symmetry_inventory(acts(m, sl['toks'], 'resid_mid'), sl['a'], P, n_freq=6) if i['present']]
        cost = np.array([sum(v * (1 - np.cos(2 * np.pi * k * D / P)) for k, v in ks) for D in range(1, P // 2 + 1)]); order = np.argsort(cost) + 1
        top3 = [int(x) for x in order[:3]]
        out[run] = dict(freqs=[k for k, _ in ks], predicted_nearest_shifts=[int(x) for x in order[:5]],
                        geodesic_shift_counts=Counter(int(x) for x in dgeo).most_common(5), logit_runnerup_counts=Counter(int(x) for x in dlog).most_common(5),
                        frac_geodesic_is_pm1=float((dgeo == 1).mean()), frac_geodesic_in_predicted_top3=float(np.isin(dgeo, top3).mean()),
                        frac_runnerup_in_predicted_top3=float(np.isin(dlog, top3).mean()), frac_geodesic_equals_runnerup=float((dgeo == dlog).mean()), chance_top3=3 / 56)
        print(run, json.dumps(out[run]))
    json.dump(out, open('results/z113_p2_boundary.json', 'w'), indent=1)


def unsup_g():
    out = {}
    for run, seed, frac in RUNS[:2]:
        ld, fit = rep_probe(final(run, seed), build_data(seed, 'clean', frac=frac), 1200)
        for metric in ('euclidean', 'g'):
            inv = p1_inventory(ld, fit, do_fibers=False, metric=metric, max_classes=8)
            out[f'{run}|{metric}'] = dict(beta1=inv['beta1'], n_named=sum(1 for c in inv['coords'] if c['naming']['named'])); print(run, metric, out[f'{run}|{metric}'])
    json.dump(out, open('results/z113_unsupervised_g.json', 'w'), indent=1)


if __name__ == '__main__':
    for name in (sys.argv[1:] or ['fibers', 'symmetry', 'boundary', 'unsup_g']):
        globals()[name]()
