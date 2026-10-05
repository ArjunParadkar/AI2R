"""Deliverable 1: the one-minute demo.  open_model() on the grokked Z113 model and on ResNet-18,
the seven answers in plain language with numbers, and the wall time.

P3 and P7 are questions about the checkpoint bundle, not about one model; their lines are read
from the bundle results computed by analysis/z113_sweep.py and analysis/z113_p3_detect.py, and
say so.  Everything else is computed live inside the timed call.
"""
import sys, os, json, time, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whiteboxtopo import open_model
from whiteboxtopo.problems import LayerData


def z113():
    from analysis.z113_common import checkpoints, load, fiber_probe, layer_from, slice_probe, acts
    from analysis.z113_static import act_logit, ood_sets, failure_radius_activation
    from systems.z113 import build_data, NULL, EQ
    run, seed = 'runs/z113_clean_s1', 1
    t0 = time.time()
    m = load(checkpoints(run)[0][-1], seed=seed); d = build_data(seed, 'clean', frac=0.3)
    ld, psi = layer_from(m, fiber_probe('clean', seed, 25, 40))
    te = d['test_idx'][:600]; a, b = d['a'], d['b']
    Xh, Lh = act_logit(m, np.stack([a[te], b[te], np.full(len(te), NULL), np.full(len(te), EQ)], 1))
    ld.logits_holdout = Lh
    ood = {k: act_logit(m, v) for k, v in ood_sets(d, te, np.random.default_rng(0)).items() if k in ('random_tokens', 'no_equals')}
    fr, _ = failure_radius_activation(m, ld.X)
    sl = slice_probe(); from systems.z113 import P
    sym = {'resid_mid': dict(X=acts(m, sl['toks'], 'resid_mid'), param=sl['a'], period=P)}
    t_prep = time.time() - t0
    rep = open_model({'resid_mid': ld}, holdout=Xh, ood=ood, failure_radius=fr, explain_inputs=[0], psi=None, symmetry=sym)
    return rep, t_prep


def resnet():
    from systems.resnet_cifar import make_model, load_cifar, MEAN, STD
    from analysis.resnet_analysis import layer, feats
    from torchvision import datasets
    t0 = time.time()
    xtr, ytr, xte, yte = load_cifar()
    m = make_model().cuda(); m.load_state_dict(torch.load(sorted(glob.glob('runs/resnet_clean_s0/ckpt_*.pt'))[-1])); m.eval()
    rng = np.random.default_rng(0)
    sel = np.concatenate([rng.choice(np.where(ytr == c)[0], 150, replace=False) for c in range(10)])
    ld, W, b = layer(m, xtr[sel], ytr[sel])
    hold = rng.choice(len(xte), 800, replace=False); Xh = feats(m, xte[hold]); ld.logits_holdout = Xh @ W.T + b
    sv = datasets.SVHN('data/svhn', split='test', download=False)
    svx = (sv.data[:800].astype(np.float32) / 255 - MEAN[None, :, None, None]) / STD[None, :, None, None]
    Xo = feats(m, svx); Xn = feats(m, rng.standard_normal((800, 3, 32, 32)).astype(np.float32))
    ood = dict(svhn=(Xo, Xo @ W.T + b), gaussian_noise=(Xn, Xn @ W.T + b))
    t_prep = time.time() - t0
    rep = open_model({'penultimate': ld}, holdout=Xh, ood=ood, explain_inputs=[0])
    return rep, t_prep


def bundle_lines():
    L = []
    try:
        s = json.load(open('results/p6_p7_score.json'))['P6']['z113_clean_s1']
        L.append(f"  P7  [from the checkpoint bundle] frequency circles born at checkpoints {s['births']} ; grokking at checkpoint {s['grok_ckpt']}")
    except Exception as e:
        L.append(f'  P7  bundle result unavailable ({e})')
    try:
        p = json.load(open('results/p3_detect.json'))['summary']
        L.append(f"  P3  [from the checkpoint bundle] clean models: {sum(p[k]['n_ckpts_flagged'] for k in p if k.startswith('clean'))} spurious classes over "
                 f"{sum(p[k]['n_ckpts'] for k in p if k.startswith('clean'))} checkpoints; planted-shortcut models first flagged at steps "
                 f"{[p[k]['first_detection_step'] for k in p if k.startswith('shortcut')]} (true onset {[p[k]['ground_truth_onset_step'] for k in p if k.startswith('shortcut')]})")
    except Exception as e:
        L.append(f'  P3  bundle result unavailable ({e})')
    return L


def main():
    out = {}
    for name, fn in (('Z113 transformer (grokked, seed 1)', z113), ('ResNet-18 / CIFAR-10 (clean)', resnet)):
        rep, t_prep = fn()
        print('=' * 100); print(f'open({name})     [probe extraction {t_prep:.1f}s, then:]'); print(rep.summary())
        if name.startswith('Z113'):
            print('\n'.join(bundle_lines()))
        print('  per-step timings:', {k: round(v, 2) for k, v in list(rep['timings'].values())[0].items()})
        out[name] = dict(wall_s=rep['wall_s'], prep_s=t_prep, timings=rep['timings'])
    json.dump(out, open('results/demo_timings.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
