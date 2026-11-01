"""Shared probes for the Z113 system."""
import sys, os, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.z113 import P, NULL, EQ, build_data, all_pairs
from systems.z113_probe import load, acts, psi_of
from whiteboxtopo.extract import jacobians, logits_of
from whiteboxtopo.problems import LayerData


def slice_probe(hint=NULL):
    """The group orbit: all 113 values of a at b = 0.  This is the probe the isotypic
    decomposition needs -- one clean orbit of the group whose action we are estimating."""
    a = np.arange(P)
    return dict(toks=np.stack([a, np.zeros(P, int), np.full(P, hint), np.full(P, EQ)], 1), a=a)


def fiber_probe(mode, seed, n_classes=25, per_class=40, rseed=0):
    """Training points only, grouped by the label the model was trained on: `n_classes`
    classes with up to `per_class` points each.

    Train points are used for both the clean and the scrambled model so the comparison is
    like for like -- off the training set a memorising model has no labels to speak of, and
    P6 would be measuring the task rather than the model.
    """
    d = build_data(seed, mode, frac=0.5 if mode == 'shortcut' else 0.3)
    toks, y = d['train_toks'], d['train_y']
    rng = np.random.default_rng(rseed)
    uniq, cnt = np.unique(y, return_counts=True)
    classes = uniq[np.argsort(-cnt)][:n_classes]
    sel = []
    for c in classes:
        m = np.where(y == c)[0]
        sel.append(rng.choice(m, size=min(per_class, len(m)), replace=False))
    sel = np.concatenate(sel)
    return dict(toks=toks[sel], labels=y[sel], a=toks[sel, 0], b=toks[sel, 1],
                c=(toks[sel, 0] + toks[sel, 1]) % P, hint=toks[sel, 2], idx=sel)


def layer_from(model, probe, layer='resid_mid', device='cuda'):
    X = acts(model, probe['toks'], layer, device)
    psi = psi_of(model, layer)
    J = jacobians(psi, X, device=device)
    lg = logits_of(psi, X, device=device)
    lab = probe.get('labels', probe.get('c'))
    variables = dict(a=probe['a'], b=probe.get('b', np.zeros(len(X), int)),
                     c=probe.get('c', lab))
    return LayerData(name=layer, X=X.astype(np.float32), J=J.astype(np.float32), logits=lg,
                     labels=np.asarray(lab), variables=variables, modular_period=P), psi


def checkpoints(run_dir):
    cks = sorted(glob.glob(f'{run_dir}/ckpt_*.pt'))
    steps = [int(os.path.basename(c).split('_')[1].split('.')[0]) for c in cks]
    return cks, np.array(steps)
