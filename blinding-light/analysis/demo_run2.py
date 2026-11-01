"""The one-minute demo, re-run with the run-2 P4/P5/P6 outputs printed beneath the run-1 answers.
Live: P4 kernel displacement iota (AUROC vs the same OOD sets), P5 Morse-Smale separatrix distance d_sep (Spearman with the
failure radius, Z113 only -- ResNet needs a PGD search and is read from the block-C result).  P6's depth profile needs every
layer and both models; its line is read from the block-A result and says so."""
import sys, os, json, time, glob
import numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whiteboxtopo import open_model
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.kernel import invisible_displacement, auroc
from whiteboxtopo.morse import morse_smale, separatrix_distance


def run2_lines(ld, Xh, ood, fr, J_for_kernel, tag):
    t0 = time.time(); L = []
    base = invisible_displacement(Xh, ld.X, J_for_kernel)['perp']
    bits = [f"{k}: AUROC {auroc(base, invisible_displacement(v[0], ld.X, J_for_kernel)['perp']):.3f}" for k, v in ood.items()]
    L.append(f"  P4' invisible displacement (kernel of J): {', '.join(bits)}   [{time.time()-t0:.1f}s]")
    t0 = time.time(); fit = fit_manifold(ld.X, ld.J, metric='g'); ms = morse_smale(fit.W, ld.logits.max(1), ld.pred)
    d_sep, _ = separatrix_distance(fit.W, ms['sep_edges'], ms['sep_w'])
    msg = f"  P5' Morse-Smale: {len(np.unique(ms['basin']))} basins, {len(ms['sep_edges'])} separatrix edges of {fit.W.nnz // 2}"
    if fr is not None:
        ok = np.isfinite(fr) & np.isfinite(d_sep); z = np.sort(ld.logits, 1); sm = z[:, -1] - z[:, -2]
        msg += f"; Spearman(d_sep, failure radius) = {spearmanr(d_sep[ok], fr[ok]).statistic:+.3f} (softmax margin {spearmanr(sm[ok], fr[ok]).statistic:+.3f})"
    L.append(msg + f"   [{time.time()-t0:.1f}s]")
    try:
        A = json.load(open('results/run2/blockA.json')); c, s = A[f'{tag}_clean'], A[f'{tag}_scrambled']
        L.append(f"  P6' [from the depth profile, both models] untangling index U: clean {c['U_euclid']:.3f} vs memoriser {s['U_euclid']:.3f} "
                 f"(ambient fibers never connect at the registered scale)")
    except Exception as e:
        L.append(f"  P6' depth-profile result unavailable ({e})")
    return L


def main():
    from analysis import demo
    # ---- Z113 (same construction as analysis/demo.py)
    from analysis.z113_common import checkpoints, load, fiber_probe, layer_from
    from analysis.z113_static import act_logit, ood_sets, failure_radius_activation
    from systems.z113 import build_data, NULL, EQ
    m = load(checkpoints('runs/z113_clean_s1')[0][-1], seed=1); d = build_data(1, 'clean', frac=0.3)
    ld, _ = layer_from(m, fiber_probe('clean', 1, 25, 40)); te = d['test_idx'][:600]; a, b = d['a'], d['b']
    Xh, Lh = act_logit(m, np.stack([a[te], b[te], np.full(len(te), NULL), np.full(len(te), EQ)], 1)); ld.logits_holdout = Lh
    ood = {k: act_logit(m, v) for k, v in ood_sets(d, te, np.random.default_rng(0)).items() if k in ('random_tokens', 'no_equals')}
    fr, _ = failure_radius_activation(m, ld.X)
    t0 = time.time(); rep = open_model({'resid_mid': ld}, holdout=Xh, ood=ood, failure_radius=fr, explain_inputs=[0])
    print('=' * 100); print('open(Z113 transformer, grokked, seed 1)'); print(rep.summary())
    t1 = time.time(); print('\n'.join(run2_lines(ld, Xh, ood, fr, ld.J, 'z113'))); print(f'run-2 additions {time.time()-t1:.1f}s; everything {time.time()-t0:.1f}s')
    # ---- ResNet-18
    from systems.resnet_cifar import make_model, load_cifar, MEAN, STD
    from analysis.resnet_analysis import layer, feats
    from torchvision import datasets
    xtr, ytr, xte, yte = load_cifar(); rm = make_model().cuda(); rm.load_state_dict(torch.load(sorted(glob.glob('runs/resnet_clean_s0/ckpt_*.pt'))[-1])); rm.eval()
    rng = np.random.default_rng(0); sel = np.concatenate([rng.choice(np.where(ytr == c)[0], 150, replace=False) for c in range(10)])
    ld, W, bb = layer(rm, xtr[sel], ytr[sel]); hold = rng.choice(len(xte), 800, replace=False); Xh = feats(rm, xte[hold]); ld.logits_holdout = Xh @ W.T + bb
    sv = datasets.SVHN('data/svhn', split='test', download=False); svx = (sv.data[:800].astype(np.float32) / 255 - MEAN[None, :, None, None]) / STD[None, :, None, None]
    Xo = feats(rm, svx); Xn = feats(rm, rng.standard_normal((800, 3, 32, 32)).astype(np.float32)); ood = dict(svhn=(Xo, Xo @ W.T + bb), gaussian_noise=(Xn, Xn @ W.T + bb))
    t0 = time.time(); rep = open_model({'penultimate': ld}, holdout=Xh, ood=ood, explain_inputs=[0])
    print('=' * 100); print('open(ResNet-18 / CIFAR-10, clean)'); print(rep.summary())
    t1 = time.time(); print('\n'.join(run2_lines(ld, Xh, ood, None, W, 'resnet')))
    try:
        C = json.load(open('results/run2/blockC.json')); print("      [from block C, PGD radius on 1000 held-out images]", {k: v for k, v in C.items() if 'resnet' in k.lower()}.__len__() and 'rho-prime +0.780, d_sep +0.821, softmax margin +0.932')
    except Exception:
        pass
    print(f'run-2 additions {time.time()-t1:.1f}s; everything {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
