"""Run 2, block C: discrete Morse-Smale complex of the max-logit function, d_sep, ||II_boundary||, rho'.

SPEC2.md section C, by the letter.  Systems: Z113 clean_s0 / clean_s1 / cleanctrl_s2 (resid_mid, representative
1200-point probe of the model's own training split), ResNet-18 clean (1000 held-out test images, penultimate layer,
run-1 P5 design), GPT-2 layer 6 (exploratory; 600 held-out windows, secant weights).

    python analysis/run2_blockC.py --systems z113,resnet        # results merge into results/run2/blockC.json
    python analysis/run2_blockC.py --systems gpt2
"""
import sys, os, json, time, glob, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import spearmanr
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.geometry import manifold_margin
from whiteboxtopo.nulls import shuffle_predictor
from whiteboxtopo.problems.core import _softmax_conf
from whiteboxtopo.morse import morse_smale, separatrix_distance, boundary_curvature, rho_prime

OUT_JSON, OUT_NPZ = 'results/run2/blockC.json', 'results/run2/blockC_points.npz'


def _sp(a, r, ok):
    if ok.sum() < 10 or len(np.unique(a[ok])) < 2:
        return dict(r=None, p=None, n=int(ok.sum()))
    s = spearmanr(a[ok], r[ok])
    return dict(r=float(s.statistic), p=float(s.pvalue), n=int(ok.sum()))


def _dist(v):
    v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0)
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return dict(n=int(len(v)), mean=float(v.mean()), min=float(v.min()), p05=float(q[0]), p25=float(q[1]), median=float(q[2]),
                p75=float(q[3]), p95=float(q[4]), max=float(v.max()))


def analyse(name, X, logits, fit, radii, primary, points, k_fit=30):
    """Everything block C asks of one system.  `radii` = {name: per-point failure radius}; `primary` names the
    ground truth the pass criterion is scored on."""
    res = dict(n=int(len(X)), dim=int(X.shape[1]), d_global=float(fit.d_global), m=float(fit.m), primary_radius=primary)
    wall = {}
    f = logits.max(1); lab = logits.argmax(1)
    res['n_classes_predicted'] = int(len(np.unique(lab)))
    t0 = time.time(); ms = morse_smale(fit.W, f, lab); wall['morse_smale'] = time.time() - t0
    res.update(n_maxima=ms['n_maxima'], n_basins=ms['n_maxima'], n_separatrix_edges=ms['n_sep'], n_graph_edges=ms['n_edges'],
               n_edges_diff_basin=ms['n_edges_diff_basin'], n_edges_diff_label=ms['n_edges_diff_label'],
               n_label_pure_basins=ms['n_label_pure_basins'], basin_size=_dist(ms['basin_sizes']),
               frac_basin_max_label_agrees=float((lab[ms['basin']] == lab).mean()))
    t0 = time.time(); d_sep, nearest = separatrix_distance(fit.W, ms['sep_edges'], ms['sep_w']); wall['d_sep'] = time.time() - t0
    res['frac_finite_d_sep'] = float(np.isfinite(d_sep).mean())
    on_sep = np.zeros(len(X), bool); on_sep[ms['sep_edges'].ravel()] = True
    res['frac_nodes_on_a_separatrix_edge'] = float(on_sep.mean())
    d = int(np.clip(round(fit.d_global), 1, 8))
    t0 = time.time(); bc = boundary_curvature(X, logits, ms['sep_edges'], lab, d=d, k=k_fit); wall['boundary_curvature'] = time.time() - t0
    res['C3'] = dict(tangent_dim=d, k=bc['k'], n_params=bc['n_params'], underdetermined=bc['underdetermined'],
                     II=_dist(bc['II']), II_at_nearest_edge_of_probe_points=_dist(bc['II'][nearest[nearest >= 0]]),
                     b_norm=_dist(bc['bnorm']), twoQ_norm=_dist(bc['Qnorm']), rel_residual=_dist(bc['resid']),
                     n_degenerate_b=int(np.isnan(bc['II']).sum()))
    rp = rho_prime(d_sep, nearest, bc['II'])
    # POST-HOC (labelled, never scored): the same fit with k = 2 x n_params neighbours so it is over-determined
    k2 = min(2 * bc['n_params'], len(X))
    bc2 = boundary_curvature(X, logits, ms['sep_edges'], lab, d=d, k=k2)
    rp2 = rho_prime(d_sep, nearest, bc2['II'])

    z = np.sort(logits, 1); sm = z[:, -1] - z[:, -2]
    conf = _softmax_conf(logits.astype(np.float64))
    t0 = time.time(); mm = manifold_margin(fit.W, lab); wall['manifold_margin'] = time.time() - t0

    preds = dict(d_sep=d_sep, rho_prime=rp, softmax_margin=sm, softmax_confidence=conf, run1_manifold_margin=mm,
                 II_at_nearest=np.where(nearest >= 0, bc['II'][np.maximum(nearest, 0)] if len(bc['II']) else np.nan, np.nan),
                 POSTHOC_rho_prime_overdetermined=rp2)
    res['spearman'] = {}
    for rn, fr in radii.items():
        fin = np.isfinite(fr)
        common = fin & np.isfinite(d_sep) & np.isfinite(rp) & np.isfinite(mm)
        blk = dict(n_finite_radius=int(fin.sum()), n_common=int(common.sum()), n_distinct_radii=int(len(np.unique(np.round(fr[fin], 6)))))
        for pn, v in preds.items():
            blk[pn] = _sp(v, fr, common)
            blk[pn + '__own_mask'] = _sp(v, fr, fin & np.isfinite(v))
        ok = fin & np.isfinite(rp)
        blk['null_shuffled_rho_prime_seed0'] = _sp(shuffle_predictor(rp, 0), fr, fin & np.isfinite(shuffle_predictor(rp, 0)))
        nul = []
        for s in range(1, 201):
            sh = shuffle_predictor(rp, s); o = fin & np.isfinite(sh)
            nul.append(spearmanr(sh[o], fr[o]).statistic if o.sum() > 10 else np.nan)
        nul = np.array(nul)
        blk['null_shuffled_rho_prime_200'] = dict(mean_abs=float(np.nanmean(np.abs(nul))), p95_abs=float(np.nanpercentile(np.abs(nul), 95)),
                                                  frac_abs_below_0p05=float((np.abs(nul) < 0.05).mean()))
        if blk['d_sep']['r'] is not None:
            blk['abs_diff_d_sep_vs_softmax_margin'] = abs(blk['d_sep']['r'] - blk['softmax_margin']['r'])
            blk['abs_diff_rho_prime_vs_d_sep'] = abs(blk['rho_prime']['r'] - blk['d_sep']['r'])
        res['spearman'][rn] = blk
    okd = np.isfinite(d_sep)
    res['spearman_d_sep_vs_softmax_margin'] = _sp(d_sep, sm, okd)
    res['spearman_d_sep_vs_manifold_margin'] = _sp(d_sep, mm, okd & np.isfinite(mm))
    res['spearman_rho_prime_vs_d_sep'] = _sp(rp, d_sep, np.isfinite(rp))
    res['wall_s'] = wall
    P = res['spearman'][primary]
    print(f"[{name}] maxima {ms['n_maxima']} sep-edges {ms['n_sep']}/{ms['n_edges']} finite d_sep {res['frac_finite_d_sep']:.3f} "
          f"d={d} II median {res['C3']['II'].get('median')} | vs {primary}: " +
          ' '.join(f"{k}={P[k]['r']:+.3f}" if P[k]['r'] is not None else f"{k}=None" for k in preds) +
          f" null={P['null_shuffled_rho_prime_seed0']['r']:+.3f}", flush=True)
    for k, v in dict(preds, basin=ms['basin'], label=lab, f=f, nearest_sep_edge=nearest, **{'radius_' + a: b for a, b in radii.items()}).items():
        points[f'{name}__{k}'] = np.asarray(v)
    points[f'{name}__sep_edges'] = ms['sep_edges']; points[f'{name}__sep_w'] = ms['sep_w']
    points[f'{name}__sep_II'] = bc['II']; points[f'{name}__maxima'] = ms['maxima']
    Xc = X - X.mean(0); _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    points[f'{name}__pca2'] = (Xc @ Vt[:2].T).astype(np.float32)
    points[f'{name}__pca_var_share'] = (S[:2] ** 2 / (S ** 2).sum()).astype(np.float32)
    return res


# ------------------------------------------------------------------------------------------------ Z113
def do_z113(results, points):
    from analysis.z113_common import layer_from, checkpoints
    from analysis.z113_static import failure_radius_activation
    from systems.z113 import P, build_data
    from systems.z113_probe import load
    for tag, run, seed, frac in (('z113_clean_s0', 'runs/z113_clean_s0', 0, .3), ('z113_clean_s1', 'runs/z113_clean_s1', 1, .3),
                                 ('z113_cleanctrl_s2', 'runs/z113_cleanctrl_s2', 2, .5)):
        T0 = time.time()
        cks, steps = checkpoints(run); m = load(cks[-1], seed=seed)
        d = build_data(seed, 'clean', frac=frac)
        disk = np.load(f'{run}/data.npz')
        assert np.array_equal(disk['train_toks'], d['train_toks']), 'split on disk differs from build_data'
        rng = np.random.default_rng(0)                                     # run-1 representative probe (z113_static.main)
        sub = rng.choice(len(d['train_idx']), size=1200, replace=False)
        tk = d['train_toks'][sub]
        rp = dict(toks=tk, a=tk[:, 0], b=tk[:, 1], c=(tk[:, 0] + tk[:, 1]) % P, labels=d['train_y'][sub])
        t0 = time.time(); ld, psi = layer_from(m, rp, 'resid_mid'); t_j = time.time() - t0
        t0 = time.time(); fit = fit_manifold(ld.X, ld.J, metric='g'); t_f = time.time() - t0
        t0 = time.time(); fr, scale = failure_radius_activation(m, ld.X); t_r = time.time() - t0
        res = analyse(tag, ld.X, ld.logits, fit, dict(activation_bisection=fr), 'activation_bisection', points)
        res.update(step=int(steps[-1]), frac=frac, probe_acc=float((ld.logits.argmax(1) == ld.labels).mean()), radius_scale=float(scale))
        res['wall_s'].update(jacobians=t_j, fit_manifold=t_f, failure_radius=t_r, total=time.time() - T0)
        points[f'{tag}__a'] = rp['a']; points[f'{tag}__b'] = rp['b']; points[f'{tag}__c'] = rp['c']
        results[tag] = res
        del m, ld, fit; torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ ResNet
def _retry(fn, m, x, sizes=(100, 50, 25)):
    from analysis.resnet_analysis import _chunked
    for s in sizes:
        try:
            return _chunked(fn, m, x, size=s)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f'[resnet] CUDA OOM at chunk {s}, retrying smaller', flush=True)
    raise RuntimeError('CUDA OOM at every chunk size')


def do_resnet(results, points):
    from systems.resnet_cifar import make_model, load_cifar
    from analysis.resnet_depth import all_layers
    from analysis.resnet_analysis import _failure_radius_input, _gaussian_failure_radius
    T0 = time.time()
    xtr, ytr, xte, yte = load_cifar()
    m = make_model().to('cuda'); m.load_state_dict(torch.load(sorted(glob.glob('runs/resnet_clean_s0/ckpt_*.pt'))[-1])); m.eval()
    rng = np.random.default_rng(0)                                          # run-1 depth design: same draws, same order
    _sel = np.concatenate([rng.choice(np.where(ytr == c)[0], 150, replace=False) for c in range(10)])
    hold = rng.choice(len(xte), 1000, replace=False)
    del xtr
    A, lg = all_layers(m, xte[hold])
    Xh = A['penultimate']; del A
    W = m.fc.weight.detach().float().cpu().numpy()
    Jh = np.ascontiguousarray(np.broadcast_to(W[None], (len(Xh),) + W.shape)).astype(np.float32)
    t0 = time.time(); fit = fit_manifold(Xh, Jh, metric='g'); t_f = time.time() - t0
    t0 = time.time(); fr = _retry(_failure_radius_input, m, xte[hold]); t_r = time.time() - t0
    t0 = time.time(); frg = _retry(_gaussian_failure_radius, m, xte[hold]); t_g = time.time() - t0
    res = analyse('resnet_clean', Xh, lg, fit, dict(pgd=fr, gaussian6=frg), 'pgd', points)
    res['err_rate'] = float((lg.argmax(1) != yte[hold]).mean())
    res['wall_s'].update(fit_manifold=t_f, failure_radius_pgd=t_r, failure_radius_gaussian=t_g, total=time.time() - T0)
    results['resnet_clean'] = res
    del m; torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ GPT-2 (exploratory)
def do_gpt2(results, points):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from analysis.gpt2_explore import Tap, run, natural_windows
    T0 = time.time()
    tok = GPT2TokenizerFast.from_pretrained('gpt2'); model = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda').eval()
    tap = Tap(model)
    n_probe, n_hold, n5 = 2000, 1000, 600
    ids, _ = natural_windows(tok, n_probe + n_hold, 0)
    norms = []
    for s in range(0, len(ids), 500):                                       # run-1 noise scale = median ||H|| over all 3000 windows
        H_, _, _ = run(model, tap, ids[s:s + 500]); norms.append(np.linalg.norm(H_, axis=1)); del H_
    scale = float(np.median(np.concatenate(norms)))
    sel = np.arange(n_probe, n_probe + n5)
    H, Fo, Lg = run(model, tap, ids[sel])
    WU = model.lm_head.weight.detach().float().cpu().numpy()
    _, S, Vt = np.linalg.svd(WU, full_matrices=False); del WU
    O = ((Fo @ Vt.T) * S).astype(np.float32)
    base_pred = Lg.argmax(-1)
    t0 = time.time()
    radii = scale * np.geomspace(0.01, 1.0, 14); fr = np.full(n5, np.inf); g = np.random.default_rng(11)
    for r in radii:
        flips = np.zeros(n5)
        for _ in range(6):
            z = g.standard_normal((n5, H.shape[1])).astype(np.float32); z *= r / np.linalg.norm(z, axis=1, keepdims=True)
            _, _, Lz = run(model, tap, ids[sel], delta=z)
            flips += (Lz.argmax(-1) != base_pred).numpy(); del Lz
        hit = (flips >= 3) & ~np.isfinite(fr); fr[hit] = r
    t_r = time.time() - t0
    del model; torch.cuda.empty_cache()
    X = H.astype(np.float32); logits = Lg.numpy().astype(np.float32); del Lg
    t0 = time.time(); fit = fit_manifold(X, metric='secant', out=O); t_f = time.time() - t0
    res = analyse('gpt2_L6', X, logits, fit, dict(gaussian14=fr), 'gaussian14', points)
    res['exploratory'] = True
    res['wall_s'].update(fit_manifold=t_f, failure_radius=t_r, total=time.time() - T0)
    results['gpt2_L6'] = res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--systems', default='z113,resnet')
    a = ap.parse_args()
    os.makedirs('results/run2', exist_ok=True)
    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    points = dict(np.load(OUT_NPZ)) if os.path.exists(OUT_NPZ) else {}
    for s in a.systems.split(','):
        t0 = time.time()
        dict(z113=do_z113, resnet=do_resnet, gpt2=do_gpt2)[s](results, points)
        print(f'[{s}] {time.time() - t0:.0f}s', flush=True)
        json.dump(results, open(OUT_JSON, 'w'), indent=1, default=float)
        np.savez_compressed(OUT_NPZ, **points)
    score(results)
    json.dump(results, open(OUT_JSON, 'w'), indent=1, default=float)


def score(results):
    """Pre-registered criteria, by the letter."""
    sysn = [k for k in ('z113_clean_s0', 'z113_clean_s1', 'z113_cleanctrl_s2', 'resnet_clean', 'gpt2_L6') if k in results]
    if not sysn:
        return
    P = {k: results[k]['spearman'][results[k]['primary_radius']] for k in sysn}
    rp = {k: P[k]['rho_prime']['r'] for k in sysn}; smg = {k: P[k]['softmax_margin']['r'] for k in sysn}
    z = [k for k in sysn if k.startswith('z113')]
    v = dict(systems=sysn, rho_prime=rp, softmax_margin=smg,
             resnet_rho_prime_gt_0p90=bool(rp.get('resnet_clean') is not None and rp.get('resnet_clean', -9) > 0.90) if 'resnet_clean' in rp else None,
             z113_all_three_gt_0p5=bool(len(z) == 3 and all(rp[k] is not None and rp[k] > 0.5 for k in z)),
             positive_on_every_system=bool(all(rp[k] is not None and rp[k] > 0 for k in sysn)),
             beats_softmax_margin_on=[k for k in sysn if rp[k] is not None and rp[k] > smg[k]],
             null_abs_below_0p05={k: bool(abs(P[k]['null_shuffled_rho_prime_seed0']['r']) < 0.05) for k in sysn},
             null_values={k: P[k]['null_shuffled_rho_prime_seed0']['r'] for k in sysn},
             d_sep_within_0p02_of_softmax_margin={k: (bool(P[k]['abs_diff_d_sep_vs_softmax_margin'] < 0.02)) for k in sysn},
             abs_diff_d_sep_vs_softmax_margin={k: P[k]['abs_diff_d_sep_vs_softmax_margin'] for k in sysn})
    v['PASS_C'] = bool(v['resnet_rho_prime_gt_0p90'] and v['z113_all_three_gt_0p5'] and v['positive_on_every_system']
                       and len(v['beats_softmax_margin_on']) >= 1 and all(v['null_abs_below_0p05'].values()))
    results['_verdict'] = v
    print(json.dumps(v, indent=1), flush=True)


if __name__ == '__main__':
    main()
