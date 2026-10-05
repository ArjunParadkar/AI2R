"""SPEC2 block A -- depth-scale bipersistence of class fibers (A1, A2, A3).

Run:  OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 python analysis/run2_blockA.py
Writes results/run2/blockA.json, results/run2/blockA_bipersistence.npz, figures/run2_bipersistence.png.

Every threshold below is SPEC2's.  Nothing is retrained; final checkpoints only.
  A1  beta0(F_y^l) at eps = 0.5 x (largest MST edge of the layer's full cloud);  (a) Euclidean = PRIMARY,
      (b) geodesic on the k=15 pullback graph (secant weights at the ResNet conv layers).
      d*(y) = first layer with beta0 = 1 (L if never);  U = 1 - E_y[d*/L].
      secondary: eps = 0.5 x the scale at which 99% of the points are in one component.
  A2  beta0(l, eps) on 40 eps values in [0, 2] x median pairwise distance of the layer's full cloud.
  A3  GW(F_y^input, F_y^l), POT gromov_wasserstein2, square loss, uniform weights, median-normalised.
"""
import sys, os, json, glob, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whiteboxtopo.depthprofile import (euclidean_distance_matrix, single_linkage, median_pairwise,
                                       fiber_beta0_profile, depth_of_connection, untangling_index,
                                       pullback_distance_matrix, gw_fibers,
                                       fiber_connection_scales, gw_product_coupling_loss, beta0_at)
from whiteboxtopo.nulls import feature_shuffle, label_permute
from whiteboxtopo.extract import jacobians
from scipy.sparse.csgraph import connected_components

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_JSON, OUT_NPZ, OUT_FIG = 'results/run2/blockA.json', 'results/run2/blockA_bipersistence.npz', 'figures/run2_bipersistence.png'
GRID_MULT = np.linspace(0.0, 2.0, 40)
N_PERM = 20                       # label-permutation seeds; seed 0 is the pre-registered number
GW_MAX_ITER = 1000
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------ one layer, both metrics
def scales_of(D):
    h, s = single_linkage(D)
    n = len(h) + 1
    hit = np.where(s >= 0.99 * n)[0]
    nn = np.sort(D, 1)[:, 1]
    sc = dict(mst_largest_edge=float(h.max()), robust_scale_99=float(h[hit[0]]), median_pairwise=median_pairwise(D),
              median_nn_distance=float(np.median(nn)))
    # diagnostics, no criterion: what the FULL cloud looks like at the pre-registered scales
    sc['full_cloud_beta0_at_eps_mst'] = int(n - np.searchsorted(h, 0.5 * sc['mst_largest_edge'], side='right'))
    sc['full_cloud_beta0_at_eps_robust'] = int(n - np.searchsorted(h, 0.5 * sc['robust_scale_99'], side='right'))
    sc['frac_points_with_any_neighbour_within_eps_mst'] = float((nn <= 0.5 * sc['mst_largest_edge']).mean())
    return sc


def layer_profile(D, labels, classes):
    """beta0 per class at the two pre-registered scales + on the A2 grid + under permuted labels."""
    sc = scales_of(D)
    eps = [0.5 * sc['mst_largest_edge'], 0.5 * sc['robust_scale_99']]
    grid = GRID_MULT * sc['median_pairwise']
    _, B, C = fiber_beta0_profile(D, labels, eps, eps_grid=grid, classes=classes)
    fc = fiber_connection_scales(D, labels, classes) / sc['mst_largest_edge']      # beta0 = 1  <=>  ratio <= 0.5
    sc['fiber_connect_scale_over_cloud_connect_scale'] = dict(mean=float(fc.mean()), min=float(fc.min()), max=float(fc.max()))
    Bperm = np.stack([fiber_beta0_profile(D, label_permute(labels, s), eps, classes=classes)[1] for s in range(N_PERM)])
    return sc, B, C, Bperm          # B (n_cls, 2), C (n_cls, 40), Bperm (N_PERM, n_cls, 2)


def analyse(tag, layer_names, get_layer, labels, input_D_of, gw_classes, shuffle_gw_layers=()):
    """get_layer(name) -> (X, J or None, out or None).  Returns (json dict, npz dict)."""
    labels = np.asarray(labels); classes = np.unique(labels); L = len(layer_names)
    res = dict(layers={}, n_points=int(len(labels)), n_classes=int(len(classes)),
               fiber_sizes={int(c): int((labels == c).sum()) for c in classes})
    b = {m: np.zeros((len(classes), L, 2), int) for m in 'eg'}
    bp = {m: np.zeros((N_PERM, len(classes), L, 2), int) for m in 'eg'}
    curves = {m: np.zeros((len(classes), L, 40), int) for m in 'eg'}
    gw = {nm: {} for nm in layer_names}; gw_ref = {nm: {} for nm in layer_names}; gw_shuf = {nm: {} for nm in shuffle_gw_layers}
    for li, nm in enumerate(layer_names):
        tl = time.time()
        X, J, out = get_layer(nm); wname = 'secant' if J is None else 'g'
        De = euclidean_distance_matrix(X)
        sc_e, b['e'][:, li], curves['e'][:, li], bp['e'][:, :, li] = layer_profile(De, labels, classes)
        t_e = time.time() - tl
        # A3 on the Euclidean fiber
        tg = time.time()
        for c in gw_classes:
            sel = np.where(labels == c)[0][:300]
            gw[nm][int(c)] = gw_fibers(input_D_of(sel), De[np.ix_(sel, sel)], max_iter=GW_MAX_ITER)[0]
            gw_ref[nm][int(c)] = gw_product_coupling_loss(input_D_of(sel), De[np.ix_(sel, sel)])
        if nm in shuffle_gw_layers:
            Ds = euclidean_distance_matrix(feature_shuffle(X, seed=0))
            for c in gw_classes:
                sel = np.where(labels == c)[0][:300]
                gw_shuf[nm][int(c)] = gw_fibers(input_D_of(sel), Ds[np.ix_(sel, sel)], max_iter=GW_MAX_ITER)[0]
            del Ds
        t_gw = time.time() - tg
        del De
        tp = time.time()
        Dg, ncomp = pullback_distance_matrix(X, J=J, out=None if J is not None else out, k=15)
        sc_g, b['g'][:, li], curves['g'][:, li], bp['g'][:, :, li] = layer_profile(Dg, labels, classes)
        if ncomp > 1:
            # POST-HOC, never scored: the cross-component fill (2 x largest finite distance) IS the largest MST
            # edge here, so the pre-registered eps equals the largest finite distance.  Same rule on finite edges:
            h, _ = single_linkage(Dg); fin = h[h < Dg.max()]
            ef = 0.5 * float(fin.max())
            bf = fiber_beta0_profile(Dg, labels, [ef], classes=classes)[1][:, 0]
            sc_g['POSTHOC_finite_edges_only'] = dict(largest_finite_mst_edge=float(fin.max()), mean_beta0=float(bf.mean()),
                                                     frac_connected=float((bf == 1).mean()),
                                                     component_sizes=sorted(np.bincount(connected_components(Dg < Dg.max(), directed=False)[1]).tolist(), reverse=True))
        del Dg, J
        res['layers'][nm] = dict(
            dim=int(X.shape[1]), euclid=sc_e, pullback=dict(sc_g, graph_components=ncomp, weights=wname),
            beta0_euclid={int(c): int(v) for c, v in zip(classes, b['e'][:, li, 0])},
            beta0_pullback={int(c): int(v) for c, v in zip(classes, b['g'][:, li, 0])},
            beta0_euclid_robust={int(c): int(v) for c, v in zip(classes, b['e'][:, li, 1])},
            beta0_pullback_robust={int(c): int(v) for c, v in zip(classes, b['g'][:, li, 1])},
            beta0_mean=dict(euclid=float(b['e'][:, li, 0].mean()), pullback=float(b['g'][:, li, 0].mean()),
                            euclid_robust=float(b['e'][:, li, 1].mean()), pullback_robust=float(b['g'][:, li, 1].mean())),
            frac_connected=dict(euclid=float((b['e'][:, li, 0] == 1).mean()), pullback=float((b['g'][:, li, 0] == 1).mean()),
                                euclid_robust=float((b['e'][:, li, 1] == 1).mean()), pullback_robust=float((b['g'][:, li, 1] == 1).mean())),
            gw_mean=float(np.mean(list(gw[nm].values()))),
            wall_s=dict(euclid_beta0=t_e, gw=t_gw, pullback_beta0=time.time() - tp, total=time.time() - tl))
        r = res['layers'][nm]
        log(f"[{tag}] {nm:11s} dim {r['dim']:4d} | eps_mst/median e={0.5 * sc_e['mst_largest_edge'] / sc_e['median_pairwise']:.3f} "
            f"g={0.5 * sc_g['mst_largest_edge'] / sc_g['median_pairwise']:.3f} | mean b0 e={r['beta0_mean']['euclid']:.1f} g={r['beta0_mean']['pullback']:.1f} "
            f"rob e={r['beta0_mean']['euclid_robust']:.1f} g={r['beta0_mean']['pullback_robust']:.1f} | connected e={r['frac_connected']['euclid']:.2f} "
            f"g={r['frac_connected']['pullback']:.2f} | GW={r['gw_mean']:.4f} | graph comps {ncomp} ({r['wall_s']['total']:.1f}s)")
    # d*, U under each (metric, scale)
    for m, mname in (('e', 'euclid'), ('g', 'pullback')):
        for si, sname in ((0, ''), (1, '_robust')):
            d, never = depth_of_connection(b[m][:, :, si])
            key = mname + sname
            res[f'dstar_{key}'] = {int(c): int(v) for c, v in zip(classes, d)}
            res[f'never_connected_{key}'] = int(never.sum())
            res[f'all_dstar_equal_L_{key}'] = bool((d == L).all())
            res[f'U_{key}'] = untangling_index(d, L)
            Un = [untangling_index(depth_of_connection(bp[m][s, :, :, si])[0], L) for s in range(N_PERM)]
            res[f'null_label_permute_U_{key}'] = dict(seed0=Un[0], mean=float(np.mean(Un)), min=float(np.min(Un)),
                                                      max=float(np.max(Un)), all_seeds=Un)
    res['gw'] = gw; res['gw_feature_shuffled'] = gw_shuf; res['gw_at_product_coupling'] = gw_ref
    npz = {f'{tag}__beta0_euclid': curves['e'], f'{tag}__beta0_pullback': curves['g'], f'{tag}__classes': classes,
           f'{tag}__fiber_sizes': np.array([(labels == c).sum() for c in classes]),
           f'{tag}__eps_mst_over_median_euclid': np.array([0.5 * res['layers'][n]['euclid']['mst_largest_edge'] / res['layers'][n]['euclid']['median_pairwise'] for n in layer_names]),
           f'{tag}__eps_mst_over_median_pullback': np.array([0.5 * res['layers'][n]['pullback']['mst_largest_edge'] / res['layers'][n]['pullback']['median_pairwise'] for n in layer_names]),
           f'{tag}__median_euclid': np.array([res['layers'][n]['euclid']['median_pairwise'] for n in layer_names]),
           f'{tag}__median_pullback': np.array([res['layers'][n]['pullback']['median_pairwise'] for n in layer_names])}
    log(f"[{tag}] U euclid={res['U_euclid']:.3f} (perm-null seed0 {res['null_label_permute_U_euclid']['seed0']:.3f}) "
        f"pullback={res['U_pullback']:.3f} | robust: euclid={res['U_euclid_robust']:.3f} pullback={res['U_pullback_robust']:.3f}")
    return res, npz


# ------------------------------------------------------------------ ResNet-18
RES_LAYERS = ['layer1', 'layer2', 'layer3', 'layer4', 'penultimate']


def resnet_acts(m, x, bs=250):
    from analysis.resnet_depth import all_layers
    while True:
        try:
            return all_layers(m, x, bs=bs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); bs = max(bs // 2, 8); log(f'CUDA OOM -> batch {bs}')


def run_resnet(tag, rundir, xtr, ytr_used):
    from systems.resnet_cifar import make_model
    t0 = time.time()
    rng = np.random.default_rng(0)                         # run-1 seed and run-1 draw order
    sel = np.concatenate([rng.choice(np.where(ytr_used == c)[0], 150, replace=False) for c in range(10)])
    lab = ytr_used[sel]
    m = make_model().to(DEV); m.load_state_dict(torch.load(sorted(glob.glob(f'{rundir}/ckpt_*.pt'))[-1], map_location=DEV)); m.eval()
    A, lg = resnet_acts(m, xtr[sel])
    W = m.fc.weight.detach().float().cpu().numpy()
    del m; torch.cuda.empty_cache()
    pca = {}
    for L in RES_LAYERS:
        if A[L].shape[1] > 1024:                           # as run 1: layers wider than 1024 -> 256 PCs
            Xc = A[L] - A[L].mean(0); U, S, _ = np.linalg.svd(Xc, full_matrices=False)
            pca[L] = dict(from_dim=int(A[L].shape[1]), variance_kept=float((S[:256] ** 2).sum() / (S ** 2).sum()))
            A[L] = (U[:, :256] * S[:256]).astype(np.float32); del U, S, Xc
    Xin = xtr[sel].reshape(len(sel), -1).astype(np.float64)

    def get_layer(nm):
        if nm == 'penultimate':
            return A[nm], np.ascontiguousarray(np.broadcast_to(W[None], (len(sel),) + W.shape)).astype(np.float32), None
        return A[nm], None, lg

    res, npz = analyse(f'resnet_{tag}', RES_LAYERS, get_layer, lab, lambda s: euclidean_distance_matrix(Xin[s]),
                       gw_classes=list(range(10)), shuffle_gw_layers=RES_LAYERS)
    res['pca'] = pca; res['probe_train_acc_vs_trained_labels'] = float((lg.argmax(1) == lab).mean())
    res['wall_s'] = time.time() - t0
    return res, npz


# ------------------------------------------------------------------ Z113
Z_LAYERS = ['embed', 'attn', 'resid_mid', 'resid_post']


def run_z113(tag, rundir, seed, mode):
    from systems.z113 import build_data, P, NULL
    from systems.z113_probe import load
    from analysis.z113_common import checkpoints
    t0 = time.time()
    d = build_data(seed, mode, frac=0.3)
    toks, y = d['train_toks'], d['train_y']
    assert (toks[:, 2] == NULL).all()
    ck = checkpoints(rundir)[0][-1]
    model = load(ck, DEV, seed=seed)
    with torch.no_grad():
        t = torch.as_tensor(toks, device=DEV)
        x0 = model.embed(t); att = model.attn(x0)
        mid = x0[:, -1] + att[:, -1]; post = mid + model.mlp(mid)
        lg = (post @ model.W_U).cpu().numpy()
        c_hint, c_eq = x0[0, 2].clone(), x0[0, 3].clone()
        assert torch.allclose(x0[:, 3], c_eq[None]) and torch.allclose(x0[:, 2], c_hint[None])
        A = dict(embed=torch.cat([x0[:, 0], x0[:, 1]], 1).cpu().numpy(), attn=att[:, -1].cpu().numpy(),
                 resid_mid=mid.cpu().numpy(), resid_post=post.cpu().numpy())
    A = {k: v.astype(np.float32) for k, v in A.items()}

    psi_mid = lambda h: (h + model.mlp(h)) @ model.W_U

    def psi_embed(e):
        n = e.shape[0]; dm = model.d_model
        x = torch.stack([e[:, :dm], e[:, dm:], c_hint[None].expand(n, -1), c_eq[None].expand(n, -1)], 1)
        x = x + model.attn(x)
        return psi_mid(x[:, -1])
    psis = dict(embed=psi_embed, attn=lambda u: psi_mid(u + c_eq[None]), resid_mid=psi_mid, resid_post=lambda h: h @ model.W_U)
    with torch.no_grad():                                   # the remaining-network maps must reproduce the logits
        for nm in Z_LAYERS:
            err = float(np.abs(psis[nm](torch.as_tensor(A[nm][:512], device=DEV)).cpu().numpy() - lg[:512]).max())
            assert err < 1e-3 * max(1.0, np.abs(lg).max()), (nm, err)

    def get_layer(nm):
        bs = 256
        while True:
            try:
                J = jacobians(psis[nm], A[nm], batch=bs, device=DEV).astype(np.float32); break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); bs = max(bs // 2, 8); log(f'CUDA OOM -> jacobian batch {bs}')
        torch.cuda.empty_cache()
        return A[nm], J, None

    a, bb = toks[:, 0], toks[:, 1]

    def input_D(s):                                         # one-hot(a) (+) one-hot(b): d^2 = 2[a!=a'] + 2[b!=b']
        return np.sqrt(2.0 * (a[s][:, None] != a[s][None]) + 2.0 * (bb[s][:, None] != bb[s][None]))

    gw_classes = [(6 * k) % P for k in range(20)]           # 0,6,...,108 and 114 mod 113 = 1  (see Deviations)
    res, npz = analyse(f'z113_{tag}', Z_LAYERS, get_layer, y, input_D, gw_classes, shuffle_gw_layers=Z_LAYERS)
    res['gw_classes'] = gw_classes; res['checkpoint'] = ck
    res['probe_train_acc_vs_trained_labels'] = float((lg.argmax(1) == y).mean())
    res['wall_s'] = time.time() - t0
    del model; torch.cuda.empty_cache()
    return res, npz


# ------------------------------------------------------------------ scoring, by the letter
def score(out):
    S = {}
    for sysname, L in (('resnet', 5), ('z113', 4)):
        c, s = out[f'{sysname}_clean'], out[f'{sysname}_scrambled']
        for key in ('euclid', 'pullback', 'euclid_robust', 'pullback_robust'):
            S[f'{sysname}_U_clean_minus_scrambled_{key}'] = c[f'U_{key}'] - s[f'U_{key}']
        S[f'{sysname}_criterion_U_gap_gt_0.4_primary'] = bool(S[f'{sysname}_U_clean_minus_scrambled_euclid'] > 0.4)
        S[f'{sysname}_null_label_permuted_U_clean_euclid'] = c['null_label_permute_U_euclid']['seed0']
        S[f'{sysname}_null_label_permuted_within_0.05'] = bool(abs(c['null_label_permute_U_euclid']['seed0']) < 0.05)
        S[f'{sysname}_kill_all_dstar_eq_L_both_models_euclid'] = bool(c['all_dstar_equal_L_euclid'] and s['all_dstar_equal_L_euclid'])
        S[f'{sysname}_never_connected_euclid'] = dict(clean=c['never_connected_euclid'], scrambled=s['never_connected_euclid'],
                                                      n_classes=c['n_classes'])
    gc = np.mean(list(out['resnet_clean']['gw']['penultimate'].values())); gs = np.mean(list(out['resnet_scrambled']['gw']['penultimate'].values()))
    S['resnet_penultimate_EGW_clean'], S['resnet_penultimate_EGW_scrambled'] = float(gc), float(gs)
    S['resnet_penultimate_EGW_ratio_scr_over_clean'] = float(gs / gc)
    S['criterion_EGW_scr_gt_2x_clean'] = bool(gs > 2 * gc)
    nc = np.mean(list(out['resnet_clean']['gw_feature_shuffled']['penultimate'].values())); ns = np.mean(list(out['resnet_scrambled']['gw_feature_shuffled']['penultimate'].values()))
    S['null_feature_shuffle_EGW_clean'], S['null_feature_shuffle_EGW_scrambled'] = float(nc), float(ns)
    S['null_feature_shuffle_EGW_ratio'] = float(ns / nc)
    S['null_feature_shuffle_ratio_in_[0.8,1.25]'] = bool(0.8 <= ns / nc <= 1.25)
    S['kill_fires'] = bool(S['resnet_kill_all_dstar_eq_L_both_models_euclid'] and S['z113_kill_all_dstar_eq_L_both_models_euclid'])
    S['kill_fires_per_system'] = dict(resnet=S['resnet_kill_all_dstar_eq_L_both_models_euclid'], z113=S['z113_kill_all_dstar_eq_L_both_models_euclid'])
    S['PASS_A_by_the_letter'] = bool(S['resnet_criterion_U_gap_gt_0.4_primary'] and S['z113_criterion_U_gap_gt_0.4_primary'] and S['criterion_EGW_scr_gt_2x_clean'])
    # GW ratio at every layer, both systems (reported, no criterion)
    for sysname, layers in (('resnet', RES_LAYERS), ('z113', Z_LAYERS)):
        S[f'{sysname}_EGW_by_layer'] = {nm: dict(
            clean=float(np.mean(list(out[f'{sysname}_clean']['gw'][nm].values()))),
            scrambled=float(np.mean(list(out[f'{sysname}_scrambled']['gw'][nm].values()))),
            shuffled_clean=float(np.mean(list(out[f'{sysname}_clean']['gw_feature_shuffled'][nm].values()))),
            shuffled_scrambled=float(np.mean(list(out[f'{sysname}_scrambled']['gw_feature_shuffled'][nm].values())))) for nm in layers}
    return S


# ------------------------------------------------------------------ figure
def make_figure(npz_path=OUT_NPZ, fig_path=OUT_FIG):
    from analysis.figstyle import setup, INK, INK2, SURFACE
    import matplotlib.pyplot as plt
    setup()
    Z = np.load(npz_path)
    rows = [('resnet_clean', 'ResNet-18 clean', RES_LAYERS, [0, 3, 8]), ('resnet_scrambled', 'ResNet-18 scrambled', RES_LAYERS, [0, 3, 8]),
            ('z113_clean', 'Z113 clean (s1)', Z_LAYERS, [0, 36, 72]), ('z113_scrambled', 'Z113 scrambled (s0)', Z_LAYERS, [0, 36, 72])]
    fig, axes = plt.subplots(4, 3, figsize=(11, 10.5), constrained_layout=True)
    im = None
    for r, (tag, title, layers, cls) in enumerate(rows):
        C = Z[f'{tag}__beta0_euclid']; classes = list(Z[f'{tag}__classes']); sizes = Z[f'{tag}__fiber_sizes']
        em = Z[f'{tag}__eps_mst_over_median_euclid']
        for k, c in enumerate(cls):
            ax = axes[r, k]; i = classes.index(c); n = max(int(sizes[i]), 2)
            img = np.log(np.maximum(C[i], 1)) / np.log(n)               # 0 = one component, 1 = every point alone
            im = ax.imshow(img, aspect='auto', origin='upper', cmap='Greys', vmin=0, vmax=1,
                           extent=(GRID_MULT[0], GRID_MULT[-1], len(layers), 0), interpolation='nearest')
            for li in range(len(layers)):                               # the A1 scale at each layer
                ax.plot([em[li], em[li]], [li + 0.12, li + 0.88], color=SURFACE, lw=3.2, solid_capstyle='butt')
                ax.plot([em[li], em[li]], [li + 0.12, li + 0.88], color=INK, lw=1.4, ls=(0, (2, 1.2)))
                first = np.where(C[i, li] == 1)[0]
                txt = f'β0=1 from {GRID_MULT[first[0]]:.2f}' if len(first) else 'never 1'
                ax.text(1.97, li + 0.5, txt, ha='right', va='center', fontsize=7, color=INK,
                        bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.2, alpha=1.0))
            ax.set_yticks(np.arange(len(layers)) + 0.5); ax.set_yticklabels(layers if k == 0 else [''] * len(layers))
            ax.grid(False); ax.set_title(f'{title} · class {c} (n={int(sizes[i])})', fontsize=9)
            if r == 3:
                ax.set_xlabel('ε / median pairwise distance')
            if k == 0:
                ax.set_ylabel('layer (depth ↓)')
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.01)
    cb.set_label('log β0 / log n   (0 = fiber is one component, 1 = every point alone)')
    fig.suptitle('Fibered barcode β0(layer, ε) of class fibers, Euclidean metric.\nDashed tick = the A1 scale (0.5 × largest MST edge of the full cloud); ε is in units of each layer\'s median pairwise distance.',
                 fontsize=10, x=0.01, ha='left', color=INK)
    fig.savefig(fig_path, dpi=300); plt.close(fig)


def main():
    os.makedirs('results/run2', exist_ok=True)
    out, npz = {}, {'eps_grid_multipliers': GRID_MULT}

    def save():
        json.dump(out, open(OUT_JSON, 'w'), indent=1, default=float); np.savez_compressed(OUT_NPZ, **npz)

    for tag, rd, seed, mode in (('clean', 'runs/z113_clean_s1', 1, 'clean'), ('scrambled', 'runs/z113_scrambled_s0', 0, 'scrambled')):
        r, z = run_z113(tag, rd, seed, mode); out[f'z113_{tag}'] = r; npz.update(z); save()
    from systems.resnet_cifar import load_cifar
    xtr, ytr, _, _ = load_cifar()
    ytr_s = np.random.default_rng(0).integers(0, 10, size=len(ytr))
    for tag, rd, yy in (('clean', 'runs/resnet_clean_s0', ytr), ('scrambled', 'runs/resnet_scrambled_s0', ytr_s)):
        r, z = run_resnet(tag, rd, xtr, yy); out[f'resnet_{tag}'] = r; npz.update(z); save()
    out['score'] = score(out)
    out['meta'] = dict(grid_multipliers=GRID_MULT.tolist(), n_label_permutations=N_PERM, gw_max_iter=GW_MAX_ITER,
                       wall_s_total=time.time() - T0, device=DEV)
    save()
    log('SCORE', json.dumps({k: v for k, v in out['score'].items() if not k.endswith('by_layer')}, indent=1))
    make_figure()
    log('done')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'figure':
        make_figure()
    else:
        main()
