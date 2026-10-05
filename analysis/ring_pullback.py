"""Ring RNN, P1: is the circle found WITHOUT being told the heading?

Three unsupervised routes, none of which sees theta until the final correlation:
  euclid   Rips on the raw hidden states (the previous attempt: r ~ 0.78)
  g-geo    Rips on all-pairs geodesic distance of the pullback-weighted kNN graph -- the
           distance the spec's object actually carries (sec 2.2-2.4)
  output   Rips on psi(h): the cloud as the output sees it (the rank-2 limit of the pullback)
and a description of WHAT ELSE the state carries (the reason the Euclidean cloud is not a circle).
"""
import sys, os, json, glob, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.ring_rnn import RingRNN
from whiteboxtopo.extract import jacobians, logits_of
from whiteboxtopo.manifold import fit_manifold, geodesic_distances
from whiteboxtopo.topology import persistence, circular_coordinates, circular_correlation, circular_rank_correlation
from whiteboxtopo.nulls import feature_shuffle


def cloud(m, n, seed, sigma=0.3, T=25):
    rng = np.random.default_rng(seed)
    v = torch.as_tensor(rng.normal(0, sigma, size=(n, T)), dtype=torch.float32, device='cuda')
    th0 = torch.as_tensor(rng.uniform(0, 2 * np.pi, size=n), dtype=torch.float32, device='cuda')
    with torch.no_grad():
        h = m(v, th0)
    th = (th0 + torch.cumsum(v, 1)[:, -1]).cpu().numpy()
    return h.cpu().numpy().astype(np.float32), np.mod(th, 2 * np.pi), v.cpu().numpy()


def top_circle(D_or_X, theta, distance_matrix=False):
    res = persistence(D_or_X, distance_matrix=distance_matrix)
    h1 = res['dgms'][1]
    if len(h1) == 0:
        return dict(r=0.0, r_rank=0.0, ratio=0.0, n_half=0)
    life = h1[:, 1] - h1[:, 0]; srt = np.sort(life); j = int(np.argmax(life))
    X = None if distance_matrix else D_or_X
    cc = circular_coordinates(X, h1, res['cocycles'][1], D=res['D'], class_idx=j)
    if cc is None:
        return dict(r=0.0, r_rank=0.0, ratio=0.0, n_half=0)
    return dict(r=float(circular_correlation(cc['theta'], theta)),
                r_rank=float(circular_rank_correlation(cc['theta'], theta)),
                ratio=float(srt[-1] / max(srt[-2], 1e-9)) if len(srt) > 1 else 99.0,
                n_half=int((life > 0.5 * srt[-1]).sum()), rel_life=float(srt[-1] / res['median_dist']))


def main(n=1200):
    out = {}
    for seed in (0, 1, 2):
        ck = sorted(glob.glob(f'runs/ring_s{seed}/ckpt_*.pt'))[-1]
        m = RingRNN(seed=seed).cuda(); m.load_state_dict(torch.load(ck)); m.eval()
        X, th, v = cloud(m, n, seed=7)
        # what the state carries: variance explained by heading harmonics, last velocities
        Xc = X - X.mean(0)
        def r2(F):
            F = np.c_[np.ones(len(F)), F]; B = np.linalg.lstsq(F, Xc, rcond=None)[0]
            return 1 - ((Xc - F @ B) ** 2).sum() / (Xc ** 2).sum()
        H = lambda K: np.concatenate([np.c_[np.cos(k * th), np.sin(k * th)] for k in range(1, K + 1)], 1)
        carry = dict(r2_heading_k1=r2(H(1)), r2_heading_k1to4=r2(H(4)),
                     r2_heading4_plus_last3v=r2(np.c_[H(4), v[:, -3:]]),
                     r2_heading4_x_last3v=r2(np.c_[H(4), v[:, -3:], (H(2)[:, :, None] * v[:, None, -3:]).reshape(len(th), -1)]))
        ev = np.linalg.svd(Xc, compute_uv=False) ** 2; ev /= ev.sum()
        carry['pca_var_top6'] = [float(e) for e in ev[:6]]
        t0 = time.time()
        J = jacobians(m.psi, X).astype(np.float32)
        fit = fit_manifold(X, J, metric='g')
        Dg = geodesic_distances(fit.W)
        fin = np.isfinite(Dg); Dg[~fin] = Dg[fin].max() * 2
        Dg = np.maximum(Dg, Dg.T); np.fill_diagonal(Dg, 0)
        Y = logits_of(m.psi, X).astype(np.float32)
        res = dict(euclid=top_circle(X, th), g_geodesic=top_circle(Dg, th, True), output=top_circle(Y, th),
                   n_components_g=int(fit.n_components()), m=fit.m, carry=carry, wall_s=time.time() - t0)
        Xs = feature_shuffle(X, 0); Js = jacobians(m.psi, Xs).astype(np.float32); fs = fit_manifold(Xs, Js, metric='g')
        Ds = geodesic_distances(fs.W); f2 = np.isfinite(Ds); Ds[~f2] = Ds[f2].max() * 2; Ds = np.maximum(Ds, Ds.T); np.fill_diagonal(Ds, 0)
        res['null_feature_shuffle_g_geodesic'] = top_circle(Ds, th, True)
        out[f's{seed}'] = res
        print(f"[ring s{seed}] euclid r={res['euclid']['r']:.4f} (ratio {res['euclid']['ratio']:.1f}) | g-geo r={res['g_geodesic']['r']:.4f} "
              f"rank {res['g_geodesic']['r_rank']:.4f} (ratio {res['g_geodesic']['ratio']:.1f}, n_half {res['g_geodesic']['n_half']}) | "
              f"output r={res['output']['r']:.4f} | null g-geo r={res['null_feature_shuffle_g_geodesic']['r']:.3f} ratio {res['null_feature_shuffle_g_geodesic']['ratio']:.1f}", flush=True)
        print('     carries:', {k: (round(v_, 3) if not isinstance(v_, list) else [round(e, 3) for e in v_]) for k, v_ in carry.items()}, flush=True)
    json.dump(out, open('results/ring_pullback.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
