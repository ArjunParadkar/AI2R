"""Run 2, Block G: m_plateau -- the intrinsic-dimension plateau score, replacing the spectral-gap m.  SPEC2.md section G.

ID(k) is the general-rank MLE at ranks (k//2, k) (identical to whiteboxtopo.manifold.intrinsic_dim's d_global), evaluated at
k in {5,10,20,40,80,160}.  m_plateau = log-length of the longest contiguous run of k's with max/min - 1 < 0.15, over log(160/5).
"""
import sys, os, json, glob, time
import numpy as np, torch
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whiteboxtopo.manifold import fit_manifold

KS = [5, 10, 20, 40, 80, 160]


def id_curve(X):
    n = len(X); ks = [k for k in KS if k <= n - 2]
    dist = NearestNeighbors(n_neighbors=max(ks) + 1).fit(X).kneighbors(X)[0]
    H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, max(ks) + 3))])
    out = []
    for k in ks:
        a = max(1, k // 2)
        lr = np.log(np.maximum(dist[:, k], 1e-12) / np.maximum(dist[:, a], 1e-12))
        out.append((H[k - 1] - H[a - 1]) / max(lr.mean(), 1e-12))
    return ks, np.array(out)


def m_plateau(ks, ids, tol=0.15):
    best = 0.0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            seg = ids[i:j + 1]
            if seg.max() / seg.min() - 1 < tol:
                best = max(best, np.log(ks[j] / ks[i]))
    return float(best / np.log(KS[-1] / KS[0]))


def score(X, name, old_m=None, n_boot=20):
    t0 = time.time(); X = np.asarray(X, np.float32)
    ks, ids = id_curve(X); m = m_plateau(ks, ids)
    rng = np.random.default_rng(0); boots = []
    for _ in range(n_boot):
        sub = rng.choice(len(X), int(0.8 * len(X)), replace=False); kb, ib = id_curve(X[sub]); boots.append(m_plateau(kb, ib))
    boots = np.array(boots); cv = float(boots.std() / boots.mean()) if boots.mean() > 0 else float('nan')
    if old_m is None:
        old_m = dict(default=fit_manifold(X).m, k15=fit_manifold(X, score_k=15).m)
    r = dict(n=int(len(X)), D=int(X.shape[1]), ks=ks, id_curve=[float(v) for v in ids], m_plateau=m, boot_mean=float(boots.mean()),
             boot_cv=cv, boot_values=[float(b) for b in boots], old_m=old_m, wall_s=time.time() - t0)
    print(f"[G] {name:22s} m_plateau={m:.3f} boot mean {boots.mean():.3f} CV {cv:.3f} | ID(k) " + ' '.join(f'{v:.1f}' for v in ids) +
          f" | old m {old_m['default']:.2f} (k15 {old_m['k15']:.2f})  ({r['wall_s']:.0f}s)", flush=True)
    return r


def main():
    out = {}
    # Z113 clean_s1 resid_mid, 1200-pt representative probe
    from analysis.z113_common import checkpoints, load, acts
    from systems.z113 import build_data
    for run, seed, frac in (('z113_clean_s1', 1, .3), ('z113_clean_s0', 0, .3), ('z113_cleanctrl_s2', 2, .5), ('z113_scrambled_s0', 0, .3)):
        d = build_data(seed, 'scrambled' if 'scrambled' in run else 'clean', frac=frac)
        sub = np.random.default_rng(0).choice(len(d['train_toks']), 1200, replace=False)
        out[run] = score(acts(load(checkpoints(f'runs/{run}')[0][-1], seed=seed), d['train_toks'][sub], 'resid_mid'), run)
    # ring
    from systems.ring_rnn import RingRNN
    from analysis.ring_pullback import cloud
    for s in (1, 0, 2):
        m = RingRNN(seed=s).cuda(); m.load_state_dict(torch.load(sorted(glob.glob(f'runs/ring_s{s}/ckpt_*.pt'))[-1])); m.eval()
        out[f'ring_s{s}'] = score(cloud(m, 1200, seed=7)[0], f'ring_s{s}')
    # ResNet clean: penultimate + layer1-4 (wide layers PCA->256 as run 1)
    from systems.resnet_cifar import make_model, load_cifar
    from analysis.resnet_depth import all_layers, LAYERS
    xtr, ytr, xte, yte = load_cifar()
    for tag, yy in (('clean', ytr), ('scrambled', np.random.default_rng(0).integers(0, 10, size=len(ytr)))):
        m = make_model().cuda(); m.load_state_dict(torch.load(sorted(glob.glob(f'runs/resnet_{tag}_s0/ckpt_*.pt'))[-1])); m.eval()
        rng = np.random.default_rng(0); sel = np.concatenate([rng.choice(np.where(yy == c)[0], 150, replace=False) for c in range(10)])
        A, _ = all_layers(m, xtr[sel])
        for L in LAYERS:
            X = A[L]
            if X.shape[1] > 1024:
                Xc = X - X.mean(0); U, S, _ = np.linalg.svd(Xc, full_matrices=False); X = (U[:, :256] * S[:256]).astype(np.float32)
            out[f'resnet_{tag}_{L}'] = score(X, f'resnet_{tag}_{L}')
        del m; torch.cuda.empty_cache()
    # GPT-2 L6 natural probe (activations cached for E2)
    from analysis import gpt2_explore as g
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained('gpt2'); gm = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda').eval(); tap = g.Tap(gm)
    ids, _ = g.natural_windows(tok, 3000, 0); H, Fo, Lg = g.run(gm, tap, ids, bs=64)
    WU = gm.lm_head.weight.detach().float().cpu().numpy(); _, S, Vt = np.linalg.svd(WU, full_matrices=False)
    np.savez_compressed('results/run2/cache_gpt2_L6.npz', H=H, O=((Fo @ Vt.T) * S).astype(np.float32), ids=ids)
    out['gpt2_L6'] = score(H[:2000], 'gpt2_L6')
    # verdict by the letter
    prim = dict(Z113='z113_clean_s1', ring='ring_s1', ResNet='resnet_clean_penultimate', GPT2='gpt2_L6')
    v = dict(primary={k: dict(m_plateau=out[r]['m_plateau'], cv=out[r]['boot_cv']) for k, r in prim.items()})
    v['cv_below_0.15_everywhere'] = bool(all(np.isfinite(out[r]['boot_cv']) and out[r]['boot_cv'] < 0.15 for r in prim.values()))
    v['resnet_penultimate_nonzero'] = bool(out['resnet_clean_penultimate']['m_plateau'] > 0)
    v['resnet_layers_nonzero'] = int(sum(out[f'resnet_clean_layer{i}']['m_plateau'] > 0 for i in range(1, 5)))
    mp = [out[prim[k]]['m_plateau'] for k in ('Z113', 'ring', 'ResNet', 'GPT2')]
    v['ordering_Z113>ring>ResNet>GPT2'] = bool(mp[0] > mp[1] > mp[2] > mp[3])
    v['PASS'] = bool(v['cv_below_0.15_everywhere'] and v['resnet_penultimate_nonzero'] and v['resnet_layers_nonzero'] >= 3 and v['ordering_Z113>ring>ResNet>GPT2'])
    out['_verdict'] = v; print(json.dumps(v, indent=1))
    json.dump(out, open('results/run2/blockG.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
