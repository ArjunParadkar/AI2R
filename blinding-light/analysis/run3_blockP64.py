"""Run 3, S7 + P6.4 (SPEC3.md): the two after-the-fact positives of run 2, pre-registered on FRESH seeds only.
connection scale of class fibers vs depth, and the ID(k) profile class with 20 bootstraps."""
import sys, os, json, glob
import numpy as np, torch
from scipy.spatial.distance import pdist, squareform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whiteboxtopo.depthprofile import fiber_connection_scales
from analysis.run2_blockG import id_curve, KS
DEV = os.environ.get('WB_DEV', 'cuda')


def id_profile(X, n_boot=20, seed=0):
    def classify(ks, ids):
        lk, li = np.log(ks), np.log(ids); slope = float(np.polyfit(lk, li, 1)[0])
        if slope < -0.2: return 'decay', slope
        if slope > 0.2: return 'growth', slope
        for i in range(len(ks) - 2):
            for j in range(i + 2, len(ks)):
                if abs(np.polyfit(lk[i:j + 1], li[i:j + 1], 1)[0]) <= 0.1: return 'plateau', slope
        return 'mixed', slope
    ks, ids = id_curve(X); cls, slope = classify(ks, ids); rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        kb, ib = id_curve(X[rng.choice(len(X), int(0.8 * len(X)), replace=False)]); boots.append(classify(kb, ib)[0])
    return dict(id_curve=[float(v) for v in ids], profile=cls, overall_slope=slope, boot_agree=int(sum(b == cls for b in boots)), boot_classes=boots)


def conn_slope(clouds, labels):
    per = []
    for X in clouds:
        D = squareform(pdist(X)).astype(np.float32); med = float(np.median(D[np.triu_indices_from(D, 1)]))
        sc = fiber_connection_scales(D, labels); v = np.array(list(sc.values()) if isinstance(sc, dict) else sc, float); per.append(float(np.mean(v) / med)); del D
    depth = np.linspace(0, 1, len(per)); return dict(per_layer=per, slope=float(np.polyfit(depth, per, 1)[0]))


def z113(run, seed, mode):
    from systems.z113 import Z113Transformer, build_data
    d = build_data(seed, mode, frac=0.3); m = Z113Transformer(seed=seed).to(DEV); m.load_state_dict(torch.load(sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))[-1], map_location=DEV)); m.eval()
    tk = torch.as_tensor(d['train_toks'], device=DEV)
    with torch.no_grad():
        x = m.embed(tk); e = x[:, :2].reshape(len(tk), -1); mid = (x + m.attn(x))[:, -1]; post = m.resid_final(tk)
    clouds = [t.float().cpu().numpy() for t in (e, mid, post)]; sub = np.random.default_rng(0).choice(len(tk), 1200, replace=False)
    return dict(connection=conn_slope(clouds, d['train_y']), id_profile=id_profile(clouds[1][sub]), layers=['embed', 'resid_mid', 'resid_post'])


def resnet(run, seed, mode):
    from systems.resnet_cifar import make_model, load_cifar
    from analysis.resnet_depth import all_layers, LAYERS
    xtr, ytr, _, _ = load_cifar(); yy = ytr if mode == 'clean' else np.random.default_rng(seed).integers(0, 10, size=len(ytr))
    m = make_model().to('cuda'); m.load_state_dict(torch.load(sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))[-1], map_location='cuda')); m.eval()   # all_layers is cuda-only
    rng = np.random.default_rng(0); sel = np.concatenate([rng.choice(np.where(yy == c)[0], 150, replace=False) for c in range(10)]); A, _ = all_layers(m, xtr[sel]); clouds = []
    for L in LAYERS:
        X = A[L]
        if X.shape[1] > 1024:
            Xc = X - X.mean(0); U, S, _ = np.linalg.svd(Xc, full_matrices=False); X = (U[:, :256] * S[:256]).astype(np.float32)
        clouds.append(X)
    return dict(connection=conn_slope(clouds, yy[sel]), id_profile=id_profile(clouds[-1]), layers=LAYERS)


def main():
    out = {}
    jobs = [(f'r3_z113_{md}_s{s}', s, md, z113) for s in (3, 4) for md in ('clean', 'scrambled')] + [(f'r3_resnet_{md}_s{s}', s, md, resnet) for s in (1, 2) for md in ('clean', 'scrambled')]
    for run, seed, mode, fn in jobs:
        hp = f'runs/{run}/history.json'
        if not os.path.exists(hp):
            print('[P6.4] not finished:', run); out[run] = dict(missing=True); continue
        try:
            out[run] = fn(run, seed, mode)
        except Exception as e:
            print('[P6.4] FAILED', run, type(e).__name__, str(e)[:150]); out[run] = dict(error=type(e).__name__); continue
        r = out[run]; print(f"[P6.4] {run}: connection scale per layer {[round(v, 3) for v in r['connection']['per_layer']]} slope {r['connection']['slope']:+.3f} | ID {[round(v, 1) for v in r['id_profile']['id_curve']]} -> {r['id_profile']['profile']} ({r['id_profile']['boot_agree']}/20)", flush=True)
        json.dump(out, open('results/run3/blockP64.json', 'w'), indent=1, default=float)
    json.dump(out, open('results/run3/blockP64.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
