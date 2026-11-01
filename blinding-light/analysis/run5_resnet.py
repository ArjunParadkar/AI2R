"""Run 5, ResNet blocks (SPEC5.md): sanity gate, P6.b shift consistency, P6.c connection-scale slope + label-permutation null; reported only:
concentration and the Euclidean tower.  Instruments imported unchanged from Run 4.
Neighbouring work: LieGG (Moskalev et al., NeurIPS 2022); shift consistency: Zhang 2019, Azulay & Weiss 2019."""
import sys, os, json, glob, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.resnet_subset import subset_data
from analysis.resnet_depth import all_layers, LAYERS
from analysis.run3_blockP64 import conn_slope
from analysis.run4_resnet import load, concentration, shift_consistency, tower

MODELS = [('r5_resnet_cleanaug_s%d' % s, 'clean', s) for s in (3, 4)] + [('r5_resnet_scrambled_s%d' % s, 'scrambled', s) for s in (3, 4)]


def main(only_gate=False):
    path = 'results/run5/resnet.json'; out = json.load(open(path)) if os.path.exists(path) else {}
    for run, mode, s in MODELS:
        if not os.path.exists(f'runs/{run}/history.json'):
            print('[r5] missing', run); continue
        if run in out and (only_gate or 'P6_c' in out[run]):
            continue
        t0 = time.time(); H = json.load(open(f'runs/{run}/history.json')); h = H['history'][-1]; x, y, _, _ = subset_data(mode, s)
        m = load(sorted(glob.glob(f'runs/{run}/ckpt_[0-9]*.pt'))[-1], s); m0 = load(f'runs/{run}/ckpt_init.pt', s)
        sc, scs = shift_consistency(m, x); sc0, _ = shift_consistency(m0, x)
        rec = dict(mode=mode, seed=s, aug=H['meta'].get('aug', False), epochs=H['meta']['epochs'], reached=H['meta']['reached'], train_acc=h['train_acc'], test_acc=h['test_acc'],
                   P6_b=dict(shift_consistency=sc, per_shift=scs, untrained=sc0))
        if not only_gate:
            sub = np.random.default_rng(0).choice(len(x), 500, replace=False); pr, pri = concentration(m, x[sub]); pr0, pri0 = concentration(m0, x[sub])
            rng = np.random.default_rng(0); sel = np.concatenate([rng.choice(np.where(y == c)[0], 150, replace=False) for c in range(10)]); A, _ = all_layers(m, x[sel]); clouds = []
            for L in LAYERS:
                X = A[L]
                if X.shape[1] > 1024:
                    Xc = X - X.mean(0); U, S, _ = np.linalg.svd(Xc, full_matrices=False); X = (U[:, :256] * S[:256]).astype(np.float32)
                clouds.append(X)
            c = conn_slope(clouds, y[sel]); r2 = np.random.default_rng(1); nul = [conn_slope(clouds, r2.permutation(y[sel]))['slope'] for _ in range(5)]
            rec.update(P6_c=dict(slope=c['slope'], per_layer=c['per_layer'], null_label_perm=nul, null_mean=float(np.mean(nul))),
                       concentration=dict(C=1 - pr / pr0, C_per_input=1 - pri / pri0), tower=tower(m, x))
        rec['wall_s'] = time.time() - t0; out[run] = rec
        print(f"[r5] {run}: {rec['epochs']} ep, acc {rec['train_acc']:.3f}/{rec['test_acc']:.3f} | shift consistency {sc:.3f} (init {sc0:.3f}) | slope {rec.get('P6_c', {}).get('slope')} null {rec.get('P6_c', {}).get('null_mean')} ({rec['wall_s']:.0f}s)", flush=True)
        json.dump(out, open(path, 'w'), indent=1, default=float); del m, m0; torch.cuda.empty_cache()
    print('done')


if __name__ == '__main__':
    os.makedirs('results/run5', exist_ok=True); main(only_gate=(len(sys.argv) > 1 and sys.argv[1] == 'gate'))
