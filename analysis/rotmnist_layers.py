"""Where does the rotation circle live?  Per layer, per orbit, in a rotation-invariant CNN."""
import sys, os, json, glob, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.rot_mnist import SmallCNN, rotate, ROOT
from torchvision import datasets, transforms
from whiteboxtopo.topology import isotypic_decomposition, persistence, circular_coordinates, circular_correlation
from whiteboxtopo.nulls import feature_shuffle

N_ANG, N_BASE = 36, 8
LAYERS = ['conv1', 'conv2', 'conv3_gap', 'fc1']


def layer_acts(m, x):
    a1 = F.max_pool2d(F.relu(m.c1(x)), 2)
    a2 = F.max_pool2d(F.relu(m.c2(a1)), 2)
    a3 = F.relu(m.c3(a2)).mean((2, 3))
    a4 = F.relu(m.fc1(a3))
    return dict(conv1=a1.flatten(1), conv2=a2.flatten(1), conv3_gap=a3, fc1=a4)


def orbit_circle(X_fit, a_fit, X_eval, a_eval):
    """Planes fitted on the other orbits of the digit; coordinates read on the held-out orbit."""
    dec, _ = isotypic_decomposition(X_fit, a_fit, N_ANG, n_freq=3, detect='dft', var_floor=0.0,
                                    X_eval=X_eval)
    best = dict(r=0.0, freq=None, var=0.0, ratio=0.0)
    for d in dec:
        Y = np.ascontiguousarray(d['proj']).astype(np.float32)
        if not np.isfinite(Y).all() or Y.std() < 1e-12:
            continue
        res = persistence(Y); h1 = res['dgms'][1]
        if len(h1) == 0:
            continue
        life = h1[:, 1] - h1[:, 0]; j = int(np.argmax(life)); srt = np.sort(life)
        cc = circular_coordinates(Y, h1, res['cocycles'][1], D=res['D'], class_idx=j)
        if cc is None:
            continue
        r = circular_correlation(cc['theta'], 2 * np.pi * d['freq'] * a_eval / N_ANG)
        ratio = float(srt[-1] / max(srt[-2], 1e-9)) if len(srt) > 1 else 99.0
        if r > best['r']:
            best = dict(r=float(r), freq=int(d['freq']), var=float(d['var']), ratio=ratio)
    return best


def main():
    te = datasets.MNIST(ROOT, train=False, download=False, transform=transforms.ToTensor())
    m = SmallCNN().cuda(); m.load_state_dict(torch.load(sorted(glob.glob('runs/rotmnist_s0/ckpt_*.pt'))[-1])); m.eval()
    ys = te.targets.numpy()
    ang = torch.arange(N_ANG, device='cuda') * (2 * np.pi / N_ANG)
    aidx = np.arange(N_ANG)
    out, t0 = {}, time.time()
    for digit in range(10):
        ids = np.where(ys == digit)[0][:N_BASE]
        per_layer = {L: [] for L in LAYERS}; null = {L: [] for L in LAYERS}
        orbits = []
        for i in ids:
            with torch.no_grad():
                acts = layer_acts(m, rotate(te[i][0][None].repeat(N_ANG, 1, 1, 1).cuda(), ang))
            orbits.append({L: acts[L].cpu().numpy().astype(np.float32) for L in LAYERS})
        for L in LAYERS:
            for o in range(len(orbits)):                      # leave one orbit out
                Xf = np.concatenate([orbits[q][L] for q in range(len(orbits)) if q != o])
                af = np.tile(aidx, len(orbits) - 1)
                Xe = orbits[o][L]
                per_layer[L].append(orbit_circle(Xf, af, Xe, aidx))
                null[L].append(orbit_circle(feature_shuffle(Xf, 0), af, feature_shuffle(Xe, 1), aidx)['r'])
        out[digit] = {L: dict(r_median=float(np.median([o['r'] for o in per_layer[L]])),
                              frac_orbits_r09=float(np.mean([o['r'] > 0.9 and o['ratio'] > 3 for o in per_layer[L]])),
                              var_median=float(np.median([o['var'] for o in per_layer[L]])),
                              null_r_median=float(np.median(null[L]))) for L in LAYERS}
        print(f"digit {digit}: " + '  '.join(f"{L} r={out[digit][L]['r_median']:.2f} var={out[digit][L]['var_median']:.2f} (null {out[digit][L]['null_r_median']:.2f})" for L in LAYERS), flush=True)
    found = {L: int(sum(out[d][L]['r_median'] > 0.9 for d in range(10))) for L in LAYERS}
    print('digits with a certified rotation circle (median orbit r>0.9), per layer:', found, f'[{time.time()-t0:.0f}s]')
    json.dump(dict(per_digit=out, found_per_layer=found), open('results/rotmnist_layers.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
