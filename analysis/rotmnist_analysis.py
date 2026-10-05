"""Rotated-MNIST CNN: P1 fibers -- is there a rotation circle inside each digit's fiber?"""
import sys, os, json, glob, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.rot_mnist import SmallCNN, rotate, ROOT
from torchvision import datasets, transforms
from whiteboxtopo.topology import (isotypic_decomposition, persistence, circular_coordinates,
                                   circular_correlation)
from whiteboxtopo.nulls import feature_shuffle

N_ANG, N_BASE = 36, 8


def fiber_cloud(m, imgs):
    """Each base image rotated through the full group: N_BASE orbits of N_ANG points."""
    ang = torch.arange(N_ANG, device='cuda') * (2 * np.pi / N_ANG)
    X, A, B = [], [], []
    with torch.no_grad():
        for bi, im in enumerate(imgs):
            x = im[None].repeat(N_ANG, 1, 1, 1).cuda()
            X.append(m.features(rotate(x, ang)).cpu().numpy())
            A.append(np.arange(N_ANG)); B.append(np.full(N_ANG, bi))
    return np.concatenate(X).astype(np.float32), np.concatenate(A), np.concatenate(B)


def circle_in_fiber(X, aidx, bidx):
    """Best certified rotation circle among the isotypic planes of the fiber."""
    dec, freqs = isotypic_decomposition(X, aidx, N_ANG, n_freq=4, detect='dft', var_floor=0.0)
    best = None
    for d in dec:
        k = d['freq']
        Y = np.ascontiguousarray(d['proj']).astype(np.float32)
        res = persistence(Y)
        h1 = res['dgms'][1]
        if len(h1) == 0:
            continue
        life = h1[:, 1] - h1[:, 0]
        j = int(np.argmax(life)); srt = np.sort(life)
        cc = circular_coordinates(Y, h1, res['cocycles'][1], D=res['D'], class_idx=j)
        if cc is None:
            continue
        true = 2 * np.pi * k * aidx / N_ANG
        r_pool = circular_correlation(cc['theta'], true)
        r_orb = float(np.median([circular_correlation(cc['theta'][bidx == b], true[bidx == b])
                                 for b in np.unique(bidx)]))
        rec = dict(freq=int(k), var=float(d['var']), r_pooled=float(r_pool), r_per_orbit=r_orb,
                   life_ratio=float(srt[-1] / max(srt[-2], 1e-9)) if len(srt) > 1 else float('inf'))
        if best is None or rec['r_pooled'] > best['r_pooled']:
            best = rec
    return best, freqs


def main():
    te = datasets.MNIST(ROOT, train=False, download=False, transform=transforms.ToTensor())
    ck = sorted(glob.glob('runs/rotmnist_s0/ckpt_*.pt'))[-1]
    m = SmallCNN().cuda(); m.load_state_dict(torch.load(ck)); m.eval()
    ys = te.targets.numpy()
    out, t0 = {}, time.time()
    for digit in range(10):
        ids = np.where(ys == digit)[0][:N_BASE]
        imgs = [te[i][0] for i in ids]
        X, aidx, bidx = fiber_cloud(m, imgs)
        best, freqs = circle_in_fiber(X, aidx, bidx)
        nb, _ = circle_in_fiber(feature_shuffle(X, 0), aidx, bidx)
        # raw Rips on the fiber, no decomposition, for the record
        res = persistence(X); h1 = res['dgms'][1]
        raw_r = 0.0
        if len(h1):
            j = int(np.argmax(h1[:, 1] - h1[:, 0]))
            cc = circular_coordinates(X, h1, res['cocycles'][1], D=res['D'], class_idx=j)
            if cc:
                raw_r = max(circular_correlation(cc['theta'], 2 * np.pi * k * aidx / N_ANG) for k in (1, 2, 3))
        out[digit] = dict(best=best, freqs=freqs, null=nb, raw_rips_r=float(raw_r),
                          found=bool(best and best['r_pooled'] > 0.9))
        print(f"digit {digit}: k={best['freq']} r_pooled={best['r_pooled']:.3f} r_per_orbit={best['r_per_orbit']:.3f} "
              f"life_ratio={best['life_ratio']:.1f} | null r={nb['r_pooled']:.3f} | raw Rips r={raw_r:.3f} "
              f"{'FOUND' if out[digit]['found'] else ''}", flush=True)
    n = sum(v['found'] for v in out.values())
    print(f"rotation circle found (r>0.9) in {n}/10 digit fibers  (pass: >=7)   [{time.time()-t0:.0f}s]")
    os.makedirs('results', exist_ok=True)
    json.dump(dict(per_digit=out, n_found=n, n_ang=N_ANG, n_base=N_BASE), open('results/rotmnist.json', 'w'),
              indent=1, default=float)


if __name__ == '__main__':
    main()
