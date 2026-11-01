"""Controls for the rotated-MNIST fiber circle.

A rotation orbit is ALREADY a circle in pixel space, so finding the rotation circle in an early
conv layer says nothing about learning unless two controls come out differently:
  pixels      the same procedure on the raw 28x28 input
  untrained   the same procedure on a randomly initialised SmallCNN
If both certify the circle, the conv1/conv2 circles are inherited from the input, and the
learned fact is the opposite one: where the trained network has DESTROYED the circle.
"""
import sys, os, json, glob, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.rot_mnist import SmallCNN, rotate, ROOT
from torchvision import datasets, transforms
from analysis.rotmnist_layers import layer_acts, orbit_circle, N_ANG, N_BASE, LAYERS


def scan(m, te, ys, with_pixels):
    ang = torch.arange(N_ANG, device='cuda') * (2 * np.pi / N_ANG); aidx = np.arange(N_ANG)
    names = (['pixels'] if with_pixels else []) + LAYERS
    out = {}
    for digit in range(10):
        ids = np.where(ys == digit)[0][:N_BASE]; orbits = []
        for i in ids:
            with torch.no_grad():
                x = rotate(te[i][0][None].repeat(N_ANG, 1, 1, 1).cuda(), ang)
                acts = layer_acts(m, x); acts['pixels'] = x.flatten(1)
            orbits.append({L: acts[L].cpu().numpy().astype(np.float32) for L in names})
        out[digit] = {}
        for L in names:
            rs = []
            for o in range(len(orbits)):
                Xf = np.concatenate([orbits[q][L] for q in range(len(orbits)) if q != o])
                rs.append(orbit_circle(Xf, np.tile(aidx, len(orbits) - 1), orbits[o][L], aidx)['r'])
            out[digit][L] = float(np.median(rs))
    found = {L: int(sum(out[d][L] > 0.9 for d in range(10))) for L in names}
    rmed = {L: float(np.median([out[d][L] for d in range(10)])) for L in names}
    return dict(per_digit=out, found=found, r_median=rmed)


def main():
    te = datasets.MNIST(ROOT, train=False, download=False, transform=transforms.ToTensor()); ys = te.targets.numpy()
    t0 = time.time()
    tr = SmallCNN().cuda(); tr.load_state_dict(torch.load(sorted(glob.glob('runs/rotmnist_s0/ckpt_*.pt'))[-1])); tr.eval()
    res = dict(trained=scan(tr, te, ys, True))
    print('trained  ', res['trained']['found'], res['trained']['r_median'], flush=True)
    for s in (0, 1, 2):
        torch.manual_seed(100 + s); un = SmallCNN().cuda().eval()
        res[f'untrained_s{s}'] = scan(un, te, ys, False)
        print(f'untrained{s}', res[f'untrained_s{s}']['found'], res[f'untrained_s{s}']['r_median'], flush=True)
    res['wall_s'] = time.time() - t0
    json.dump(res, open('results/rotmnist_controls.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
