"""Small CNN on rotated MNIST.  Ground truth for P1 fibers: inside each digit's fiber there
must be a rotation circle, named by the rotation angle."""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = '/home/god/.pytorch/MNIST_data'


class SmallCNN(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64, 64); self.fc2 = nn.Linear(64, 10)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.relu(self.c3(x)).mean((2, 3))
        return F.relu(self.fc1(x))               # 64-d penultimate: the layer we open

    def psi(self, h):
        return self.fc2(h)

    def forward(self, x):
        return self.psi(self.features(x))


def rotate(x, ang):
    """Rotate a batch of images by per-sample angles (radians)."""
    c, s = torch.cos(ang), torch.sin(ang)
    z = torch.zeros_like(c)
    th = torch.stack([torch.stack([c, -s, z], 1), torch.stack([s, c, z], 1)], 1)
    grid = F.affine_grid(th, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


def run(seed, epochs, outdir, device='cuda', bs=256):
    os.makedirs(outdir, exist_ok=True)
    tr = datasets.MNIST(ROOT, train=True, download=False, transform=transforms.ToTensor())
    te = datasets.MNIST(ROOT, train=False, download=False, transform=transforms.ToTensor())
    trl = torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True, num_workers=2)
    tel = torch.utils.data.DataLoader(te, batch_size=512, num_workers=2)
    torch.manual_seed(seed)
    m = SmallCNN(seed).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    hist, t0 = [], time.time()
    for ep in range(epochs):
        m.train()
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            ang = torch.rand(len(x), device=device) * 2 * np.pi     # full rotation group
            loss = F.cross_entropy(m(rotate(x, ang)), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        m.eval(); cor = tot = 0
        with torch.no_grad():
            for x, y in tel:
                x, y = x.to(device), y.to(device)
                ang = torch.rand(len(x), device=device) * 2 * np.pi
                cor += (m(rotate(x, ang)).argmax(1) == y).sum().item(); tot += len(y)
        hist.append(dict(epoch=ep, test_acc=cor / tot))
        torch.save(m.state_dict(), f'{outdir}/ckpt_{ep:03d}.pt')
        print(f'[rotmnist s{seed}] ep {ep} acc {cor/tot:.4f}', flush=True)
    json.dump(dict(meta=dict(seed=seed, epochs=epochs, wall_s=time.time() - t0), history=hist),
              open(f'{outdir}/history.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args(); run(a.seed, a.epochs, a.outdir)
