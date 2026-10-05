"""ResNet-18 on CIFAR-10, clean and scrambled-label.

Ground truth: P6 (clean must show geodesic closure, scrambled must not), P4 (SVHN and
Gaussian noise as OOD), P5 (perturbation failure radius), P1 fibers.
"""
import argparse, json, os, time, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18

DATA = "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - Archive/v1/experiments/data/cifar-10-batches-py"
MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], np.float32)


def load_cifar():
    xs, ys = [], []
    for i in range(1, 6):
        with open(f'{DATA}/data_batch_{i}', 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        xs.append(d[b'data']); ys += d[b'labels']
    with open(f'{DATA}/test_batch', 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    xtr = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32) / 255
    xte = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255
    norm = lambda a: (a - MEAN[None, :, None, None]) / STD[None, :, None, None]
    return norm(xtr), np.array(ys), norm(xte), np.array(d[b'labels'])


def make_model(seed=0):
    torch.manual_seed(seed)
    m = resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)     # CIFAR stem
    m.maxpool = nn.Identity()
    return m


def features(m, x):
    """Penultimate 512-d representation (after global average pool).  psi from here is the
    final linear layer, so the pullback metric is a single fixed quadratic form -- reported
    with the results, since it makes the Lipschitz field constant."""
    x = m.relu(m.bn1(m.conv1(x)))
    x = m.layer4(m.layer3(m.layer2(m.layer1(m.maxpool(x)))))
    return torch.flatten(m.avgpool(x), 1)


def run(mode, seed, epochs, outdir, device='cuda', bs=256, lr=0.05):
    os.makedirs(outdir, exist_ok=True)
    xtr, ytr, xte, yte = load_cifar()
    rng = np.random.default_rng(seed)
    if mode == 'scrambled':
        ytr = rng.integers(0, 10, size=len(ytr))
    Xtr = torch.as_tensor(xtr); Ytr = torch.as_tensor(ytr, dtype=torch.long)
    Xte = torch.as_tensor(xte).to(device); Yte = torch.as_tensor(yte, dtype=torch.long).to(device)
    m = make_model(seed).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    nstep = (len(Xtr) // bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=nstep)
    scaler = torch.amp.GradScaler('cuda')
    hist, t0, step = [], time.time(), 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr) - bs + 1, bs):
            idx = perm[s:s + bs]
            x = Xtr[idx].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = Ytr[idx].to(device, non_blocking=True)
            if rng.random() < 1.0:                       # standard augmentation
                if rng.random() < 0.5:
                    x = torch.flip(x, [3])
            with torch.amp.autocast('cuda'):
                loss = F.cross_entropy(m(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if step < nstep - 1:
                sched.step()
            step += 1
        m.eval(); cor = 0
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for s in range(0, len(Xte), 1000):
                cor += (m(Xte[s:s + 1000]).argmax(1) == Yte[s:s + 1000]).sum().item()
            tr_cor = 0
            for s in range(0, 10000, 1000):
                xb = Xtr[s:s + 1000].to(device)
                tr_cor += (m(xb).argmax(1) == Ytr[s:s + 1000].to(device)).sum().item()
        hist.append(dict(epoch=ep, test_acc=cor / len(Xte), train_acc=tr_cor / 10000))
        torch.save(m.state_dict(), f'{outdir}/ckpt_{ep:03d}.pt')
        print(f'[resnet {mode} s{seed}] ep {ep} train {tr_cor/10000:.4f} test {cor/len(Xte):.4f} '
              f'({time.time()-t0:.0f}s)', flush=True)
    json.dump(dict(meta=dict(mode=mode, seed=seed, epochs=epochs, wall_s=time.time() - t0),
                   history=hist), open(f'{outdir}/history.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='clean', choices=['clean', 'scrambled'])
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args(); run(a.mode, a.seed, a.epochs, a.outdir)
