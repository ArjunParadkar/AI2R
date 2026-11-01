"""Run 4 specimens: TRUE memorisers.  ResNet-18 on a fixed 5,000-image CIFAR-10 subset, no augmentation, weight decay 0, trained until
train accuracy >= 99% (cap 200 epochs).  mode = clean (true labels) | scrambled (uniform random labels, drawn with the run seed).
The subset is the same for every model: the first 5,000 indices of np.random.default_rng(12345).permutation(50000)."""
import argparse, json, os, time
import numpy as np, torch, torch.nn.functional as F
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.resnet_cifar import make_model, load_cifar

SUBSET = np.random.default_rng(12345).permutation(50000)[:5000]


def subset_data(mode, seed):
    xtr, ytr, xte, yte = load_cifar(); x = xtr[SUBSET]; y = ytr[SUBSET].copy()
    if mode == 'scrambled':
        y = np.random.default_rng(seed).integers(0, 10, size=len(y))
    return x, y, xte, yte


def augment(xb):
    """Run 5: standard CIFAR augmentation on a GPU batch -- 4-px zero-pad random crop + horizontal flip, fresh per batch."""
    n = len(xb); xp = F.pad(xb, (4, 4, 4, 4)); i = torch.randint(0, 9, (n,), device=xb.device); j = torch.randint(0, 9, (n,), device=xb.device)
    ar = torch.arange(32, device=xb.device); rows = (i[:, None] + ar)[:, :, None].expand(n, 32, 32); cols = (j[:, None] + ar)[:, None, :].expand(n, 32, 32)
    out = xp[torch.arange(n, device=xb.device)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(n, device=xb.device) < 0.5; out[flip] = out[flip].flip(3); return out.contiguous()


def run(mode, seed, outdir, device='cuda', bs=128, lr=1e-3, max_epochs=200, target=0.99, aug=False):
    os.makedirs(outdir, exist_ok=True); x, y, xte, yte = subset_data(mode, seed)
    X = torch.as_tensor(x).to(device); Y = torch.as_tensor(y, dtype=torch.long).to(device)
    Xte = torch.as_tensor(xte[:5000]).to(device); Yte = torch.as_tensor(yte[:5000], dtype=torch.long).to(device)
    torch.manual_seed(seed); m = make_model(seed).to(device); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=0.0)
    torch.save(m.state_dict(), f'{outdir}/ckpt_init.pt'); hist, t0 = [], time.time()
    def acc(A, B):
        m.eval(); c = 0
        with torch.no_grad():
            for s in range(0, len(A), 1000):
                c += (m(A[s:s + 1000]).argmax(1) == B[s:s + 1000]).sum().item()
        return c / len(A)
    for ep in range(1, max_epochs + 1):
        m.train(); perm = torch.randperm(len(X), device=device)
        for s in range(0, len(X), bs):
            idx = perm[s:s + bs]; xb = augment(X[idx]) if aug else X[idx]; loss = F.cross_entropy(m(xb), Y[idx]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tr, te = acc(X, Y), acc(Xte, Yte); hist.append(dict(epoch=ep, train_acc=tr, test_acc=te))
        if ep % 5 == 0 or tr >= target:
            torch.save(m.state_dict(), f'{outdir}/ckpt_{ep:03d}.pt')
        if ep % 5 == 0 or tr >= target:
            print(f'[subset {mode} s{seed}] ep {ep} train {tr:.4f} test {te:.4f} ({time.time()-t0:.0f}s)', flush=True)
        if tr >= target:
            break
    json.dump(dict(meta=dict(mode=mode, seed=seed, aug=bool(aug), epochs=ep, reached=bool(tr >= target), wall_s=time.time() - t0), history=hist), open(f'{outdir}/history.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', required=True); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--outdir', required=True); ap.add_argument('--aug', action='store_true'); ap.add_argument('--max-epochs', type=int, default=200)
    a = ap.parse_args(); run(a.mode, a.seed, a.outdir, max_epochs=a.max_epochs, aug=a.aug)
