"""Phase 3a (prereg/PHASE_3.md): learned foliation / rotation symmetry on rotated MNIST.  Access level B (input->output Jacobian).
Chart: p = (t1, t2, a) -> image rendered from one fixed digit by translation t and rotation a (differentiable grid_sample).  Model metric = Fisher-Rao pullback
g = J^T (diag p - p p^T) J in the chart (J = d logits / d p, forward-mode exact).  Per model: r_a = mean g_aa / tr g ; kernel alignment = mean cos^2(smallest
eigenvector of g, e_a).  Conditions: rotaug (generalises over SO(2)), noaug (does not), scrambled (memoriser, labels random, rotation-augmented, small subset).
seeds 1-5.  python analysis/phase3_rotmnist.py [out.json]    Authors: Arjun Paradkar and Reva Paradkar."""
import sys, os, json, time
import numpy as np, torch, torch.nn.functional as F
from torch.func import vmap, jacrev
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.rot_mnist import SmallCNN, rotate, ROOT
from torchvision import datasets, transforms
from bench import stats as S
DEV = 'cuda'; OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/phase3_rotmnist.json'


def render(x0, p):
    """x0: (1,28,28); p=(t1,t2,a) -> (1,1,28,28) via one affine grid (rotation about the centre then translation)."""
    c, s = torch.cos(p[2]), torch.sin(p[2]); th = torch.stack([torch.stack([c, -s, p[0]]), torch.stack([s, c, p[1]])])[None]
    return F.grid_sample(x0[None], F.affine_grid(th, (1, 1, 28, 28), align_corners=False), align_corners=False)


def render_b(x0, P):
    """batched render: P (n,3) -> (n,1,28,28)."""
    c, sn = torch.cos(P[:, 2]), torch.sin(P[:, 2]); th = torch.stack([torch.stack([c, -sn, P[:, 0]], 1), torch.stack([sn, c, P[:, 1]], 1)], 1)
    return F.grid_sample(x0[None].expand(len(P), -1, -1, -1), F.affine_grid(th, (len(P), 1, 28, 28), align_corners=False), align_corners=False)


def train(cond, seed, epochs=8):
    tr = datasets.MNIST(ROOT, train=True, download=False, transform=transforms.ToTensor()); te = datasets.MNIST(ROOT, train=False, download=False, transform=transforms.ToTensor())
    X = torch.stack([tr[i][0] for i in range(len(tr))]); Y = torch.tensor([tr[i][1] for i in range(len(tr))]); g = torch.Generator().manual_seed(seed)
    if cond == 'scrambled': idx = torch.randperm(len(X), generator=g)[:3000]; X, Y = X[idx], torch.randint(0, 10, (3000,), generator=g); epochs = 80
    X, Y = X.to(DEV), Y.to(DEV); m = SmallCNN(seed).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-3); torch.manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=DEV); m.train()
        for i in range(0, len(X), 256):
            xb, yb = X[perm[i:i + 256]], Y[perm[i:i + 256]]
            if cond in ('rotaug', 'scrambled'): xb = rotate(xb, torch.rand(len(xb), device=DEV) * 2 * np.pi)
            loss = F.cross_entropy(m(xb), yb); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval(); Xt = torch.stack([te[i][0] for i in range(2000)]).to(DEV); Yt = torch.tensor([te[i][1] for i in range(2000)]).to(DEV)
    with torch.no_grad():
        acc0 = float((m(Xt).argmax(1) == Yt).float().mean()); acc_rot = float((m(rotate(Xt, torch.rand(len(Xt), device=DEV) * 2 * np.pi)).argmax(1) == Yt).float().mean())
    return m, dict(test_acc_upright=acc0, test_acc_rotated=acc_rot)


def chart_stats(m, digits, n_pts=40, seed=0):
    rng = np.random.default_rng(seed); ra, al, tn = [], [], []
    for x0 in digits:
        P = torch.tensor(np.c_[rng.uniform(-0.15, 0.15, (n_pts, 2)), rng.uniform(0, 2 * np.pi, n_pts)], dtype=torch.float32, device=DEV)
        P = P.clone().requires_grad_(True); lg = m(render_b(x0, P)); J = torch.stack([torch.autograd.grad(lg[:, k].sum(), P, retain_graph=k < 9)[0] for k in range(10)], 1)   # (n,10,3) exact, one backward per logit
        with torch.no_grad():
            pr = torch.softmax(lg.detach(), 1); Fm = torch.diag_embed(pr) - pr[:, :, None] * pr[:, None, :]; J = J.detach()
        g = (J.transpose(1, 2) @ Fm @ J).double().cpu().numpy()
        for gi in g:
            w, V = np.linalg.eigh(gi); tr = np.trace(gi); tn.append(tr)
            if tr < 1e-12: continue
            ra.append(gi[2, 2] / tr); al.append(V[2, 0] ** 2)                                 # V[:,0] = smallest-eigenvalue eigenvector; component along e_a
    return dict(r_a=float(np.mean(ra)), kernel_align=float(np.mean(al)), trace=float(np.mean(tn)))


def main():
    te = datasets.MNIST(ROOT, train=False, download=False, transform=transforms.ToTensor()); digits = [te[i][0].to(DEV) for i in range(12)]; rows = []
    for cond in ('rotaug', 'noaug', 'scrambled'):
        for seed in range(1, 6):
            t0 = time.time(); m, acc = train(cond, seed); st = chart_stats(m, digits, seed=seed); rows.append(dict(cond=cond, seed=seed, **acc, **st)); print(rows[-1], round(time.time() - t0), 's', flush=True)
            json.dump(rows, open(OUT, 'w'), indent=1)
    R = {}
    for k in ('r_a', 'kernel_align', 'test_acc_rotated'):
        a = [r[k] for r in rows if r['cond'] == 'rotaug']; b = [r[k] for r in rows if r['cond'] == 'noaug']; c = [r[k] for r in rows if r['cond'] == 'scrambled']
        R[k] = dict(rotaug_vs_noaug=S.perm_p_auroc(a, b), rotaug_vs_scrambled=S.perm_p_auroc(a, c), means=[float(np.mean(a)), float(np.mean(b)), float(np.mean(c))])
    json.dump(dict(rows=rows, tests=R), open(OUT, 'w'), indent=1, default=float); print(json.dumps(R, indent=1, default=float))


if __name__ == '__main__': main()
