"""Orbit coherence and equivariance defect (Run 6, SPEC6.md).

A network's learned group G acts on inputs; rho(g) is the induced action on outputs.  For an input x:
    kappa(x)   = (1/|G|) sum_g 1[ argmax f(gx) == rho(g)(argmax f(x)) ]          (hard orbit coherence)
    kappa_s(x) = (1/|G|) sum_g cos( f(gx), rho(g) f(x) )                          (soft)
    eps(x)     = max_g || f(gx) - rho(g) f(x) ||_2 / || f(x) ||_2                 (equivariance defect)
    eps_bar(x) = mean_g of the same.
A group element is a pair (T, R): T maps a batch of inputs to transformed inputs, R maps a batch of logit vectors to rho(g)-relabelled
logit vectors (R = identity for a symmetry).  Neighbours: Golan & El-Yaniv 2018 (designer transformations); LieGG (Moskalev et al. 2022)."""
import torch, torch.nn.functional as F


def orbit_stats(f, x, group, chunk=500):
    """f: batched logits map. x: (N, ...) tensor. group: list of (T, R). Returns dict of kappa, kappa_s, eps, eps_bar, each (N,)."""
    out = dict(kappa=[], kappa_s=[], eps=[], eps_bar=[])
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk]
        with torch.no_grad():
            fx = f(xb).float(); base = fx.argmax(1); nrm = fx.norm(dim=1).clamp_min(1e-12); hard, soft, dev = [], [], []
            for T, R in group:
                fg = f(T(xb)).float(); ref = R(fx)
                hard.append((fg.argmax(1) == ref.argmax(1)).float()); soft.append(F.cosine_similarity(fg, ref, dim=1)); dev.append((fg - ref).norm(dim=1) / nrm)
            hard, soft, dev = torch.stack(hard), torch.stack(soft), torch.stack(dev)
        out['kappa'].append(hard.mean(0)); out['kappa_s'].append(soft.mean(0)); out['eps'].append(dev.max(0).values); out['eps_bar'].append(dev.mean(0))
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def eps_bar_grad(f, x, group):
    """Gradient of the soft mean defect w.r.t. the input (for the direction clause).  x: (n, ...) requires no grad."""
    xt = x.clone().requires_grad_(True); fx = f(xt).float(); nrm = fx.norm(dim=1).clamp_min(1e-12); tot = 0
    for T, R in group:
        tot = tot + (f(T(xt)).float() - R(fx)).norm(dim=1) / nrm
    g, = torch.autograd.grad((tot / len(group)).sum(), xt); return g.detach()


# ---- groups -------------------------------------------------------------------------------------------------------------
def resnet_group(shifts=(1, 2, 3)):
    """The architecture's own: circular shifts of +-1,2,3 px along rows and columns, and a horizontal flip.  rho = id.  |G| = 13."""
    I = lambda l: l; G = []
    for s in shifts:
        for sg in (s, -s):
            G.append(((lambda x, sg=sg: torch.roll(x, sg, 3)), I)); G.append(((lambda x, sg=sg: torch.roll(x, sg, 2)), I))
    G.append(((lambda x: x.flip(3)), I)); return G


def golan_group():
    """Golan & El-Yaniv 2018's 72 designer transformations: {id, hflip} x translations {0,+-8}^2 x rotations {0,90,180,270}. Consistency form."""
    I = lambda l: l; G = []
    for fl in (False, True):
        for dx in (0, 8, -8):
            for dy in (0, 8, -8):
                for k in range(4):
                    def T(x, fl=fl, dx=dx, dy=dy, k=k):
                        if fl: x = x.flip(3)
                        x = torch.roll(x, (dy, dx), (2, 3)); return torch.rot90(x, k, (2, 3))
                    G.append((T, I))
    return G


def random_group_resnet(n=13, seed=0):
    """Non-group null: random 3-px translations with random sign + a random permutation of the pixels of one 4x4 patch (fixed per element)."""
    g = torch.Generator().manual_seed(seed); I = lambda l: l; G = []
    for _ in range(n):
        dx, dy = [int(v) for v in torch.randint(-3, 4, (2,), generator=g)]; r, c = [int(v) for v in torch.randint(0, 28, (2,), generator=g)]; perm = torch.randperm(16, generator=g)
        def T(x, dx=dx, dy=dy, r=r, c=c, perm=perm):
            x = torch.roll(x, (dy, dx), (2, 3)).clone(); p = x[:, :, r:r + 4, c:c + 4].reshape(len(x), x.shape[1], 16)[:, :, perm.to(x.device)]; x[:, :, r:r + 4, c:c + 4] = p.reshape(len(x), x.shape[1], 4, 4); return x
        G.append((T, I))
    return G


def z113_group(u, P=113, valid_slot_only=False):
    """Z113 group on token tensors (N, 4) = [a, b, hint, EQ].  Elements: translation (a+-u, b-+u) rho=id; swap rho=id; orbit translation of
    slot a by +-u, rho = relabel c -> c+-u; of slot b likewise.  |G| = 7.  valid_slot_only: only swap and the two slot-b orbit translations
    (for inputs whose slot a is out of vocabulary)."""
    def relabel(k):
        return lambda l: torch.roll(l[:, :P], k, 1) if l.shape[1] == P else torch.cat([torch.roll(l[:, :P], k, 1), l[:, P:]], 1)
    I = lambda l: l
    def tr(da, db):
        def T(x):
            y = x.clone(); y[:, 0] = (y[:, 0] + da) % P; y[:, 1] = (y[:, 1] + db) % P; return y
        return T
    swap = lambda x: torch.stack([x[:, 1], x[:, 0], x[:, 2], x[:, 3]], 1)
    G = [(swap, I), (tr(0, u), relabel(u)), (tr(0, -u), relabel(-u))]
    if not valid_slot_only:
        G += [(tr(u, -u), I), (tr(-u, u), I), (tr(u, 0), relabel(u)), (tr(-u, 0), relabel(-u))]
    return G
