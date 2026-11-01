"""Task designed to VIOLATE 'metric isometry => output equivariance' (see M1, Prop 1.4).
Input: angle theta as (cos, sin) in R^2.  Target: soft label p*(theta) = (sqrt-p circle of half-angle alpha around the barycentre)^2.
G = SO(2) acts on inputs by rotation; the *target-induced* metric is exactly rotation invariant, the target is equivariant only for Z_3.
A trained MLP should show: metric isometry defect small for ALL shifts c, equivariance defect ~0 only for c in (2 pi/3) Z."""
import torch, math, itertools, json, sys
torch.set_default_dtype(torch.float64)
alpha = 0.4
def target(th):
    k = torch.arange(3.)
    s = math.cos(alpha) / math.sqrt(3) + math.sin(alpha) * math.sqrt(2 / 3) * torch.cos(th[:, None] - 2 * math.pi * k / 3)
    return s ** 2
def emb(th): return torch.stack([torch.cos(th), torch.sin(th)], -1)
res = []
for seed in range(3):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 3))
    th = torch.rand(2000) * 2 * math.pi
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for it in range(4000):
        idx = torch.randint(0, 2000, (256,)); p = torch.softmax(net(emb(th[idx])), -1)
        loss = (target(th[idx]) * (target(th[idx]).log() - p.log())).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    grid = torch.linspace(0, 2 * math.pi, 181)[:-1]
    def g_theta(t):          # metric along the circle, coordinate theta
        f = lambda s: net(emb(s.reshape(1)))[0]
        J = torch.autograd.functional.jacobian(f, t).reshape(3)
        p = torch.softmax(f(t), -1); return (J @ (torch.diag(p) - torch.outer(p, p)) @ J).item()
    gv = torch.tensor([g_theta(t) for t in grid])
    perms = [torch.eye(3)[list(q)] for q in itertools.permutations(range(3))]
    P0 = torch.softmax(net(emb(grid)), -1)
    row = {"seed": seed, "final_KL": loss.item(), "g_mean": gv.mean().item(), "g_cv": (gv.std() / gv.mean()).item()}
    for name, c in [("2pi/3", 2 * math.pi / 3), ("1.0", 1.0), ("pi/2", math.pi / 2)]:
        # isometry defect of T_c: mean |g(theta+c)-g(theta)| / mean g
        gs = torch.tensor([g_theta(t + c) for t in grid]); iso = ((gs - gv).abs().mean() / gv.mean()).item()
        Pc = torch.softmax(net(emb(grid + c)), -1)
        eq = min(((Pc - P0 @ Q.T).norm(dim=1).mean().item()) for Q in perms)
        row[name] = {"isometry_defect": iso, "equivariance_defect": eq}
    res.append(row)
    print(json.dumps(row))
