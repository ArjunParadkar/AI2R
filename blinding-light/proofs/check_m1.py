"""M1 checks: (1) Frobenius/Lie-bracket test (kernel distribution involutive, horizontal one not)
(2) Killing-algebra solver validated on the FR simplex image (expect so(3), dim 3) and on a generic metric (dim 0)
(3) counterexample: isometry of g WITHOUT output equivariance (circle of sqrt-p around the barycentre)
(4) lemma: vertex-complete + orthogonal R preserving orthant  =>  permutation."""
import torch, math, itertools, numpy as np
torch.set_default_dtype(torch.float64)
torch.manual_seed(1)

# ---------- (1) Frobenius -------------------------------------------------
n, K = 5, 3
A1, A2 = torch.randn(7, n), torch.randn(K, 7)
f = lambda x: A2 @ torch.tanh(A1 @ x)
Pc = torch.eye(K) - torch.ones(K, K) / K
def Jc(x): return Pc @ torch.autograd.functional.jacobian(f, x)
def proj_ker(x):
    U, S, Vh = torch.linalg.svd(Jc(x)); r = int((S > 1e-9).sum()); R = Vh[:r].T
    return torch.eye(n) - R @ R.T, r
def proj_hor(x):
    P, r = proj_ker(x); return torch.eye(n) - P
def bracket(field, x, a, b, h=1e-5):
    Ea = lambda y: field(y) @ torch.eye(n)[:, a]
    Eb = lambda y: field(y) @ torch.eye(n)[:, b]
    def D(E, v):  return (E(x + h * v) - E(x - h * v)) / (2 * h)
    return D(Eb, Ea(x)) - D(Ea, Eb(x))
x0 = torch.randn(n)
P, r = proj_ker(x0)
print(f"(1) rank J_c = {r} (K-1 = {K-1}); dim D = {n-r}")
res_ker, res_hor = [], []
for a, b in itertools.combinations(range(n), 2):
    Bk = bracket(lambda y: proj_ker(y)[0], x0, a, b)
    Bh = bracket(proj_hor, x0, a, b)
    res_ker.append((Jc(x0) @ Bk).norm().item() / (Jc(x0).norm() * (Bk.norm() + 1e-30)).item() if Bk.norm() > 1e-6 else 0.0)
    # horizontal: component of bracket outside H = component inside D
    Ph = proj_hor(x0)
    res_hor.append(((torch.eye(n) - Ph) @ Bh).norm().item() / (Bh.norm().item() + 1e-30))
print(f"    kernel distribution : Frobenius residual |J_c [E_a,E_b]| (rel) max = {max(res_ker):.2e}   (should be ~0)")
print(f"    horizontal distr.   : normal component of bracket (rel)    max = {max(res_hor):.2e}   mean = {np.mean(res_hor):.2e}  (O(1) = non-involutive)")

# ---------- (2) Killing solver --------------------------------------------
def stereo_to_sqrtp(z):   # inverse stereographic projection, centred on the barycentre, unit sphere in R^3
    s = torch.stack([2 * z[0], 2 * z[1], 1 - z @ z]) / (1 + z @ z)
    e3 = torch.ones(3) / math.sqrt(3); a = torch.tensor([1., -1., 0.]) / math.sqrt(2); b = torch.linalg.cross(e3, a)
    return s[0] * a + s[1] * b + s[2] * e3
def f_sphere(z, scale=0.35):
    y = stereo_to_sqrtp(scale * z)       # input z in unit disc -> small cap around barycentre (stays in orthant)
    return torch.log(y ** 2)
def metric_fn(f):
    def g(z):
        J = torch.autograd.functional.jacobian(f, z, create_graph=True)
        p = torch.softmax(f(z), -1); F = torch.diag(p) - torch.outer(p, p)
        return J.T @ F @ J
    return g
def killing_nullity(f, deg=2, npts=60, seed=0):
    g = metric_fn(f); gen = torch.Generator().manual_seed(seed)
    mons = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    nb = len(mons); rows = []
    for _ in range(npts):
        z = (torch.rand(2, generator=gen) * 2 - 1) * 0.8
        G = g(z)
        dG = torch.autograd.functional.jacobian(g, z)       # 2x2x2 : dG[i,j,k] = d_k g_ij
        phi = torch.tensor([z[0] ** i * z[1] ** j for i, j in mons])
        dphi = torch.stack([torch.tensor([ (i * z[0] ** (i-1) * z[1] ** j if i > 0 else 0.) for i, j in mons]),
                            torch.tensor([ (j * z[0] ** i * z[1] ** (j-1) if j > 0 else 0.) for i, j in mons])])  # dphi[k,alpha]
        for (i, j) in [(0, 0), (0, 1), (1, 1)]:
            row = torch.zeros(2 * nb)           # unknown c[k, alpha] flattened k*nb+alpha
            for k in range(2):
                # (L_Y g)_ij = Y^k d_k g_ij + g_kj d_i Y^k + g_ik d_j Y^k
                row[k * nb:(k + 1) * nb] += dG[i, j, k] * phi + G[k, j] * dphi[i] + G[i, k] * dphi[j]
            rows.append(row)
    M = torch.stack(rows); S = torch.linalg.svdvals(M)
    return S / S[0]
S = killing_nullity(f_sphere)
print("(2) FR-simplex cap (sphere): smallest normalised singular values of the Killing system, deg-2 polynomial fields:")
print("    ", [f"{s:.1e}" for s in S[-6:].tolist()], "-> nullity", int((S < 1e-8).sum()), "(expect 3 = dim so(3))")
Wg1, Wg2 = torch.randn(6, 2), torch.randn(3, 6)
f_gen = lambda z: Wg2 @ torch.tanh(Wg1 @ z)
S2 = killing_nullity(f_gen)
print("    generic tanh net     :", [f"{s:.1e}" for s in S2[-4:].tolist()], "-> nullity", int((S2 < 1e-8).sum()), "(expect 0)")

# ---------- (3) counterexample: circle -------------------------------------
alpha = 0.4
def sqrtp(theta):
    k = torch.arange(3.)
    return math.cos(alpha) / math.sqrt(3) + math.sin(alpha) * math.sqrt(2 / 3) * torch.cos(theta - 2 * math.pi * k / 3)
def f_circ(theta): return torch.log(sqrtp(theta) ** 2)
th = torch.linspace(0, 2 * math.pi, 400)[:-1]
def gth(t):   # metric in coordinate theta
    J = torch.autograd.functional.jacobian(f_circ, t).reshape(3)
    p = torch.softmax(f_circ(t), -1); return J @ (torch.diag(p) - torch.outer(p, p)) @ J
gv = torch.stack([gth(t) for t in th])
print(f"(3) circle example: g_theta constant? max/min = {gv.max().item():.6f}/{gv.min().item():.6f}  (analytic 4 sin^2(alpha) = {4*math.sin(alpha)**2:.6f})")
perms = [torch.eye(3)[list(p)] for p in itertools.permutations(range(3))]
P0 = torch.stack([torch.softmax(f_circ(t), -1) for t in th])
for c in [2 * math.pi / 3, 4 * math.pi / 3, 1.0, math.pi / 2, 0.3]:
    Pc_ = torch.stack([torch.softmax(f_circ(t + c), -1) for t in th])
    best = min(((Pc_ - P0 @ Q.T).norm(dim=1).mean().item()) for Q in perms)
    print(f"    shift c={c:.4f}: metric defect = 0 (isometry), best permutation equivariance defect = {best:.2e}")

# ---------- (4) lemma sanity -----------------------------------------------
# near-permutation orthogonal matrices: Q = P exp(eps A). If Q maps the vertices into the closed orthant it must be a permutation.
viol = 0; exact = 0
for _ in range(5000):
    eps = 10 ** torch.empty(1).uniform_(-8, -1).item()
    A = torch.randn(4, 4); A = A - A.T
    Q = torch.eye(4)[torch.randperm(4)] @ torch.linalg.matrix_exp(eps * A)
    ok = bool((Q >= -1e-15).all())
    isperm = bool(((Q.abs() > 1e-9).sum(0) == 1).all())
    viol += ok and not isperm
    exact += ok and isperm
print(f"(4) 5000 near-permutation orthogonal Q: {viol} violate the lemma (nonneg columns but not a permutation); {exact} were (numerically) permutations")
