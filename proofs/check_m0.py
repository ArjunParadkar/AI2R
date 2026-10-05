"""M0 numerical sanity checks (CPU, float64).
 (a) g = J^T F(p) J is invariant under logit shift f -> f + c(x) 1
 (b) g invariant under a rewriting group G (cyclic shifts of the input, permutation of logits) when f is equivariant
 (c) rank g <= K-1, kernel = {v : J v in span 1}
 (d) control: a non-equivariant net is NOT invariant.
"""
import torch, math
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
n = 12            # input dim = |Z_n|, K = n logits
K = n

def circ(w):      # circulant matrix from first row w
    return torch.stack([torch.roll(w, i) for i in range(len(w))])

W1 = circ(torch.randn(n)) ; b1 = torch.randn(1) ; W2 = circ(torch.randn(n))
def f_eq(x):      # circular conv - tanh - circular conv : equivariant to cyclic shift, rho(T)=T on logits
    return torch.tanh(x @ W1.T + b1) @ W2.T
A1 = torch.randn(n, n); A2 = torch.randn(n, n)
def f_non(x):
    return torch.tanh(x @ A1.T) @ A2.T

def metric(f, x):
    J = torch.autograd.functional.jacobian(f, x)           # K x n
    p = torch.softmax(f(x), -1)
    F = torch.diag(p) - torch.outer(p, p)
    return J.T @ F @ J, J, F

T = lambda x, s=1: torch.roll(x, s, -1)
x = torch.randn(n)
worst_eq, worst_non, worst_shift = 0., 0., 0.
for s in range(1, n):
    g0, *_ = metric(f_eq, x); gT, *_ = metric(f_eq, T(x, s))
    # T^*g at x: dT^T g_{Tx} dT with dT = permutation matrix P_s
    P = torch.roll(torch.eye(n), s, 0)
    pb = P.T @ gT @ P
    worst_eq = max(worst_eq, ((pb - g0).norm() / g0.norm()).item())
    g0n, *_ = metric(f_non, x); gTn, *_ = metric(f_non, T(x, s))
    worst_non = max(worst_non, ((P.T @ gTn @ P - g0n).norm() / g0n.norm()).item())
print(f"(b) equivariant net: max rel. defect |T*g-g|/|g| over s = {worst_eq:.2e}   (control non-equivariant: {worst_non:.2e})")

# (a) logit shift by a smooth non-constant scalar function c(x)
c = lambda x: 3.0 * torch.sin(x.sum()) + (x ** 2).sum()
f_sh = lambda x: f_eq(x) + c(x)
g0, J, F = metric(f_eq, x); g1, J1, _ = metric(f_sh, x)
print(f"(a) logit shift by c(x): |g'-g|/|g| = {((g1-g0).norm()/g0.norm()).item():.2e}  (J changed by {((J1-J).norm()/J.norm()).item():.2f} relative)")

# (c) rank / kernel with input dim n=12 > K-1 : use K=4 head
Wh = torch.randn(4, n)
f_h = lambda x: Wh @ torch.tanh(x @ A1.T)
g, J, F = metric(f_h, x)
ev = torch.linalg.eigvalsh(g)
print("(c) K=4, n=12: eigenvalues of g (top 5):", [f"{e:.2e}" for e in ev.flip(0)[:5].tolist()], " numerical rank =", int((ev > 1e-12 * ev.max()).sum()), "<= K-1 = 3")
Jc = (torch.eye(4) - torch.ones(4, 4) / 4) @ J
U, S, Vh = torch.linalg.svd(Jc)
ker = Vh[3:].T   # kernel of Jc, dim n-3 (generic)
print(f"    kernel dim of g: {int((ev < 1e-12*ev.max()).sum())} (expect n-(K-1)=9); |g v| on ker(J_c) basis: {(g @ ker).norm().item():.2e}")
