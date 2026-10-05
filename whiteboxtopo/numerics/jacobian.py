"""N1: Jacobian-based access (level B).

Conventions: f maps a single point x (1-D tensor, length n) to a 1-D tensor (length m).  Everything is
dtype-agnostic; use float64 for certification.  CPU-only.

Cost model (n = input dim, m = output dim, one f-pass = one "query" for gradient-free methods,
one forward+backward for autograd):
  exact Jacobian ........ min(n,m) autograd passes (jacrev: m VJPs)
  JVP / VJP sketch ...... k passes, error ~ 1/sqrt(k) in Frobenius norm
  Hutchinson tr g ....... N JVPs, relative error eps with prob 1-delta for N = probe_count(eps, delta, a, b)
  randomized SVD ........ (rank+p)(2q+2) matvecs
  gradient-free FD ...... 2*L queries per direction (L Richardson levels); full J: 2*n*L
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.func import jacrev, jvp as _jvp, vjp as _vjp


# ---------------------------------------------------------------- exact products
def jacobian_exact(f, x):
    return jacrev(f)(x)


def jvp(f, x, v):
    return _jvp(f, (x,), (v,))[1]


def vjp(f, x, u):
    _, pull = _vjp(f, x)
    return pull(u)[0]


# ---------------------------------------------------------------- sketches
def vjp_sketch(f, x, k, gen: torch.Generator):
    """S J with S (k x m) Gaussian / sqrt(k): k VJPs.  g_hat = (SJ)^T (SJ) is unbiased for J^T J."""
    y, pull = _vjp(f, x)
    m = y.shape[0]
    S = torch.randn(k, m, generator=gen, dtype=x.dtype) / math.sqrt(k)
    SJ = torch.stack([pull(S[i])[0] for i in range(k)])
    return SJ


def vjp_sketch_gram(f, x, k, gen):
    SJ = vjp_sketch(f, x, k, gen)
    return SJ.T @ SJ


def jvp_sketch(f, x, k, gen):
    """J Omega, Omega (n x k) Gaussian / sqrt(k): k JVPs.  (J Omega)(J Omega)^T is unbiased for J J^T."""
    n = x.shape[0]
    Om = torch.randn(n, k, generator=gen, dtype=x.dtype) / math.sqrt(k)
    return torch.stack([jvp(f, x, Om[:, i]) for i in range(k)], dim=1)


# ---------------------------------------------------------------- Hutchinson
def probe_count(eps: float, delta: float, a: float = 1.0, b: float = 1.0) -> int:
    """Gaussian-probe Hutchinson probe count for a PSD matrix A (here A = g).

    z^T A z - tr A = sum_i lam_i (xi_i^2 - 1).  Laurent-Massart (chi-square Bernstein): for the average of N probes,
        P( T_hat - tr A >= 2||A||_F sqrt(x/N) + 2||A||_2 x/N ) <= e^-x,   lower tail: only the first term.
    Two-sided with x = ln(2/delta).  With a = ||A||_F/tr A, b = ||A||_2/tr A the requirement
        2 a u + 2 b u^2 <= eps,  u = sqrt(x/N)   gives   u = (-a + sqrt(a^2 + 2 b eps)) / (2 b),  N = ceil(x/u^2).
    Worst case (rank one, a=b=1) is used when A is unknown; a,b from a pilot estimate shrink N by the stable rank.
    """
    x = math.log(2.0 / delta)
    if b <= 0: raise ValueError
    u = (-a + math.sqrt(a * a + 2 * b * eps)) / (2 * b)
    return int(math.ceil(x / (u * u)))


def _probes(kind, N, d, gen, dtype):
    if kind == 'gaussian': return torch.randn(N, d, generator=gen, dtype=dtype)
    if kind == 'rademacher': return (torch.randint(0, 2, (N, d), generator=gen) * 2 - 1).to(dtype)
    raise ValueError(kind)


def _quad_forms(f, x, V):
    """||J v||^2 for each row v of V via JVPs (vectorised)."""
    from torch.func import vmap
    JV = vmap(lambda v: _jvp(f, (x,), (v,))[1])(V)
    return (JV ** 2).sum(dim=1)


def hutchinson_trace_g(f, x, N, gen, basis=None, kind='gaussian'):
    """Estimate tr(P^T g P) with g = J^T J.  basis: (n x r) orthonormal columns (default identity -> tr g).
    Returns (estimate, per-probe samples)."""
    n = x.shape[0]
    r = n if basis is None else basis.shape[1]
    Z = _probes(kind, N, r, gen, x.dtype)
    V = Z if basis is None else Z @ basis.T
    q = _quad_forms(f, x, V)
    return q.mean(), q


def slot_shares(f, x, slots, N, gen, kind='gaussian'):
    """Share of tr g carried by each input slot (a partition of coordinate indices):
    tr(P_s g P_s)/tr g, shared probes, shares normalised by the sum of slot estimates (sum to 1 exactly)."""
    n = x.shape[0]
    Z = _probes(kind, N, n, gen, x.dtype)
    ests = []
    for s in slots:
        mask = torch.zeros(n, dtype=x.dtype); mask[list(s)] = 1
        ests.append(_quad_forms(f, x, Z * mask).mean())
    ests = torch.stack(ests)
    return ests / ests.sum(), ests


# ---------------------------------------------------------------- randomized SVD
def rsvd(matvec, rmatvec, m, n, rank, oversample=10, n_iter=2, seed=0):
    """Halko-Martinsson-Tropp randomized SVD from products only.  matvec: (n,k)->(m,k); rmatvec: (m,k)->(n,k)."""
    rng = np.random.default_rng(seed)
    l = min(rank + oversample, min(m, n))
    Y = matvec(rng.standard_normal((n, l)))
    Q, _ = np.linalg.qr(Y)
    for _ in range(n_iter):
        Z, _ = np.linalg.qr(rmatvec(Q))
        Q, _ = np.linalg.qr(matvec(Z))
    B = rmatvec(Q).T                     # (l x n) = Q^T A
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :rank], s[:rank], Vt[:rank]


def rsvd_jacobian(f, x, rank, oversample=10, n_iter=2, seed=0):
    """rSVD of J using batched JVP/VJP (numpy float64 interface)."""
    from torch.func import vmap
    dt = x.dtype
    y, pull = _vjp(f, x)
    m, n = y.shape[0], x.shape[0]

    def mv(M):
        V = torch.as_tensor(M.T, dtype=dt)
        return vmap(lambda v: _jvp(f, (x,), (v,))[1])(V).T.detach().double().numpy()

    def rmv(M):
        U = torch.as_tensor(M.T, dtype=dt)
        return vmap(lambda u: pull(u)[0])(U).T.detach().double().numpy()
    return rsvd(mv, rmv, m, n, rank, oversample, n_iter, seed)


# ---------------------------------------------------------------- gradient-free finite differences
class QueryCounter:
    """Wraps f and counts evaluations (the cost currency of gradient-free access)."""
    def __init__(self, f): self.f, self.count = f, 0
    def __call__(self, x): self.count += 1; return self.f(x)


def fd_step(x, dtype=None):
    """h = eps^(1/3) ||x|| (optimal for plain central differences when the function is O(1)-scaled)."""
    dt = dtype or x.dtype
    eps = torch.finfo(dt).eps
    return float(eps ** (1.0 / 3.0) * max(float(torch.linalg.norm(x)), 1e-8))


def fd_directional(f, x, v, h, levels=1):
    """Central difference of f along v with Richardson extrapolation.
    levels L: steps h, h/2, ..., h/2^(L-1); 2L queries.  Error O(h^(2L))."""
    R = []
    for l in range(levels):
        hl = h / 2 ** l
        R.append([(f(x + hl * v) - f(x - hl * v)) / (2 * hl)])
    for j in range(1, levels):
        for l in range(j, levels):
            R[l].append((4 ** j * R[l][j - 1] - R[l - 1][j - 1]) / (4 ** j - 1))
    return R[levels - 1][levels - 1]


def fd_jacobian(f, x, h=None, levels=1):
    """Gradient-free Jacobian; 2*n*levels queries."""
    h = fd_step(x) if h is None else h
    n = x.shape[0]
    I = torch.eye(n, dtype=x.dtype)
    return torch.stack([fd_directional(f, x, I[i], h, levels) for i in range(n)], dim=1)


def fd_hutchinson_trace(f, x, N, gen, h=None, levels=1):
    """Gradient-free tr g by Hutchinson with FD directional derivatives: 2*N*levels queries."""
    h = fd_step(x) if h is None else h
    n = x.shape[0]
    q = []
    for _ in range(N):
        z = torch.randn(n, generator=gen, dtype=x.dtype)
        q.append((fd_directional(f, x, z, h, levels) ** 2).sum())
    return torch.stack(q).mean()


def query_cost_curve(f, x, J_ref, levels_list=(1, 2, 3), h=None):
    """(queries, rel. error) for full FD Jacobians at increasing Richardson levels."""
    out = []
    for L in levels_list:
        qc = QueryCounter(f)
        J = fd_jacobian(qc, x, h=h, levels=L)
        out.append((qc.count, float((J - J_ref).abs().max() / J_ref.abs().max())))
    return out


# ---------------------------------------------------------------- pullback metrics
def euclid_pullback(f, x):
    J = jacobian_exact(f, x)
    return J.T @ J


def fisher_rao_pullback(f, x, form='logit'):
    """g = J^T G_FR J with the categorical Fisher metric in logit coordinates G_FR = diag(p) - p p^T.
    form='sqrt': 4 J_s^T J_s with s = sqrt(softmax(f)) (same tensor, different arithmetic).
    G_FR 1 = 0, so f -> f + c(x) 1 leaves g unchanged (logit-shift invariance)."""
    if form == 'sqrt':
        Js = jacrev(lambda z: torch.sqrt(torch.softmax(f(z), dim=0)))(x)
        return 4 * Js.T @ Js
    J = jacobian_exact(f, x)
    p = torch.softmax(f(x), dim=0)
    G = torch.diag(p) - torch.outer(p, p)
    return J.T @ G @ J
