"""N3: delta-symmetry search.

Defect of a transformation T for a map F:  delta(T) = RMS_x ||F(Tx) - F(x)|| / RMS_x ||F(x) - mean F||.

(1) Lie-algebra search.  Candidate generators A_1..A_p (d x d matrices acting x -> A x, Frobenius-orthonormalised).
    For a unit generator A = sum c_i A_i,  delta(exp(tA)) = t sqrt(c^T Q c) + O(t^2),
        Q_ij = E_x < dF(x) A_i x , dF(x) A_j x > / Var(F).
    Eigenvectors of Q with eigenvalue <= lam_tol span the delta-symmetry subspace.  Lie closure under the
    commutator [A,B]=AB-BA is iterated, so non-abelian algebras are found from generator pairs.
    Cost: Jacobians O(n m d) (one vmap(jacrev) pass), Q: O(n p m d + n p^2 m), eigh O(p^3), closure O(r^2 d^3 + r^2 p) per round.
(2) Finite groups: closure of generator matrices under products, abelian test, exact defect for callable F.
(3) Calibration from a random-relabelling null (dataset form): F values shuffled over the points, delta recomputed
    through the same kNN-lookup estimator; threshold = alpha-quantile of the null with a DKW confidence band
    (eps = sqrt(ln(2/a')/(2R)), rank band R(alpha -/+ eps)).  Cost O(R n log n) (KD-tree built once).
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.func import jacrev, vmap
from scipy.spatial import cKDTree


# ------------------------------------------------------------------ Lie algebra
def orthonormalize(mats, tol=1e-12):
    M = np.array([np.asarray(m, float).ravel() for m in mats])
    if len(M) == 0: return []
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    r = int((s > tol * max(s[0], 1e-300)).sum())
    d = np.asarray(mats[0]).shape[0]
    return [Vt[i].reshape(d, d) for i in range(r)]


def commutator(A, B): return A @ B - B @ A


def lie_derivative_gram(F, X, basis):
    """Q (p x p, relative to Var F), eigen-decomposition ascending."""
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    J = vmap(jacrev(F))(Xt)                         # (n, m, d)
    Fx = vmap(F)(Xt)
    var = ((Fx - Fx.mean(0)) ** 2).sum(1).mean()
    L = []
    for A in basis:
        v = Xt @ torch.as_tensor(A, dtype=torch.float64).T
        L.append(torch.einsum('nmd,nd->nm', J, v).reshape(len(Xt), -1))
    L = torch.stack(L, 0)                           # (p, n, m)
    Q = torch.einsum('inm,jnm->ij', L, L) / (len(Xt) * var)
    Q = 0.5 * (Q + Q.T)
    lam, V = torch.linalg.eigh(Q)
    return Q.numpy(), lam.numpy(), V.numpy()


def span_residual(M, basis):
    """Frobenius norm of M outside span(basis) (basis Frobenius-orthonormal)."""
    v = M.ravel().copy()
    for B in basis: v -= (B.ravel() @ v) * B.ravel()
    return float(np.linalg.norm(v))


def closure_residual(basis):
    r = 0.0
    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            r = max(r, span_residual(commutator(basis[i], basis[j]), basis))
    return r


def lie_closure(basis, tol=1e-8, max_rounds=10):
    basis = list(basis)
    for _ in range(max_rounds):
        added = False
        for i in range(len(basis)):
            for j in range(i + 1, len(basis)):
                c = commutator(basis[i], basis[j])
                res = c.ravel().copy()
                for B in basis: res -= (B.ravel() @ res) * B.ravel()
                if np.linalg.norm(res) > tol:
                    basis.append(res.reshape(c.shape) / np.linalg.norm(res)); added = True
        if not added: break
    return basis


def is_abelian(basis, tol=1e-8):
    """Abelian iff all brackets vanish (relative to the generator norms, which are 1)."""
    return all(np.linalg.norm(commutator(a, b)) <= tol for i, a in enumerate(basis) for b in basis[i + 1:])


def max_bracket(basis):
    return max([np.linalg.norm(commutator(a, b)) for i, a in enumerate(basis) for b in basis[i + 1:]] or [0.0])


def bracket_defects(F, X, gens):
    """Defect eigenvalue (Q-form) of each generator and of each pairwise bracket: tests whether approximate
    symmetries compose into approximate symmetries."""
    out = {}
    def q(A):
        A = A / max(np.linalg.norm(A), 1e-300)
        return float(lie_derivative_gram(F, X, [A])[0][0, 0])
    for i, A in enumerate(gens):
        out[(i,)] = q(A)
        for j in range(i + 1, len(gens)):
            C = commutator(A, gens[j])
            out[(i, j)] = q(C) if np.linalg.norm(C) > 1e-12 else 0.0
    return out


def search_lie(F, X, candidates, lam_tol=1e-10, close=True, closure_tol=1e-8):
    basis = orthonormalize(candidates)
    Q, lam, V = lie_derivative_gram(F, X, basis)
    k = int((lam <= lam_tol).sum())
    sym = [sum(V[i, a] * basis[i] for i in range(len(basis))) for a in range(k)]
    sym = orthonormalize(sym) if sym else []
    if close and sym: sym = lie_closure(sym, tol=closure_tol)
    gap = float(lam[k] / max(lam[k - 1], 1e-300)) if 0 < k < len(lam) else float('inf')
    return {'dim': len(sym), 'eigenvalues': lam, 'gap_ratio': gap, 'basis': sym, 'candidates': basis,
            'abelian': is_abelian(sym), 'max_bracket': max_bracket(sym),
            'closure_residual': closure_residual(sym) if sym else 0.0}


def principal_sine(Ab, Bb):
    """Sine of the largest principal angle between two subspaces of matrix space (lists of matrices)."""
    A = np.array([a.ravel() for a in Ab]).T; B = np.array([b.ravel() for b in Bb]).T
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(math.sqrt(max(0.0, 1 - s.min() ** 2)))


# ------------------------------------------------------------------ finite groups
def group_closure(gens, max_size=1000, tol=1e-9):
    d = gens[0].shape[0]
    els = [np.eye(d)]
    frontier = [np.eye(d)]
    while frontier and len(els) <= max_size:
        new = []
        for e in frontier:
            for g in gens:
                p = g @ e
                if not any(np.allclose(p, q, atol=tol) for q in els):
                    els.append(p); new.append(p)
        frontier = new
    return els


def group_is_abelian(els, tol=1e-9):
    return all(np.allclose(a @ b, b @ a, atol=tol) for a in els for b in els)


def delta_exact(F, X, T):
    """delta(T) for a callable F (batch: (n,d)->(n,m)) and callable T on batches."""
    X = np.asarray(X, float)
    Fx = np.asarray(F(X)); FT = np.asarray(F(T(X)))
    num = np.sqrt(np.mean(np.sum((FT - Fx) ** 2, 1)))
    den = np.sqrt(np.mean(np.sum((Fx - Fx.mean(0)) ** 2, 1)))
    return float(num / den)


# ------------------------------------------------------------------ dataset form + calibration
class DatasetDelta:
    """delta(T) from samples only: F(Tx_i) is read from the k nearest data points to T x_i."""
    def __init__(self, X, k=3):
        self.X = np.asarray(X, float); self.tree = cKDTree(self.X); self.k = k
    def prepare(self, T):
        Y = T(self.X)
        _, idx = self.tree.query(Y, self.k)
        return idx.reshape(len(self.X), -1)
    def delta(self, Fvals, idx):
        Fvals = np.asarray(Fvals, float).reshape(len(self.X), -1)
        Fhat = Fvals[idx].mean(1)
        num = np.sqrt(np.mean(np.sum((Fhat - Fvals) ** 2, 1)))
        den = np.sqrt(np.mean(np.sum((Fvals - Fvals.mean(0)) ** 2, 1)))
        return float(num / den)


def null_deltas(dd: DatasetDelta, Fvals, T, R, rng):
    idx = dd.prepare(T)
    Fvals = np.asarray(Fvals)
    return np.array([dd.delta(Fvals[rng.permutation(len(Fvals))], idx) for _ in range(R)])


def calibrate(null, alpha=0.05, conf=0.95):
    """alpha-quantile of the null with a DKW band: returns (point, lo, hi, eps)."""
    null = np.sort(np.asarray(null)); R = len(null)
    eps = math.sqrt(math.log(2.0 / (1 - conf)) / (2 * R))
    r = lambda a: int(min(max(math.floor(R * a), 0), R - 1))
    return float(null[r(alpha)]), float(null[r(alpha - eps)]), float(null[r(alpha + eps)]), eps
