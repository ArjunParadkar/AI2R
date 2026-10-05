"""N4: meshfree least-squares Killing-field solver (Gaussian RBF collocation).

Killing equation  (L_v g)_ij = v^k d_k g_ij + g_kj d_i v^k + g_ik d_j v^k = 0.
Unknown field  v^c(x) = sum_a coef[c,a] phi_a(x),  phi_a(x) = exp(-|psi(x) - psi(c_a)|^2 / (2 s^2)),
where psi is a feature map (identity on a patch; (cos,sin) on periodic coordinates, so periodic domains are exact).
Collocating at n points gives A (n*npairs x d*N).  Because Gaussian bases are nearly dependent the *value*
matrix V (field values at the points) is whitened by an SVD, V = U S W^T (columns with S_i < rtol S_0 dropped):
coef = W S^-1 eta with eta orthonormal in the L2(points) sense.  The singular values sigma_i of
A_red = A W S^-1 are then ||L_v g||_rms / ||v||_rms minimised over unit fields, ascending; Killing fields are the
(near-)null right singular vectors.  Null dimension = #{sigma < tau} with a reported gap.
Cost: building A O(n N d^2 npairs), SVD of V and A_red O(n^2 d^2 N) -> fine up to n ~ 2000 on CPU.
Convergence: for smooth g and fields, Gaussian RBF interpolation converges spectrally in the fill distance h
(until conditioning/rtol saturate); the observed order is measured by the tests, not assumed.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import qmc


def halton(n, lo, hi, seed=0):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return qmc.scale(qmc.Halton(len(lo), scramble=True, seed=seed).random(n), lo, hi)


def feat_identity(X):
    X = np.asarray(X, float)
    return X, np.broadcast_to(np.eye(X.shape[1]), (X.shape[0], X.shape[1], X.shape[1])).copy()


def feat_periodic(periodic_mask):
    """psi: periodic coords -> (cos, sin), others unchanged.  dpsi: (n, p, d)."""
    pm = np.asarray(periodic_mask, bool)
    def feat(X):
        X = np.asarray(X, float); n, d = X.shape
        cols, dcols = [], []
        for c in range(d):
            if pm[c]:
                cols += [np.cos(X[:, c]), np.sin(X[:, c])]
                for tr in (-np.sin(X[:, c]), np.cos(X[:, c])):
                    D = np.zeros((n, d)); D[:, c] = tr; dcols.append(D)
            else:
                cols.append(X[:, c]); D = np.zeros((n, d)); D[:, c] = 1; dcols.append(D)
        return np.stack(cols, 1), np.stack(dcols, 1)
    return feat


def metric_and_derivative(g_fn, X, h=1e-5):
    X = np.asarray(X, float); n, d = X.shape
    G = np.asarray(g_fn(X))
    dG = np.zeros((n, d, d, d))                      # dG[n,k,i,j] = d_k g_ij
    for k in range(d):
        e = np.zeros(d); e[k] = h
        dG[:, k] = (np.asarray(g_fn(X + e)) - np.asarray(g_fn(X - e))) / (2 * h)
    return G, dG


class KillingSolver:
    def __init__(self, centers, feat, shape_mult=2.0, domain_volume=None, rtol=1e-9, fill=None):
        self.C = np.asarray(centers, float); self.feat = feat
        N, d = self.C.shape
        if fill is None:
            if domain_volume is None: raise ValueError('give domain_volume or fill')
            fill = (domain_volume / N) ** (1.0 / d)
        self.h = float(fill); self.s = shape_mult * self.h; self.rtol = rtol
        self.PC, _ = feat(self.C)
        self.d, self.N = d, N

    def _basis(self, X):
        P, dP = self.feat(X)                                       # (n,p), (n,p,d)
        diff = P[:, None, :] - self.PC[None, :, :]                 # (n,N,p)
        phi = np.exp(-(diff ** 2).sum(2) / (2 * self.s ** 2))      # (n,N)
        dphi = -np.einsum('nap,npk->nak', diff, dP) / self.s ** 2 * phi[:, :, None]   # (n,N,d)
        return phi, dphi

    def assemble(self, X, g_fn):
        X = np.asarray(X, float); n, d, N = len(X), self.d, self.N
        G, dG = metric_and_derivative(g_fn, X)
        sc = np.abs(G).mean()                      # Killing's equation is scale-free: normalise g to O(1)
        G, dG = G / sc, dG / sc
        phi, dphi = self._basis(X)
        pairs = [(i, j) for i in range(d) for j in range(i, d)]
        # unknown index = c*N + a
        V = np.zeros((n, d, d * N))
        for c in range(d): V[:, c, c * N:(c + 1) * N] = phi
        A = np.zeros((n, len(pairs), d * N))
        for p_, (i, j) in enumerate(pairs):
            for k in range(d):
                # v^k d_k g_ij
                A[:, p_, k * N:(k + 1) * N] += dG[:, k, i, j][:, None] * phi
                # g_kj d_i v^k + g_ik d_j v^k
                A[:, p_, k * N:(k + 1) * N] += G[:, k, j][:, None] * dphi[:, :, i] + G[:, i, k][:, None] * dphi[:, :, j]
        return V.reshape(n * d, d * N), A.reshape(n * len(pairs), d * N)

    def solve(self, X, g_fn):
        V, A = self.assemble(X, g_fn)
        U, S, Wt = np.linalg.svd(V, full_matrices=False)
        keep = S > self.rtol * S[0]
        W, Sk = Wt[keep].T, S[keep]
        Ared = A @ (W / Sk)
        Ub, sig, Vt = np.linalg.svd(Ared, full_matrices=False)
        order = np.argsort(sig)
        sig = sig[order]; Vt = Vt[order]
        coefs = (W / Sk) @ Vt.T                                     # columns: coefficient vectors, ascending sigma
        self.sigma, self.coefs, self.rank = sig, coefs, int(keep.sum())
        return sig, coefs

    def null_dim(self, tau=1e-3, min_gap=100.0):
        sig = self.sigma
        K = int((sig < tau).sum())
        gap = float(sig[K] / max(sig[K - 1], 1e-300)) if 0 < K < len(sig) else (float('inf') if K == 0 else float('nan'))
        return K, gap

    def evaluate(self, coef, X):
        phi, _ = self._basis(np.asarray(X, float))
        return np.stack([phi @ coef[c * self.N:(c + 1) * self.N] for c in range(self.d)], 1)   # (n,d)

    def null_fields(self, K, X):
        """values of the K smallest-sigma fields at X: (n*d, K)."""
        return np.stack([self.evaluate(self.coefs[:, i], X).ravel() for i in range(K)], 1)

    def residual(self, coef, X, g_fn):
        """rms ||L_v g|| / rms ||v|| of one field on (independent) points."""
        V, A = self.assemble(X, g_fn)
        return float(np.linalg.norm(A @ coef) / max(np.linalg.norm(V @ coef), 1e-300))


def subspace_error(F_found, F_exact):
    """sine of largest principal angle between column spaces (inputs: (m,K) arrays on the same points)."""
    Qa, _ = np.linalg.qr(F_found); Qb, _ = np.linalg.qr(F_exact)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.sqrt(max(0.0, 1 - s.min() ** 2)))


def lie_derivative_residual(v_fn, g_fn, X, h=1e-5):
    """||L_v g||_max / ||g||_max for an analytic field v_fn(X)->(n,d), by central FD (for reference fields)."""
    X = np.asarray(X, float); n, d = X.shape
    G, dG = metric_and_derivative(g_fn, X)
    v = v_fn(X)
    dv = np.zeros((n, d, d))                                         # dv[n,i,k] = d_i v^k
    for i in range(d):
        e = np.zeros(d); e[i] = h
        dv[:, i] = (v_fn(X + e) - v_fn(X - e)) / (2 * h)
    L = np.einsum('nk,nkij->nij', v, dG) + np.einsum('nkj,nik->nij', G, dv) + np.einsum('nik,njk->nij', G, dv)
    return float(np.abs(L).max() / np.abs(G).max())
