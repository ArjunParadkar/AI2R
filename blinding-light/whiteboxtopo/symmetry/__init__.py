"""whiteboxtopo.symmetry -- continuous symmetries of the pullback metric g = J^T J  (SPEC3 S2, S4, S6).

 * null_space      -- the null distribution of g at a point: what the network is blind to (S2).
 * killing         -- Lie derivative of g along linear vector fields V(x) = A x + b; per-generator
                      Killing residual and the dimension of the Killing algebra inside the span (S4).
 * quotient_depth  -- how far a known transformation T moves a representation, layer by layer (S6).
 * image_generators-- the 6 affine image-warp generators + brightness + contrast as linear fields.

Nothing here forms a d_in x d_in matrix.  The finite (automorphism) side lives in `finite.py`.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

from ..metric import pullback_input, gram_spectrum, g_matvec

try:                                     # the finite-group side is added by another file
    from .finite import *                # noqa: F401,F403
except ImportError:
    pass


# ----------------------------------------------------------------------------- S2
def null_space(J, tol=1e-3):
    """Null distribution of g = J^T J at one point (J: d_out x d_in) or a batch (N x d_out x d_in).

    An eigenvalue of g is 'null' when it is below tol x the top eigenvalue.  g has d_in eigenvalues of
    which at most d_out are nonzero, and those are the eigenvalues of J J^T (exact).
    Returns dict(n0, nu = n0/d_in, rank, eigenvalues, V) with V (d_in x rank) an orthonormal basis of the
    NON-null (row) space, so the null projector is I - V V^T (never formed).  For a batch, n0/nu/rank are
    arrays and V is a list."""
    single = J.dim() == 2
    Jb = J[None] if single else J
    d_in = Jb.shape[-1]
    lam, U = gram_spectrum(Jb, return_vectors=True)
    keep = lam >= tol * lam[:, :1].clamp_min(1e-300)
    keep &= lam > 0
    rank = keep.sum(1)
    Vs = []
    for i in range(len(Jb)):
        r = int(rank[i])
        Vi = (Jb[i].transpose(0, 1) @ U[i, :, :r]) / lam[i, :r].clamp_min(1e-300).sqrt()
        Vs.append(Vi)
    n0 = d_in - rank
    if single:
        return dict(n0=int(n0[0]), nu=float(n0[0]) / d_in, rank=int(rank[0]), eigenvalues=lam[0], V=Vs[0])
    return dict(n0=n0.cpu().numpy(), nu=(n0.double() / d_in).cpu().numpy(), rank=rank.cpu().numpy(),
                eigenvalues=lam, V=Vs)


def null_project(V, v):
    """(I - V V^T) v."""
    return v - V @ (V.transpose(-1, -2) @ v)


# ----------------------------------------------------------------------------- linear fields
class LinearField:
    """V(x) = A x + b on flattened inputs of dimension d_in.

    A is one of: None (A = 0), a dense (d_in x d_in) matrix, a per-channel matrix M (P x P) acting
    identically on each of C channels (A = I_C (x) M, d_in = C P; never formed), or a scalar multiple
    of the identity.  `normalise()` rescales to ||A||_F = 1 (or ||b|| = 1 when A = 0), SPEC3 S4."""

    def __init__(self, name, d_in, dense=None, channel=None, n_channels=1, identity=None, b=None):
        self.name, self.d_in = name, d_in
        self.dense, self.channel, self.C, self.identity, self.b = dense, channel, n_channels, identity, b

    # -- A x, for x of shape (..., d_in)
    def Ax(self, X):
        if self.dense is not None:
            return X @ self.dense.transpose(0, 1)
        if self.channel is not None:
            sh = X.shape
            return (X.reshape(*sh[:-1], self.C, -1) @ self.channel.transpose(0, 1)).reshape(sh)
        if self.identity is not None:
            return self.identity * X
        return torch.zeros_like(X)

    # -- A^T r, for r of shape (..., d_in); (J A) has rows A^T J_k
    def ATr(self, R):
        if self.dense is not None:
            return R @ self.dense
        if self.channel is not None:
            sh = R.shape
            return (R.reshape(*sh[:-1], self.C, -1) @ self.channel).reshape(sh)
        if self.identity is not None:
            return self.identity * R
        return torch.zeros_like(R)

    def V(self, X):
        out = self.Ax(X)
        return out + self.b if self.b is not None else out

    def fro(self):
        if self.dense is not None:
            return float(self.dense.norm())
        if self.channel is not None:
            return float(self.channel.norm()) * math.sqrt(self.C)
        if self.identity is not None:
            return abs(float(self.identity)) * math.sqrt(self.d_in)
        return 0.0

    def op_norm(self):
        if self.dense is not None:
            return float(torch.linalg.matrix_norm(self.dense.double(), 2))
        if self.channel is not None:
            return float(torch.linalg.matrix_norm(self.channel.double(), 2))
        if self.identity is not None:
            return abs(float(self.identity))
        return 0.0

    def normalise(self):
        nA = self.fro()
        s = nA if nA > 0 else float(self.b.norm())
        if self.dense is not None:
            self.dense = self.dense / s
        if self.channel is not None:
            self.channel = self.channel / s
        if self.identity is not None:
            self.identity = self.identity / s
        if self.b is not None:
            self.b = self.b / s
        return self

    def to(self, device=None, dtype=None):
        for k in ('dense', 'channel', 'b'):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, v.to(device=device, dtype=dtype))
        return self


def field(name, A=None, b=None, d_in=None):
    """Convenience: a normalised dense linear field from numpy/torch A and/or b."""
    A = None if A is None else torch.as_tensor(A, dtype=torch.float64)
    b = None if b is None else torch.as_tensor(b, dtype=torch.float64)
    d = d_in or (A.shape[0] if A is not None else b.shape[0])
    return LinearField(name, d, dense=A, b=b).normalise()


# ----------------------------------------------------------------------------- S4
def killing(J_fn, xs, generators, n_sketch=8, seed=0, chunk=32, rel_tol=0.1, fd_rel=1e-2):
    """Killing residuals of g = J^T J along linear fields, and the Killing algebra inside their span.

    J_fn(x_batch) -> (B, d_out, d_in); xs (N, *shape); generators: list of (normalised) LinearField.
    L_V g = K^T J + J^T K,  K = d_V J + J A,  d_V J by central difference with eps = fd_rel ||x|| / ||A x + b||.
    r_i = E||L_{V_i} g||_F / (2 E||g||_F), with ||K^T J + J^T K||_F^2 = 2 tr(K K^T J J^T) + 2 tr((K J^T)^2).
    Algebra: singular values s of c -> sum_i c_i L_{V_i} g, from n_sketch Gaussian sketches (L_V g) z per
    sample; dim k = #{ s_i < rel_tol * s_1 }.
    """
    n_gen = len(generators)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    fro = [[] for _ in range(n_gen)]
    gfro = []
    gram = None
    shape = xs.shape[1:]
    for s in range(0, len(xs), chunk):
        xb = xs[s:s + chunk]
        B = len(xb)
        xf = xb.reshape(B, -1)
        J = J_fn(xb)
        dt, dev = J.dtype, J.device
        d_in = J.shape[-1]
        Jt = J.transpose(1, 2)
        JJt = J @ Jt
        gfro.append(JJt.flatten(1).norm(dim=1).double().cpu())
        z = torch.randn(B, n_sketch, d_in, generator=gen, dtype=torch.float64).to(device=dev, dtype=dt)
        Jz = z @ Jt                                              # (B, k, d_out)
        Lz_all = []
        xn = xf.norm(dim=1)
        for i, fld in enumerate(generators):
            V = fld.V(xf.to(dt))
            eps = fd_rel * xn.to(dt) / V.norm(dim=1).clamp_min(1e-30)
            step = (eps[:, None] * V).reshape(B, *shape).to(xb.dtype)
            Jp = J_fn(xb + step)
            Jm = J_fn(xb - step)
            K = (Jp - Jm) / (2 * eps[:, None, None]) + fld.ATr(J)
            del Jp, Jm
            KJt = K @ Jt
            KKt = K @ K.transpose(1, 2)
            f2 = 2 * (KKt * JJt).flatten(1).sum(1) + 2 * (KJt * KJt.transpose(1, 2)).flatten(1).sum(1)
            fro[i].append(f2.clamp_min(0).sqrt().double().cpu())
            Lz = Jz @ K + (z @ K.transpose(1, 2)) @ J            # (B, k, d_in) = (L_V g) z
            Lz_all.append(Lz.reshape(-1).double())
            del K, KJt, KKt, Lz
        M = torch.stack(Lz_all, 0)
        Gc = (M @ M.transpose(0, 1)).cpu()
        gram = Gc if gram is None else gram + Gc
        del M, Lz_all, J, Jt, JJt, z, Jz
        if xs.is_cuda:
            torch.cuda.empty_cache()
    fro = [torch.cat(f) for f in fro]
    gfro = torch.cat(gfro)
    Eg = float(gfro.mean())
    res = [float(f.mean()) / (2 * Eg) for f in fro]
    ev, W = torch.linalg.eigh(gram)
    sv = ev.flip(0).clamp_min(0).sqrt()
    W = W.flip(1)
    n_sk = len(xs) * n_sketch
    # machine-zero guard: if even the largest singular value is numerically zero, every field is Killing
    if float(sv[0]) <= 1e-9 * Eg * math.sqrt(n_sk):
        dim_k = n_gen
    else:
        dim_k = int((sv < rel_tol * sv[0]).sum())
    return dict(names=[g.name for g in generators], residuals=res, dim_k=dim_k,
                singular_values=sv.tolist(), singular_vectors=W.tolist(), E_g_fro=Eg,
                per_sample_Lg_fro=[f.numpy() for f in fro], per_sample_g_fro=gfro.numpy(), n=len(xs))


# ----------------------------------------------------------------------------- S6
def _jvp_norm(tail, h, delta):
    """|| J_tail(h) . delta || per sample, by forward-mode (no Jacobian formed)."""
    from torch.func import jvp
    _, t = jvp(tail, (h,), (delta,))
    return t.flatten(1).norm(dim=1)


def quotient_depth(layer_fns, tail_fns, x, Tx, thresh=0.05, groups=None, max_pairs=8):
    """q_l(T) = E_x d_l(x, Tx) / E_{x,x'} d_l(x, x'), layer by layer; invariance depth = first l with q < thresh.

    layer_fns[l](x) -> h_l (B, ...);  tail_fns[l](h_l) -> network output (the remainder of the network).
    Primary d_l = linearised pullback || J_l(h_l(x)) . (h_l(y) - h_l(x)) ||, J_l = d tail_l / d h_l (SPEC3 fact 5:
    the secant pullback is the same number at every layer, so it cannot locate anything);
    secondary d_l = Euclidean || h_l(y) - h_l(x) ||.  The denominator pairs each x with up to `max_pairs` other
    probe points (cyclic shifts).  groups: optional integer labels -> per-group numerators over the global
    denominator.  depth is None when no layer is below thresh."""
    N = len(x)
    shifts = list(range(1, min(N - 1, max_pairs) + 1))
    out = dict(q_pullback=[], q_euclid=[], num_pullback=[], den_pullback=[], num_euclid=[], den_euclid=[])
    per_group = {} if groups is not None else None
    for lf, tf in zip(layer_fns, tail_fns):
        with torch.no_grad():
            h, hT = lf(x), lf(Tx)
        d = hT - h
        nP = _jvp_norm(tf, h, d).detach()
        nE = d.flatten(1).norm(dim=1)
        dP, dE = [], []
        for k in shifts:
            dd = torch.roll(h, k, 0) - h
            dP.append(_jvp_norm(tf, h, dd).detach())
            dE.append(dd.flatten(1).norm(dim=1))
        dP, dE = torch.stack(dP).mean(), torch.stack(dE).mean()
        out['num_pullback'].append(float(nP.mean())); out['den_pullback'].append(float(dP))
        out['num_euclid'].append(float(nE.mean())); out['den_euclid'].append(float(dE))
        out['q_pullback'].append(float(nP.mean() / dP.clamp_min(1e-300)))
        out['q_euclid'].append(float(nE.mean() / dE.clamp_min(1e-300)))
        if groups is not None:
            gr = np.asarray(groups)
            for c in np.unique(gr):
                m = torch.as_tensor(gr == c, device=nP.device)
                pg = per_group.setdefault(int(c), dict(q_pullback=[], q_euclid=[]))
                pg['q_pullback'].append(float(nP[m].mean() / dP.clamp_min(1e-300)))
                pg['q_euclid'].append(float(nE[m].mean() / dE.clamp_min(1e-300)))

    def first(q):
        for l, v in enumerate(q):
            if v < thresh:
                return l
        return None
    out['depth_pullback'] = first(out['q_pullback'])
    out['depth_euclid'] = first(out['q_euclid'])
    if groups is not None:
        for c, pg in per_group.items():
            pg['depth_pullback'] = first(pg['q_pullback'])
            pg['depth_euclid'] = first(pg['q_euclid'])
        out['per_group'] = per_group
    return out


# ----------------------------------------------------------------------------- image generators
_WARPS = {                                   # d(theta)/d(param) of the 2x3 affine grid, image moving "forward"
    'translate_x': [[0., 0., -1.], [0., 0., 0.]],
    'translate_y': [[0., 0., 0.], [0., 0., -1.]],
    'rotation':    [[0., -1., 0.], [1., 0., 0.]],     # same sense as systems.rot_mnist.rotate
    'scale':       [[-1., 0., 0.], [0., -1., 0.]],
    'shear_1':     [[-1., 0., 0.], [0., 1., 0.]],
    'shear_2':     [[0., -1., 0.], [-1., 0., 0.]],
}
WARP_NAMES = list(_WARPS)


def warp(x, name, param):
    """Bilinear affine warp of a batch of images (B, C, H, W) by `param` along the named one-parameter family."""
    G = torch.tensor(_WARPS[name], dtype=x.dtype, device=x.device)
    if name == 'rotation':
        c, s = math.cos(param), math.sin(param)
        th = torch.tensor([[c, -s, 0.], [s, c, 0.]], dtype=x.dtype, device=x.device)
    else:
        th = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=x.dtype, device=x.device) + param * G
    grid = F.affine_grid(th[None].expand(len(x), 2, 3), x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


def warp_generator_matrix(name, H, W, delta=0.02, dtype=torch.float64):
    """Per-channel matrix M (HW x HW) of A = d/dtheta warp_theta at theta = 0, by central difference of the
    (linear-in-the-image) bilinear warp applied to the pixel basis: column j = (warp_+d(e_j) - warp_-d(e_j)) / 2d."""
    P = H * W
    E = torch.eye(P, dtype=dtype).reshape(P, 1, H, W)
    D = (warp(E, name, delta) - warp(E, name, -delta)) / (2 * delta)
    return D.reshape(P, P).transpose(0, 1).contiguous()          # M[:, j] = A e_j


def image_generators(C, H, W, delta=0.02, device='cpu', dtype=torch.float32, which=None):
    """The 8 generators of SPEC3 fact 6 as normalised LinearFields on flattened (C, H, W) images:
    translate_x, translate_y, rotation, scale, shear_1, shear_2 (image warps; A = I_C (x) M), brightness
    (b = constant image, A = 0), contrast (A = I)."""
    d_in = C * H * W
    out = []
    for nm in WARP_NAMES:
        M = warp_generator_matrix(nm, H, W, delta)
        out.append(LinearField(nm, d_in, channel=M, n_channels=C))
    out.append(LinearField('brightness', d_in, b=torch.ones(d_in, dtype=torch.float64)))
    out.append(LinearField('contrast', d_in, identity=1.0))
    out = [g.normalise().to(device=device, dtype=dtype) for g in out]
    if which is not None:
        out = [g for g in out if g.name in which]
    return out


__all__ = [n for n in dir() if not n.startswith('_') and n not in ('math', 'np', 'torch', 'F')]
