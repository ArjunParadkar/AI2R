"""Run 3, SPEC3 S1/S2/S4/S6 -- planted truth for the continuous-symmetry instruments, before any network.

Run from inside tests/ with:  python -m pytest test_run3_symmetry_continuous.py -q --noconftest
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util
import math
import numpy as np
import pytest
import torch


def _load(name, rel):
    """Import the two run-3 packages without importing the whole whiteboxtopo namespace (keeps the test light)."""
    import types
    if 'whiteboxtopo' not in sys.modules:
        pkg = types.ModuleType('whiteboxtopo'); pkg.__path__ = [str(ROOT / 'whiteboxtopo')]
        sys.modules['whiteboxtopo'] = pkg
    return importlib.import_module(name)


metric = _load('whiteboxtopo.metric', 'metric')
sym = _load('whiteboxtopo.symmetry', 'symmetry')
torch.set_default_dtype(torch.float64)

ROT2 = np.array([[0., -1.], [1., 0.]])


def gens_r2():
    return [sym.field('rotation', A=ROT2), sym.field('translate_x', b=[1., 0.]),
            sym.field('translate_y', b=[0., 1.]), sym.field('scaling', A=np.eye(2))]


def Jfn(f, method='backward'):
    return lambda xb: metric.pullback_input(f, xb, batch=64, method=method)


def pts(n, d, seed=0, lo=0.6, hi=1.6):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g)
    r = lo + (hi - lo) * torch.rand(n, 1, generator=g)
    return v / v.norm(dim=1, keepdim=True) * r


# ------------------------------------------------------------------ S1
def test_pullback_matches_analytic_and_both_methods_agree():
    W = torch.randn(5, 8, generator=torch.Generator().manual_seed(1))
    f = lambda x: torch.tanh(x @ W.T)
    x = pts(7, 8)
    Ja = metric.pullback_input(f, x, batch=3)
    Jb = metric.pullback_input(f, x, batch=4, method='func')
    exact = (1 - torch.tanh(x @ W.T) ** 2)[:, :, None] * W[None]
    assert torch.allclose(Ja, exact, atol=1e-12) and torch.allclose(Jb, exact, atol=1e-12)


def test_gram_spectrum_is_the_nonzero_spectrum_of_g():
    J = torch.randn(4, 30, generator=torch.Generator().manual_seed(2))
    lam = metric.gram_spectrum(J)
    full = torch.linalg.eigvalsh(J.T @ J).flip(0)
    assert torch.allclose(lam, full[:4], atol=1e-10) and full[4:].abs().max() < 1e-10
    v = torch.randn(30, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(metric.g_matvec(J, v), J.T @ (J @ v), atol=1e-12)


# ------------------------------------------------------------------ S2 (a)
def test_null_space_of_radial_function_is_the_sphere_tangent():
    f = lambda x: torch.sin((x ** 2).sum(1, keepdim=True))
    x = pts(5, 8, seed=4, lo=0.5, hi=1.1)
    J = metric.pullback_input(f, x)
    for i in range(len(x)):
        ns = sym.null_space(J[i], tol=1e-3)
        assert ns['n0'] == 7 and abs(ns['nu'] - 7 / 8) < 1e-12 and ns['rank'] == 1
        xhat = x[i] / x[i].norm()
        P_null = torch.eye(8) - ns['V'] @ ns['V'].T
        P_tan = torch.eye(8) - torch.outer(xhat, xhat)
        assert torch.allclose(P_null, P_tan, atol=1e-9)
        assert sym.null_project(ns['V'], x[i]).norm() < 1e-9          # the radial direction is NOT null
    nb = sym.null_space(J, tol=1e-3)
    assert (nb['n0'] == 7).all() and np.allclose(nb['nu'], 7 / 8)


# ------------------------------------------------------------------ S4 (b)
def test_killing_radial_map_only_rotation():
    w = torch.tensor([0.7, -1.3, 0.4])
    f = lambda x: torch.tanh((x ** 2).sum(1, keepdim=True)) * w[None]
    out = sym.killing(Jfn(f), pts(64, 2, seed=5), gens_r2())
    r = dict(zip(out['names'], out['residuals']))
    assert r['rotation'] < 1e-3, r
    assert min(r['translate_x'], r['translate_y'], r['scaling']) >= 0.1, r
    assert out['dim_k'] == 1, out['singular_values']
    c = np.abs(np.array(out['singular_vectors'])[:, -1])              # the Killing combination IS the rotation
    assert c[0] > 0.999


# ------------------------------------------------------------------ S4 (c)
def test_killing_equivariant_map():
    def f(x):                                                         # f(Rx) = R f(x): rho = the rotation itself
        return torch.sin(1.3 * (x ** 2).sum(1, keepdim=True)) * x + 0.5 * x
    R = torch.tensor([[math.cos(0.7), -math.sin(0.7)], [math.sin(0.7), math.cos(0.7)]])
    x = pts(64, 2, seed=6)
    assert torch.allclose(f(x @ R.T), f(x) @ R.T, atol=1e-12)
    out = sym.killing(Jfn(f), x, gens_r2())
    r = dict(zip(out['names'], out['residuals']))
    assert r['rotation'] < 1e-3 and min(r['translate_x'], r['translate_y'], r['scaling']) >= 0.1, r
    assert out['dim_k'] == 1


# ------------------------------------------------------------------ S4 (d)
def test_killing_identity_map_isometries():
    out = sym.killing(Jfn(lambda x: x), pts(32, 2, seed=7), gens_r2())
    r = dict(zip(out['names'], out['residuals']))
    assert max(r['rotation'], r['translate_x'], r['translate_y']) < 1e-9, r
    assert abs(r['scaling'] - 1 / math.sqrt(2)) < 1e-9, r            # L_V g = 2I/sqrt(2), ||g||_F = sqrt(2)
    assert out['dim_k'] == 3
    iso = sym.killing(Jfn(lambda x: x), pts(32, 2, seed=7), gens_r2()[:3])     # all-Killing: the machine-zero guard
    assert iso['dim_k'] == 3


def test_killing_frobenius_identity_matches_dense():
    """The d_out-sized trace formula equals the Frobenius norm of the dense d_in x d_in Lie derivative."""
    g = torch.Generator().manual_seed(8)
    W1, W2 = torch.randn(6, 5, generator=g), torch.randn(3, 6, generator=g)
    f = lambda x: torch.tanh(x @ W1.T) @ W2.T
    A = torch.randn(5, 5, generator=g); b = torch.randn(5, generator=g)
    fld = sym.LinearField('generic', 5, dense=A.clone(), b=b.clone()).normalise()
    x = pts(4, 5, seed=9)
    out = sym.killing(Jfn(f), x, [fld])
    J = metric.pullback_input(f, x)
    for i in range(len(x)):
        V = fld.V(x[i:i + 1])[0]
        eps = 1e-2 * x[i].norm() / V.norm()
        gp = metric.pullback_input(f, (x[i] + eps * V)[None])[0]; gm = metric.pullback_input(f, (x[i] - eps * V)[None])[0]
        L = (gp.T @ gp - gm.T @ gm) / (2 * eps) + fld.dense.T @ J[i].T @ J[i] + J[i].T @ J[i] @ fld.dense
        assert abs(out['per_sample_Lg_fro'][0][i] - float(L.norm())) < 2e-3 * float(L.norm())


# ------------------------------------------------------------------ S6 (e)
def _swap_mlp(seed=10, nb=4, nh=6):
    g = torch.Generator().manual_seed(seed)
    W = 1.5 * torch.randn(nh, nb, generator=g); B = 1.5 * torch.randn(nh, nh, generator=g)
    W3 = torch.randn(5, nh, generator=g); W4 = torch.randn(3, 5, generator=g)
    l1 = lambda h: torch.cat([torch.tanh(h[:, :nb] @ W.T), torch.tanh(h[:, nb:] @ W.T)], 1)
    l2 = lambda h: torch.tanh(h[:, :nh] @ B.T) + torch.tanh(h[:, nh:] @ B.T)      # weights shared by the two blocks
    l3 = lambda h: torch.tanh(h @ W3.T)
    l4 = lambda h: h @ W4.T
    return [l1, l2, l3, l4]


def _compose(fs):
    def f(h):
        for fn in fs:
            h = fn(h)
        return h
    return f


def test_quotient_depth_block_swap_is_two_generic_is_none():
    Ls = _swap_mlp()
    layer_fns = [_compose(Ls[:k]) for k in range(5)]                  # 0 = input ... 4 = output
    tail_fns = [_compose(Ls[k:]) for k in range(5)]
    x = torch.randn(200, 8, generator=torch.Generator().manual_seed(11))
    swap = torch.cat([x[:, 4:], x[:, :4]], 1)
    q = sym.quotient_depth(layer_fns, tail_fns, x, swap)
    assert q['depth_pullback'] == 2 and q['depth_euclid'] == 2, q
    assert max(q['q_pullback'][2:]) < 1e-12 and min(q['q_pullback'][:2]) > 0.05
    Q, _ = torch.linalg.qr(torch.randn(8, 8, generator=torch.Generator().manual_seed(12)))
    qg = sym.quotient_depth(layer_fns, tail_fns, x, x @ Q.T, groups=np.arange(200) % 2)
    assert qg['depth_pullback'] is None and qg['depth_euclid'] is None, qg
    assert all(v['depth_pullback'] is None for v in qg['per_group'].values())


# ------------------------------------------------------------------ image generators (f)
def _grid(H, W):
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing='ij')
    return ys - (H - 1) / 2, xs - (W - 1) / 2


def test_image_generators_rotation_kills_radial_translation_is_minus_dx():
    H = W = 32
    gens = {g.name: g for g in sym.image_generators(1, H, W, dtype=torch.float64)}
    assert list(gens) == ['translate_x', 'translate_y', 'rotation', 'scale', 'shear_1', 'shear_2', 'brightness', 'contrast']
    for g in gens.values():
        n = g.fro() if g.fro() > 0 else float(g.b.norm())
        assert abs(n - 1) < 1e-9
    ys, xs = _grid(H, W)
    radial = torch.exp(-(xs ** 2 + ys ** 2) / (2 * 5.0 ** 2)).reshape(1, -1)
    blob = torch.exp(-((xs - 3) ** 2 / (2 * 4.0 ** 2) + (ys + 2) ** 2 / (2 * 3.0 ** 2))).reshape(1, -1)
    rot = gens['rotation']
    assert rot.Ax(radial).norm() < 0.02 * rot.Ax(blob).norm()         # rotation generator ~ 0 on a radial image
    img = blob.reshape(H, W)
    ddx = torch.zeros_like(img); ddx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
    ddy = torch.zeros_like(img); ddy[1:-1] = (img[2:] - img[:-2]) / 2
    cos = lambda a, b: float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()))
    assert cos(gens['translate_x'].Ax(blob), -ddx) > 0.999
    assert cos(gens['translate_y'].Ax(blob), -ddy) > 0.999
    # rotation generator = d/dtheta of the warp; and A^T is the true transpose
    fd = (sym.warp(blob.reshape(1, 1, H, W), 'rotation', 1e-3) - sym.warp(blob.reshape(1, 1, H, W), 'rotation', -1e-3)) / 2e-3
    assert cos(rot.Ax(blob), fd) > 0.999
    u, v = torch.randn(2, H * W, generator=torch.Generator().manual_seed(13))
    assert abs(float(rot.Ax(u) @ v) - float(u @ rot.ATr(v))) < 1e-9
    # multi-channel fields act channel by channel
    g3 = sym.image_generators(3, 8, 8, dtype=torch.float64, which=['rotation'])[0]
    x3 = torch.randn(2, 3 * 64, generator=torch.Generator().manual_seed(14))
    assert torch.allclose(g3.Ax(x3).reshape(2, 3, 64)[:, 1], x3.reshape(2, 3, 64)[:, 1] @ g3.channel.T)


def test_killing_on_images_identity_metric_sees_rotation_as_isometry():
    """With g = I on 16x16 images, rotation and translations are (near-)isometries of pixel space; scale, shear
    and contrast are not.  Bilinear interpolation makes this approximate, hence the loose bound."""
    gens = sym.image_generators(1, 16, 16, dtype=torch.float64)
    ys, xs = _grid(16, 16)
    imgs = torch.stack([torch.exp(-((xs - a) ** 2 + (ys - b) ** 2) / (2 * 3.0 ** 2)) for a, b in [(0, 0), (2, -1), (-2, 2), (1, 3)]])
    out = sym.killing(lambda xb: torch.eye(256)[None].expand(len(xb), 256, 256), imgs.reshape(4, 1, 16, 16), gens)
    r = dict(zip(out['names'], out['residuals']))
    assert r['brightness'] < 1e-12
    assert r['rotation'] < 0.35 * r['scale'] and r['translate_x'] < 0.35 * r['scale'], r
