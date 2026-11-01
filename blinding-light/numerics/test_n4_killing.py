import math, numpy as np, torch
from torch.func import vmap, jacrev, jacfwd
from common import *
from whiteboxtopo.numerics import killing as K

TWO_PI = 2 * math.pi


def g_torus(X): return np.broadcast_to(np.eye(2), (len(X), 2, 2)).copy()
def g_sphere(X):
    G = np.zeros((len(X), 2, 2)); G[:, 0, 0] = 1; G[:, 1, 1] = np.sin(X[:, 0]) ** 2; return G
def sph_fields(X):
    th, ph = X[:, 0], X[:, 1]
    return [np.stack([-np.sin(ph), -np.cos(ph) / np.tan(th)], 1), np.stack([np.cos(ph), -np.sin(ph) / np.tan(th)], 1),
            np.stack([0 * th, 0 * th + 1], 1)]


def exact_cols(fields):          # list of (n,d) -> (n*d, K)
    return np.stack([f.ravel() for f in fields], 1)


def solve_case(n, g_fn, lo, hi, feat, vol, seed=0, shape_mult=2.0):
    X = K.halton(n, lo, hi, seed); C = K.halton(n // 2, lo, hi, seed + 1)
    sol = K.KillingSolver(C, feat, shape_mult=shape_mult, domain_volume=vol)
    sol.solve(X, g_fn)
    return sol, X


def test_n4_01_torus(cell):
    lo, hi = [0, 0], [TWO_PI] * 2
    sol, X = solve_case(400, g_torus, lo, hi, K.feat_periodic([1, 1]), TWO_PI ** 2)
    Kd, gap = sol.null_dim(1e-3)
    Xt = K.halton(300, lo, hi, 99)
    ex = exact_cols([np.tile([1.0, 0.0], (300, 1)), np.tile([0.0, 1.0], (300, 1))])
    err = K.subspace_error(sol.null_fields(2, Xt), ex)
    cell('T-N4-01', {'n': 400, 'shape_mult': 2.0}, 0, {'null_dim': Kd, 'gap': gap, 'sigma_first6': sol.sigma[:6], 'subspace_err': err},
         {'dim=2': Kd == 2, 'gap>=100': gap >= 100, 'err<=1e-4': err <= 1e-4})


def sphere_err(n, shape_mult=2.0):
    lo, hi = [0.3, 0.0], [math.pi - 0.3, TWO_PI]
    vol = (math.pi - 0.6) * TWO_PI
    sol, X = solve_case(n, g_sphere, lo, hi, K.feat_periodic([0, 1]), vol, shape_mult=shape_mult)
    Xt = K.halton(300, lo, hi, 99)
    err = K.subspace_error(sol.null_fields(3, Xt), exact_cols(sph_fields(Xt)))
    return sol, err, math.sqrt(vol / n)


def test_n4_02_05_sphere(cell):
    # reference fields really are Killing (checks the exact answer, not the solver)
    Xr = K.halton(200, [0.3, 0.0], [math.pi - 0.3, TWO_PI], 5)
    ref = [K.lie_derivative_residual(lambda Y, i=i: sph_fields(Y)[i], g_sphere, Xr) for i in range(3)]
    assert max(ref) < 1e-7, ref
    sol, err, h = sphere_err(600)
    Kd, gap = sol.null_dim(1e-3)
    sweep = {}
    for n in (100, 200, 400, 800):
        _, e, hh = sphere_err(n); sweep[n] = (hh, e)
    hs = np.log([v[0] for v in sweep.values()]); es = np.log([max(v[1], 1e-16) for v in sweep.values()])
    order = float(np.polyfit(hs, es, 1)[0])
    conv = sweep[800][1] <= 0.1 * sweep[100][1] or sweep[800][1] <= 1e-8
    cell('T-N4-02', {'n': 600, 'sweep': [100, 200, 400, 800]}, 0,
         {'null_dim': Kd, 'gap': gap, 'sigma_first6': sol.sigma[:6], 'subspace_err': err, 'sweep_h_err': {str(k): v for k, v in sweep.items()},
          'observed_order': order, 'ref_fields_residual': ref},
         {'dim=3': Kd == 3, 'gap>=100': gap >= 100, 'err<=1e-3': err <= 1e-3, 'convergence': conv})
    Xi = K.halton(300, [0.3, 0.0], [math.pi - 0.3, TWO_PI], 123)
    res = [sol.residual(sol.coefs[:, i], Xi, g_sphere) for i in range(3)]
    cell('T-N4-05', {'n_test': 300}, 0, {'residuals': res}, {'<=1e-3': max(res) <= 1e-3})


# ---------------------------------------------------------------- rotated / translated image manifold
GRID = torch.linspace(-4, 4, 53, dtype=T64); GX, GY = torch.meshgrid(GRID, GRID, indexing='ij')
MU = torch.tensor([[0.9, 0.2], [-0.7, 0.8], [-0.3, -0.9]], dtype=T64)
SG = torch.tensor([[0.5, 0.35], [0.4, 0.5], [0.3, 0.45]], dtype=T64)
AMP = torch.tensor([1.0, 0.7, 0.5], dtype=T64)
DA2 = float((GRID[1] - GRID[0]) ** 2)


def render(p):
    t1, t2, a = p[0], p[1], p[2]
    c, s = torch.cos(a), torch.sin(a)
    x, y = GX - t1, GY - t2
    u, v = c * x + s * y, -s * x + c * y            # R_{-a} (x - t)
    img = 0
    for b in range(3):
        img = img + AMP[b] * torch.exp(-0.5 * (((u - MU[b, 0]) / SG[b, 0]) ** 2 + ((v - MU[b, 1]) / SG[b, 1]) ** 2))
    return img.reshape(-1)


def g_img(X):
    Jm = vmap(jacfwd(render))(torch.as_tensor(X, dtype=T64))   # 3 inputs: forward mode (jacrev vmaps over 2809 outputs -> ~60 GB)
    return (torch.einsum('npi,npj->nij', Jm, Jm) * DA2).numpy()


def test_n4_03_images(cell):
    lo, hi = [-0.5, -0.5, 0.0], [0.5, 0.5, TWO_PI]
    sol, X = solve_case(500, g_img, lo, hi, K.feat_periodic([0, 0, 1]), 1.0 * TWO_PI)
    Kd, gap = sol.null_dim(1e-3)
    Xt = K.halton(300, lo, hi, 99)
    ex = exact_cols([np.tile([1.0, 0, 0], (300, 1)), np.tile([0, 1.0, 0], (300, 1)), np.stack([-Xt[:, 1], Xt[:, 0], np.ones(300)], 1)])
    err = K.subspace_error(sol.null_fields(3, Xt), ex)
    ref = [K.lie_derivative_residual(lambda Y, i=i: [np.tile([1.0, 0, 0], (len(Y), 1)), np.tile([0, 1.0, 0], (len(Y), 1)),
                                                      np.stack([-Y[:, 1], Y[:, 0], np.ones(len(Y))], 1)][i], g_img, Xt[:60]) for i in range(3)]
    cell('T-N4-03', {'n': 500, 'grid': 53}, 0, {'null_dim': Kd, 'gap': gap, 'sigma_first6': sol.sigma[:6], 'subspace_err': err, 'ref_fields_residual': ref},
         {'dim=3': Kd == 3, 'gap>=20': gap >= 20, 'err<=1e-2': err <= 1e-2})


def test_n4_04_negative_control(cell):
    def g_gen(X):
        u = 0.8 * np.sin(3 * X[:, 0] + 1) * np.cos(2 * X[:, 1]) + 0.5 * X[:, 0] * X[:, 1]
        return np.exp(2 * u)[:, None, None] * np.eye(2)[None]
    sol, X = solve_case(400, g_gen, [0, 0], [2, 2], K.feat_identity, 4.0)
    Kd, gap = sol.null_dim(1e-3)
    cell('T-N4-04', {'n': 400}, 0, {'null_dim': Kd, 'sigma_min': sol.sigma[0], 'sigma_first6': sol.sigma[:6]},
         {'dim=0': Kd == 0, 'sigma_min>=1e-3': sol.sigma[0] >= 1e-3})
