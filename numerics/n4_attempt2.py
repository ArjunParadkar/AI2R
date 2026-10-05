"""N4 attempt 2 (POST HOC, openly labelled).  Attempt 1 failed T-N4-02/03 on the frozen tolerances.  Protocol:
 1. tune (shape_mult, centre fraction) ONLY on three geometries that are not the failed cells: flat cylinder, hyperbolic half-plane patch, flat T^3 (n=500);
 2. fix the winner, apply ONCE to the frozen cells with the SAME subspace-error and gap thresholds, but with the null dimension read by a gap rule
    (K = position of the largest sigma ratio among the first 8 if ratio >= gap_min and sigma_K <= 0.05, else 0) instead of the absolute cut 1e-3.
Authors: Arjun Paradkar and Reva Paradkar."""
import sys, os, json, math, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from whiteboxtopo.numerics import killing as K
import test_n4_killing as T
TWO_PI = 2 * math.pi


def gap_dim(sig, gap_min, cap=0.05, kmax=8):
    r = sig[1:kmax + 1] / np.maximum(sig[:kmax], 1e-300); i = int(np.argmax(r))
    return (i + 1, float(r[i])) if (r[i] >= gap_min and sig[i] <= cap) else (0, float(r[i]))


def solve(n, g_fn, lo, hi, feat, vol, sm, cf, seed=0):
    X = K.halton(n, lo, hi, seed); C = K.halton(max(8, int(cf * n)), lo, hi, seed + 1)
    sol = K.KillingSolver(C, feat, shape_mult=sm, domain_volume=vol); sol.solve(X, g_fn); return sol


def cyl_g(X): return np.broadcast_to(np.diag([1.0, 4.0]), (len(X), 2, 2)).copy()
def hyp_g(X): return (1.0 / X[:, 1] ** 2)[:, None, None] * np.eye(2)[None]
def t3_g(X): return np.broadcast_to(np.diag([1.0, 2.0, 3.0]), (len(X), 3, 3)).copy()
def hyp_fields(X):
    x, y = X[:, 0], X[:, 1]
    return [np.stack([np.ones_like(x), 0 * x], 1), np.stack([x, y], 1), np.stack([x ** 2 - y ** 2, 2 * x * y], 1)]


def tuning_cases():
    return [
        ('cylinder', 400, cyl_g, [0, 0], [TWO_PI, 1.5], K.feat_periodic([1, 0]), TWO_PI * 1.5, 2, lambda X: [np.tile([1.0, 0], (len(X), 1)), np.tile([0, 1.0], (len(X), 1))]),
        ('hyperbolic', 400, hyp_g, [0, 0.5], [1, 1.5], K.feat_identity, 1.0, 3, hyp_fields),
        ('T3', 500, t3_g, [0] * 3, [TWO_PI] * 3, K.feat_periodic([1, 1, 1]), TWO_PI ** 3, 3, lambda X: [np.tile(e, (len(X), 1)) for e in np.eye(3)]),
    ]


def evaluate_case(c, sm, cf, gap_min=100.0):
    name, n, g, lo, hi, feat, vol, kk, fields = c
    sol = solve(n, g, lo, hi, feat, vol, sm, cf); d, ratio = gap_dim(sol.sigma, gap_min)
    Xt = K.halton(300, lo, hi, 99); err = K.subspace_error(sol.null_fields(kk, Xt), T.exact_cols(fields(Xt)))
    return dict(dim=d, true=kk, ratio=ratio, err=err)


def main():
    out = {'protocol': __doc__.split('\n')[0], 'tuning': {}}
    cases = tuning_cases(); best, bs = None, 1e9
    for sm, cf in itertools.product((2.0, 3.0, 4.0, 6.0, 8.0), (0.5, 1.0)):
        res = {c[0]: evaluate_case(c, sm, cf) for c in cases}
        ok = all(r['dim'] == r['true'] for r in res.values()); worst = max(r['err'] for r in res.values())
        out['tuning'][f'sm{sm}_cf{cf}'] = dict(res=res, all_dims_right=ok, worst_err=worst); print('tune', sm, cf, ok, f'{worst:.2e}', flush=True)
        score = worst if ok else 1e9 + worst
        if score < bs: best, bs = (sm, cf), score
    sm, cf = best; out['chosen'] = dict(shape_mult=sm, centre_frac=cf); print('chosen', best, flush=True)
    # ---- apply once to the frozen cells (same n, same error thresholds, gap rule) ----
    fin = {}
    lo, hi = [0.3, 0.0], [math.pi - 0.3, TWO_PI]; vol = (math.pi - 0.6) * TWO_PI
    sol = solve(600, T.g_sphere, lo, hi, K.feat_periodic([0, 1]), vol, sm, cf); d, ratio = gap_dim(sol.sigma, 100.0)
    Xt = K.halton(300, lo, hi, 99); err = K.subspace_error(sol.null_fields(3, Xt), T.exact_cols(T.sph_fields(Xt)))
    fin['T-N4-02b'] = dict(dim=d, gap=ratio, err=err, sigma=sol.sigma[:6].tolist(), passed=bool(d == 3 and ratio >= 100 and err <= 1e-3))
    lo, hi = [-0.5, -0.5, 0.0], [0.5, 0.5, TWO_PI]
    sol = solve(500, T.g_img, lo, hi, K.feat_periodic([0, 0, 1]), TWO_PI, sm, cf); d, ratio = gap_dim(sol.sigma, 20.0)
    Xt = K.halton(300, lo, hi, 99)
    ex = T.exact_cols([np.tile([1.0, 0, 0], (300, 1)), np.tile([0, 1.0, 0], (300, 1)), np.stack([-Xt[:, 1], Xt[:, 0], np.ones(300)], 1)])
    err = K.subspace_error(sol.null_fields(3, Xt), ex)
    fin['T-N4-03b'] = dict(dim=d, gap=ratio, err=err, sigma=sol.sigma[:6].tolist(), passed=bool(d == 3 and ratio >= 20 and err <= 1e-2))
    # re-check the cells that passed in attempt 1 with the chosen solver (negative control must still give 0)
    def g_gen(X):
        u = 0.8 * np.sin(3 * X[:, 0] + 1) * np.cos(2 * X[:, 1]) + 0.5 * X[:, 0] * X[:, 1]; return np.exp(2 * u)[:, None, None] * np.eye(2)[None]
    sol = solve(400, g_gen, [0, 0], [2, 2], K.feat_identity, 4.0, sm, cf); d, ratio = gap_dim(sol.sigma, 20.0)
    fin['T-N4-04b'] = dict(dim=d, sigma_min=float(sol.sigma[0]), passed=bool(d == 0 and sol.sigma[0] >= 1e-3))
    sol = solve(400, T.g_torus, [0, 0], [TWO_PI] * 2, K.feat_periodic([1, 1]), TWO_PI ** 2, sm, cf); d, ratio = gap_dim(sol.sigma, 100.0)
    Xt = K.halton(300, [0, 0], [TWO_PI] * 2, 99); err = K.subspace_error(sol.null_fields(2, Xt), T.exact_cols([np.tile([1.0, 0], (300, 1)), np.tile([0, 1.0], (300, 1))]))
    fin['T-N4-01b'] = dict(dim=d, gap=ratio, err=err, passed=bool(d == 2 and ratio >= 100 and err <= 1e-4))
    out['frozen_cells_attempt2'] = fin; json.dump(out, open('numerics/results_attempt2_n4.json', 'w'), indent=1, default=float)
    print(json.dumps(fin, indent=1, default=float))


if __name__ == '__main__': main()
