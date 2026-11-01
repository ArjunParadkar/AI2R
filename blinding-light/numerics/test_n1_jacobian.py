import math, numpy as np, torch
from torch.func import vmap, jacrev
from common import *
from whiteboxtopo.numerics import jacobian as J


def mx(a): return float(a.abs().max())


def test_n1_01_02_exact_vs_analytic(cell):
    Ws = mlp_weights([6, 20, 20, 5], 1)
    g = torch.Generator().manual_seed(0)
    errs = {}; rel2 = {}
    A = torch.randn(4, 5, generator=g, dtype=T64)
    cases = {'affine': (lambda x: A @ x + 1.0, lambda x: A, 5), 'poly': (poly_map, poly_map_J, 5),
             'tanh_mlp': (tanh_mlp(Ws), lambda x: tanh_mlp_J(Ws, x), 6)}
    for name, (f, Jf, n) in cases.items():
        e = 0; r = 0
        for _ in range(10):
            x = torch.randn(n, generator=g, dtype=T64); Ja = Jf(x)
            e = max(e, mx(J.jacobian_exact(f, x) - Ja) / (1 + mx(Ja)))
            v = torch.randn(n, generator=g, dtype=T64); u = torch.randn(Ja.shape[0], generator=g, dtype=T64)
            r = max(r, float(torch.linalg.norm(J.jvp(f, x, v) - Ja @ v) / torch.linalg.norm(Ja @ v)),
                    float(torch.linalg.norm(J.vjp(f, x, u) - Ja.T @ u) / torch.linalg.norm(Ja.T @ u)))
        errs[name] = e; rel2[name] = r
    cell('T-N1-01', {'cases': list(cases)}, 0, errs, {'abs<=1e-12(1+|J|)': all(v <= 1e-12 for v in errs.values())})
    cell('T-N1-02', {'cases': list(cases)}, 0, rel2, {'rel<=1e-12': all(v <= 1e-12 for v in rel2.values())})


def _mlp_g(n=12, m=6, seed=3):
    Ws = mlp_weights([n, 24, m], seed)
    f = tanh_mlp(Ws); x = torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=T64)
    Jx = J.jacobian_exact(f, x); return f, x, Jx.T @ Jx


def test_n1_03_vjp_sketch(cell):
    f, x, g = _mlp_g(6, 10)
    gen = torch.Generator().manual_seed(11)
    mean = sum(J.vjp_sketch_gram(f, x, 8, gen) for _ in range(400)) / 400
    rel = float(torch.linalg.norm(mean - g) / torch.linalg.norm(g))
    med = {}
    for k in (4, 64):
        med[k] = float(np.median([float(torch.linalg.norm(J.vjp_sketch_gram(f, x, k, gen) - g) / torch.linalg.norm(g)) for _ in range(20)]))
    cell('T-N1-03', {'k': 8, 'R': 400}, 11, {'rel_mean_err': rel, 'median_err_k4': med[4], 'median_err_k64': med[64]},
         {'rel<=0.15': rel <= 0.15, 'k64<k4': med[64] < med[4]})


def test_n1_04_hutchinson_probe_count(cell):
    f, x, g = _mlp_g(12, 6)
    tr = float(torch.trace(g)); a = float(torch.linalg.norm(g) / tr); b = float(torch.linalg.eigvalsh(g).max() / tr)
    eps, delta = 0.1, 0.05
    N = J.probe_count(eps, delta, a, b)
    gen = torch.Generator().manual_seed(5)
    fails = 0; R = 400
    for _ in range(R):
        est, _ = J.hutchinson_trace_g(f, x, N, gen)
        fails += abs(float(est) - tr) > eps * tr
    rate = fails / R
    cell('T-N1-04', {'eps': eps, 'delta': delta, 'N': N, 'a': a, 'b': b}, 5,
         {'N': N, 'a': a, 'b': b, 'failure_rate': rate, 'worst_case_N': J.probe_count(eps, delta)}, {'rate<=delta': rate <= delta})


def test_n1_05_06_slots_and_subspace(cell):
    f, x, g = _mlp_g(12, 6)
    slots = [range(0, 3), range(3, 6), range(6, 9), range(9, 12)]
    gen = torch.Generator().manual_seed(7)
    sh, _ = J.slot_shares(f, x, slots, 4000, gen)
    ex = torch.stack([sum(g[i, i] for i in s) for s in slots]); ex = ex / ex.sum()
    err = float((sh - ex).abs().max())
    cell('T-N1-05', {'N': 4000}, 7, {'max_abs_err': err, 'sum_minus_1': float(sh.sum() - 1)},
         {'err<=0.03': err <= 0.03, 'sum=1': abs(float(sh.sum()) - 1) <= 1e-12})
    U, _ = torch.linalg.qr(torch.randn(12, 3, generator=gen, dtype=T64))
    est, _ = J.hutchinson_trace_g(f, x, 4000, gen, basis=U)
    exact = float(torch.trace(U.T @ g @ U)); rel = abs(float(est) - exact) / exact
    cell('T-N1-06', {'N': 4000, 'r': 3}, 7, {'rel_err': rel}, {'rel<=0.05': rel <= 0.05})


def test_n1_07_rsvd(cell):
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.standard_normal((60, 60))); V, _ = np.linalg.qr(rng.standard_normal((80, 60)))
    s = 2.0 ** -np.arange(60); A = (U * s) @ V.T
    Uh, sh, Vh = J.rsvd(lambda M: A @ M, lambda M: A.T @ M, 60, 80, 10, 10, 2, seed=1)
    rel = np.abs(sh[:5] - s[:5]) / s[:5]
    spec = np.linalg.norm(A - (Uh * sh) @ Vh, 2)
    cell('T-N1-07', {'rank': 10, 'p': 10, 'q': 2}, 1, {'rel_top5': rel.max(), 'spec_err': spec, 'sigma_11': s[10]},
         {'rel<=1e-6': rel.max() <= 1e-6, 'spec<=2sigma11': spec <= 2 * s[10]})


def _rel(Jh, Jr): return float((Jh - Jr).abs().max() / Jr.abs().max())


def test_n1_08_09_10_fd(cell):
    g = torch.Generator().manual_seed(2)
    x = torch.randn(5, generator=g, dtype=T64) * 1.5
    Jr = poly_map_J(x)
    h = J.fd_step(x)
    e1, e2 = _rel(J.fd_jacobian(poly_map, x, h, 1), Jr), _rel(J.fd_jacobian(poly_map, x, h, 2), Jr)
    cell('T-N1-08', {'h': h}, 2, {'plain': e1, 'richardson': e2}, {'plain<=1e-8': e1 <= 1e-8, 'rich<=1e-8': e2 <= 1e-8})
    hb = 1e-2 * float(torch.linalg.norm(x))
    p, r = _rel(J.fd_jacobian(poly_map, x, hb, 1), Jr), _rel(J.fd_jacobian(poly_map, x, hb, 2), Jr)
    cell('T-N1-09', {'h': hb}, 2, {'plain': p, 'richardson': r, 'ratio': p / r}, {'ratio>=100': p / r >= 100})
    h3 = 1e-1 * float(torch.linalg.norm(x))
    curve = J.query_cost_curve(poly_map, x, Jr, (1, 2, 3), h=h3)
    exact_counts = all(q == 2 * 5 * L for (q, _), L in zip(curve, (1, 2, 3)))
    errs = [e for _, e in curve]
    cell('T-N1-10', {'h': h3}, 2, {'curve_queries_err': curve}, {'counts_exact': exact_counts, 'monotone': errs[1] <= errs[0] and errs[2] <= errs[1]})


def test_n1_11_float_drift(cell):
    Ws = mlp_weights([6, 32, 32, 5], 4)
    x = torch.randn(6, generator=torch.Generator().manual_seed(4), dtype=T64)
    J64 = J.jacobian_exact(tanh_mlp(Ws), x)
    Ws32 = [(W.float(), b.float()) for W, b in Ws]; x32 = x.float(); f32 = tanh_mlp(Ws32)
    J32 = J.jacobian_exact(f32, x32)
    drift = float((J32.double() - J64).abs().max() / J64.abs().max())
    Jfd = J.fd_jacobian(f32, x32)                       # h = eps32^(1/3)||x||
    fd = float((Jfd.double() - J64).abs().max() / J64.abs().max())
    cell('T-N1-11', {'dtype': 'float32 vs float64'}, 4, {'autograd_drift': drift, 'fd32_err': fd, 'fd_over_autograd': fd / max(drift, 1e-300)},
         {'drift<=1e-5': drift <= 1e-5, 'fd32<=5e-2': fd <= 5e-2})


def test_n1_12_13_fisher_rao(cell):
    g = torch.Generator().manual_seed(6)
    W = torch.randn(5, 7, generator=g, dtype=T64); b = torch.randn(5, generator=g, dtype=T64)
    f = lambda x: W @ x + b
    errs = []; sq = []; chg = []; ech = []
    for _ in range(10):
        x = torch.randn(7, generator=g, dtype=T64)
        p = torch.softmax(W @ x + b, 0); ga = W.T @ (torch.diag(p) - torch.outer(p, p)) @ W
        gf = J.fisher_rao_pullback(f, x); gs = J.fisher_rao_pullback(f, x, 'sqrt')
        errs.append(float((gf - ga).abs().max() / ga.abs().max())); sq.append(float((gs - gf).abs().max() / gf.abs().max()))
        f2 = lambda z: f(z) + (torch.sin(z.sum()) * 3 + z[0] ** 2) * torch.ones(5, dtype=T64)
        chg.append(float((J.fisher_rao_pullback(f2, x) - gf).abs().max() / gf.abs().max()))
        e1, e2 = J.euclid_pullback(f, x), J.euclid_pullback(f2, x)
        ech.append(float((e2 - e1).abs().max() / e1.abs().max()))
    cell('T-N1-12', {}, 6, {'analytic_rel': max(errs), 'sqrt_form': max(sq)}, {'<=1e-12': max(errs) <= 1e-12 and max(sq) <= 1e-12})
    cell('T-N1-13', {}, 6, {'fr_change': max(chg), 'euclid_change_min': min(ech)}, {'fr<=1e-12': max(chg) <= 1e-12, 'euclid>=1e-2': min(ech) >= 1e-2})


def test_n1_14_rewriting_group(cell):
    base = mlp_weights([6, 16, 16, 16, 5], 8)
    f0 = relu_mlp(base)
    g = torch.Generator().manual_seed(9)
    worst_in = 0.0; worst_h = 0.0; worst_fr = 0.0; fdiff = 0.0
    def readout(Ws, l):                       # hidden layer l (post-ReLU) -> logits
        def r(h):
            a = h
            for W, b in Ws[l + 1:-1]: a = torch.relu(W @ a + b)
            W, b = Ws[-1]; return W @ a + b
        return r
    def hidden(Ws, l, x):
        a = x
        for W, b in Ws[:l + 1]: a = torch.relu(W @ a + b)
        return a
    for r in range(20):
        Wn, Ms = rewrite(base, 100 + r); fn = relu_mlp(Wn)
        for _ in range(50 if r < 4 else 5):
            x = torch.randn(6, generator=g, dtype=T64)
            fdiff = max(fdiff, float((f0(x) - fn(x)).abs().max()))
            g0, g1 = J.euclid_pullback(f0, x), J.euclid_pullback(fn, x)
            worst_in = max(worst_in, float((g0 - g1).abs().max() / g0.abs().max()))
            q0, q1 = J.fisher_rao_pullback(f0, x), J.fisher_rao_pullback(fn, x)
            worst_fr = max(worst_fr, float((q0 - q1).abs().max() / q0.abs().max()))
            for l in (0, 1):
                M = Ms[l]
                # hidden metric of the layer-l activations, covariance under h' = M h (only this layer rewritten in the chart)
                h0 = hidden(base, l, x); h1 = hidden(Wn, l, x)
                if l == 0:
                    # layers > l are rewritten too, so compare via the layer-l readout including downstream rewrites:
                    pass
                G0 = J.euclid_pullback(readout(base, l), h0); G1 = J.euclid_pullback(readout(Wn, l), h1)
                Mi = torch.linalg.inv(M)
                if l == len(Ms) - 1:
                    cov = Mi.T @ G0 @ Mi
                    worst_h = max(worst_h, float((G1 - cov).abs().max() / G0.abs().max()))
    cell('T-N1-14', {'layers': [6, 16, 16, 16, 5], 'rewritings': 20}, 8,
         {'f_diff': fdiff, 'euclid_in': worst_in, 'fr_in': worst_fr, 'hidden_cov': worst_h},
         {'in<=1e-10': worst_in <= 1e-10 and worst_fr <= 1e-10, 'hidden<=1e-10': worst_h <= 1e-10})
