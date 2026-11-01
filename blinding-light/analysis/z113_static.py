"""P2, P4, P5 and their nulls/baselines on the final Z113 checkpoints."""
import sys, os, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.z113_common import *
from systems.z113 import P, NULL, EQ, build_data
from systems.z113_probe import psi_of
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.problems import p2_explain, p4_ignorance, p5_failure, p6_geodesic_generalization
from whiteboxtopo.problems.core import _softmax_conf
from whiteboxtopo.nulls import feature_shuffle, shuffle_predictor
from whiteboxtopo.extract import jacobians, logits_of
from whiteboxtopo.geometry import reach, ignorance
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr


def act_logit(m, toks, layer='resid_mid'):
    X = acts(m, toks, layer)
    psi = psi_of(m, layer)
    return X.astype(np.float32), logits_of(psi, X)


def failure_radius_activation(m, X, layer='resid_mid', lo=1e-3, hi=20.0, iters=18, seed=0):
    """Smallest perturbation that flips the prediction, found by bisection along the
    adversarial direction.

    Z113 inputs are discrete tokens, so a Gaussian input perturbation is not defined; we
    perturb the activation at the probe layer instead, which asks the same question on the
    manifold -- how far can this point move before the network changes its mind.

    The direction is the gradient of the top-versus-runner-up logit margin (a one-step PGD
    direction), and the radius is then bisected.  A grid of random directions, which is what
    this did first, returns the same coarse value for every point that flips at all and makes
    the rank correlation undefined.
    """
    psi = psi_of(m, layer)
    Xt = torch.as_tensor(X, device='cuda', dtype=torch.float32).requires_grad_(True)
    lg = psi(Xt)
    top2 = lg.topk(2, dim=1)
    margin = top2.values[:, 0] - top2.values[:, 1]
    gexp, = torch.autograd.grad(margin.sum(), Xt)
    d = -gexp / gexp.norm(dim=1, keepdim=True).clamp_min(1e-12)      # steepest margin descent
    base = lg.argmax(1).detach()
    d = d.detach()
    Xd = Xt.detach()
    scale = float(np.median(NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)[0][:, 1]))
    lo_t = torch.full((len(X),), lo, device='cuda')
    hi_t = torch.full((len(X),), hi, device='cuda')
    with torch.no_grad():
        flips_at_hi = psi(Xd + (hi_t[:, None] * scale) * d).argmax(1) != base
        for _ in range(iters):
            mid = 0.5 * (lo_t + hi_t)
            f = psi(Xd + (mid[:, None] * scale) * d).argmax(1) != base
            hi_t = torch.where(f, mid, hi_t)
            lo_t = torch.where(f, lo_t, mid)
    r = (0.5 * (lo_t + hi_t) * scale).cpu().numpy()
    r[~flips_at_hi.cpu().numpy()] = np.inf
    return r, scale


def ood_sets(d, te, rng):
    """Candidate out-of-distribution inputs, all built from tokens the model has seen in
    *some* position but never in this arrangement, plus uniform token noise."""
    a, b, n = d['a'], d['b'], len(te)
    full = lambda v: np.full(n, v)
    return {
        'random_tokens': np.stack([rng.integers(0, P + 2, n), rng.integers(0, P + 2, n),
                                   rng.integers(0, P + 2, n), rng.integers(0, P + 2, n)], 1),
        'no_equals': np.stack([a[te], b[te], full(NULL), a[te]], 1),
        'repeated_token': np.stack([a[te], a[te], a[te], a[te]], 1),
        'operand_is_EQ': np.stack([full(EQ), b[te], full(NULL), full(EQ)], 1),
    }


def constant_pattern_ig(psi_t, x0, x1, model, steps=32):
    """P2 anchor on a CONSTANT-PATTERN segment.

    The path product A(x) is the network's exact linear map only inside one activation cell.
    We walk from x0 towards x1, find the largest t at which the ReLU pattern still equals the
    pattern at x0, and compare A(x0) . delta against integrated gradients on [0, t].  Comparing
    across the whole edge instead mixes several cells, which is why the agreement was 0.89.
    """
    with torch.no_grad():
        pat0 = (torch.as_tensor(x0, device='cuda') @ model.W_in + model.b_in) > 0
        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            xm = torch.as_tensor(x0 + mid * (x1 - x0), device='cuda', dtype=torch.float32)
            same = bool((((xm @ model.W_in + model.b_in) > 0) == pat0).all())
            lo, hi = (mid, hi) if same else (lo, mid)
    t = lo
    if t <= 1e-6:
        return None
    delta = (x1 - x0) * t
    J0 = jacobians(psi_t, x0[None].astype(np.float32), device='cuda')[0]
    lin = J0 @ delta
    al = (np.arange(steps) + 0.5) / steps
    pts = (x0[None] + al[:, None] * delta[None]).astype(np.float32)
    Js = jacobians(psi_t, pts, device='cuda')
    ig = np.einsum('sod,d->o', Js, delta) / steps
    r = float(np.corrcoef(lin, ig)[0, 1]) if lin.std() > 0 and ig.std() > 0 else np.nan
    return dict(ig_r=r, t_constant=float(t))


def main(run, mode, seed, out=None, n_probe=1200):
    cks, steps = checkpoints(run)
    m = load(cks[-1], seed=seed)
    d = build_data(seed, mode, frac=0.5 if mode == 'shortcut' else 0.3)
    res = dict(run=run, mode=mode, seed=seed, step=int(steps[-1]))
    t0 = time.time()
    rng = np.random.default_rng(0)

    # ---- two probes, for two different jobs.
    # fiber probe: training points grouped by label -- P6 needs many points per class.
    # representative probe: uniform training points over ALL answers -- P4 needs a cloud that
    # actually samples the input distribution, or held-out points look "unknown" simply
    # because their answer is missing from the probe.
    fp = fiber_probe(mode, seed, n_classes=25, per_class=40)
    ldf, psif = layer_from(m, fp)
    fitf = fit_manifold(ldf.X, ldf.J, metric='g')

    tr = d['train_idx']
    sub = rng.choice(len(tr), size=min(n_probe, len(tr)), replace=False)
    rp_toks = d['train_toks'][sub]
    rp = dict(toks=rp_toks, a=rp_toks[:, 0], b=rp_toks[:, 1],
              c=(rp_toks[:, 0] + rp_toks[:, 1]) % P, labels=d['train_y'][sub])
    ld, psi = layer_from(m, rp)
    fit = fit_manifold(ld.X, ld.J, metric='g')
    res['m_fiber'] = fitf.m
    res['m_repr'] = fit.m
    res['n_classes_repr'] = int(len(np.unique(ld.labels)))

    # ---------------- P2: the geodesic to the boundary
    inputs = rng.choice(len(ld.X), size=60, replace=False)
    ex = p2_explain(ld, fit, inputs)
    marg = np.array([e['margin'] for e in ex if np.isfinite(e['margin'])])
    eucl = np.array([e['euclid_margin'] for e in ex if np.isfinite(e['margin'])])
    igs = []
    for e in ex[:25]:
        if e['path'] is None or len(e['path']) < 2:
            continue
        r = constant_pattern_ig(psi, ld.X[e['path'][0]], ld.X[e['path'][1]], m)
        if r:
            igs.append(r)
    igr = np.array([g['ig_r'] for g in igs if np.isfinite(g['ig_r'])])
    res['P2'] = dict(n=len(marg), margin_mean=float(marg.mean()),
                     margin_cv=float(marg.std() / marg.mean()),
                     euclid_margin_cv=float(eucl.std() / eucl.mean()),
                     ig_r_mean=float(igr.mean()) if len(igr) else None,
                     ig_r_min=float(igr.min()) if len(igr) else None,
                     ig_r_frac_above_095=float((igr > 0.95).mean()) if len(igr) else None,
                     t_constant_mean=float(np.mean([g['t_constant'] for g in igs])) if igs else None,
                     n_steps_mean=float(np.mean([e['n_steps'] for e in ex])))

    # ---------------- P4: what it does not know
    te = d['test_idx'][:1000]
    a, b = d['a'], d['b']
    hold_toks = np.stack([a[te], b[te], np.full(len(te), NULL), np.full(len(te), EQ)], 1)
    Xh, Lh = act_logit(m, hold_toks)
    nn_e = NearestNeighbors(n_neighbors=1).fit(fit.X)
    eps, _, nnb = reach(fit.X, fit.bases, fit.J, Xh, 99)
    ig_in, dg_in = ignorance(Xh, fit.X, fit.bases, fit.J, eps, nnb)
    e_in = nn_e.kneighbors(Xh)[0][:, 0]
    sm_in = _softmax_conf(Lh)
    p4 = dict(reach=float(eps), sets={})
    for nm, tk in ood_sets(d, te, rng).items():
        Xo, Lo = act_logit(m, tk)
        ig_o, dg_o = ignorance(Xo, fit.X, fit.bases, fit.J, eps, nnb)
        y = np.r_[np.zeros(len(dg_in)), np.ones(len(dg_o))]
        p4['sets'][nm] = dict(
            auc_ignorance=float(roc_auc_score(y, np.r_[dg_in, dg_o])),
            auc_softmax=float(roc_auc_score(y, np.r_[-sm_in, -_softmax_conf(Lo)])),
            auc_euclid_knn=float(roc_auc_score(y, np.r_[e_in, nn_e.kneighbors(Xo)[0][:, 0]])),
            beats_softmax=None, frac_outside_reach=float((ig_o > 1).mean()),
            frac_in_outside=float((ig_in > 1).mean()))
        sset = p4['sets'][nm]
        sset['beats_softmax'] = bool(sset['auc_ignorance'] > sset['auc_softmax'])
        sset['beats_euclid'] = bool(sset['auc_ignorance'] > sset['auc_euclid_knn'])
    res['P4'] = p4

    # ---------------- P5: where it will fail
    fr, scale = failure_radius_activation(m, ld.X)
    p5 = p5_failure(ld, fit, fr)
    ok = np.isfinite(fr)
    null_rho = spearmanr(shuffle_predictor(p5['rho'], 0)[ok], fr[ok])
    res['P5'] = dict(spearman_rho=p5['spearman_rho'], p_rho=p5['p_rho'],
                     spearman_softmax_margin=p5['spearman_softmax_margin'],
                     p_margin=p5['p_margin'], n_saddles=int(len(p5['saddles'])),
                     n_finite_radius=int(ok.sum()), n=int(len(fr)),
                     radius_distinct=int(len(np.unique(np.round(fr[ok], 6)))),
                     null_spearman_shuffled_rho=float(null_rho.statistic),
                     lip_cv=float(np.std(p5['lip']) / np.mean(p5['lip'])),
                     curv_cv=float(np.std(p5['curv']) / np.mean(p5['curv'])))

    # ---------------- P6 and its null
    g = p6_geodesic_generalization(ldf, fitf)
    from whiteboxtopo.problems import fiber_beta0
    b0 = fiber_beta0(ldf, fitf)
    res['P6'] = dict(G=g['G'], beta0_fiber_mean=float(np.mean(list(b0.values()))), m=fitf.m)
    Xs = feature_shuffle(ldf.X, seed=0)
    Js = jacobians(psif, Xs, device='cuda').astype(np.float32)
    lds = type(ldf)(name='null', X=Xs, J=Js, logits=logits_of(psif, Xs), labels=ldf.labels,
                    variables=ldf.variables, modular_period=P)
    fits = fit_manifold(Xs, Js, metric='g')
    res['null_feature_shuffle'] = dict(G=p6_geodesic_generalization(lds, fits)['G'], m=fits.m)

    res['wall_s'] = time.time() - t0
    out = out or f'{run}/static.json'
    json.dump(res, open(out, 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != 'run'}, indent=1, default=float), flush=True)
    print(f'-> {out} ({res["wall_s"]:.0f}s)', flush=True)


def _jac(psi, pts):
    return jacobians(psi, np.asarray(pts, dtype=np.float32), device='cuda')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--mode', required=True)
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args(); main(a.run, a.mode, a.seed)
