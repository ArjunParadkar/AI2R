"""The seven definitions, each computed from one manifold fit.

Every routine takes a LayerData (activations, Jacobians, cached logits, labels, the variables
we are allowed to name coordinates with) plus the ManifoldFit, and returns a typed dict that
carries the manifold score with it.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.stats import spearmanr

from ..manifold import fit_manifold, geodesic
from ..topology import (persistence, significant_classes, normalized_persistence,
                        circular_coordinates, circular_correlation, name_coordinate,
                        toroidal_decorrelate, fiber_topology, match_classes)
from ..geometry import (tubular_distance, second_fundamental_form, lipschitz_field, reach, ignorance,
                        failure_risk, confidence_saddles)


@dataclass
class LayerData:
    name: str
    X: np.ndarray                      # (n, d_l) activations on the probe
    J: np.ndarray                      # (n, d_out, d_l) Jacobians of the remaining network
    logits: np.ndarray                 # (n, d_out) cached
    labels: np.ndarray                 # (n,)
    variables: dict = field(default_factory=dict)      # for naming coordinates
    modular_period: int = None

    @property
    def pred(self):
        return self.logits.argmax(1)


# ---------------------------------------------------------------- P1
def geodesic_distance_matrix(fit):
    """All-pairs geodesic distance on the pullback-weighted graph, as a finite symmetric matrix
    (points in different components are put at twice the largest finite distance)."""
    D = dijkstra(fit.W, directed=False)
    fin = np.isfinite(D)
    if not fin.all():
        D[~fin] = 2.0 * D[fin].max()
    D = np.maximum(D, D.T)
    np.fill_diagonal(D, 0.0)
    return D


def p1_inventory(ld, fit, noise_band=None, maxdim=1, do_fibers=True, max_classes=6,
                 name_threshold=0.8, fiber_min=40, metric='euclidean'):
    """What it learned: Betti numbers, the coordinate of each persistent class, the fiber
    topology and coordinates per concept, and the manifold score.

    metric='g' computes the cohomology of the cloud as a metric space under the pullback
    geodesic distance (sec 2.2-2.4 taken together) instead of the ambient Euclidean one.  The
    two differ whenever the layer carries variables the output does not read: a ring RNN's
    hidden state is only ~50% heading (the rest is velocity history), Euclidean Rips sees a
    thick blob (coordinate r = 0.80 with heading), and the g-geodesic cloud is one clean circle
    (lifetime ratio 20, r = 0.98 = the network's own output accuracy).  `noise_band` is in the
    units of the chosen metric; a Euclidean null band must not be passed with metric='g'.
    """
    if metric == 'g':
        res = persistence(geodesic_distance_matrix(fit), maxdim=maxdim, distance_matrix=True)
    else:
        res = persistence(ld.X, maxdim=maxdim)
    h1 = res['dgms'][1]
    sig, band = significant_classes(h1, noise_band)
    order = sig[np.argsort(-(h1[sig, 1] - h1[sig, 0]))][:max_classes] if len(sig) else sig
    coords = []
    for ci in order:
        cc = circular_coordinates(ld.X, h1, res['cocycles'][1], D=res['D'], class_idx=int(ci))
        if cc is None:
            continue
        nm = name_coordinate(cc['theta'], ld.variables, modular_period=ld.modular_period,
                             threshold=name_threshold)
        cc.update(naming=nm, class_idx=int(ci))
        coords.append(cc)
    kept, dropped = toroidal_decorrelate([c['theta'] for c in coords]) if coords else ([], [])

    beta0 = connected_components(fit.A, directed=False)[0]
    out = dict(layer=ld.name, m=fit.m, beta0=int(beta0), beta1=int(len(sig)),
               noise_band=band, tildeC=normalized_persistence(h1, res['median_dist']),
               coords=coords, independent_coords=kept, dependent_coords=dropped,
               d_intrinsic=float(np.nanmedian(fit.d_local)), dgm1=h1)
    if do_fibers:
        fib = fiber_topology(ld.X, ld.labels, maxdim=1, noise_band=noise_band, min_points=fiber_min)
        fsum = {}
        for y, f in fib.items():
            if f.get('skipped'):
                fsum[y] = f
                continue
            entry = dict(n=f['n'], beta0=f['beta0'], beta1=f['beta1'], tildeC=f['tildeC'])
            if f['beta1'] >= 1:
                h1f = f['res']['dgms'][1]
                s, _ = significant_classes(h1f, noise_band)
                best = s[np.argmax(h1f[s, 1] - h1f[s, 0])]
                cc = circular_coordinates(ld.X[f['idx']], h1f, f['res']['cocycles'][1],
                                          D=f['res']['D'], class_idx=int(best))
                if cc is not None:
                    fvars = {k: np.asarray(v)[f['idx']] for k, v in ld.variables.items()}
                    cc['naming'] = name_coordinate(cc['theta'], fvars,
                                                   modular_period=ld.modular_period,
                                                   threshold=name_threshold)
                    entry['coord'] = cc
            fsum[y] = entry
        out['fibers'] = fsum
    return out


def p1_symmetry_inventory(X, param, period, n_freq=6, r_pass=0.9, life_ratio=3.0, var_floor=0.02, X_eval=None, param_eval=None):
    """P1 when the task's symmetry is known: which characters of a cyclic group does the layer use,
    and does each one carry a certified circle?

    `X` are activations on an orbit probe (inputs that sweep the group parameter `param` in
    Z_period).  This is SYMMETRY-INFORMED, not unsupervised: the group parameter is supplied; what
    is discovered is the set of frequencies, and each is certified independently by persistent
    H^1 (one dominant class, lifetime ratio > `life_ratio`) whose DSMVJ coordinate must track
    k*param at circular r > `r_pass`.  Pass X_eval/param_eval (held-out orbit) when D >> n.
    """
    from ..topology import isotypic_decomposition
    dec, _ = isotypic_decomposition(X, param, period, n_freq=n_freq, detect='dft', var_floor=0.0, X_eval=X_eval)
    pe = np.asarray(param if X_eval is None else param_eval)
    out = []
    for d in dec:
        Y = np.ascontiguousarray(d['proj']).astype(np.float32)
        res = persistence(Y); h1 = res['dgms'][1]
        rec = dict(freq=int(d['freq']), var=float(d['var']), r=0.0, life_ratio=0.0, present=False)
        if len(h1):
            life = h1[:, 1] - h1[:, 0]; srt = np.sort(life)
            rec['life_ratio'] = float(srt[-1] / max(srt[-2], 1e-9)) if len(srt) > 1 else float('inf')
            cc = circular_coordinates(Y, h1, res['cocycles'][1], D=res['D'], class_idx=int(np.argmax(life)))
            if cc is not None:
                rec['r'] = float(circular_correlation(cc['theta'], 2 * np.pi * d['freq'] * pe / period))
            rec['present'] = bool(rec['r'] > r_pass and rec['life_ratio'] > life_ratio and rec['var'] > var_floor)
        out.append(rec)
    return out


# ---------------------------------------------------------------- P6
def p6_geodesic_generalization(ld, fit, pairs_per_class=200, seed=0, eps=0.1):
    """Understanding vs memorization: the fraction of the geodesic between two same-class
    points on which the network still answers that class."""
    rng = np.random.default_rng(seed)
    pred = ld.pred
    per_class, all_frac = {}, []
    interior, n_direct, n_pairs = [], 0, 0
    for y in np.unique(ld.labels):
        if y < 0:                      # unlabeled points: path nodes only, never endpoints
            continue
        members = np.where(ld.labels == y)[0]
        if len(members) < 3:
            continue
        a = rng.choice(members, size=min(pairs_per_class, len(members) * 4))
        b = rng.choice(members, size=len(a))
        keep = a != b
        a, b = a[keep], b[keep]
        srcs = np.unique(a)
        dist, pred_tree = dijkstra(fit.W, directed=False, indices=srcs, return_predecessors=True)
        row = {s: i for i, s in enumerate(srcs)}
        fr = []
        for ai, bi in zip(a, b):
            r = row[ai]
            if not np.isfinite(dist[r, bi]):
                fr.append(0.0)
                continue
            path, cur = [bi], bi
            while cur != ai:
                cur = pred_tree[r, cur]
                if cur < 0:
                    break
                path.append(cur)
            path = np.array(path)
            fr.append(float((pred[path] == y).mean()))
            n_pairs += 1
            if len(path) > 2:
                interior.append(float((pred[path[1:-1]] == y).mean()))
            else:
                n_direct += 1
        if fr:
            per_class[int(y)] = float(np.mean(fr))
            all_frac.extend(fr)
    G = float(np.mean(all_frac)) if all_frac else float('nan')
    # G_interior: the same expectation over INTERIOR path nodes only.  The two endpoints are
    # training points of class y, which a memorising network labels correctly by construction,
    # so they put a floor of 2/|gamma| under G that has nothing to do with understanding.
    G_int = float(np.mean(interior)) if interior else float('nan')
    return dict(G=G, G_interior=G_int, frac_direct=n_direct / max(n_pairs, 1),
                per_class=per_class, m=fit.m, n_pairs=len(all_frac), eps=eps,
                understood={y: v > 1 - eps for y, v in per_class.items()})


def fiber_beta0(ld, fit):
    """beta_0 of each fiber in the kNN graph: is the class one geodesically connected piece?"""
    out = {}
    for y in np.unique(ld.labels):
        sel = np.where(ld.labels == y)[0]
        if len(sel) < 3 or y < 0:
            continue
        out[int(y)] = int(connected_components(fit.A[sel][:, sel], directed=False)[0])
    return out


# ---------------------------------------------------------------- P2
def p2_explain(ld, fit, inputs, coords=None, ig_steps=32, psi=None):
    """Why this decision: the geodesic from the input to the nearest point the network calls
    something else.  Its g-length is the manifold margin; its first edge is what would have
    to change."""
    pred = ld.pred
    comp_n, comp_lab = connected_components(fit.A, directed=False)
    out = []
    for i in inputs:
        yhat = pred[i]
        dist, pt = dijkstra(fit.W, directed=False, indices=i, return_predecessors=True)
        other = np.where((pred != yhat) & np.isfinite(dist))[0]
        if len(other) == 0:
            out.append(dict(i=int(i), pred=int(yhat), margin=np.inf, path=None))
            continue
        j = other[np.argmin(dist[other])]
        path, cur = [j], j
        while cur != i:
            cur = pt[cur]
            if cur < 0:
                break
            path.append(cur)
        path = np.array(path[::-1])
        tangent = ld.X[path[1]] - ld.X[path[0]] if len(path) > 1 else np.zeros(ld.X.shape[1])
        rec = dict(i=int(i), pred=int(yhat), flips_to=int(pred[j]), margin=float(dist[j]),
                   path=path, n_steps=len(path) - 1, component=int(comp_lab[i]),
                   tangent=tangent, euclid_margin=float(np.linalg.norm(ld.X[j] - ld.X[i])))
        if coords:
            rec['coords'] = {f"c{c['class_idx']}": float(c['theta'][i]) for c in coords}
        if psi is not None and len(path) > 1:
            rec.update(path_product_check(ld, psi, path, ig_steps))
        out.append(rec)
    return out


def path_product_check(ld, psi, path, ig_steps=32):
    """The P2 anchor: on the first segment of the geodesic the network is a single linear map
    A(x) (the path product, exactly gauge-invariant).  Its attribution must agree with
    integrated gradients."""
    x0, x1 = ld.X[path[0]], ld.X[path[1]]
    delta = x1 - x0
    A = ld.J[path[0]]                       # the path product at x0 equals the Jacobian there
    lin = A @ delta                         # (d_out,) predicted logit change
    alphas = (np.arange(ig_steps) + 0.5) / ig_steps
    pts = x0[None] + alphas[:, None] * delta[None]
    Js = psi(pts)                           # (steps, d_out, d_l)
    ig = np.einsum('sod,d->o', Js, delta) / ig_steps
    r = float(np.corrcoef(lin, ig)[0, 1]) if lin.std() > 0 and ig.std() > 0 else np.nan
    return dict(ig_r=r, lin_attr=lin, ig_attr=ig)


# ---------------------------------------------------------------- P4
def p4_ignorance(ld, fit, holdout_acts, ood_sets, percentile=99):
    """What it does not know: distance from the tubular neighbourhood, against the softmax
    and raw-Euclidean baselines."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import roc_auc_score
    eps, dg_hold, nn = reach(fit.X, fit.bases, fit.J, holdout_acts, percentile)
    nn_e = NearestNeighbors(n_neighbors=1).fit(fit.X)
    res = dict(reach=eps, m=fit.m, holdout_dg=dg_hold, sets={})
    for nm, (acts, logits) in ood_sets.items():
        ig, dg = ignorance(acts, fit.X, fit.bases, fit.J, eps, nn)
        ig_in, dg_in = ignorance(holdout_acts, fit.X, fit.bases, fit.J, eps, nn)
        y = np.r_[np.zeros(len(dg_in)), np.ones(len(dg))]
        auc_ours = roc_auc_score(y, np.r_[dg_in, dg])
        e_in = nn_e.kneighbors(holdout_acts)[0][:, 0]
        e_out = nn_e.kneighbors(acts)[0][:, 0]
        auc_euclid = roc_auc_score(y, np.r_[e_in, e_out])
        # POST-HOC companion to the spec's d_g: the same normal displacement measured in the
        # activation space itself, not through J (see geometry.tubular_distance for why).
        t_in = tubular_distance(holdout_acts, fit.X, fit.bases)
        t_out = tubular_distance(acts, fit.X, fit.bases)
        res['sets'][nm] = dict(auc_ignorance=float(auc_ours), auc_euclid_knn=float(auc_euclid),
                               auc_tubular=float(roc_auc_score(y, np.r_[t_in, t_out])),
                               ignorance=ig, frac_outside=float((ig > 1).mean()))
        if logits is not None:
            sm_in = _softmax_conf(ld.logits_holdout) if hasattr(ld, 'logits_holdout') else None
            sm_out = _softmax_conf(logits)
            if sm_in is not None:
                res['sets'][nm]['auc_softmax'] = float(roc_auc_score(y, np.r_[-sm_in, -sm_out]))
    return res


def _softmax_conf(logits):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p.max(1)


# ---------------------------------------------------------------- P5
def p5_failure(ld, fit, failure_radius, curv_k=30, dim_cap=None):
    """Where it will fail: rho = ||II|| * Lip, against the softmax-margin baseline.

    The tangent dimension of the curvature fit follows the manifold's own estimated dimension.
    It used to be fixed at 4: on a manifold of lower dimension the tangent fit then swallows the
    normal direction, the curvature is round-off, and the sign of the correlation is arbitrary
    (caught by tests/test_problems.py on two circles of different radius)."""
    if dim_cap is None:
        dim_cap = int(np.clip(np.round(fit.d_global), 1, 8))
    curv = second_fundamental_form(fit.X, k=max(curv_k, dim_cap * (dim_cap + 1) // 2 + 8), dim_cap=dim_cap)
    lip = lipschitz_field(fit.J)
    rho = failure_risk(curv, lip)
    maxlogit = ld.logits.max(1)
    saddles, ncomp = confidence_saddles(fit.W, maxlogit)
    ok = np.isfinite(failure_radius)
    s_rho = spearmanr(rho[ok], failure_radius[ok])
    z = np.sort(ld.logits, 1)
    margin = z[:, -1] - z[:, -2]
    s_margin = spearmanr(margin[ok], failure_radius[ok])
    return dict(rho=rho, curv=curv, lip=lip, saddles=saddles, m=fit.m, curv_dim=int(dim_cap),
                spearman_rho=float(s_rho.statistic), p_rho=float(s_rho.pvalue),
                spearman_softmax_margin=float(s_margin.statistic), p_margin=float(s_margin.pvalue),
                n=int(ok.sum()))


# ---------------------------------------------------------------- P3
def p3_spurious(ld, fit, nuisance, coords=None, tau=None, corr_thresh=0.5, noise_band=None):
    """When bias entered: a persistent class whose coordinate tracks a nuisance variable,
    that the output actually depends on."""
    if coords is None:
        inv = p1_inventory(ld, fit, noise_band=noise_band, do_fibers=False)
        coords = inv['coords']
    found = []
    for c in coords:
        th = c['theta']
        rz = {}
        for nm, z in nuisance.items():
            z = np.asarray(z)
            if ld.modular_period and np.issubdtype(z.dtype, np.integer):
                best = max(abs(circular_correlation(th, 2 * np.pi * k * z / ld.modular_period))
                           for k in range(1, ld.modular_period // 2 + 1))
            else:
                from ..topology import circular_linear_correlation
                best = circular_linear_correlation(th, z.astype(float))
            rz[nm] = float(best)
        best_nm = max(rz, key=rz.get)
        # does the output depend on movement along this coordinate?
        v = _coordinate_tangent(fit, th)
        sens = float(np.mean(np.linalg.norm(np.einsum('nod,nd->no', fit.J, v), axis=1)))
        found.append(dict(class_idx=c['class_idx'], lifetime=c['lifetime'], corr=rz,
                          best_nuisance=best_nm, best_corr=rz[best_nm], sensitivity=sens,
                          spurious=bool(rz[best_nm] > corr_thresh and (tau is None or sens > tau))))
    return dict(classes=found, m=fit.m,
                n_spurious=int(sum(f['spurious'] for f in found)), tau=tau)


def _coordinate_tangent(fit, theta):
    """Unit tangent along the level structure of a circular coordinate: the graph gradient of
    theta, which is the direction the harmonic representative points."""
    W = fit.A.tocoo()
    n, D = fit.X.shape
    v = np.zeros((n, D))
    wsum = np.zeros(n)
    d = np.mod(theta[W.col] - theta[W.row] + np.pi, 2 * np.pi) - np.pi
    delta = fit.X[W.col] - fit.X[W.row]
    np.add.at(v, W.row, d[:, None] * delta)
    np.add.at(wsum, W.row, np.abs(d))
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(nrm, 1e-12)


# ---------------------------------------------------------------- P7
def p7_formation(inventories, steps, coord_match_r=0.8):
    """How it evolves: track each class across checkpoints and record when it was born.

    Classes are matched between consecutive checkpoints by Wasserstein matching of the
    diagrams with coordinate correlation as the tie-break (the spec's fallback for zigzag).
    A class's birth time is the first checkpoint at which it exists and from which it
    persists to the end.
    """
    tracks = []      # each: dict(birth_idx, last_idx, members=[(ckpt, class_idx)], names=[])
    active = {}
    for t, inv in enumerate(inventories):
        cur = inv['coords']
        if t == 0 or not active:
            for c in cur:
                tracks.append(dict(birth=t, last=t, name=_cname(c), lifetimes=[c['lifetime']]))
                active[len(tracks) - 1] = c
            continue
        prev_ids = list(active.keys())
        prev = [active[i] for i in prev_ids]
        pairs, un_a, un_b = match_classes(
            np.array([[p['birth'], p['death']] for p in prev]),
            np.array([[c['birth'], c['death']] for c in cur]),
            [p['theta'] for p in prev], [c['theta'] for c in cur])
        new_active = {}
        for ia, ib, cost in pairs:
            tid = prev_ids[ia]
            tracks[tid]['last'] = t
            tracks[tid]['lifetimes'].append(cur[ib]['lifetime'])
            new_active[tid] = cur[ib]
        for ib in un_b:
            tracks.append(dict(birth=t, last=t, name=_cname(cur[ib]), lifetimes=[cur[ib]['lifetime']]))
            new_active[len(tracks) - 1] = cur[ib]
        active = new_active
    T = len(inventories) - 1
    for tr in tracks:
        tr['survives'] = tr['last'] >= T
        tr['birth_step'] = int(steps[tr['birth']])
    return dict(tracks=tracks, n_surviving=sum(t['survives'] for t in tracks),
                birth_order=[t['birth_step'] for t in tracks if t['survives']],
                names=[t['name'] for t in tracks if t['survives']])


def _cname(c):
    nm = c.get('naming', {})
    if nm and nm.get('named'):
        b = nm['named']
        return f"{b['name']}" + (f"@freq{b['freq']}" if b.get('freq') else "")
    b = (nm or {}).get('best')
    return f"unnamed(best {b['name']} r={b['r']:.2f})" if b else "unnamed"
