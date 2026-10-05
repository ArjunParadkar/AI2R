"""Section 2.4-2.6: persistent cohomology, circular coordinates, fibers, the training bundle."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from ripser import ripser

PRIME = 47          # Z_47 coefficients, as specified


def persistence(X, maxdim=1, coeff=PRIME, thresh_pct=95, distance_matrix=False, do_cocycles=True):
    """Vietoris-Rips persistent cohomology of the cloud.

    The Rips threshold is the `thresh_pct` percentile of pairwise distances: without it the
    full complex on 2,000 points is unaffordable, and classes born past that scale are
    artifacts of the cloud's diameter rather than structure.
    """
    D = X if distance_matrix else squareform(pdist(X))
    finite = D[np.triu_indices_from(D, 1)]
    thresh = float(np.percentile(finite, thresh_pct))
    res = ripser(D, maxdim=maxdim, coeff=coeff, thresh=thresh,
                 distance_matrix=True, do_cocycles=do_cocycles)
    res['thresh'] = thresh
    res['median_dist'] = float(np.median(finite))
    res['D'] = D
    return res


def significant_classes(dgm, noise_band=None, min_lifetime_frac=0.0, essential_always=False):
    """Indices of H^k classes whose lifetime clears the noise band.

    `noise_band` should come from the matched null (feature-shuffled activations run through
    the identical pipeline): any lifetime the null can produce is not structure.  With no null
    supplied we fall back to the largest RATIO gap in the sorted finite lifetimes, and only if
    that gap is at least 2x -- a cloud with no such gap has no significant finite class (the old
    largest-difference rule always returned at least two, so a connected torus read beta_0=624).

    essential_always: classes that never die (the H^0 component bars) are counted as
    significant regardless of the band.  Off by default so H^1 bars that merely outlive the
    Rips threshold are still judged by lifetime.
    """
    if len(dgm) == 0:
        return np.array([], dtype=int), 0.0
    raw = dgm[:, 1] - dgm[:, 0]
    fin = np.isfinite(raw)
    top = np.nanmax(raw[fin], initial=0.0)
    life = np.where(fin, raw, top)
    if noise_band is None:
        s = np.sort(raw[fin]) if essential_always else np.sort(life)
        s = s[s > 0]
        if len(s) < 2:
            noise_band = s[-1] * 0.5 if len(s) and not essential_always else (s[-1] if len(s) else 0.0)
        else:
            # only the 20 longest bars compete: among the thousands of tiny bars at the bottom of a
            # diagram two neighbours can differ by 2x by accident, which would make everything
            # above them "significant" (measured: beta_1 = 2180 on a 1200-point Z113 cloud)
            top = s[-20:]
            ratios = top[1:] / top[:-1]
            i = int(ratios.argmax())
            noise_band = top[i] if ratios[i] >= 2.0 else s[-1]
    band = max(noise_band, min_lifetime_frac * life.max() if len(life) else 0.0)
    keep = life > band
    if essential_always:
        keep = np.where(fin, raw > band, True)
    return np.where(keep)[0], float(band)


def normalized_persistence(dgm, median_dist):
    """Scale-free total persistence: lifetimes divided by the median pairwise distance,
    then summed in quadrature.  Dividing first is what makes it invariant to a global
    rescaling of the activations -- the artifact that made the v1 number meaningless."""
    if len(dgm) == 0 or median_dist <= 0:
        return 0.0
    life = dgm[:, 1] - dgm[:, 0]
    life = life[np.isfinite(life)] / median_dist
    return float((life ** 2).sum())


def _rips_edges(D, alpha):
    i, j = np.triu_indices_from(D, 1)
    keep = D[i, j] <= alpha
    return i[keep], j[keep]


def circular_coordinates(X, dgm1, cocycles1, D=None, class_idx=0, alpha_frac=0.35,
                         coeff=PRIME, max_edges=4_000_000):
    """The circular coordinate of a persistent H^1 class (de Silva-Morozov-Vejdemo-Johansson).

    Take the Z_p cocycle ripser returns for the class, lift its values to (-p/2, p/2], and
    find the real function f on vertices whose coboundary is closest to it:

        min_f || zbar - delta^0 f ||_2 ,     theta = f mod 1.

    The residual zbar - delta^0 f is the harmonic representative; theta is the map to S^1 it
    integrates to.  Evaluated at a threshold inside the class's lifetime, so the cocycle is
    still a cocycle there.
    """
    if D is None:
        D = squareform(pdist(X))
    birth, death = dgm1[class_idx]
    if not np.isfinite(death):
        death = D.max()
    alpha = birth + alpha_frac * (death - birth)
    alpha = min(alpha, death - 1e-9)

    ei, ej = _rips_edges(D, alpha)
    if len(ei) > max_edges:      # keep the solve tractable: tighten toward birth
        alpha = birth + 1e-6 * (death - birth)
        ei, ej = _rips_edges(D, alpha)
    n, m = len(D), len(ei)          # D, not X: the cloud may be given only as a metric space
    if m == 0:
        return None

    eindex = {}
    for e, (a, b) in enumerate(zip(ei, ej)):
        eindex[(int(a), int(b))] = e

    z = np.zeros(m)
    cc = cocycles1[class_idx]
    for row in np.atleast_2d(cc):
        a, b, v = int(row[0]), int(row[1]), float(row[2])
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        e = eindex.get((a, b))
        if e is not None:
            v = v if v <= coeff / 2 else v - coeff      # lift Z_p -> (-p/2, p/2]
            z[e] = s * v

    rows = np.concatenate([np.arange(m), np.arange(m)])
    cols = np.concatenate([ej, ei])
    vals = np.concatenate([np.ones(m), -np.ones(m)])
    delta = coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsr()

    f = lsqr(delta, z, atol=1e-10, btol=1e-10, iter_lim=3000)[0]
    resid = z - delta @ f
    theta = np.mod(f, 1.0) * 2 * np.pi
    return dict(theta=theta, alpha=float(alpha), n_edges=int(m), birth=float(birth),
                death=float(death), lifetime=float(death - birth),
                harmonic_residual=float(np.linalg.norm(resid) / max(np.linalg.norm(z), 1e-12)))


def circular_correlation(a, b, allow_reflection=True):
    """Agreement between two circular variables, invariant to rotation and reflection.

        r = | mean_i exp( i (a_i - b_i) ) |        (and the same for a_i + b_i)

    r = 1 iff a = b + const (or a = -b + const), 0 for unrelated angles.

    This replaces the Jammalamadaka-Sarma coefficient, which is WRONG for this use.  That
    estimator centres each variable on its mean direction; for angles spread uniformly around
    the circle the resultant vector is ~0, so the mean direction is numerically arbitrary, and
    two *identical* angle sequences get centred by different offsets and score near zero.
    Uniform marginals are not an edge case here -- a circular coordinate on a well-sampled
    circle always has them.  Measured: an isotypic coordinate that reproduces cos/sin(k a) at
    component correlation 0.997 scored 0.145 under Jammalamadaka and 1.000 under this.

    A circular coordinate is only defined up to rotation (and up to orientation, which is the
    sign of the cocycle), so rotation invariance is not a convenience -- it is the only way the
    comparison is well posed.
    """
    a, b = np.asarray(a), np.asarray(b)
    r_minus = abs(np.mean(np.exp(1j * (a - b))))
    if not allow_reflection:
        return float(r_minus)
    r_plus = abs(np.mean(np.exp(1j * (a + b))))
    return float(max(r_minus, r_plus))


def circular_rank_correlation(a, b):
    """Circular correlation after uniformising both variables (a circular Spearman).

    A network has no reason to parametrise a ring at unit speed: its coordinate can be any
    monotone warp of the true angle, which caps the linear circular correlation below 1 even
    when the coordinate orders every point correctly.  Replacing each angle by its circular
    rank removes the warp and leaves only the question "same cyclic order?".
    """
    def uni(x):
        x = np.mod(np.asarray(x), 2 * np.pi)
        r = np.argsort(np.argsort(x))
        return 2 * np.pi * (r + 0.5) / len(x)
    return circular_correlation(uni(a), uni(b))


def circular_linear_correlation(theta, x):
    """Correlation between a circular variable and a linear one (Mardia's R)."""
    x = np.asarray(x, dtype=float)
    if x.std() < 1e-12:
        return 0.0
    rxc = np.corrcoef(x, np.cos(theta))[0, 1]
    rxs = np.corrcoef(x, np.sin(theta))[0, 1]
    rcs = np.corrcoef(np.cos(theta), np.sin(theta))[0, 1]
    num = rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs
    den = 1 - rcs ** 2
    return float(np.sqrt(max(num / den, 0.0))) if den > 1e-12 else 0.0


def name_coordinate(theta, variables, modular_period=None, max_freq=56, threshold=0.8):
    """Name a coordinate by correlating it with every variable we know about.

    `variables` maps name -> array.  A variable declared modular (period p) is tested against
    every frequency k: the angle 2*pi*k*v/p.  That is what lets a coordinate be reported as
    "frequency 17 of a", which is the Z113 ground truth.  The best match above `threshold`
    names the coordinate; everything else is reported as discovered structure.
    """
    out = []
    for nm, v in variables.items():
        v = np.asarray(v)
        if modular_period and np.issubdtype(v.dtype, np.integer):
            best = (0.0, None)
            for k in range(1, max_freq + 1):
                r = abs(circular_correlation(theta, 2 * np.pi * k * v / modular_period))
                if r > best[0]:
                    best = (r, k)
            out.append(dict(name=nm, r=best[0], freq=best[1], kind='modular'))
        else:
            out.append(dict(name=nm, r=circular_linear_correlation(theta, v.astype(float)),
                            freq=None, kind='linear'))
    out.sort(key=lambda d: -d['r'])
    named = out[0] if out and out[0]['r'] >= threshold else None
    return dict(best=out[0] if out else None, named=named, all=out, threshold=threshold)


def toroidal_decorrelate(thetas, max_coeff=3, r_thresh=0.9):
    """Drop coordinates that are integer combinations of ones already kept.

    Two H^1 classes of the same torus can carry coordinates that are Z-linear combinations of
    each other; reporting both would double-count the structure.  For each candidate we search
    small integer combinations of the coordinates already kept and drop it if one of them
    reproduces it at circular correlation > r_thresh.
    """
    TWO_PI = 2 * np.pi
    kept, dropped = [], []
    for i, th in enumerate(thetas):
        explained = None
        for a in kept:                                   # single-generator multiples
            for k1 in range(-max_coeff, max_coeff + 1):
                if k1 == 0:
                    continue
                if abs(circular_correlation(th, np.mod(k1 * thetas[a], TWO_PI))) > r_thresh:
                    explained = dict(by=[a], coeffs=[k1])
                    break
            if explained:
                break
        if explained is None:                            # pairs of generators
            for ii in range(len(kept)):
                for jj in range(ii + 1, len(kept)):
                    for k1 in range(-max_coeff, max_coeff + 1):
                        for k2 in range(-max_coeff, max_coeff + 1):
                            if k1 == 0 and k2 == 0:
                                continue
                            comb = np.mod(k1 * thetas[kept[ii]] + k2 * thetas[kept[jj]], TWO_PI)
                            if abs(circular_correlation(th, comb)) > r_thresh:
                                explained = dict(by=[kept[ii], kept[jj]], coeffs=[k1, k2])
                                break
                        if explained:
                            break
                    if explained:
                        break
                if explained:
                    break
        if explained is None:
            kept.append(i)
        else:
            explained['idx'] = i
            dropped.append(explained)
    return kept, dropped


def fiber_topology(X, labels, maxdim=1, coeff=PRIME, min_points=40, noise_band=None):
    """H_*(F_y) for every class y: what the model represents *within* a concept."""
    out = {}
    for y in np.unique(labels):
        sel = np.where(labels == y)[0]
        if len(sel) < min_points:
            out[int(y)] = dict(n=len(sel), skipped=True)
            continue
        r = persistence(X[sel], maxdim=maxdim, coeff=coeff, do_cocycles=True)
        h0, h1 = r['dgms'][0], r['dgms'][1] if maxdim >= 1 else np.zeros((0, 2))
        sig0, b0 = significant_classes(h0, noise_band, essential_always=True)
        sig1, b1 = significant_classes(h1, noise_band)
        out[int(y)] = dict(n=len(sel), idx=sel, res=r,
                           beta0=int(len(sig0)), beta1=int(len(sig1)),
                           band0=b0, band1=b1,
                           tildeC=normalized_persistence(h1, r['median_dist']))
    return out


def match_classes(dgm_a, dgm_b, thetas_a=None, thetas_b=None, tie_break_weight=0.3):
    """Match persistent classes across consecutive checkpoints (spec 2.6).

    Dionysus zigzag is not available in this environment, so we use the spec's stated
    fallback: Wasserstein matching of consecutive diagrams, with the matched classes'
    coordinate correlation as the tie-break.  Cost is the L-inf distance between diagram
    points, reduced by the circular correlation of their coordinates when those exist.
    """
    na, nb = len(dgm_a), len(dgm_b)
    if na == 0 or nb == 0:
        return [], list(range(na)), list(range(nb))
    C = np.abs(dgm_a[:, None, :] - dgm_b[None, :, :]).max(-1)
    C = np.where(np.isfinite(C), C, 1e6)
    if thetas_a is not None and thetas_b is not None:
        for i in range(na):
            for j in range(nb):
                if thetas_a[i] is not None and thetas_b[j] is not None:
                    r = abs(circular_correlation(thetas_a[i], thetas_b[j]))
                    C[i, j] *= (1 - tie_break_weight * r)
    ri, ci = linear_sum_assignment(C)
    life_a = dgm_a[:, 1] - dgm_a[:, 0]
    cutoff = np.nanmax(np.where(np.isfinite(life_a), life_a, 0)) * 0.5 + 1e-12
    pairs = [(int(i), int(j), float(C[i, j])) for i, j in zip(ri, ci) if C[i, j] < cutoff]
    matched_a = {p[0] for p in pairs}; matched_b = {p[1] for p in pairs}
    return pairs, [i for i in range(na) if i not in matched_a], [j for j in range(nb) if j not in matched_b]
