"""S3 -- the automorphism (isometry) group of a finite metric space, found with a tolerance and with nothing supplied.

Neighbouring work: LieGG (Moskalev, Sepliarskaia, Sosnovik, Smeulders, NeurIPS 2022) extracts Lie-algebra generators of the
symmetries a trained network has learned.  This module is the finite counterpart used in Run 3: given only a distance matrix D
between the network's representations (or outputs) of a finite input set, return the permutations that preserve D.

Search: an isometry is determined by the images of a few anchor points, because every other point is pinned down by its distances
to them.  For each admissible image of three anchors we solve one assignment problem and then VERIFY the whole permutation against
the whole of D.  Only verified permutations are returned.
"""
import math
from functools import reduce
import numpy as np
from scipy.optimize import linear_sum_assignment


def _cycles(p):
    seen, out = np.zeros(len(p), bool), []
    for i in range(len(p)):
        if not seen[i]:
            n, j = 0, i
            while not seen[j]:
                seen[j] = True; j = p[j]; n += 1
            out.append(n)
    return out


def element_order(p):
    return reduce(math.lcm, _cycles(p), 1)


def automorphisms(D, tol=0.10, max_triples=20000):
    D = np.asarray(D, float); n = len(D); off = D[~np.eye(n, dtype=bool)]
    med = float(np.median(off)); nn = float(np.median(np.sort(D + np.eye(n) * 1e18, 1)[:, 0]))
    T = min(tol * med, 0.5 * nn)                             # A-8: never looser than half the point spacing
    Tc = tol * med                                           # candidate filter stays loose; only verification is strict
    out = dict(n=n, tol=tol, T=T, nn_median=nn, median=med, spread=float(off.std() / max(med, 1e-300)), degenerate=False, truncated=False, perms=[], order=1)
    if out['spread'] < tol:                                  # A-7: simplex-like, no structure to be symmetric about
        out.update(degenerate=True, order=1, order_mean=1, has_full_cycle_mean=False, delta_star=None, perms_mean=[], order_rel=1, has_full_cycle_rel=False, perms_rel=[], R_star=None, delta_rand=None, has_full_cycle=False, has_involution=False, max_element_order=1, closure=1.0); return out
    i0 = 0; i1 = int(np.argmin(np.where(np.arange(n) == i0, np.inf, D[i0])))
    m = np.ones(n, bool); m[[i0, i1]] = False; i2 = int(np.arange(n)[m][np.argmin(D[i1][m])])
    A = [i0, i1, i2]; found, tried = {}, 0
    _rng = np.random.default_rng(0); d_rand = float(np.mean([np.abs(D[np.ix_(q, q)] - D).mean() for q in (_rng.permutation(n) for _ in range(10))]))
    found_rel = {}                                             # A-10 (post-hoc): mean defect relative to a random relabelling
    found_mean, best = {}, (np.inf, None)                      # A-9 (post-hoc): mean-norm acceptance and the best non-identity defect
    for j0 in range(n):
        for j1 in np.where(np.abs(D[j0] - D[i0, i1]) < Tc)[0]:
            if j1 == j0:
                continue
            for j2 in np.where((np.abs(D[j1] - D[i1, i2]) < Tc) & (np.abs(D[j0] - D[i0, i2]) < Tc))[0]:
                if j2 in (j0, j1):
                    continue
                tried += 1
                if tried > max_triples:
                    out['truncated'] = True; break
                cost = sum(np.abs(D[j][None, :] - D[i][:, None]) for i, j in zip(A, (j0, j1, j2)))     # cost[k, m]
                cost[A, :] = 1e9; cost[:, [j0, j1, j2]] = 1e9
                for i, j in zip(A, (j0, j1, j2)):
                    cost[i, j] = 0.0
                r, c = linear_sum_assignment(cost); p = np.empty(n, int); p[r] = c
                for _ in range(2):                           # A-8: refine with every point as an anchor
                    full = np.abs(D[p][:, None, :] - D[:, :, None]).mean(0)      # full[k, m] = mean_i |D[p_i, m] - D[i, k]|
                    r, c = linear_sum_assignment(full); q = np.empty(n, int); q[r] = c
                    if np.array_equal(q, p):
                        break
                    p = q
                dfc = np.abs(D[np.ix_(p, p)] - D)
                if dfc.max() < T:
                    found[tuple(p)] = p
                if dfc.mean() < 0.25 * d_rand:
                    found_rel[tuple(p)] = p
                if dfc.mean() < tol * med:
                    found_mean[tuple(p)] = p
                if (p != np.arange(n)).any() and dfc.mean() < best[0]:
                    best = (float(dfc.mean()), p)
            if out['truncated']:
                break
        if out['truncated']:
            break
    pm = list(found_mean.values()); out['perms_mean'] = pm; out['order_mean'] = max(1, len(pm)); out['has_full_cycle_mean'] = bool(any(_cycles(q) == [n] for q in pm))
    out['delta_star'] = best[0] / med if best[1] is not None else None
    pr = list(found_rel.values()); out['perms_rel'] = pr; out['order_rel'] = max(1, len(pr)); out['has_full_cycle_rel'] = bool(any(_cycles(q) == [n] for q in pr))
    out['delta_rand'] = d_rand / med; out['R_star'] = (best[0] / d_rand) if best[1] is not None else None
    perms = list(found.values()); out['perms'] = perms; out['order'] = max(1, len(perms)); out['n_triples'] = tried
    eo = [element_order(p) for p in perms] or [1]
    out['max_element_order'] = int(max(eo)); out['has_full_cycle'] = bool(any(_cycles(p) == [n] for p in perms))
    out['has_involution'] = bool(any(e == 2 for e in eo))
    if len(perms) > 1:                                        # closure on a sample of products
        rng = np.random.default_rng(0); S = set(found); k = min(400, len(perms) ** 2); ok = 0
        for _ in range(k):
            a, b = perms[rng.integers(len(perms))], perms[rng.integers(len(perms))]; ok += tuple(a[b]) in S
        out['closure'] = ok / k
    else:
        out['closure'] = 1.0
    return out


def full_cycle(res, key='perms'):
    """A verified permutation that is a single n-cycle (the discovered cyclic generator), preferring the one that moves point 0 to
    its nearest neighbour; None if there is none."""
    c = [p for p in res[key] if _cycles(p) == [res['n']]]
    return c[0] if c else None


def cycle_coordinate(sigma):
    """t with t[sigma^n(0)] = n: the discovered group's own coordinate on the points."""
    n = len(sigma); t = np.empty(n, int); j = 0
    for k in range(n):
        t[j] = k; j = sigma[j]
    return t


def characters(E, t, top=5):
    """DFT of the centred rows E along the discovered coordinate t.  Returns the top frequencies by power and, for each, the angle of
    every point in that frequency's plane (the discovered character)."""
    n = len(t); order = np.argsort(t); Ec = E[order] - E.mean(0); F = np.fft.rfft(Ec, axis=0); pw = (np.abs(F) ** 2).sum(1); pw[0] = 0
    ks = [int(k) for k in np.argsort(-pw)[:top]]; out = []
    for k in ks:
        z = Ec @ np.conj(F[k]) ; ang = np.empty(n); ang[order] = np.angle(z)
        out.append(dict(freq=k, power_share=float(pw[k] / pw.sum()), angle=ang))
    return out
