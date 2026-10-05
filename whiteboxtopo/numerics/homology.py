"""N5: persistent homology (Vietoris-Rips via ripser) with certified bars.

Stability facts used (both theorems, not heuristics):
 (S1) Rips filtration value of a simplex = its largest edge weight.  If two symmetric weight matrices differ by at
      most eta entrywise, the filtration functions differ by <= eta in sup norm, hence
      bottleneck(dgm(D), dgm(D')) <= eta   (Cohen-Steiner, Edelsbrunner, Harer 2007).
 (S2) For S subset of X (metric spaces, Rips parametrised by distance): d_B(dgm S, dgm X) <= 2 d_GH(S,X) <= 2 d_H(S,X)
      (Chazal, de Silva, Oudot 2014).  For S subset X, d_H(S,X) is the covering radius max_x min_s d(x,s).
Certification: if d_B(dgm S, dgm M) <= eta_c then a bar (b,d) of S with d-b > 2 eta_c cannot be matched to the diagonal,
so M has a bar (b',d') with |b-b'|,|d-d'| <= eta_c: it is alive on [b+eta_c, d-eta_c].  `certified_alive(dgm, eta_c, r)`
counts the bars guaranteed to exist in M's diagram at scale r (a lower bound on the persistent Betti number).
ripser works in float32 internally: all comparisons carry 1e-6 slack.
Cost: ripser VR is O(n^(2+...)) worst-case in memory for H2; use maxmin subsampling (O(n m)) for big clouds.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


def persistence(X=None, D=None, maxdim=1, thresh=np.inf):
    from ripser import ripser
    if D is not None:
        return ripser(np.asarray(D, np.float64), maxdim=maxdim, thresh=thresh, distance_matrix=True)['dgms']
    return ripser(np.asarray(X, np.float64), maxdim=maxdim, thresh=thresh)['dgms']


def finite(dgm):
    return dgm[np.isfinite(dgm[:, 1])]


def bottleneck(d1, d2):
    from persim import bottleneck as _b
    a, b = finite(d1), finite(d2)
    if len(a) == 0 and len(b) == 0: return 0.0
    if len(a) == 0: return float(((b[:, 1] - b[:, 0]) / 2).max())
    if len(b) == 0: return float(((a[:, 1] - a[:, 0]) / 2).max())
    return float(_b(a, b))


def maxmin_subsample(X, m, seed=0):
    """Farthest-point sampling; returns indices and covering radius of the subset in X (= d_H(S,X))."""
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(X)))]
    d = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(m - 1):
        i = int(d.argmax()); idx.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return np.array(idx), float(d.max())


def covering_radius(S, X):
    return float(cdist(X, S).min(1).max())


def hausdorff_to_manifold(S, ref, ref_spacing=0.0):
    """Upper bound on d_H(S, M): covering radius of a dense reference sample of M plus the reference's own
    covering radius (ref_spacing) — conservative."""
    return covering_radius(S, ref) + ref_spacing


def certified_alive(dgm, eta_c, r=None):
    """Bars guaranteed to exist in the target's diagram: life > 2 eta_c; if r given, also alive at r with margin."""
    b, d = dgm[:, 0], dgm[:, 1]
    ok = (d - b) > 2 * eta_c + 1e-6
    if r is not None: ok &= (b + eta_c <= r + 1e-9) & (r <= d - eta_c + 1e-9)
    return int(ok.sum())


def uncertified_long(dgm, eta_c):
    """Finite bars longer than 2 eta_c (all certified); returned for the 'no further long bar' check."""
    f = finite(dgm); return f[(f[:, 1] - f[:, 0]) > 2 * eta_c + 1e-6]


def perturb_distance_matrix(D, eta, rng):
    E = rng.uniform(-eta, eta, D.shape); E = np.triu(E, 1); E = E + E.T
    return np.maximum(D + E, 0)
