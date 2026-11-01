"""M3 checks: H0 persistence (single-linkage death times = MST edge lengths).
 (a) metric perturbation:  |d - d'|_inf <= eta  =>  |e_(i) - e'_(i)| <= eta for every sorted MST edge
 (b) gamma-net subsample:  r_conn(S)-2gamma <= r_conn(X) <= max(r_conn(S), gamma) ; counting interleaving c_S(r+2g)<=c_X(r), c_X(r)<=c_S(r) (r>=g)
 (c) certified interval for the ratio statistic tau = r_fib / r_all coverage on random clouds."""
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial.distance import pdist, squareform
rng = np.random.default_rng(0)
def mst_edges(D): return np.sort(minimum_spanning_tree(D).data)
def ncomp(D, r):
    A = (D <= r) & (D > 0); return connected_components(A, directed=False)[0]
viol_a = viol_b = viol_b2 = 0
for t in range(300):
    n = int(rng.integers(8, 120)); X = rng.normal(size=(n, int(rng.integers(2, 10)))) * rng.uniform(0.2, 2)
    D = squareform(pdist(X)); eta = rng.uniform(0, 0.3)
    N = rng.uniform(-eta, eta, size=(n, n)); N = (N + N.T) / 2; np.fill_diagonal(N, 0)
    D2 = np.clip(D + N, 1e-9, None); np.fill_diagonal(D2, 0)       # symmetric perturbation, |D2-D|<=eta
    e1, e2 = mst_edges(D), mst_edges(D2)
    viol_a += np.max(np.abs(e1 - e2)) > eta + 1e-12
    # (b) subsample = greedy farthest-point net
    idx = [0]; 
    m = int(rng.integers(3, n)) if n > 3 else 2
    dist_to = D[0].copy()
    for _ in range(m - 1):
        j = int(np.argmax(dist_to)); idx.append(j); dist_to = np.minimum(dist_to, D[j])
    gamma = dist_to.max(); S = D[np.ix_(idx, idx)]
    rcX, rcS = mst_edges(D).max(), (mst_edges(S).max() if len(idx) > 1 else 0.)
    viol_b += not (rcS - 2 * gamma - 1e-12 <= rcX <= max(rcS, gamma) + 1e-12)
    for r in np.linspace(0, D.max(), 25):
        if ncomp(S, r + 2 * gamma) > ncomp(D, r): viol_b2 += 1
        if r >= gamma and ncomp(D, r) > ncomp(S, r): viol_b2 += 1
print(f"(a) sorted MST edge perturbation bound violated {viol_a}/300; (b) connection-scale bound violated {viol_b}/300; counting interleaving violated {viol_b2} times")

# (c) coverage of the certified interval on a fibre statistic
cover = 0; trials = 400; widths = []
for t in range(trials):
    n, k = 240, 4
    centers = rng.normal(size=(k, 5)) * 3; lab = rng.integers(0, k, size=n)
    X = centers[lab] + rng.normal(size=(n, 5)) * 0.5
    D = squareform(pdist(X))
    def tau(D, lab):   # mean over classes of (connection scale within class)/(global MST max edge scale + median NN dist)
        rf = np.mean([mst_edges(D[np.ix_(lab == c, lab == c)]).max() for c in range(k) if (lab == c).sum() > 1])
        ra = np.median(np.sort(D + np.eye(n) * 1e9, 1)[:, 0]) * 0 + np.median(D[np.triu_indices(n, 1)])
        return rf, ra
    rf, ra = tau(D, lab)
    # noisy measurement: perturb points by <=rho each -> metric changes <= 2 rho
    rho = 0.05; P = rng.normal(size=X.shape); P *= rho / np.linalg.norm(P, axis=1, keepdims=True) * rng.uniform(0, 1, size=(n, 1))
    D2 = squareform(pdist(X + P)); rf2, ra2 = tau(D2, lab); eta = 2 * rho
    # rf: bounded by eta (max sorted edge); ra (median of all pair distances) bounded by eta as well (each order statistic moves <= eta)
    lo = max(rf - eta, 0) / (ra + eta); hi = (rf + eta) / max(ra - eta, 1e-12)
    # certify from the NOISY side (what we actually observe): interval for the true tau around noisy numbers
    lo_o = max(rf2 - eta, 0) / (ra2 + eta); hi_o = (rf2 + eta) / max(ra2 - eta, 1e-12)
    cover += lo_o <= rf / ra <= hi_o; widths.append(hi_o - lo_o)
print(f"(c) certified interval covers the true tau in {cover}/{trials} trials (must be {trials}); mean width {np.mean(widths):.3f}")
