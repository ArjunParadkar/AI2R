"""Isotypic decomposition of the representation manifold.

When the probe carries a group action -- a shift of a modular input, a rotation of an image, a
turn of a heading -- the network's activations carry a representation of that group.  A
representation of a compact abelian group splits into two-dimensional invariant subspaces, one
per character, and *each of those planes is where a circle lives*.

This matters because persistent cohomology on the raw cloud cannot see those circles.  If the
model uses several frequencies at once, the image of the input torus winds densely: two inputs
are close in activation space only when every frequency component agrees, so the Rips complex
jumps from totally disconnected to nearly complete with no scale in between, and H^1 fills up
with hundreds of meaningless classes.  Projecting first is not a convenience, it is what makes
the cohomology of the object visible at all.

PCA alone does not do it: the frequency planes have nearly equal variance, and inside a
degenerate eigenspace the PCA basis is arbitrary, so it returns mixtures.  The group action
breaks that degeneracy exactly -- the invariant planes are the eigenspaces of the induced
shift operator, and its eigenvalues are the characters themselves.
"""
import numpy as np


def induced_shift_operator(X, param, period, energy=0.99, min_rel_sv=0.02):
    """Least-squares estimate of the linear map T with T h(t) ~= h(t+1).

    X is the activation cloud, param the integer group coordinate of each point (a in Z_p, or
    a discretised angle).  Points are grouped by param, averaged, and T is regressed on the
    shifted pairs.  T is the group action *as the network implements it*; everything else here
    is read off its spectrum.

    The regression is solved in a TRUNCATED subspace.  The pseudo-inverse divides by the
    singular values, so any direction the cloud barely occupies is amplified by 1/s and returns
    a rotation angle that is pure noise -- which is what produced a spurious cluster of nearby
    "frequencies" before this truncation was added.  We keep the directions holding `energy` of
    the variance, and never any with a singular value below `min_rel_sv` of the largest.
    """
    param = np.asarray(param).astype(int) % period
    D = X.shape[1]
    M = np.zeros((period, D))
    cnt = np.zeros(period)
    np.add.at(M, param, X)
    np.add.at(cnt, param, 1)
    ok = cnt > 0
    M[ok] /= cnt[ok, None]
    M = M - M[ok].mean(0)
    idx = np.where(ok)[0]
    nxt = (idx + 1) % period
    keep = ok[nxt]
    A, B = M[idx[keep]], M[nxt[keep]]

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    var = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-30)
    r = int(np.searchsorted(var, energy) + 1)
    r = min(r, int((s > min_rel_sv * s[0]).sum()))
    r = max(r, 2)
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    Tt = Vt.T @ ((U.T @ B) / s[:, None])        # = pinv(A) @ B = T^T  (D x D)
    return Tt.T, Vt, M, idx, r


def isotypic_planes(X, param, period, n_planes=8, var_floor=0.005):
    """The invariant 2-planes of the induced action, with the frequency of each.

    Diagonalise the induced action, T = V L V^-1.  In the eigenbasis the coefficients obey
    z(t+1) = L z(t) exactly, so for the eigenvalue exp(2 pi i k / period) the coefficient
    traces a circle at angle 2 pi k t / period: the frequency and the circle both fall out of
    the spectrum.

    The coefficients must be read off with V^-1, the dual basis.  Projecting onto the real span
    of (Re v, Im v) instead -- orthogonally, or in that frame -- is wrong whenever the frequency
    subspaces are not mutually orthogonal, which in a trained network they never are: the other
    frequencies leak in and the angle stops tracking the group coordinate.  V^-1 removes that
    leakage by construction, because it inverts the whole basis at once.
    """
    T, basis, M, idx, rank = induced_shift_operator(X, param, period)
    Ts = basis @ T @ basis.T                  # the action restricted to the data subspace
    w, V = np.linalg.eig(Ts)
    Vinv = np.linalg.inv(V)
    Xc = X - X.mean(0)
    S = Xc @ basis.T                          # subspace coordinates, (n, r)
    Z = S @ Vinv.T                            # eigenbasis coefficients, (n, r) complex
    total = float((Xc ** 2).sum())

    out, used = [], np.zeros(len(w), bool)
    order = np.argsort(-np.abs(Z).sum(0))     # by how much of the cloud each mode carries
    for i in order:
        if used[i] or abs(w[i].imag) < 1e-8:
            continue
        ang = float(np.angle(w[i]))
        k = int(np.round(ang / (2 * np.pi) * period)) % period
        j = int(np.argmin(np.abs(w - np.conj(w[i])) + 1e9 * used))
        used[i] = True
        if j != i:
            used[j] = True
        z = Z[:, i]
        Y = np.stack([z.real, z.imag], 1)     # the circle, in the eigen-coefficient plane
        E = basis.T @ np.stack([V[:, i].real, V[:, i].imag]).T
        Q, _ = np.linalg.qr(E)                # orthonormal basis of the same plane, for tangents
        var = float(((Xc @ Q) ** 2).sum() / total)
        if var < var_floor:
            continue
        out.append(dict(freq=min(k, period - k), signed_freq=k, plane=Q, proj=Y, var=var,
                        eigenvalue=complex(w[i]), modulus=float(abs(w[i])), rank=rank))
        if len(out) >= n_planes:
            break
    out.sort(key=lambda d: -d['var'])
    return out


def isotypic_decomposition(X, param, period, n_freq=6, var_floor=0.01, detect='action',
                           X_eval=None, ridge=1e-6):
    """Split the cloud into isotypic components, robustly.

    Two steps, and they do different jobs:

    1. DISCOVER the characters the network uses.  Either from the spectrum of the induced
       action (`detect='action'`) or from the power spectrum of the cloud over the group
       (`detect='dft'`).  Both read the model only; neither uses the labels or the task.
    2. SEPARATE them by one joint least-squares fit of the cloud onto the character basis of
       all discovered frequencies at once.  This is the step that matters.  Diagonalising the
       estimated action instead works in exact arithmetic but not on a trained network: two
       frequencies whose eigenvalues are close (50 and 51 of 113) have eigenvectors that mix
       under estimation noise, and the coordinate of the contaminated plane stops tracking the
       group.  Fitting all characters together gives the joint dual basis, so each plane is
       cleaned of every other one by construction.

    OUT-OF-SAMPLE MODE.  When the layer is wider than the orbit is long (D >> n) the dual basis
    can map ANY n points onto a circle -- the feature-shuffle null scored r = 0.97 on a
    3136-dimensional conv layer with 36-point orbits.  Pass `X_eval` (points not used in the
    fit, e.g. a held-out orbit) and the coordinates returned are those of X_eval under planes
    fitted on X: noise cannot transfer, so the null dies.  In-sample coordinates are only safe
    when n >> number of fitted characters and the null has been run, as on Z113.

    Returns one entry per frequency: the plane, the coefficient pair (the circle), and the
    fraction of the cloud's variance it carries.
    """
    param = np.asarray(param).astype(int) % period
    Xc = X - X.mean(0)

    if detect == 'dft':
        N = period
        M = np.zeros((N, X.shape[1])); cnt = np.zeros(N)
        np.add.at(M, param, Xc); np.add.at(cnt, param, 1)
        M[cnt > 0] /= cnt[cnt > 0, None]
        pw = (np.abs(np.fft.rfft(M, axis=0)) ** 2).sum(1)
        pw[0] = 0
        freqs = [int(k) for k in np.argsort(-pw)[:n_freq]]
    else:
        pl = isotypic_planes(X, param, period, n_planes=n_freq, var_floor=0.0)
        freqs = []
        for d in pl:
            if d['freq'] not in freqs and d['freq'] != 0:
                freqs.append(int(d['freq']))
        freqs = freqs[:n_freq]

    cols, meta = [], []
    for k in freqs:
        ang = 2 * np.pi * k * param / period
        cols += [np.cos(ang), np.sin(ang)]
        meta.append(k)
    C = np.stack(cols, 1)                                  # (n, 2F) character design matrix
    B, *_ = np.linalg.lstsq(C, Xc, rcond=None)             # (2F, D) isotypic basis vectors
    target = Xc if X_eval is None else (X_eval - X.mean(0))
    Gm = B @ B.T
    dual = B.T @ np.linalg.inv(Gm + ridge * np.trace(Gm) / len(Gm) * np.eye(len(Gm)))
    coef = target @ dual                                   # joint dual coordinates

    total = float((Xc ** 2).sum())
    out = []
    for i, k in enumerate(meta):
        Y = coef[:, 2 * i:2 * i + 2]
        E = B[2 * i:2 * i + 2].T                           # (D, 2) plane
        Q, _ = np.linalg.qr(E)
        var = float(((Xc @ Q) ** 2).sum() / total)
        if var < var_floor:
            continue
        out.append(dict(freq=int(k), proj=Y, plane=Q, var=var, basis=E, dual=dual[:, 2 * i:2 * i + 2], mean=X.mean(0),
                        recon_r2=float(1 - ((Xc - C @ B) ** 2).sum() / total)))
    out.sort(key=lambda d: -d['var'])
    return out, freqs
