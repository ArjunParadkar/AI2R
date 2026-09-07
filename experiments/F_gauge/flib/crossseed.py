"""F4a — cross-seed gauge residual between circuits (SPEC_F4_reformulated §F4a)."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .gauge import sub_implementation
from .interpret import circuit_bits
from .enumerate import circuit_layers, _offsets


def _normalize(Ws, bs):
    """Gauge fix: unit-normalise every hidden unit's incoming (w, b), push the norm downstream; unit-normalise the output row."""
    Ws = [W.copy() for W in Ws]; bs = [b.copy() for b in bs]
    for l in range(len(Ws) - 1):
        V = np.vstack([Ws[l], bs[l][None, :]]); n = np.linalg.norm(V, axis=0); n[n == 0] = 1.0
        Ws[l] = Ws[l] / n[None, :]; bs[l] = bs[l] / n; Ws[l + 1] = Ws[l + 1] * n[:, None]
    on = np.sqrt((Ws[-1] ** 2).sum() + (bs[-1] ** 2).sum()) or 1.0
    Ws[-1] = Ws[-1] / on; bs[-1] = bs[-1] / on
    return Ws, bs


def effective_subimpl(model, X, mid, scheme, collapse=True):
    """Gauge-fixed sub-implementation of circuit ``mid``: per hidden layer the unit-normalised incoming
    vectors (k_l, fan_in+1) and the unit-normalised output row (k_d + 1,). With ``collapse`` the
    shared-activation-pattern groups (functional-dimension correction) are replaced by rank-r effective
    units computed from the group's effective map M = Σ_u (w_u, b_u) ⊗ w_out[u] IN THE GAUGE-FIXED FRAME
    (so M is a gauge invariant), followed by re-normalisation; every step is a deterministic function
    of gauge invariants, so exact relabellings give residual 0."""
    Ws, bs, layers = sub_implementation(model, mid)
    Ws, bs = _normalize(Ws, bs)
    if collapse:
        bits, _ = circuit_bits(model, X, mid, scheme); off = _offsets(model.widths)
        pat = [[bits[:, off[l] + u].tobytes() for u in layers[l]] for l in range(len(layers))]
        for l in range(len(Ws) - 1):
            if len(pat[l]) < 2: continue
            groups = {}
            for j, key in enumerate(pat[l]): groups.setdefault(key, []).append(j)
            groups = [g for g in groups.values() if len(g) > 1]
            if not groups: continue
            keep = [j for j in range(Ws[l].shape[1]) if not any(j in g for g in groups)]
            new_in, new_b, new_out, new_pat = [Ws[l][:, keep]], [bs[l][keep]], [Ws[l + 1][keep, :]], [pat[l][j] for j in keep]
            for g in groups:
                A = np.vstack([Ws[l][:, g], bs[l][g][None, :]]); B = Ws[l + 1][g, :]; M = A @ B        # (fan_in+1, fan_out)
                # the children's gauge rescales M's columns, so the invariant object is col(M): take the SVD of the
                # column-normalised M (invariant), use its left singular vectors as effective incoming units and
                # project the raw M onto them for the outgoing weights (M is reproduced exactly)
                cn = np.linalg.norm(M, axis=0); cn[cn == 0] = 1.0; Mn = M / cn[None, :]
                U, S, Vt = np.linalg.svd(Mn, full_matrices=False); r = int((S > 1e-9 * max(S[0], 1e-300)).sum()) if S.size else 0
                for i in range(r):
                    v = U[:, i]; sgn = np.sign(v[np.argmax(np.abs(v))]) or 1.0; v = v * sgn; o = v @ M
                    new_in.append(v[:-1][:, None]); new_b.append(np.array([v[-1]])); new_out.append(o[None, :]); new_pat.append(pat[l][g[0]] + bytes([i]))
            Ws[l] = np.hstack(new_in); bs[l] = np.concatenate(new_b); Ws[l + 1] = np.vstack(new_out); pat[l] = new_pat
            Ws, bs = _normalize(Ws, bs)
    inc = [np.vstack([Ws[l], bs[l][None, :]]).T for l in range(len(Ws) - 1)]
    out = np.concatenate([Ws[-1].ravel(), bs[-1]])
    return inc, out


def residual(sa, sb):
    """Gauge residual between two effective sub-implementations (Hungarian matching layer by layer).
    Returns (r, perms); r = inf if layer sizes differ."""
    inc_a, out_a = sa; inc_b, out_b = sb
    if len(inc_a) != len(inc_b) or any(a.shape != b.shape for a, b in zip(inc_a, inc_b)): return float("inf"), None
    perms = []; num = 0.0; den = 0.0; prev_perm = None
    for l, (A, B) in enumerate(zip(inc_a, inc_b)):
        Bp = B if prev_perm is None else np.hstack([B[:, :-1][:, prev_perm], B[:, -1:]])   # reorder B's inputs to A's parent order
        cost = ((A[:, None, :] - Bp[None, :, :]) ** 2).sum(-1); ri, ci = linear_sum_assignment(cost)
        p = np.empty(A.shape[0], int); p[ri] = ci; perms.append(p)
        num += cost[ri, ci].sum(); den += (A ** 2).sum() + (Bp ** 2).sum(); prev_perm = p
    ob = np.concatenate([out_b[:-1][prev_perm], out_b[-1:]])
    num += ((out_a - ob) ** 2).sum(); den += (out_a ** 2).sum() + (ob ** 2).sum()
    return float(np.sqrt(2 * num / den)) if den > 0 else 0.0, perms


def size_signature(sa):
    return tuple(a.shape[0] for a in sa[0])
