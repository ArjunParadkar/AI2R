"""Symmetry group G (hidden-unit permutations within layers, positive scalings through ReLU),
canonical forms of circuit sub-implementations, orbit keys, and random gauge transforms."""
import hashlib
import numpy as np
from .enumerate import circuit_layers, binarised_acts, interpretations, _offsets


def apply_gauge(model, perms=None, scales=None):
    """Return a new MLP with hidden units permuted (perms[l] = new order) and rescaled
    (scales[l] > 0 per unit). f is unchanged."""
    m = model.copy()
    for l in range(m.depth):
        w = m.Ws[l].shape[1]
        if scales is not None:
            s = np.asarray(scales[l], dtype=np.float64)
            m.Ws[l] = m.Ws[l] * s[None, :]
            m.bs[l] = m.bs[l] * s
            m.Ws[l + 1] = m.Ws[l + 1] / s[:, None]
        if perms is not None:
            p = np.asarray(perms[l])
            assert sorted(p.tolist()) == list(range(w))
            m.Ws[l] = m.Ws[l][:, p]
            m.bs[l] = m.bs[l][p]
            m.Ws[l + 1] = m.Ws[l + 1][p, :]
    return m


def random_gauge(model, rng, log_scale_sd=1.0):
    perms = [rng.permutation(w) for w in model.widths]
    scales = [np.exp(rng.normal(0, log_scale_sd, size=w)) for w in model.widths]
    return apply_gauge(model, perms, scales), perms, scales


def permute_mask_id(mid, perms, widths):
    """Image of circuit id under the permutation used by apply_gauge (unit p[j] moves to slot j)."""
    off = _offsets(widths)
    out = 0
    for l, p in enumerate(perms):
        for j, src in enumerate(p):
            if (mid >> (off[l] + int(src))) & 1:
                out |= 1 << (off[l] + j)
    return out


def sub_implementation(model, mid):
    """Weights restricted to the circuit: list of (W_l[in_units, out_units], b_l[out_units])
    in original unit order."""
    layers = circuit_layers(mid, model.widths)
    prev = np.arange(model.n_in)
    Ws, bs = [], []
    for l, units in enumerate(layers):
        Ws.append(model.Ws[l][np.ix_(prev, units)])
        bs.append(model.bs[l][units])
        prev = units
    Ws.append(model.Ws[-1][prev, :])
    bs.append(model.bs[-1])
    return Ws, bs, layers


def canonical_form(model, mid, eps):
    """Scaling-normalised, permutation-sorted, eps-quantised sub-implementation.
    Returns (key: bytes, order: list per layer of the sort order applied to circuit units,
    normalised Ws/bs)."""
    Ws, bs, layers = sub_implementation(model, mid)
    Ws = [W.copy() for W in Ws]
    bs = [b.copy() for b in bs]
    orders = []
    for l in range(len(Ws) - 1):
        # scaling gauge: unit-normalise (w_in, b) of each unit, push norm downstream
        norms = np.sqrt((Ws[l] ** 2).sum(0) + bs[l] ** 2)
        norms = np.where(norms == 0, 1.0, norms)
        Ws[l] = Ws[l] / norms[None, :]
        bs[l] = bs[l] / norms
        Ws[l + 1] = Ws[l + 1] * norms[:, None]
        # permutation gauge: sort units by quantised incoming vector
        q = np.round(np.vstack([Ws[l], bs[l][None, :]]) / eps).astype(np.int64)  # (fan_in+1, k)
        order = np.lexsort(q[::-1])          # lexicographic on rows, first row most significant
        orders.append(order)
        Ws[l] = Ws[l][:, order]
        bs[l] = bs[l][order]
        Ws[l + 1] = Ws[l + 1][order, :]
    # output row: normalise scale (does not change f), quantise
    on = np.sqrt((Ws[-1] ** 2).sum() + bs[-1] ** 2)
    on = 1.0 if on == 0 else on
    Ws[-1] = Ws[-1] / on
    bs[-1] = bs[-1] / on
    h = hashlib.blake2b(digest_size=16)
    h.update(str([W.shape for W in Ws]).encode())
    for W, b in zip(Ws, bs):
        h.update(np.round(W / eps).astype(np.int64).tobytes())
        h.update(np.round(b / eps).astype(np.int64).tobytes())
    return h.digest(), orders, Ws, bs


def functional_key(model, X, mid, scheme):
    """Exactly gauge-invariant coarse key: per layer the sorted multiset of binarised truth tables."""
    B, out_bits = binarised_acts(model, X, mid, scheme)
    off = _offsets(model.widths)
    layers = circuit_layers(mid, model.widths)
    parts = []
    for l, units in enumerate(layers):
        parts.append(tuple(sorted(tuple(int(v) for v in B[:, off[l] + u]) for u in units)))
    parts.append(tuple(int(v) for v in out_bits))
    return tuple(parts)


def explanation_keys(model, X, mid, scheme, eps_list, max_leaves=10_000_000):
    """For one circuit: interpretation count, per-eps gauge keys (canonical form of the circuit's
    parameter implementation + the frozenset of its gate-multiset interpretations, which is
    gauge-invariant by construction), and the functional key."""
    count, gkeys = interpretations(model, X, mid, scheme, max_leaves)
    fkey = functional_key(model, X, mid, scheme)
    ik = frozenset(gkeys)
    keys = {}
    for eps in eps_list:
        ck, _, _, _ = canonical_form(model, mid, eps)
        keys[eps] = (ck, ik)
    return count, keys, fkey


def orbit_counts(records, eps_list):
    """records: list of (count, keys, fkey) over circuits (possibly pooled across networks).
    Explanations of a circuit with the same gauge key are the same orbit *per interpretation*:
    a circuit with k interpretations contributes k explanations; two circuits with equal
    (canonical form, canonical option set) contribute the same k orbits. Returns dict."""
    n_expl = sum(c for c, _, _ in records)
    out = {"n_expl": n_expl, "n_circuits": len(records)}
    for eps in eps_list:
        seen = {}
        for c, keys, _ in records:
            seen[keys[eps]] = c
        out[f"orbits_eps{eps}"] = sum(seen.values())
        out[f"circuit_orbits_eps{eps}"] = len(seen)
    fseen = {}
    for c, _, f in records:
        fseen[f] = c
    out["orbits_functional"] = sum(fseen.values())
    out["circuit_orbits_functional"] = len(fseen)
    return out
