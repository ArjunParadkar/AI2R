"""Méloux-style exhaustive enumeration of valid circuits and gate interpretations.
Neuron indexing: hidden neuron (layer l, unit j) has global id offset[l] + j."""
import itertools
import numpy as np


def _offsets(widths):
    off = [0]
    for w in widths:
        off.append(off[-1] + w)
    return off


def all_masks(n_hidden, chunk=1 << 16):
    """Yield boolean mask arrays (B, n_hidden) covering all 2^n_hidden subsets."""
    total = 1 << n_hidden
    bits = np.arange(n_hidden)
    for start in range(0, total, chunk):
        ids = np.arange(start, min(start + chunk, total), dtype=np.int64)
        yield ids, ((ids[:, None] >> bits[None, :]) & 1).astype(bool)


def ablated_forward(model, X, masks, scheme, means=None):
    """Run the network with complement-of-mask neurons ablated in every hidden layer.
    masks: (B, n_hidden) bool. Returns logits (B, n_inputs) and per-layer acts list of (B, n_inputs, w)."""
    widths = model.widths
    off = _offsets(widths)
    B = masks.shape[0]
    h = np.broadcast_to(X, (B,) + X.shape)
    acts = []
    for l, (W, b) in enumerate(zip(model.Ws[:-1], model.bs[:-1])):
        h = np.maximum(h @ W + b, 0.0)                      # (B, N, w)
        m = masks[:, off[l]:off[l + 1]][:, None, :]         # (B, 1, w)
        if scheme == "zero":
            h = np.where(m, h, 0.0)
        elif scheme == "mean":
            h = np.where(m, h, means[l][None, None, :])
        else:
            raise ValueError(scheme)
        acts.append(h)
    logits = (h @ model.Ws[-1] + model.bs[-1])[..., 0]      # (B, N)
    return logits, acts


def enumerate_valid(model, X, y, scheme):
    """Return sorted array of mask ids (ints) whose ablated network reproduces y."""
    means = [a.mean(0) for a in model.hidden_acts(X)]
    n_hidden = sum(model.widths)
    valid = []
    for ids, masks in all_masks(n_hidden):
        logits, _ = ablated_forward(model, X, masks, scheme, means)
        ok = ((logits > 0).astype(np.int8) == y[None, :]).all(1)
        valid.append(ids[ok])
    return np.concatenate(valid) if valid else np.zeros(0, dtype=np.int64)


def minimal_ids(valid_ids):
    """Subsets (as bitmask ints) with no proper valid subset."""
    vs = set(int(v) for v in valid_ids)
    out = []
    for v in sorted(vs, key=lambda t: bin(t).count("1")):
        sub = v
        minimal = True
        # iterate over proper submasks
        s = (v - 1) & v
        while True:
            if s in vs:
                minimal = False
                break
            if s == 0:
                break
            s = (s - 1) & v
        if minimal:
            out.append(v)
    return np.array(sorted(out), dtype=np.int64)


def mask_from_id(mid, n_hidden):
    return ((mid >> np.arange(n_hidden)) & 1).astype(bool)


def circuit_layers(mid, widths):
    """List per hidden layer of the unit indices in the circuit."""
    off = _offsets(widths)
    m = mask_from_id(mid, off[-1])
    return [np.flatnonzero(m[off[l]:off[l + 1]]) for l in range(len(widths))]


def binarised_acts(model, X, mid, scheme):
    """Binarised activations of every hidden unit (N, n_hidden) under ablation of mid's complement,
    plus the output bit (N,)."""
    means = [a.mean(0) for a in model.hidden_acts(X)]
    n_hidden = sum(model.widths)
    masks = mask_from_id(mid, n_hidden)[None, :]
    logits, acts = ablated_forward(model, X, masks, scheme, means)
    B = np.concatenate([(a[0] > 0) for a in acts], axis=1)
    return B.astype(np.int8), (logits[0] > 0).astype(np.int8)


def separations(parent_bits, acts):
    """Méloux-style interpretation of one neuron given its parents' bits.
    Group inputs by parent pattern; a separation is a threshold t between two consecutive distinct
    activation values such that every parent pattern lies entirely on one side (consistency,
    Definition 3 of Méloux et al. 2025); 1 = above t (their 'larger interval' convention).
    Returns list of (gap_index, bits (N,), gate = tuple(sorted partial truth table)))."""
    N = len(acts)
    vals = np.unique(np.round(acts, 9))
    out = []
    pats = [tuple(int(v) for v in row) for row in parent_bits]
    for gi in range(len(vals) - 1):
        t = 0.5 * (vals[gi] + vals[gi + 1])
        bits = (acts > t).astype(np.int8)
        table = {}
        ok = True
        for r in range(N):
            if table.setdefault(pats[r], int(bits[r])) != bits[r]:
                ok = False
                break
        if ok:
            out.append((gi, bits, tuple(sorted(table.items()))))
    return out


def _ablated_acts(model, X, mid, scheme):
    means = [a.mean(0) for a in model.hidden_acts(X)]
    masks = mask_from_id(mid, sum(model.widths))[None, :]
    logits, acts = ablated_forward(model, X, masks, scheme, means)
    return [a[0] for a in acts], logits[0]


def interpretations(model, X, mid, scheme, max_leaves=10_000_000):
    """Enumerate consistent gate interpretations of circuit `mid` (Méloux et al. 2025, Sec. 3):
    layer by layer, each circuit neuron gets a threshold separating its activations into 0/1 such
    that the bits are a function of its circuit-parents' bits; the output bit is fixed (logit > 0)
    and must be a function of the last layer's bits. Returns (count, gate_multiset_keys) where
    gate_multiset_keys is the set of distinct per-layer gate-table multisets found (for orbit keys)."""
    acts, logits = _ablated_acts(model, X, mid, scheme)
    layers = circuit_layers(mid, model.widths)
    out_bits = (logits > 0).astype(np.int8)
    count = 0
    gate_keys = set()

    def rec(l, prev_bits, chosen):
        nonlocal count
        if l == len(layers):
            # output consistency
            table = {}
            perm = sorted(range(prev_bits.shape[1]), key=lambda j: tuple(prev_bits[:, j]))
            for r in range(len(X)):
                k = tuple(int(prev_bits[r, j]) for j in perm)
                if table.setdefault(k, int(out_bits[r])) != out_bits[r]:
                    return
            count += 1
            gate_keys.add(tuple(chosen) + (tuple(sorted(table.items())),))
            return
        units = layers[l]
        # canonical parent order: sort parent columns by their bit-tuples (permutation-invariant)
        perm = sorted(range(prev_bits.shape[1]), key=lambda j: tuple(prev_bits[:, j]))
        pb = prev_bits[:, perm] if len(perm) else prev_bits
        opts = [separations(pb, acts[l][:, u]) for u in units]
        if any(len(o) == 0 for o in opts):
            return
        for combo in itertools.product(*opts):
            bits = np.stack([c[1] for c in combo], axis=1) if len(combo) else np.zeros((len(X), 0), np.int8)
            key = tuple(sorted(c[2] for c in combo))     # multiset of gates in this layer (permutation-invariant)
            rec(l + 1, bits, chosen + [key])
            if count > max_leaves:
                return

    rec(0, X.astype(np.int8), [])
    return count, gate_keys
