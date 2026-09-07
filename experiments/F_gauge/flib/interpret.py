"""Gate interpretations of a circuit under the handbook definition (F2), plus gate naming (F6)."""
import numpy as np
from .enumerate import _offsets, circuit_layers, mask_from_id, ablated_forward


def circuit_bits(model, X, mid, scheme):
    """Bits 1[pre-activation > 0] of every hidden unit under ablation of the complement of mid,
    (N, n_hidden) int8, plus output bit (N,)."""
    means = [a.mean(0) for a in model.hidden_acts(X)]
    n_hidden = sum(model.widths)
    masks = mask_from_id(mid, n_hidden)[None, :]
    off = _offsets(model.widths)
    # recompute pre-activations layer by layer under ablation
    h = X; bits = np.zeros((X.shape[0], n_hidden), dtype=np.int8)
    for l, (W, b) in enumerate(zip(model.Ws[:-1], model.bs[:-1])):
        pre = h @ W + b
        bits[:, off[l]:off[l + 1]] = (pre > 0)
        h = np.maximum(pre, 0.0)
        m = masks[0, off[l]:off[l + 1]]
        h = np.where(m[None, :], h, 0.0 if scheme == "zero" else means[l][None, :])
    logit = (h @ model.Ws[-1] + model.bs[-1])[:, 0]
    return bits, (logit > 0).astype(np.int8)


def _is_function_of(child_bits, parent_bits):
    """True iff child bit is a function of the parent bit pattern over all rows. Returns (ok, table)."""
    table = {}
    for r in range(len(child_bits)):
        key = tuple(int(v) for v in parent_bits[r])
        if table.setdefault(key, int(child_bits[r])) != child_bits[r]:
            return False, None
    return True, table


def handbook_interpretation(model, X, mid, scheme):
    """Returns (valid: bool, gates: list per layer of dict unit -> partial table, out_table).
    Valid iff every circuit unit's bit is a function of its circuit parents' bits and the
    output bit is a function of the last circuit layer's bits."""
    bits, out = circuit_bits(model, X, mid, scheme)
    layers = circuit_layers(mid, model.widths); off = _offsets(model.widths)
    parents = X.astype(np.int8)
    gates = []
    for l, units in enumerate(layers):
        g = {}
        for u in units:
            ok, table = _is_function_of(bits[:, off[l] + u], parents)
            if not ok:
                return False, None, None
            g[int(u)] = table
        gates.append(g)
        parents = bits[:, [off[l] + u for u in units]] if len(units) else np.zeros((X.shape[0], 0), np.int8)
    ok, out_table = _is_function_of(out, parents)
    return ok, gates, out_table


NAMED = {}


def _register():
    import itertools
    for p in (1, 2, 3):
        pats = list(itertools.product([0, 1], repeat=p))
        def tab(fn): return tuple(fn(*x) for x in pats)
        NAMED[(p, tab(lambda *x: int(all(x))))] = "AND" if p > 1 else "ID"
        NAMED[(p, tab(lambda *x: int(any(x))))] = "OR" if p > 1 else "ID"
        NAMED[(p, tab(lambda *x: int(sum(x) % 2)))] = "XOR" if p > 1 else "ID"
        NAMED[(p, tab(lambda *x: 1 - int(all(x))))] = "NAND" if p > 1 else "NOT"
        NAMED[(p, tab(lambda *x: 1 - int(any(x))))] = "NOR" if p > 1 else "NOT"
        NAMED[(p, tab(lambda *x: 1 - int(sum(x) % 2)))] = "XNOR" if p > 1 else "NOT"
        if p == 3:
            NAMED[(p, tab(lambda *x: int(sum(x) >= 2)))] = "MAJ"
        for k in range(1, p + 1):
            NAMED.setdefault((p, tab(lambda *x, k=k: int(sum(x) >= k))), f"THR{k}")
        for i in range(p):
            NAMED.setdefault((p, tab(lambda *x, i=i: x[i])), f"x{i}")
            NAMED.setdefault((p, tab(lambda *x, i=i: 1 - x[i])), f"not_x{i}")
        NAMED.setdefault((p, tuple(0 for _ in pats)), "CONST0"); NAMED.setdefault((p, tuple(1 for _ in pats)), "CONST1")
_register()


def gate_name(table):
    """Name of the simplest named gate consistent with a partial table (observed patterns only)."""
    if not table:
        return "CONST"
    p = len(next(iter(table)))
    if p > 3:
        return f"partial({len(table)} pats)"
    import itertools
    pats = list(itertools.product([0, 1], repeat=p))
    for (pp, full), name in NAMED.items():
        if pp != p:
            continue
        if all(full[pats.index(k)] == v for k, v in table.items()):
            return name
    return "unnamed"


def algorithm_string(gates, out_table, layers):
    parts = []
    for l, g in enumerate(gates):
        parts.append("[" + ",".join(f"u{u}:{gate_name(t)}" for u, t in sorted(g.items())) + "]")
    parts.append("out:" + gate_name(out_table))
    return "->".join(parts)
