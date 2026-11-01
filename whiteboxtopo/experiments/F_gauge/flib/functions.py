"""Target Boolean functions. 3-input: XOR3, MAJ3, AND_OR3. 4-input: the fixed list of 16
non-degenerate functions (every input matters; asserted by ``check_nondegenerate``)."""
import itertools
import numpy as np


def all_inputs(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.float64)


def _f3():
    X = all_inputs(3); a, b, c = X.T.astype(int)
    return {"XOR3": a ^ b ^ c, "MAJ3": ((a + b + c) >= 2).astype(int), "AND_OR3": (a & b) | c}


def _f4():
    X = all_inputs(4); a, b, c, d = X.T.astype(int)
    s = a + b + c + d
    F = {
        "PARITY4": a ^ b ^ c ^ d,
        "THR2": (s >= 2).astype(int),
        "THR3": (s >= 3).astype(int),
        "AND4": a & b & c & d,
        "OR4": a | b | c | d,
        "XORPAIR_AND": (a ^ b) & (c ^ d),
        "ANDPAIR_XOR": (a & b) ^ (c & d),
        "ANDPAIR_OR": (a & b) | (c & d),
        "ORPAIR_AND": (a | b) & (c | d),
        "MUX_XOR": np.where(a == 1, b, c) ^ d,
        "EQPAIR_AND": (1 - (a ^ b)) & (1 - (c ^ d)),
        "ONEHOT": (s == 1).astype(int),
        "EXACT2": (s == 2).astype(int),
        "XOR12_AND34": (a ^ b) ^ (c & d),
        "AND123_OR4": (a & b & c) | d,
        "NANDTREE": 1 - ((1 - (a & b)) & (1 - (c & d))),
    }
    return {k: v.astype(int) for k, v in F.items()}


FUNCTIONS = {**_f3(), **_f4()}
NAMES3 = list(_f3()); NAMES4 = list(_f4())


def truth_table(name):
    y = FUNCTIONS[name]; n = 3 if name in NAMES3 else 4
    return all_inputs(n), y.astype(np.int8)


def check_nondegenerate():
    bad = []
    for name in FUNCTIONS:
        X, y = truth_table(name); n = X.shape[1]
        for i in range(n):
            # flipping input i must change y for at least one x
            Xf = X.copy(); Xf[:, i] = 1 - Xf[:, i]
            idx = (Xf @ (2 ** np.arange(n)[::-1])).astype(int)
            if np.array_equal(y[idx], y):
                bad.append((name, i))
    return bad


assert not check_nondegenerate(), check_nondegenerate()
assert len(NAMES4) == 16
