"""Hand-built networks with known circuits for the F2 / F4 kill tests."""
import numpy as np
from .models import MLP


def xor3_single():
    """XOR3 with exactly one 2-hidden-layer circuit: layer 1 computes p = x1 XOR x2 via two units
    (x1∧¬x2, ¬x1∧x2), layer 2 computes p XOR x3 via two units ((u1+u2)∧¬x3, ¬(u1+u2)∧x3).
    Because there are no spare units, the only valid circuit is the full set."""
    # layer 1: units a = relu(x1 - x2), b = relu(x2 - x1); a + b = x1 xor x2
    W1 = np.array([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]]); b1 = np.zeros(2)
    # layer 2: p = a + b; c = relu(p - x3), d = relu(x3 - p)  -> c + d = p xor x3
    # need x3 in layer 2: route it through layer 1 with a third unit e = relu(x3)
    W1 = np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]); b1 = np.zeros(3)
    W2 = np.array([[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]); b2 = np.zeros(2)
    W3 = np.array([[1.0], [1.0]]); b3 = np.array([-0.5])
    return MLP([W1, W2, W3], [b1, b2, b3])


def _xor3_alg_A():
    """(x1∧¬x2)∨(¬x1∧x2) then xor x3 — the network above, as sub-blocks."""
    return xor3_single()


def _xor3_alg_B():
    """XOR via (x1∨x2)∧¬(x1∧x2): u1 = relu(x1+x2) (OR-ish: 0,1,2), u2 = relu(x1+x2-1) (AND-ish);
    p = u1 - 2*u2 = x1 xor x2 ; then same trick with x3."""
    W1 = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]); b1 = np.array([0.0, -1.0, 0.0])
    # p = u1 - 2 u2 ; c = relu(p - x3), d = relu(x3 - p)
    W2 = np.array([[1.0, -1.0], [-2.0, 2.0], [-1.0, 1.0]]); b2 = np.zeros(2)
    W3 = np.array([[1.0], [1.0]]); b3 = np.array([-0.5])
    return MLP([W1, W2, W3], [b1, b2, b3])


def _xor3_alg_C():
    """Threshold decomposition of parity: s = x1+x2+x3; parity = [s>=1] - [s>=2] + [s>=3];
    units t1=relu(s), t2=relu(s-1), t3=relu(s-2) -> t1 - 2 t2 + 2 t3 = parity (values 0/1)."""
    W1 = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]); b1 = np.array([0.0, -1.0, -2.0])
    W2 = np.array([[1.0, 0.0], [-2.0, 0.0], [2.0, 0.0]]); b2 = np.array([0.0, 0.0])   # second unit dead (bias 0, weight 0) -> relu(0)=0
    # make the second layer-2 unit carry nothing but stay "in circuit" semantics: give it a real role: copy parity again
    W2 = np.array([[1.0, 1.0], [-2.0, -2.0], [2.0, 2.0]]); b2 = np.zeros(2)
    W3 = np.array([[0.5], [0.5]]); b3 = np.array([-0.5])
    return MLP([W1, W2, W3], [b1, b2, b3])


def parallel(algs, rng, seed=0):
    """Wire the algorithms in parallel (block-diagonal), each duplicated once by a permuted,
    rescaled copy; the output sums all copies' logits / n_copies so the truth table holds.
    Returns the MLP and the list of block unit ranges per copy."""
    from .gauge import apply_gauge
    rng = np.random.default_rng(seed)
    copies = []
    for m in algs:
        perms = [rng.permutation(w) for w in m.widths]
        scales = [np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=w)) for w in m.widths]
        copies += [m, apply_gauge(m, perms, scales)]
    depth = copies[0].depth; n_in = copies[0].n_in
    Ws, bs = [], []
    for l in range(depth + 1):
        blocks = [c.Ws[l] for c in copies]
        if l == 0:
            W = np.hstack(blocks)
        elif l == depth:
            W = np.vstack(blocks) / len(copies)
        else:
            W = np.zeros((sum(b.shape[0] for b in blocks), sum(b.shape[1] for b in blocks)))
            r = c = 0
            for B in blocks:
                W[r:r + B.shape[0], c:c + B.shape[1]] = B; r += B.shape[0]; c += B.shape[1]
        Ws.append(W)
        bs.append(np.concatenate([cp.bs[l] for cp in copies]) if l < depth else np.mean([cp.bs[l] for cp in copies], 0))
    return MLP(Ws, bs), len(copies)


def planted_k(k, seed=0):
    algs = [_xor3_alg_A(), _xor3_alg_B(), _xor3_alg_C()][:k]
    return parallel(algs, None, seed)
