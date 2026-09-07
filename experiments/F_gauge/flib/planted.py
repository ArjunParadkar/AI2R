"""Hand-built networks with known circuits for the F2 / F4 kill tests."""
import numpy as np
from .models import MLP


def xor3_single():
    """XOR3 with exactly one 2-hidden-layer circuit, every pre-activation >= 0.5 from 0 on the
    truth table, and the output built so that NO proper subset is valid under zero ablation:
    Layer 1: a = relu(x1-x2-.5), b = relu(x2-x1-.5) (a+b = .5*(x1 xor x2)), e = relu(x3-.5).
    Layer 2 computes the COMPLEMENT c'+d' = .5*[p == x3]: c' = relu(2a+2b+2e-1.5), d' = relu(.5-2a-2b-2e).
    Output: -2(c'+d') + 0.5  = +0.5 (xor true) / -0.5 (xor false). Dropping the bias (the const unit
    added by ``with_bias_unit``) leaves {0, -1}: 0 is not > 0, so the circuit is invalid without it."""
    W1 = np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]); b1 = np.array([-0.5, -0.5, -0.5])
    W2 = np.array([[2.0, -2.0], [2.0, -2.0], [2.0, -2.0]]); b2 = np.array([-1.5, 0.5])
    W3 = np.array([[-2.0], [-2.0]]); b3 = np.array([0.5])
    return MLP([W1, W2, W3], [b1, b2, b3])


def _xor3_alg_A():
    return xor3_single()


def _xor3_alg_B():
    """XOR via OR-minus-AND: u1 = relu(x1+x2-.5) in {0,.5,1.5}, u2 = relu(x1+x2-1.5) in {0,.5};
    2p = 2u1 - 6u2 = x1 xor x2; e = relu(x3-.5); complement units as in alg A."""
    W1 = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]); b1 = np.array([-0.5, -1.5, -0.5])
    W2 = np.array([[2.0, -2.0], [-6.0, 6.0], [2.0, -2.0]]); b2 = np.array([-1.5, 0.5])
    W3 = np.array([[-2.0], [-2.0]]); b3 = np.array([0.5])
    return MLP([W1, W2, W3], [b1, b2, b3])


def _xor3_alg_C():
    """Threshold decomposition of parity: s = x1+x2+x3; t_k = relu(s - k + .5), k = 1, 2, 3;
    t1 - 3 t2 + 5 t3 = .5 * parity(s); complement c' = relu(.25 - (t1 - 3t2 + 5t3)) in {.25, 0};
    output -4c' + .5."""
    W1 = np.ones((3, 3)); b1 = np.array([-0.5, -1.5, -2.5])
    W2 = np.array([[-1.0], [3.0], [-5.0]]); b2 = np.array([0.25])
    W3 = np.array([[-4.0]]); b3 = np.array([0.5])
    return MLP([W1, W2, W3], [b1, b2, b3])


def with_bias_unit(m):
    """Append a constant-1 chain (one unit per hidden layer) carrying the output bias as a weight,
    so the algorithm's contribution to the logit is self-contained (+/-margin, zero bias)."""
    Ws, bs = [W.copy() for W in m.Ws], [b.copy() for b in m.bs]
    d = m.depth
    for l in range(d):
        fan_in, w = Ws[l].shape
        col = np.zeros((fan_in + (1 if l > 0 else 0), 1))
        if l > 0:
            Ws[l] = np.vstack([Ws[l], np.zeros((1, w))])       # previous const unit -> nothing else
            col[-1, 0] = 1.0                                   # const chain
        Ws[l] = np.hstack([Ws[l], col]); bs[l] = np.append(bs[l], 1.0 if l == 0 else 0.0)
    Ws[d] = np.vstack([Ws[d], bs[d][None, :]]); bs[d] = np.zeros_like(bs[d])
    return MLP(Ws, bs)


def parallel(algs, rng, seed=0):
    """Wire the algorithms in parallel (block-diagonal), each duplicated once by a permuted,
    rescaled copy. Each copy carries its own bias unit, so a copy alone gives the right sign
    (+/-0.5), a zero-ablated copy contributes 0 and a mean-ablated copy contributes its mean
    (0 for a balanced target). Output bias 0. Returns (MLP, n_copies)."""
    from .gauge import apply_gauge
    rng = np.random.default_rng(seed)
    copies = []
    for m in algs:
        m = with_bias_unit(m)
        perms = [rng.permutation(w) for w in m.widths]
        scales = [np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=w)) for w in m.widths]
        copies += [m, apply_gauge(m, perms, scales)]
    depth = copies[0].depth
    Ws, bs = [], []
    blocks_of = [[(sum(c.widths[l] for c in copies[:j]), sum(c.widths[l] for c in copies[:j + 1])) for l in range(depth)]
                 for j in range(len(copies))]
    for l in range(depth + 1):
        blocks = [c.Ws[l] for c in copies]
        if l == 0:
            W = np.hstack(blocks)
        elif l == depth:
            W = np.vstack(blocks)
        else:
            W = np.zeros((sum(b.shape[0] for b in blocks), sum(b.shape[1] for b in blocks)))
            r = c = 0
            for B in blocks:
                W[r:r + B.shape[0], c:c + B.shape[1]] = B; r += B.shape[0]; c += B.shape[1]
        Ws.append(W)
        bs.append(np.concatenate([cp.bs[l] for cp in copies]) if l < depth else np.zeros(1))
    return MLP(Ws, bs), blocks_of


def planted_k(k, seed=0):
    """Returns (MLP, blocks) with blocks[j][l] = (lo, hi) unit range of copy j in hidden layer l."""
    algs = [_xor3_alg_A(), _xor3_alg_B(), _xor3_alg_C()][:k]
    return parallel(algs, None, seed)
