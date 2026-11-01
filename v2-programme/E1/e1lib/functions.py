import itertools
import numpy as np


def all_inputs(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.float32)


def _rand4():
    rng = np.random.default_rng(4)
    return rng.integers(0, 2, size=16).astype(np.int8)


FUNCTIONS = {
    # name: (n_inputs, truth table over all_inputs(n) order)
    "XOR3": (3, lambda X: X.sum(1) % 2),
    "MAJ3": (3, lambda X: (X.sum(1) >= 2).astype(int)),
    "AND_OR3": (3, lambda X: ((X[:, 0] * X[:, 1]) + X[:, 2] > 0).astype(int)),
    "PARITY4": (4, lambda X: X.sum(1) % 2),
    "THR2_4": (4, lambda X: (X.sum(1) >= 2).astype(int)),
    "RAND4": (4, lambda X: _rand4()),
}


def truth_table(name):
    n, fn = FUNCTIONS[name]
    X = all_inputs(n)
    return X, np.asarray(fn(X), dtype=np.int8)
