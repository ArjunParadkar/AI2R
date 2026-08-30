"""The gauge-invariant metrics that survived E2 (whitebox-v2), and nothing else.

Each function takes ``(model, X)`` — a ReLU MLP (see ``model.py`` for the
accepted types) and an input batch ``X`` of shape (N, n_in) — and returns a
quantity that is exactly invariant (orbit CV at the 1e-15 float floor on 687
trained and 840 untrained nets, E2 table) under hidden-unit permutations and
per-unit positive scalings. The unit tests re-check every one of them by
orbit sampling (``tests/test_invariants.py``).

| function                 | E2 id | quantity                                                          |
|--------------------------|-------|-------------------------------------------------------------------|
| ``logits``               | I1    | the network outputs on X (sanity: f itself)                       |
| ``input_jacobian``       | I2    | d output / d x at each input (the per-region linear map)          |
| ``weight_product``       | I3    | W0 W1 ... Wd — the *linear part* only (equals the end-to-end map  |
|                          |       | only inside the all-active region; this is a sanity invariant)    |
| ``standardized_tc``      | I8    | TC(H0), TC(H1) of the per-unit standardised activation cloud of   |
|                          |       | each hidden layer (units z-scored over X; dead units dropped)     |
| ``sorted_gradxact``      | N2    | gradient x activation attribution per unit, summed over X, SORTED  |
|                          |       | within each layer (the multiset, not the assignment to indices)   |
| ``sorted_ablation``      | N3    | zero-ablation effect per unit, sorted within each layer           |
| ``gate_multiset``        | N6    | per-layer sorted multiset of binarised unit truth tables [a > 0]  |

What is deliberately NOT here (E2 "not invariant" column; documentation, not
API — computing them is easy, the point is that they are not well-defined
statements about the function):

* TC / beta / persistent entropy of the RAW activation cloud (I5-I7): exactly
  permutation-invariant but broken by per-unit scaling, orbit CV 1.4-3.0 for
  TC — larger than TC's variation between independently trained nets
  (0.6-1.6). Under *uniform* scaling TC is covariant of degree 2 and
  beta/PE invariant, but uniform scaling is a measure-zero subgroup.
* singular-value spectra of individual layer maps W_l (I4): scale CV ~ 1.
* per-neuron |activation| (N1) and raw weight magnitudes / incoming norms
  (N4): broken by scaling even after sorting (CV 0.9 / 1.1) — no function-
  level meaning at all.
* any per-neuron vector read at its raw index — grad x act, ablation, gate
  of neuron i (N2, N3, N6 raw): the VALUES are scaling-invariant per unit
  (alpha cancels analytically; measured 1e-15), the LABELS are relabelled
  by permutation. "The largest ablation effect is 2.3" is well-defined;
  "neuron 7 is the most important" is not (raw agreement 0.17).
* "top neuron" by |activation| (N5): not even scaling-stable (agreement 0.30).

Multi-output networks: attribution and ablation are computed for the summed
output ``sum_k f_k`` (grad x act) and for the output vector norm (ablation).
"""
from __future__ import annotations
from typing import Dict
import numpy as np

from .model import as_relu_mlp
from ..core.persistence import compute_diagrams
from ..metrics.tc import topological_complexity


def logits(model, X) -> np.ndarray:
    """I1: outputs on X, shape (N, n_out)."""
    return as_relu_mlp(model).logits(X)


def input_jacobian(model, X) -> np.ndarray:
    """I2: d f_k / d x at every row of X, shape (N, n_out, n_in); exact for a ReLU net
    inside each activation region."""
    m = as_relu_mlp(model)
    masks, _, _ = m.forward(X)
    N = np.asarray(X).shape[0]
    # G: (N, n_out, fan_in of current layer)
    G = np.broadcast_to(m.Ws[-1].T[None], (N, m.n_out, m.Ws[-1].shape[0])).copy()
    for l in range(m.depth - 1, -1, -1):
        G = G * masks[l][:, None, :]
        G = G @ m.Ws[l].T
    return G


def weight_product(model) -> np.ndarray:
    """I3: W0 W1 ... Wd, the linear part only (shape (n_in, n_out))."""
    m = as_relu_mlp(model)
    P = m.Ws[0]
    for W in m.Ws[1:]:
        P = P @ W
    return P


def standardize_units(A: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Per-unit standardisation of a cloud (rows = points, columns = units): each
    column z-scored over the rows; columns with zero spread (dead units) dropped.
    This removes exactly the per-unit scale (and shift) that the ReLU gauge acts
    on; it is a different metric from the raw cloud, not a normalisation of it."""
    A = np.asarray(A, dtype=np.float64)
    sd = A.std(0)
    keep = sd > eps
    if not keep.any():
        return np.zeros((A.shape[0], 1))
    return (A[:, keep] - A[:, keep].mean(0)) / sd[keep]


def standardized_tc(model, X, layers=None, thresh_percentile: float = 95.0, backend: str = "ripser") -> np.ndarray:
    """I8: TC(H0), TC(H1) of the per-unit standardised post-ReLU cloud of each hidden
    layer on X, shape (n_layers, 2). Uses the library's diagram convention
    (VR, 95th-percentile threshold)."""
    m = as_relu_mlp(model)
    _, acts, _ = m.forward(X)
    ls = range(m.depth) if layers is None else layers
    out = []
    for l in ls:
        D = compute_diagrams(standardize_units(acts[l]), None, maxdim=1, thresh_percentile=thresh_percentile, backend=backend)
        out.append([topological_complexity(D[0]), topological_complexity(D[1])])
    return np.asarray(out, dtype=np.float64)


def _hidden_gradients(m, masks):
    """d (sum_k f_k) / d a_l for each hidden layer (post-ReLU), list of (N, w_l)."""
    N = masks[0].shape[0]
    g = np.broadcast_to(m.Ws[-1].sum(1)[None], (N, m.Ws[-1].shape[0])).copy()
    grads = [None] * m.depth
    grads[m.depth - 1] = g
    for l in range(m.depth - 1, 0, -1):
        g = (g * masks[l]) @ m.Ws[l].T
        grads[l - 1] = g
    return grads


def _sorted_per_layer(vec, widths):
    out, off = [], 0
    for w in widths:
        out.append(np.sort(vec[off:off + w])); off += w
    return np.concatenate(out)


def gradxact(model, X) -> np.ndarray:
    """N2 raw: per-unit gradient x activation attribution to sum_k f_k, summed over X
    (global unit index). NOT invariant under permutation — use ``sorted_gradxact``."""
    m = as_relu_mlp(model)
    masks, acts, _ = m.forward(X)
    grads = _hidden_gradients(m, masks)
    return np.concatenate([(g * a).sum(0) for g, a in zip(grads, acts)])


def sorted_gradxact(model, X) -> np.ndarray:
    """N2 sorted: the per-layer sorted multiset of grad x act attributions."""
    return _sorted_per_layer(gradxact(model, X), as_relu_mlp(model).widths)


def ablation(model, X) -> np.ndarray:
    """N3 raw: mean over X of ||f(x) - f(x; unit j zeroed)|| per hidden unit (global
    index). NOT invariant under permutation — use ``sorted_ablation``."""
    m = as_relu_mlp(model)
    X = np.asarray(X, dtype=np.float64)
    base = m.logits(X)
    out = []
    for l in range(m.depth):
        for j in range(m.widths[l]):
            h = X
            for k, (W, b) in enumerate(zip(m.Ws[:-1], m.bs[:-1])):
                h = np.maximum(h @ W + b, 0.0)
                if k == l:
                    h = h.copy(); h[:, j] = 0.0
            out.append(np.mean(np.linalg.norm(h @ m.Ws[-1] + m.bs[-1] - base, axis=1)))
    return np.asarray(out)


def sorted_ablation(model, X) -> np.ndarray:
    """N3 sorted: the per-layer sorted multiset of zero-ablation effects."""
    return _sorted_per_layer(ablation(model, X), as_relu_mlp(model).widths)


def gate_bits(model, X) -> np.ndarray:
    """N6 raw: [a > 0] per hidden unit over X, shape (N, n_hidden), int8 — the unit's
    binarised truth table. Scaling-invariant per unit, relabelled by permutation."""
    m = as_relu_mlp(model)
    _, acts, _ = m.forward(X)
    return np.concatenate([(a > 0).astype(np.int8) for a in acts], axis=1)


def gate_multiset(model, X) -> np.ndarray:
    """N6 sorted: per hidden layer, the columns of ``gate_bits`` sorted
    lexicographically — "this layer contains an AND, two ORs and a constant" —
    shape (N, n_hidden). Exactly invariant (CV 0)."""
    m = as_relu_mlp(model)
    B = gate_bits(model, X)
    cols, off = [], 0
    for w in m.widths:
        blk = B[:, off:off + w]
        cols.append(blk[:, np.lexsort(blk[::-1])]); off += w
    return np.concatenate(cols, axis=1)


INVARIANTS = {
    "logits": ("I1", logits), "input_jacobian": ("I2", input_jacobian), "weight_product": ("I3", lambda m, X: weight_product(m)),
    "standardized_tc": ("I8", standardized_tc), "sorted_gradxact": ("N2", sorted_gradxact),
    "sorted_ablation": ("N3", sorted_ablation), "gate_multiset": ("N6", gate_multiset),
}


def invariant_quantities(model, X, with_topology: bool = True) -> Dict[str, np.ndarray]:
    """All shipped invariants in one call (for ``audit_metrics``)."""
    q = {k: f(model, X) for k, (_, f) in INVARIANTS.items() if with_topology or k != "standardized_tc"}
    return q
