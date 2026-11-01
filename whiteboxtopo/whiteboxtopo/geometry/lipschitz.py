"""Local Lipschitz structure of a ReLU MLP from its exact path product (SPEC 2.11).

On the open region where the ReLU gates are constant the network is exactly
affine and its Jacobian is the path product

    A(x) = W_L D_{L-1}(x) W_{L-1} ... D_1(x) W_1 ,

``D_l(x)`` the 0/1 gate mask of hidden layer l. ``Lip(x) = ||A(x)||_2`` is then
the exact local Lipschitz constant on that region, and
``||A(x)-A(x')||_2 / ||x-x'||`` over the k nearest neighbours measures how fast
it changes across region boundaries.

Null / anchor it was validated against (tests/test_geometry.py): the finite-
difference Jacobian at points in the interior of a gate region (relative error
< 1e-6), and exact invariance under random elements of the ReLU-MLP gauge
group ``whiteboxtopo.invariants.model.apply_gauge`` (max relative deviation
< 1e-9) - A(x) is a function of the network, not of its parameterisation.

Failure boundary. (1) Exact only for ReLU MLPs: any other non-linearity,
LayerNorm, attention or gating makes the path product input-dependent in a way
this does not model, and ``as_relu_mlp`` will refuse those. (2) At a gate
boundary (a pre-activation exactly 0) the gate mask is a choice and A(x) is the
one-sided Jacobian; the finite-difference anchor disagrees there by design.
(3) The neighbour variation is a difference quotient over whatever points you
passed - it is not a bound on the derivative of A, and it is dominated by how
densely X samples the region boundaries.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..invariants.model import as_relu_mlp


@dataclass
class LipResult:
    """Exact local Lipschitz field of a ReLU MLP (SPEC 2.11)."""
    lip: np.ndarray            # (n,) ||A(x)||_2
    variation: np.ndarray      # (n,) max_j ||A(x)-A(x_j)||_2 / ||x-x_j|| over the k nearest neighbours
    mean_variation: np.ndarray  # (n,) the same quotient averaged over neighbours
    A: np.ndarray              # (n, n_out, n_in) the path products themselves
    k: int

    def to_dict(self) -> Dict[str, object]:
        return {"n": int(self.lip.size), "k": int(self.k),
                "lip_median": float(np.median(self.lip)), "lip_max": float(self.lip.max()),
                "variation_median": float(np.median(self.variation)),
                "lip": self.lip.tolist(), "variation": self.variation.tolist(),
                "mean_variation": self.mean_variation.tolist()}


def path_products(model, X: np.ndarray) -> np.ndarray:
    """``A(x)`` for every row of X, shape (n, n_out, n_in). Exact, no autodiff."""
    m = as_relu_mlp(model)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    masks, _, _ = m.forward(X)
    M = np.broadcast_to(m.Ws[0], (X.shape[0],) + m.Ws[0].shape).copy()   # (n, n_in, w0)
    for l in range(m.depth):
        M = M * masks[l][:, None, :]
        M = M @ m.Ws[l + 1]
    return np.transpose(M, (0, 2, 1))                                    # (n, n_out, n_in)


def lipschitz_field(model, X, k: int = 15) -> LipResult:
    """Exact local Lipschitz constant and its neighbour variation (SPEC 2.11).

    ``model`` is anything ``whiteboxtopo.invariants.model.as_relu_mlp`` accepts
    (numpy ``ReluMLP``, an object with ``.Ws``/``.bs``, or a torch
    ``nn.Sequential`` of Linear/ReLU - torch is imported lazily there, never
    here). Deterministic: no randomness.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    A = path_products(model, X)
    lip = np.linalg.svd(A, compute_uv=False)[:, 0]
    n = X.shape[0]
    k = int(min(k, n - 1))
    if k < 1:
        z = np.zeros(n)
        return LipResult(lip=lip, variation=z, mean_variation=z, A=A, k=0)
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    dA = A[:, None, :, :] - A[idx]                                       # (n, k, n_out, n_in)
    num = np.linalg.svd(dA.reshape(n * k, A.shape[1], A.shape[2]), compute_uv=False)[:, 0].reshape(n, k)
    q = num / np.where(dist > 0, dist, np.inf)
    return LipResult(lip=lip, variation=q.max(axis=1), mean_variation=q.mean(axis=1), A=A, k=k)
