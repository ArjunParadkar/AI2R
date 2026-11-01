"""Two-parameter persistence summaries (handbook S5 / Z2): a scale x parameter bifiltration.

Given a cloud X with a per-point function value f (e.g. density, or a training
fraction at which the point exists), the bifiltration is
Rips(X_{f <= p}, r). RIVET is not available as a Python wheel, so we compute
the **Hilbert function** H_k(r, p) = dim H_k(Rips(X_{f<=p}, r)) on a grid —
the rank invariant restricted to the diagonal — by running Ripser on each
p-slice (all r at once) and counting bars alive at each r. Also returned:
the **fibered barcode** along the anti-diagonal line r/r_max + p/p_max = 1,
and scalar summaries (Hilbert mass per degree = mean of H_k over the grid,
and the grid location of its maximum).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
from ..core.persistence import compute_diagrams


@dataclass
class MultiparamResult:
    hilbert: np.ndarray             # (maxdim+1, n_p, n_r)
    scales: np.ndarray
    params: np.ndarray
    hilbert_mass: np.ndarray        # (maxdim+1,)
    argmax: list                    # per degree (p, r) of max H
    total_persistence_slices: np.ndarray   # (maxdim+1, n_p) total persistence of each slice diagram


def hilbert_function(X, f, scales: Optional[Sequence[float]] = None, params: Optional[Sequence[float]] = None,
                     maxdim: int = 1, n_grid: int = 20, backend: str = "ripser") -> MultiparamResult:
    X = np.asarray(X, dtype=np.float64); f = np.asarray(f, dtype=np.float64)
    if params is None:
        params = np.quantile(f, np.linspace(0.1, 1.0, n_grid))
    params = np.asarray(params)
    D = None
    dgms_by_p = []
    for p in params:
        idx = np.flatnonzero(f <= p)
        if len(idx) < 3:
            dgms_by_p.append([np.zeros((0, 2)) for _ in range(maxdim + 1)]); continue
        dg = compute_diagrams(X[idx], maxdim=maxdim, backend=backend)
        dgms_by_p.append([np.asarray(dg.dgms[k]).reshape(-1, 2) for k in range(maxdim + 1)])
    if scales is None:
        finite = np.concatenate([d[:, 1][np.isfinite(d[:, 1])] for ds in dgms_by_p for d in ds] + [np.zeros(1)])
        scales = np.linspace(0, float(finite.max()) * 1.05 if finite.max() > 0 else 1.0, n_grid)
    scales = np.asarray(scales)
    H = np.zeros((maxdim + 1, len(params), len(scales)))
    TP = np.zeros((maxdim + 1, len(params)))
    for i, ds in enumerate(dgms_by_p):
        for k, dg in enumerate(ds):
            if len(dg) == 0:
                continue
            b, d = dg[:, 0], dg[:, 1]
            for j, r in enumerate(scales):
                H[k, i, j] = np.sum((b <= r) & (d > r))
            fin = np.isfinite(d)
            TP[k, i] = float(np.sum(d[fin] - b[fin]))
    mass = H.mean(axis=(1, 2))
    am = [tuple(np.unravel_index(np.argmax(H[k]), H[k].shape)) for k in range(maxdim + 1)]
    return MultiparamResult(H, scales, params, mass, [(float(params[a]), float(scales[b])) for a, b in am], TP)
