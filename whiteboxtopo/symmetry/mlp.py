"""ReLU-MLP backend for the symmetry group (wraps ``whiteboxtopo.invariants.model``)."""
from __future__ import annotations
import numpy as np
from ..invariants.model import as_relu_mlp, apply_gauge, ReluMLP, _is_torch_sequential
from .group import GaugeElement


def handles(model) -> bool:
    if isinstance(model, ReluMLP) or (hasattr(model, "Ws") and hasattr(model, "bs")):
        return True
    if _is_torch_sequential(model):
        try:
            as_relu_mlp(model); return True
        except Exception:
            return False
    return False


def describe(model):
    m = as_relu_mlp(model)
    return {"type": "relu_mlp", "widths": m.widths, "keys": list(range(m.depth)),
            "known_dim": int(sum(m.widths)),           # one scaling generator per hidden unit
            "n_params": int(sum(W.size + b.size for W, b in zip(m.Ws, m.bs)))}


def sample(model, rng, kind="combined", log_range=(0.1, 10.0), orthogonal=True) -> GaugeElement:
    m = as_relu_mlp(model)
    g = GaugeElement(kind=kind)
    lo, hi = np.log(log_range[0]), np.log(log_range[1])
    for l, w in enumerate(m.widths):
        if kind in ("perm", "combined"):
            g.perms[l] = rng.permutation(w)
        if kind in ("scale", "combined"):
            g.scales[l] = np.exp(rng.uniform(lo, hi, size=w))
    return g


def uniform_scaling(model, alpha, layer=None) -> GaugeElement:
    m = as_relu_mlp(model)
    g = GaugeElement(kind="scale")
    for l, w in enumerate(m.widths):
        if layer is None or l == layer:
            g.scales[l] = np.full(w, alpha)
    return g


def apply(model, g: GaugeElement):
    m = as_relu_mlp(model)
    perms = [g.perms.get(l, np.arange(w)) for l, w in enumerate(m.widths)] if g.perms else None
    scales = [g.scales.get(l, np.ones(w)) for l, w in enumerate(m.widths)] if g.scales else None
    if perms is None and scales is None:
        return model
    return apply_gauge(model, perms, scales)


def forward(model, X):
    return as_relu_mlp(model).logits(np.asarray(X, dtype=np.float64))


def param_vector(model):
    m = as_relu_mlp(model)
    return np.concatenate([p.ravel() for p in list(m.Ws) + list(m.bs)])
