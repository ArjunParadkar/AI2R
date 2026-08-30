"""ReLU-MLP adapter, the symmetry group G, and orbit sampling.

Scope (stated, not implied): the symmetry group implemented here is the one
of a plain ReLU MLP —

    G = (hidden-unit permutations within each hidden layer)
        x (per-hidden-unit positive scalings  (w_in, b) -> a (w_in, b), w_out -> w_out / a)

Both leave the network function exactly unchanged. Nothing else is sampled:
E1 found infinitesimal flat directions beyond G in 92 % of small nets, and
those are *not* known to integrate to symmetries, so they are excluded (a
finite step along one would change f at second order). Architectures with
LayerNorm, attention heads, residual streams, gated units or non-ReLU
activations have a different (and larger) group and need their own sampler:
verdicts from this module are "invariant under G", never "invariant".

Supported model objects
-----------------------
* any object with ``.Ws`` / ``.bs`` lists (W[l] of shape (fan_in, fan_out),
  b[l] of shape (fan_out,), ReLU between, last layer linear) — the E1/E2
  numpy ``MLP`` — transformed copies are produced with ``model.copy()``;
* a ``torch.nn.Sequential`` whose modules are ``Linear`` and ``ReLU`` only,
  in alternation and ending in a ``Linear``; transformed copies are new
  ``nn.Sequential`` modules with the same structure, so a user metric written
  for torch keeps working on every orbit representative;
* :class:`ReluMLP` (this module's own container).

``as_relu_mlp(model)`` returns a :class:`ReluMLP` view for computation and
``rebuild(model, Ws, bs)`` returns a transformed model *of the input's type*.
"""
from __future__ import annotations
import copy as _copy
from typing import Iterator, List, Optional, Sequence, Tuple
import numpy as np

GROUPS = ("perm", "scale", "combined")


class ReluMLP:
    """Plain-numpy ReLU MLP: hidden layers 0..d-1 (ReLU), output layer d (linear).
    ``Ws[l]`` has shape (fan_in, fan_out), ``bs[l]`` shape (fan_out,)."""

    def __init__(self, Ws: Sequence[np.ndarray], bs: Sequence[np.ndarray]):
        self.Ws = [np.asarray(W, dtype=np.float64) for W in Ws]
        self.bs = [np.asarray(b, dtype=np.float64) for b in bs]
        if len(self.Ws) != len(self.bs) or len(self.Ws) < 2:
            raise ValueError("need >= 1 hidden layer and one output layer, with one bias per layer")
        for l in range(len(self.Ws) - 1):
            if self.Ws[l].shape[1] != self.Ws[l + 1].shape[0]:
                raise ValueError(f"layer {l} fan_out {self.Ws[l].shape[1]} != layer {l+1} fan_in {self.Ws[l+1].shape[0]}")

    @property
    def depth(self) -> int:
        return len(self.Ws) - 1

    @property
    def widths(self) -> List[int]:
        return [W.shape[1] for W in self.Ws[:-1]]

    @property
    def n_in(self) -> int:
        return self.Ws[0].shape[0]

    @property
    def n_out(self) -> int:
        return self.Ws[-1].shape[1]

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Returns (pre-ReLU masks per hidden layer, post-ReLU activations per hidden layer, outputs (N, n_out))."""
        h = np.asarray(X, dtype=np.float64)
        masks, acts = [], []
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            pre = h @ W + b
            masks.append((pre > 0).astype(np.float64))
            h = np.maximum(pre, 0.0)
            acts.append(h)
        return masks, acts, h @ self.Ws[-1] + self.bs[-1]

    def hidden_acts(self, X):
        return self.forward(X)[1]

    def logits(self, X) -> np.ndarray:
        """(N, n_out) outputs."""
        return self.forward(X)[2]

    def copy(self) -> "ReluMLP":
        return ReluMLP([W.copy() for W in self.Ws], [b.copy() for b in self.bs])

    def __repr__(self):
        return f"ReluMLP(n_in={self.n_in}, widths={self.widths}, n_out={self.n_out})"


# ----------------------------------------------------------------------------- adapters
def _is_torch_sequential(model) -> bool:
    try:
        import torch.nn as nn
    except ImportError:
        return False
    return isinstance(model, nn.Sequential)


def _torch_layers(model):
    import torch.nn as nn
    linears, expect_linear = [], True
    for m in model:
        if isinstance(m, nn.Linear):
            if not expect_linear:
                raise ValueError("nn.Sequential must alternate Linear / ReLU (two Linears in a row is not a ReLU MLP)")
            linears.append(m); expect_linear = False
        elif isinstance(m, nn.ReLU):
            if expect_linear:
                raise ValueError("nn.Sequential must alternate Linear / ReLU (ReLU without a preceding Linear)")
            expect_linear = True
        elif isinstance(m, (nn.Identity, nn.Flatten)):
            continue
        else:
            raise ValueError(f"unsupported module {type(m).__name__}: the ReLU-MLP symmetry group does not apply. "
                             "Other architectures need their own group (see whitebox.invariants docs).")
    if expect_linear or len(linears) < 2:
        raise ValueError("nn.Sequential must end in a Linear layer and have >= 1 hidden ReLU layer")
    return linears


def as_relu_mlp(model) -> ReluMLP:
    """A numpy :class:`ReluMLP` view of ``model`` (copy of the weights)."""
    if isinstance(model, ReluMLP):
        return model
    if _is_torch_sequential(model):
        Ls = _torch_layers(model)
        return ReluMLP([L.weight.detach().cpu().double().numpy().T for L in Ls],
                       [(L.bias.detach().cpu().double().numpy() if L.bias is not None else np.zeros(L.out_features)) for L in Ls])
    if hasattr(model, "Ws") and hasattr(model, "bs"):
        return ReluMLP(model.Ws, model.bs)
    raise TypeError(f"cannot interpret {type(model).__name__} as a ReLU MLP: pass a torch nn.Sequential of Linear/ReLU, "
                    "an object with .Ws/.bs lists, or a whitebox.invariants.ReluMLP")


def rebuild(model, Ws, bs):
    """A transformed model of the same type as ``model`` carrying weights (Ws, bs)."""
    if isinstance(model, ReluMLP):
        return ReluMLP(Ws, bs)
    if _is_torch_sequential(model):
        import torch
        new = _copy.deepcopy(model)
        with torch.no_grad():
            for L, W, b in zip(_torch_layers(new), Ws, bs):
                L.weight.copy_(torch.as_tensor(np.asarray(W).T, dtype=L.weight.dtype))
                if L.bias is not None:
                    L.bias.copy_(torch.as_tensor(np.asarray(b), dtype=L.bias.dtype))
        return new
    new = model.copy() if hasattr(model, "copy") else _copy.deepcopy(model)
    new.Ws = [np.asarray(W, dtype=np.float64) for W in Ws]
    new.bs = [np.asarray(b, dtype=np.float64) for b in bs]
    return new


# ----------------------------------------------------------------------------- the group
def apply_gauge(model, perms: Optional[Sequence[np.ndarray]] = None, scales: Optional[Sequence[np.ndarray]] = None):
    """Apply an element of G. ``perms[l]`` is the new unit order of hidden layer l,
    ``scales[l] > 0`` the per-unit factor. Returns a model of the same type as the
    input with the same function. (Same convention as E1's ``e1lib.gauge.apply_gauge``.)"""
    m = as_relu_mlp(model)
    Ws = [W.copy() for W in m.Ws]; bs = [b.copy() for b in m.bs]
    for l in range(m.depth):
        w = Ws[l].shape[1]
        if scales is not None:
            s = np.asarray(scales[l], dtype=np.float64)
            if s.shape != (w,) or (s <= 0).any():
                raise ValueError(f"scales[{l}] must be {w} positive numbers")
            Ws[l] = Ws[l] * s[None, :]; bs[l] = bs[l] * s; Ws[l + 1] = Ws[l + 1] / s[:, None]
        if perms is not None:
            p = np.asarray(perms[l])
            if sorted(p.tolist()) != list(range(w)):
                raise ValueError(f"perms[{l}] must be a permutation of range({w})")
            Ws[l] = Ws[l][:, p]; bs[l] = bs[l][p]; Ws[l + 1] = Ws[l + 1][p, :]
    return rebuild(model, Ws, bs)


def random_gauge_element(widths: Sequence[int], rng: np.random.Generator, group: str = "combined", log_scale_sd: float = 1.0):
    """Sample (perms, scales) from the chosen group; identity where the group does not act."""
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
    perms = [rng.permutation(w) if group in ("perm", "combined") else np.arange(w) for w in widths]
    scales = [np.exp(rng.normal(0.0, log_scale_sd, size=w)) if group in ("scale", "combined") else np.ones(w) for w in widths]
    return perms, scales


def param_vector(model) -> np.ndarray:
    m = as_relu_mlp(model)
    return np.concatenate([p.ravel() for p in list(m.Ws) + list(m.bs)])


def param_distance(model_a, model_b) -> float:
    """Relative L2 distance in parameter space ||theta_b - theta_a|| / ||theta_a||."""
    a, b = param_vector(model_a), param_vector(model_b)
    return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-300))


def sample_orbit(model, n_reps: int = 100, group: str = "combined", seed: int = 0, log_scale_sd: float = 1.0
                 ) -> Iterator[Tuple[int, object, list, list, float]]:
    """Yield ``(rep_index, transformed_model, perms, scales, rel_param_distance)``.
    Representative 0 is the base model itself with the identity element."""
    widths = as_relu_mlp(model).widths
    rng = np.random.default_rng(seed)
    yield 0, model, [np.arange(w) for w in widths], [np.ones(w) for w in widths], 0.0
    for r in range(1, n_reps):
        perms, scales = random_gauge_element(widths, rng, group, log_scale_sd)
        m = apply_gauge(model, perms, scales)
        yield r, m, perms, scales, param_distance(model, m)
