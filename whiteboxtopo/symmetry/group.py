"""The gauge group G of a network, as one object every experiment shares (handbook S1).

A *renaming* g ∈ G is a transformation of the parameters that leaves the
function computed on every input exactly unchanged. Three families are
implemented, each exact (float round-off only):

* **permutations** — hidden units within a layer (MLP / transformer MLP),
  attention heads within a layer;
* **scalings** — positive per-unit rescaling through a positively homogeneous
  nonlinearity (ReLU): (w_in, b) -> a (w_in, b), w_out -> w_out / a; and
  LayerNorm-compatible rescalings: (gamma_i, beta_i) -> a_i (gamma_i, beta_i)
  with row i of the following linear map divided by a_i;
* **rotations** — within-head invertible changes of basis: (W_Q, W_K) ->
  (W_Q R, W_K R^{-T}), (W_V, W_O) -> (W_V R, R^{-1} W_O) (biases likewise).

``GaugeElement`` is a plain container; each backend (``mlp``, ``transformer``,
``gpt2``) knows how to sample and apply one for its model type. The public
functions dispatch on the model type.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np

KINDS = ("perm", "scale", "rot", "combined")


@dataclass
class GaugeElement:
    """One renaming. Keys are backend-specific layer identifiers.

    ``perms[key]`` — new unit order (``np.ndarray`` of ints);
    ``scales[key]`` — positive per-unit factors;
    ``rots[key]`` — list (per head) of invertible (d_head, d_head) matrices.
    Missing keys mean "identity there"."""
    perms: Dict[Any, np.ndarray] = field(default_factory=dict)
    scales: Dict[Any, np.ndarray] = field(default_factory=dict)
    rots: Dict[Any, list] = field(default_factory=dict)
    kind: str = "combined"

    def is_identity(self) -> bool:
        return not self.perms and not self.scales and not self.rots

    def log_scale_summary(self) -> Tuple[float, float]:
        """(mean, max) of |log a| over all scaling factors (0, 0 if none)."""
        if not self.scales:
            return 0.0, 0.0
        la = np.concatenate([np.abs(np.log(np.asarray(s, dtype=np.float64))).ravel() for s in self.scales.values()])
        return float(la.mean()), float(la.max())


def _backend(model):
    from . import mlp as _mlp
    if _mlp.handles(model):
        return _mlp
    try:
        from . import transformer as _tf
        if _tf.handles(model):
            return _tf
    except ImportError:
        pass
    try:
        from . import gpt2 as _g2
        if _g2.handles(model):
            return _g2
    except ImportError:
        pass
    try:
        from . import llama as _ll
        if _ll.handles(model):
            return _ll
    except ImportError:
        pass
    try:
        from . import neox as _nx
        if _nx.handles(model):
            return _nx
    except ImportError:
        pass
    raise TypeError(f"no symmetry backend for {type(model).__name__}; supported: ReLU MLPs (numpy .Ws/.bs, "
                    "torch nn.Sequential Linear/ReLU), whiteboxtopo.symmetry.transformer.ToyTransformer, "
                    "HF GPT2LMHeadModel, Llama/Gemma-2/Qwen2/Mistral ForCausalLM, "
                    "GPTNeoXForCausalLM / GPTNeoXModel (Pythia)")


def describe(model) -> Dict[str, Any]:
    """What G looks like for this model: which layers/heads exist, and the *known* continuous
    dimension of G (scalings + rotations) — the number ``hidden()`` compares against."""
    return _backend(model).describe(model)


def permutations(model, rng=None, seed: int = 0) -> GaugeElement:
    """A random permutation-only element (hidden units within layers; heads within layers)."""
    rng = np.random.default_rng(seed) if rng is None else rng
    return _backend(model).sample(model, rng, kind="perm")


def scalings(model, rng=None, seed: int = 0, log_range: Tuple[float, float] = (0.1, 10.0)) -> GaugeElement:
    """A random scaling-only element, per-unit factors log-uniform in ``log_range``."""
    rng = np.random.default_rng(seed) if rng is None else rng
    return _backend(model).sample(model, rng, kind="scale", log_range=log_range)


def rotations(model, rng=None, seed: int = 0, orthogonal: bool = True) -> GaugeElement:
    """A random within-head-rotation element (attention models only; identity for MLPs)."""
    rng = np.random.default_rng(seed) if rng is None else rng
    return _backend(model).sample(model, rng, kind="rot", orthogonal=orthogonal)


def random_element(model, rng=None, seed: int = 0, kind: str = "combined",
                   log_range: Tuple[float, float] = (0.1, 10.0), orthogonal: bool = True) -> GaugeElement:
    """A random element of the chosen sub-group (``perm`` | ``scale`` | ``rot`` | ``combined``)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    rng = np.random.default_rng(seed) if rng is None else rng
    return _backend(model).sample(model, rng, kind=kind, log_range=log_range, orthogonal=orthogonal)


def uniform_scaling(model, alpha: float, layer=None) -> GaugeElement:
    """The element that rescales every unit of one hidden layer (or every hidden layer if
    ``layer is None``) by the same factor ``alpha`` — used to measure covariance laws (Z5)."""
    return _backend(model).uniform_scaling(model, float(alpha), layer)


def apply(model, g: GaugeElement):
    """The transformed model (same type as ``model``); ``model`` is not modified."""
    return _backend(model).apply(model, g)


def inverse_on_units(g: GaugeElement, key, values: np.ndarray, axis: int = 0, scale_power: float = 0.0) -> np.ndarray:
    """Undo a renaming on a per-unit array: un-permute ``values`` along ``axis`` for layer ``key``
    (back to the base model's unit order) and then, if ``scale_power != 0``, divide by
    ``a ** scale_power`` (``a`` indexed in base order). Used by ``align="gauge"``.
    Convention (``apply``): scalings act first, then the permutation, so transformed unit j is
    base unit p[j] scaled by a[p[j]]."""
    v = np.asarray(values, dtype=np.float64)
    if key in g.perms:
        p = np.asarray(g.perms[key])
        inv = np.empty_like(p); inv[p] = np.arange(len(p))
        v = np.take(v, inv, axis=axis)
    if key in g.scales and scale_power != 0.0:
        a = np.asarray(g.scales[key], dtype=np.float64) ** scale_power
        shape = [1] * v.ndim; shape[axis] = -1
        v = v / a.reshape(shape)
    return v


def output_deviation(model, g: GaugeElement, inputs, forward=None) -> float:
    """max |f_g(x) - f(x)| over ``inputs`` — the identity check every sampler is tested with."""
    be = _backend(model)
    fwd = forward or be.forward
    y0 = np.asarray(fwd(model, inputs), dtype=np.float64)
    y1 = np.asarray(fwd(be.apply(model, g), inputs), dtype=np.float64)
    return float(np.max(np.abs(y1 - y0)))
