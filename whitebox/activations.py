"""Pull per-layer activation point clouds out of a torch model on a probe set.

Pooling conventions carried over from the original experiments:
  * 4-D conv maps (N,C,H,W)  -> global average pool -> (N,C)
  * 3-D sequence (N,T,D)      -> attention-mask-weighted mean over T -> (N,D)
  * 2-D                       -> as is
One row per probe example; that row-set is the point cloud.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Union
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None


def _pool(v, mask=None):
    if v.ndim == 4:
        return F.adaptive_avg_pool2d(v, 1).flatten(1)
    if v.ndim == 3:
        if mask is None:
            return v.mean(1)
        m = mask.unsqueeze(-1).to(v.dtype)
        return (v * m).sum(1) / m.sum(1).clamp(min=1)
    return v.flatten(1)


class ActivationExtractor:
    """Forward-hook extractor for named submodules.

    >>> ex = ActivationExtractor(model, ["layer3", "fc"])
    >>> acts = ex(probe_loader)      # {layer: (n_probe, d) float32}
    """

    def __init__(self, model, layers: Sequence[str], device: Optional[str] = None):
        if torch is None:
            raise ImportError("torch is required for ActivationExtractor")
        self.model = model
        self.layers = list(layers)
        self.device = device or next(model.parameters()).device
        self._store: Dict[str, List[np.ndarray]] = {}
        self._mask = None
        mods = dict(model.named_modules())
        missing = [l for l in self.layers if l not in mods]
        if missing:
            raise KeyError(f"layers not found in model: {missing}")
        self._handles = [mods[l].register_forward_hook(self._hook(l)) for l in self.layers]

    def _hook(self, name):
        def h(_m, _i, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            self._store.setdefault(name, []).append(_pool(o.detach(), self._mask).float().cpu().numpy())
        return h

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def __call__(self, batches: Iterable) -> Dict[str, np.ndarray]:
        """``batches`` yields tensors, (x, y) tuples, or dicts of model kwargs
        (HF style: input_ids / attention_mask)."""
        self.model.eval()
        self._store = {}
        for b in batches:
            if isinstance(b, dict):
                kw = {k: v.to(self.device) for k, v in b.items()}
                self._mask = kw.get("attention_mask")
                self.model(**kw)
            else:
                x = b[0] if isinstance(b, (tuple, list)) else b
                self._mask = None
                self.model(x.to(self.device))
        return {l: np.concatenate(v, 0) for l, v in self._store.items()}


@torch.no_grad()
def hf_hidden_states(model, input_ids, attention_mask, layers: Sequence[int],
                     batch_size: int = 64, device: Optional[str] = None) -> Dict[int, np.ndarray]:
    """Mask-weighted mean of HF ``hidden_states[l]`` per probe sentence
    (the expA/phase1 convention). ``layers`` index into hidden_states, so 0 is
    the embedding output and L is the final block."""
    device = device or next(model.parameters()).device
    model.eval()
    out = {l: [] for l in layers}
    for i in range(0, len(input_ids), batch_size):
        x = input_ids[i:i + batch_size].to(device); m = attention_mask[i:i + batch_size].to(device)
        hs = model(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states
        for l in layers:
            out[l].append(_pool(hs[l], m).float().cpu().numpy())
    return {l: np.concatenate(v, 0) for l, v in out.items()}
