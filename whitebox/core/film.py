"""The topological film: one persistence diagram per (layer, epoch).

Definition 2: F^l = (D^l_k(t))_t. This module only stores/indexes; the
instruments in ``whitebox.metrics`` read it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence
import numpy as np

from .persistence import Diagrams, compute_diagrams
from .reduce import Reducer, fit_reducer


@dataclass
class Film:
    """frames[layer][t] -> Diagrams. Epoch index t is positional; ``epochs``
    records the actual epoch labels (e.g. checkpoint numbers)."""
    frames: Dict[Hashable, List[Diagrams]] = field(default_factory=dict)
    epochs: List = field(default_factory=list)
    reducers: Dict[Hashable, Reducer] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def layers(self) -> list:
        return list(self.frames.keys())

    def __len__(self) -> int:
        return len(self.epochs)

    def add_frame(self, epoch, per_layer: Dict[Hashable, Diagrams]) -> None:
        for layer, D in per_layer.items():
            self.frames.setdefault(layer, []).append(D)
        self.epochs.append(epoch)

    def diagrams(self, layer, k: int = 1) -> List[np.ndarray]:
        """Sequence of dim-k diagrams across epochs for one layer."""
        return [D[k] for D in self.frames[layer]]


def film_from_activations(
    activations_per_epoch: Sequence[Dict[Hashable, np.ndarray]],
    epochs: Optional[Sequence] = None,
    n_components: int = 30,
    maxdim: int = 1,
    thresh_percentile: float = 95.0,
    backend: str = "auto",
    reducers: Optional[Dict[Hashable, Reducer]] = None,
) -> Film:
    """Build a Film from already-extracted activations.

    ``activations_per_epoch[t][layer]`` is an (n_probe, d) array. The reducer
    for each layer is fitted on epoch 0 (t=0) and held fixed, unless
    ``reducers`` is supplied.
    """
    if not activations_per_epoch:
        raise ValueError("need at least one epoch of activations")
    layers = list(activations_per_epoch[0].keys())
    if reducers is None:
        reducers = {l: fit_reducer(activations_per_epoch[0][l], n_components) for l in layers}
    film = Film(reducers=reducers, meta=dict(n_components=n_components, maxdim=maxdim,
                                              thresh_percentile=thresh_percentile, backend=backend))
    if epochs is None:
        epochs = list(range(len(activations_per_epoch)))
    for ep, acts in zip(epochs, activations_per_epoch):
        film.add_frame(ep, {l: compute_diagrams(acts[l], reducers[l], maxdim, thresh_percentile, backend)
                            for l in layers})
    return film
