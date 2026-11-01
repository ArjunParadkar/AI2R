"""The topological film: one persistence diagram per (layer, epoch), with
subsampling uncertainty.

Definition 2: F^l = (D^l_k(t))_t. ``Film`` stores/indexes; ``FilmRecorder``
builds one live during training; ``topological_film`` is the one-call entry
point for a list of checkpoints. The instruments in ``whiteboxtopo.metrics`` read
the result.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Sequence
import numpy as np

from .persistence import Diagrams, compute_diagrams
from .reduce import Reducer, fit_reducer
from .bootstrap import BootEstimate, subsample_indices, summarize
from ..metrics.tc import topological_complexity
from ..metrics.sv import structural_velocity
from ..metrics.betti import betti_at_scale, median_scale
from ..metrics.fpl import top_k_lifetime


@dataclass
class Film:
    """frames[layer][t] -> Diagrams (full probe). boots[layer][t] -> list of
    Diagrams on the subsample index sets (same sets at every epoch)."""
    frames: Dict[Hashable, List[Diagrams]] = field(default_factory=dict)
    boots: Dict[Hashable, List[List[Diagrams]]] = field(default_factory=dict)
    epochs: List = field(default_factory=list)
    reducers: Dict[Hashable, Reducer] = field(default_factory=dict)
    timings: List[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def layers(self) -> list:
        return list(self.frames.keys())

    def __len__(self) -> int:
        return len(self.epochs)

    def add_frame(self, epoch, per_layer: Dict[Hashable, Diagrams],
                  per_layer_boots: Optional[Dict[Hashable, List[Diagrams]]] = None, seconds: float = 0.0) -> None:
        for layer, D in per_layer.items():
            self.frames.setdefault(layer, []).append(D)
            if per_layer_boots is not None:
                self.boots.setdefault(layer, []).append(per_layer_boots[layer])
        self.epochs.append(epoch); self.timings.append(seconds)

    def diagrams(self, layer, k: int = 1) -> List[np.ndarray]:
        return [D[k] for D in self.frames[layer]]

    # ---- instruments with uncertainty -------------------------------------
    def _per_epoch(self, layer, k, f: Callable[[np.ndarray], float]) -> List[BootEstimate]:
        out = []
        for t, D in enumerate(self.frames[layer]):
            point = f(D[k])
            if self.boots.get(layer):
                vals = [f(B[k]) for B in self.boots[layer][t]]
                out.append(summarize(vals, point, self.boots[layer][t][0].n_points))
            else:
                out.append(summarize([point], point, D.n_points))
        return out

    def tc(self, layer, k: int = 1) -> List[BootEstimate]:
        return self._per_epoch(layer, k, topological_complexity)

    def betti(self, layer, k: int = 1, scale: Optional[float] = None) -> List[BootEstimate]:
        f = (lambda D: betti_at_scale(D, median_scale(D))) if scale is None else (lambda D: betti_at_scale(D, scale))
        return self._per_epoch(layer, k, f)

    def fpl(self, layer, k: int = 1, top: int = 3) -> List[BootEstimate]:
        return self._per_epoch(layer, k, lambda D: top_k_lifetime(D, top))

    def sv(self, layer, k: int = 1) -> List[BootEstimate]:
        """SV(t) = W2(D_t, D_{t-1}); bootstrap pairs use identical index sets."""
        out = [summarize([0.0], 0.0, self.frames[layer][0].n_points)]
        for t in range(1, len(self.frames[layer])):
            point = structural_velocity(self.frames[layer][t][k], self.frames[layer][t - 1][k])
            if self.boots.get(layer):
                vals = [structural_velocity(a[k], b[k]) for a, b in zip(self.boots[layer][t], self.boots[layer][t - 1])]
                out.append(summarize(vals, point, self.boots[layer][t][0].n_points))
            else:
                out.append(summarize([point], point, self.frames[layer][t].n_points))
        return out

    def summary(self, k: int = 1) -> dict:
        """{layer: {metric: {'point': [...], 'lo': [...], 'hi': [...]}}}"""
        out = {}
        for l in self.layers:
            out[l] = {}
            for name, fn in (("TC", self.tc), ("SV", self.sv), ("B1", self.betti), ("FPL", self.fpl)):
                es = fn(l, k)
                out[l][name] = {"point": [e.point for e in es], "lo": [e.ci_low for e in es], "hi": [e.ci_high for e in es]}
        return out


class FilmRecorder:
    """Record a Film during training.

    >>> rec = FilmRecorder(model, probe_loader, layers=["layer3", "fc"])
    >>> for epoch in range(E):
    ...     train(...); rec.record(epoch)
    >>> rec.film.tc("fc")

    The PCA reducer per layer is fitted on the FIRST recorded frame and held
    fixed; the bootstrap index sets are drawn once and reused so SV
    uncertainty is a paired comparison.
    """

    def __init__(self, model, probe, layers: Sequence, n_components: int = 30, maxdim: int = 1,
                 thresh_percentile: float = 95.0, backend: str = "auto", n_boot: int = 20,
                 subsample: Optional[int] = None, seed: int = 0, extractor: Optional[Callable] = None,
                 distance_device: str = "cpu", **ph_kw):
        self.model, self.probe, self.layers = model, probe, list(layers)
        self.ph = dict(maxdim=maxdim, thresh_percentile=thresh_percentile, backend=backend,
                       distance_device=distance_device, **ph_kw)
        self.n_components, self.n_boot, self.subsample, self.seed = n_components, n_boot, subsample, seed
        self.film = Film(meta=dict(n_components=n_components, n_boot=n_boot, subsample=subsample, **self.ph))
        self._idx = None
        if extractor is None:
            from ..activations import ActivationExtractor
            self._ex = ActivationExtractor(model, self.layers)
            self.extract = lambda: self._ex(self.probe)
        else:
            self.extract = lambda: extractor(self.model, self.probe, self.layers)

    def record(self, epoch=None) -> Film:
        t0 = time.time()
        acts = self.extract()
        epoch = len(self.film) if epoch is None else epoch
        self.record_activations(acts, epoch)
        self.film.timings[-1] = time.time() - t0
        return self.film

    def record_activations(self, acts: Dict[Hashable, np.ndarray], epoch=None) -> Film:
        t0 = time.time()
        epoch = len(self.film) if epoch is None else epoch
        if not self.film.reducers:
            self.film.reducers = {l: fit_reducer(acts[l], self.n_components) for l in self.layers}
            n = len(next(iter(acts.values())))
            m = self.subsample or max(50, int(0.5 * n))
            self._idx = subsample_indices(n, m, self.n_boot, self.seed) if self.n_boot > 0 else None
        frame, boots = {}, {}
        for l in self.layers:
            r = self.film.reducers[l]
            frame[l] = compute_diagrams(acts[l], r, **self.ph)
            if self._idx is not None:
                boots[l] = [compute_diagrams(acts[l][i], r, **self.ph) for i in self._idx]
        self.film.add_frame(epoch, frame, boots if self._idx is not None else None, time.time() - t0)
        return self.film


def topological_film(model, probe, layers: Sequence, checkpoints: Optional[Iterable] = None,
                     load_checkpoint: Optional[Callable] = None, **kw) -> Film:
    """One-call film over a sequence of checkpoints.

    ``checkpoints`` yields (epoch, state) pairs; ``load_checkpoint(model, state)``
    puts the model in that state (default: ``model.load_state_dict(state)``).
    With ``checkpoints=None`` the model's current state is recorded as a
    single frame (useful for a quick uncertainty-bearing snapshot).
    All other keyword arguments go to ``FilmRecorder``.
    """
    rec = FilmRecorder(model, probe, layers, **kw)
    if checkpoints is None:
        rec.record(0)
        return rec.film
    load = load_checkpoint or (lambda m, s: m.load_state_dict(s))
    for epoch, state in checkpoints:
        load(model, state); rec.record(epoch)
    return rec.film


def film_from_activations(activations_per_epoch: Sequence[Dict[Hashable, np.ndarray]], epochs=None,
                          n_components: int = 30, **kw) -> Film:
    """Build a Film from already-extracted activations
    (``activations_per_epoch[t][layer]`` = (n_probe, d)). Same reducer/bootstrap
    policy as ``FilmRecorder``."""
    if not activations_per_epoch:
        raise ValueError("need at least one epoch of activations")
    layers = list(activations_per_epoch[0].keys())
    rec = FilmRecorder(None, None, layers, n_components=n_components, extractor=lambda *a: None, **kw)
    epochs = list(range(len(activations_per_epoch))) if epochs is None else list(epochs)
    for ep, acts in zip(epochs, activations_per_epoch):
        rec.record_activations(acts, ep)
    return rec.film
