"""System abstraction used by every estimator.

A *system* exposes:
  sites            ordered list of site names (the fixed linear extension of theory Def. 1.1)
  site_dim[site]   dimension of the state at that site
  input_names      names of the input nodes
  behaviours       names of behaviours; each has a finite readout (argmax of logits)
  input_values(X)  dict input name -> int array of input-node values
  run(X, patches)  -> (logits: dict behaviour -> (N, n_classes) array,
                       states: dict site -> (N, d) array)   [states of the patched run]

A patch is a realiser patch (theory Def. 1.3): at `site`, the coordinates of the
state in the orthonormal basis U (d x k) are replaced by those of `src`
(N x d, clean source states aligned with X rows):  h <- h + U U^T (src - h).
Every later computation is recomputed. Several patches are applied jointly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch


@dataclass
class Patch:
    site: str
    U: np.ndarray            # (d, k) orthonormal columns
    src: np.ndarray          # (N, d) source states, aligned with the batch
    match_U: Optional[np.ndarray] = None   # if set: rescale each row's delta to the norm it has under match_U


def apply_patches_torch(h: torch.Tensor, site: str, patches: Sequence[Patch], rows=None) -> torch.Tensor:
    for p in patches:
        if p.site != site:
            continue
        U = torch.as_tensor(p.U, dtype=h.dtype, device=h.device)
        src = torch.as_tensor(p.src if rows is None else p.src[rows], dtype=h.dtype, device=h.device)
        delta = ((src - h) @ U) @ U.T
        if p.match_U is not None:
            M = torch.as_tensor(p.match_U, dtype=h.dtype, device=h.device)
            target = ((src - h) @ M).norm(dim=-1, keepdim=True)
            delta = delta * (target / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        h = h + delta
    return h


# ---------------------------------------------------------------------------
# Residual MLP system: the planted networks and small trained networks.
# ---------------------------------------------------------------------------

def clamp01(x):
    return torch.clamp(x, 0.0, 1.0)


@dataclass
class Block:
    """h <- h + W3 act2(W2 act1(W1 h + b1) + b2)   ('and' block)
       h <- h + W3 relu(W1 h + b1)                  ('relu' block; W2 unused)"""
    kind: str
    W1: torch.Tensor
    b1: torch.Tensor
    W3: torch.Tensor
    W2: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None

    def __call__(self, h):
        if self.kind == "and":
            u = clamp01(h @ self.W1.T + self.b1)
            a = torch.relu(u @ self.W2.T + self.b2)
            return h + a @ self.W3.T
        u = torch.relu(h @ self.W1.T + self.b1)
        return h + u @ self.W3.T

    def params(self):
        return [t for t in (self.W1, self.b1, self.W2, self.b2, self.W3) if t is not None]


class ResidualMLPSystem:
    """Residual stream of fixed width D. Site 'h0' is the embedding, site 'h{l}'
    is the stream after block l. Readouts are linear maps of the last stream."""

    def __init__(self, emb: List[torch.Tensor], blocks: List[Block], readouts: Dict[str, tuple],
                 input_names: List[str], extra: Optional[Callable] = None, name: str = "resmlp"):
        self.emb = emb                    # list over inputs: (card_j, D)
        self.blocks = blocks
        self.readouts = readouts          # behaviour -> (W (C, D), b (C,))
        self.input_names = list(input_names)
        self.behaviours = list(readouts.keys())
        self.D = emb[0].shape[1]
        self.sites = [f"h{l}" for l in range(len(blocks) + 1)]
        self.site_dim = {s: self.D for s in self.sites}
        self.extra = extra                # optional deterministic per-input stream perturbation at h0
        self.name = name

    def input_values(self, X):
        X = np.asarray(X)
        return {n: X[:, j].astype(np.int64) for j, n in enumerate(self.input_names)}

    @torch.no_grad()
    def run(self, X, patches: Sequence[Patch] = (), batch: int = 8192):
        X = np.asarray(X)
        outs_l = {b: [] for b in self.behaviours}
        outs_s = {s: [] for s in self.sites}
        for i in range(0, len(X), batch):
            rows = slice(i, i + batch)
            lg, st = self._run(X[rows], patches, rows)
            for b in lg:
                outs_l[b].append(lg[b])
            for s in st:
                outs_s[s].append(st[s])
        return ({b: np.concatenate(v) for b, v in outs_l.items()},
                {s: np.concatenate(v) for s, v in outs_s.items()})

    def edit_inputs(self, X, edits):
        X = np.array(X, copy=True)
        for name, vals in edits.items():
            X[:, self.input_names.index(name)] = vals
        return X

    def _embed(self, X):
        Xt = torch.as_tensor(X, dtype=torch.long)
        h = sum(self.emb[j][Xt[:, j]] for j in range(len(self.emb)))
        if self.extra is not None:
            h = h + torch.as_tensor(self.extra(X), dtype=h.dtype)
        return h

    def _run(self, X, patches, rows):
        h = self._embed(X)
        states = {}
        h = apply_patches_torch(h, "h0", patches, rows)
        states["h0"] = h.numpy().copy()
        for l, blk in enumerate(self.blocks):
            h = blk(h)
            h = apply_patches_torch(h, f"h{l+1}", patches, rows)
            states[f"h{l+1}"] = h.numpy().copy()
        logits = {b: (h @ W.T + bb).numpy() for b, (W, bb) in self.readouts.items()}
        return logits, states

    # differentiable forward for trained variants and the pullback proposer
    def forward_torch(self, X, h_override: Optional[Dict[str, torch.Tensor]] = None):
        h = self._embed(np.asarray(X))
        if h_override and "h0" in h_override:
            h = h_override["h0"]
        for l, blk in enumerate(self.blocks):
            h = blk(h)
            if h_override and f"h{l+1}" in h_override:
                h = h_override[f"h{l+1}"]
        return {b: h @ W.T + bb for b, (W, bb) in self.readouts.items()}

    def downstream_from(self, site: str, h: torch.Tensor):
        """Logits as a function of the state at `site` (for Jacobians)."""
        l0 = self.sites.index(site)
        for blk in self.blocks[l0:]:
            h = blk(h)
        return {b: h @ W.T + bb for b, (W, bb) in self.readouts.items()}

    def params(self):
        ps = list(self.emb)
        for blk in self.blocks:
            ps += blk.params()
        for W, b in self.readouts.values():
            ps += [W, b]
        return ps

    def untrained_copy(self, seed: int = 0) -> "ResidualMLPSystem":
        """Same architecture and shapes, random weights (Kaiming-normal-style)."""
        g = torch.Generator().manual_seed(seed)

        def rnd(t, fan_in):
            return torch.randn(t.shape, generator=g, dtype=t.dtype) / np.sqrt(max(fan_in, 1))

        emb = [rnd(e, 1) for e in self.emb]
        blocks = []
        for blk in self.blocks:
            W1 = rnd(blk.W1, blk.W1.shape[1])
            b1 = torch.zeros_like(blk.b1)
            W3 = rnd(blk.W3, blk.W3.shape[1])
            if blk.kind == "and":
                W2 = rnd(blk.W2, blk.W2.shape[1])
                blocks.append(Block("and", W1, b1, W3, W2, torch.zeros_like(blk.b2)))
            else:
                blocks.append(Block("relu", W1, b1, W3))
        ro = {b: (rnd(W, W.shape[1]), torch.zeros_like(bb)) for b, (W, bb) in self.readouts.items()}
        return ResidualMLPSystem(emb, blocks, ro, self.input_names, None, self.name + "_untrained")


def readout(logits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {b: v.argmax(-1) for b, v in logits.items()}


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)
