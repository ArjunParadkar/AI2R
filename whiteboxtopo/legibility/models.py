"""Node-instrumented model protocol, and a reference two-hidden-layer ReLU MLP that follows it.

A *node* is a unit whose output can be mean-ablated and attributed. ``legibility_profile``
and ``discover_circuit`` accept any ``torch.nn.Module`` with:

    n_nodes                                   int
    forward(x, keep=None, means=None, record=False) -> logits (..., C)
        keep : bool tensor (n_nodes,) — False nodes are replaced by ``means``
        record=True stores the per-node output tensors in ``self._node_out``
    node_means(x)                             list of per-node mean outputs (batch-averaged)
    node_scores(x, targets, means, metric)    (n_nodes,) node attribution-patching scores
    edge_scores(x, targets, means, metric)    ((n_nodes, n_nodes) EAP scores u -> v, node scores)

E4's transformer models (attention heads + MLP neurons as nodes) live in
whitebox-v2/E4/e4lib/models.py and follow the same protocol; they are not
vendored here (they are tied to E4's tasks).
"""
from __future__ import annotations
import numpy as np

from .discovery import logprob_metric

try:
    import torch, torch.nn as nn, torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None; nn = object


class NodeMLP(nn.Module if torch is not None else object):
    """x (B, in_dim) float -> ReLU(l1) -> ReLU(l2) -> l3 -> logits (B, n_classes).
    Nodes: the 2 x width hidden neurons (layer-1 first)."""
    arch = "mlp"

    def __init__(self, in_dim: int, width: int = 64, n_classes: int = 2, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.in_dim, self.width, self.C = in_dim, width, n_classes
        self.l1 = nn.Linear(in_dim, width); self.l2 = nn.Linear(width, width); self.l3 = nn.Linear(width, n_classes)
        self.n_nodes = 2 * width
        self.node_layer = [0] * width + [1] * width
        self.node_names = [f"L{l}.n{i}" for l in range(2) for i in range(width)]

    def forward(self, x, keep=None, means=None, record=False):
        W = self.width
        a1 = F.relu(self.l1(x))
        if keep is not None:
            k1 = keep[:W].to(a1.device).float(); a1 = a1 * k1 + torch.cat(means[:W], 1) * (1 - k1)
        pre2 = self.l2(a1)
        if record and pre2.requires_grad: pre2.retain_grad()
        a2 = F.relu(pre2)
        if keep is not None:
            k2 = keep[W:].to(a2.device).float(); a2 = a2 * k2 + torch.cat(means[W:], 1) * (1 - k2)
        if record:
            self._node_out = [a1[:, i:i + 1] for i in range(W)] + [a2[:, i:i + 1] for i in range(W)]
            self._pre2 = pre2
        return self.l3(a2)

    @torch.no_grad()
    def node_means(self, x):
        self.forward(x, record=True)
        return [o.mean(0, keepdim=True).detach() for o in self._node_out]

    def node_scores(self, x, targets, means, metric=logprob_metric):
        self.zero_grad(set_to_none=True)
        m = metric(self.forward(x, record=True), targets); outs = self._node_out
        grads = torch.autograd.grad(m, outs, allow_unused=True)
        return torch.tensor([0.0 if g is None else ((mu - o) * g).sum().item() for o, g, mu in zip(outs, grads, means)])

    def edge_scores(self, x, targets, means, metric=logprob_metric):
        """EAP scores for the layer-1 -> layer-2 edges: E[u, v] = W2[v, u] sum_b d1_u(b) dm/dpre2_v(b)."""
        self.zero_grad(set_to_none=True)
        m = metric(self.forward(x, record=True), targets); outs = self._node_out; W = self.width
        grads = torch.autograd.grad(m, outs + [self._pre2], allow_unused=True)
        g_pre2 = grads[-1]
        d1 = torch.cat([mu - o for o, mu in zip(outs[:W], means[:W])], 1)
        E = torch.zeros(self.n_nodes, self.n_nodes)
        E[:W, W:] = (self.l2.weight.T * (d1.T @ g_pre2)).detach().cpu()
        node = torch.tensor([0.0 if g is None else ((mu - o) * g).sum().item() for o, g, mu in zip(outs, grads[:-1], means)])
        return E, node


def node_importance(model, x) -> np.ndarray:
    """Mean output norm per node on ``x`` (input of ``sparsity``)."""
    with torch.no_grad():
        model.forward(x, record=True)
        return np.array([o.detach().reshape(o.shape[0], -1).norm(dim=-1).mean().item() for o in model._node_out])
