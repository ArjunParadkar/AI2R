"""Shared helpers for Block O on the T1 specimens trained by sweep_tosho.py.

* ``load_grok(name, step=None)`` -> (ToyTransformer, cfg, data)      data = dict(a, b, c, tr, te, tokens_fn)
* ``load_ring(name, step=None)`` -> (RingRNN, cfg)
* ``grok_logits(model, data, s=None)``  logits at the '=' position for every (a, b) pair (numpy)
* ``NodeView``: heads + MLP units of the 1-layer toy transformer as ablatable nodes (mean ablation), with
  ``forward(tokens, keep=None, means=None, record=False)`` — the interface ``whiteboxtopo.legibility.discovery`` expects.
* ``attribution_graph(model, tokens, targets)`` -> (E, names): gradient x (activation - mean) edge scores between
  nodes (heads -> MLP units -> logit node), the input of ``whiteboxtopo.legibility.graph`` (P0).
* ``gauge_samples(model, n, seed)``: n random gauge elements of the toy transformer (heads, rotations, MLP perms +
  scalings, LN rescalings) and ``on_orbit(model, fn, n, seed)`` -> list of fn(g·model) — the certification loop.
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from whiteboxtopo.symmetry.transformer import ToyTransformer  # noqa: E402
from whiteboxtopo.symmetry import transformer as tf_gauge  # noqa: E402
from sweep_tosho import P, EQ, SHORT0, N_SHORT, RES, CK, grok_data, RingRNN  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def checkpoints(name):
    """Sorted (step, path) of a specimen's checkpoints (excluding final.pt)."""
    fs = sorted(glob.glob(os.path.join(CK, name, "step_*.pt")))
    return [(int(os.path.basename(f)[5:-3]), f) for f in fs]


def load_grok(name, step=None, device=DEV):
    path = os.path.join(CK, name, "final.pt" if step is None else f"step_{step:06d}.pt")
    ck = torch.load(path, map_location=device); cfg = ck["cfg"]
    m = ToyTransformer(cfg["vocab"], cfg["seq_len"], d_model=128, n_heads=4, n_layers=1, d_mlp=512, n_out=P, seed=cfg["seed"]).to(device)
    m.load_state_dict(ck["state"]); m.eval()
    a, b, c, tr, te = grok_data(cfg["seed"], cfg["condition"])
    data = dict(a=a, b=b, c=c, tr=tr, te=te, planted=cfg["condition"] == "planted", onset=cfg.get("onset"),
                s_leak=SHORT0 + ((a + b) % N_SHORT))
    return m, ck, data


def load_ring(name, step=None, device=DEV):
    path = os.path.join(CK, name, "final.pt" if step is None else f"step_{step:06d}.pt")
    ck = torch.load(path, map_location=device); m = RingRNN(seed=ck["cfg"]["seed"]).to(device); m.load_state_dict(ck["state"]); m.eval()
    return m, ck


def grok_tokens(data, idx=None, shortcut="leak", device=DEV):
    """Token matrix for pairs idx (default all). shortcut: 'leak' (true residue token), 'random', or None (clean/memorise)."""
    a, b = data["a"], data["b"]
    if idx is not None: a, b = a[idx], b[idx]
    cols = [a, b]
    if data["planted"]:
        if shortcut == "leak": s = (data["s_leak"] if idx is None else data["s_leak"][idx])
        else: s = np.random.default_rng(0).integers(SHORT0, SHORT0 + N_SHORT, size=a.shape)
        cols.append(s)
    cols.append(np.full_like(a, EQ))
    return torch.as_tensor(np.stack(cols, 1), dtype=torch.long, device=device)


@torch.no_grad()
def grok_logits(model, data, idx=None, shortcut="leak", batch=4096):
    toks = grok_tokens(data, idx, shortcut); out = []
    for i in range(0, len(toks), batch): out.append(model(toks[i:i + batch])[:, -1].cpu())
    return torch.cat(out).numpy()


def accuracy(model, data, idx, shortcut="leak"):
    return float((grok_logits(model, data, idx, shortcut).argmax(1) == data["c"][idx]).mean())


# ------------------------------------------------------------------------------------------------ node-ablatable view
class NodeView(torch.nn.Module):
    """Nodes 0..H-1 = attention heads (their residual contribution), H..H+m-1 = MLP hidden units.
    forward(tokens, keep=None, means=None, record=False): keep = bool/int mask over nodes (1 = keep, 0 = mean-ablate);
    means = list of per-node mean outputs (as returned by ``node_means``). Logits at the last position."""
    def __init__(self, model: ToyTransformer):
        super().__init__(); self.m = model; self.H, self.n_mlp = model.H, model.m; self.n_nodes = self.H + self.n_mlp
        self._node_out = []

    def forward(self, tokens, keep=None, means=None, record=False):
        m = self.m; B, T = tokens.shape; l = 0
        x = m.wte[tokens] + m.wpe[:T][None]
        h = m._ln(x, m.ln1_g[l], m.ln1_b[l])
        q = torch.einsum("btd,dhk->bthk", h, m.Wq[l]) + m.bq[l]; k = torch.einsum("btd,dhk->bthk", h, m.Wk[l]) + m.bk[l]
        v = torch.einsum("btd,dhk->bthk", h, m.Wv[l]) + m.bv[l]
        att = (torch.einsum("bthk,bshk->bhts", q, k) / np.sqrt(m.dh)).masked_fill(~m.mask[:T, :T], float("-inf")).softmax(-1)
        z = torch.einsum("bhts,bshk->bthk", att, v)
        head_out = torch.einsum("bthk,hkd->bthd", z, m.Wo[l])                    # (B,T,H,d)
        outs = []
        if keep is not None or record:
            ho = []
            for hh in range(self.H):
                o = head_out[:, :, hh]
                if keep is not None and not bool(keep[hh]): o = means[hh].expand_as(o) if torch.is_tensor(means[hh]) else o * 0 + means[hh]
                if record: o = o.clone().requires_grad_(True)
                ho.append(o); outs.append(o)
            head_sum = torch.stack(ho, 2).sum(2)
        else: head_sum = head_out.sum(2)
        x = x + head_sum + m.bo[l]
        h2 = m._ln(x, m.ln2_g[l], m.ln2_b[l]); a = torch.relu(h2 @ m.W1[l] + m.b1[l])          # (B,T,m)
        if keep is not None:
            km = torch.as_tensor(np.asarray(keep[self.H:], dtype=bool), device=a.device)
            mu = torch.stack([means[self.H + j] if torch.is_tensor(means[self.H + j]) else torch.as_tensor(means[self.H + j], device=a.device) for j in range(self.n_mlp)], -1)
            a = torch.where(km, a, mu.expand_as(a) if mu.dim() < a.dim() else mu)
        if record:
            units = [a[..., j].clone().requires_grad_(True) for j in range(self.n_mlp)]; a = torch.stack(units, -1); outs += units
        x = x + a @ m.W2[l] + m.b2[l]
        self._node_out = outs
        return (m._ln(x, m.lnf_g, m.lnf_b) @ m.Wu + m.bu)[:, -1]          # logits at the "=" position

    @torch.no_grad()
    def node_means(self, tokens):
        """Per-node mean output over ``tokens`` (heads: (T,d) mean; MLP units: (T,) mean) for mean-ablation."""
        self.forward(tokens, record=True); outs = self._node_out
        return [o.detach().mean(0) for o in outs]


def attribution_graph(view: NodeView, tokens, targets, means=None, metric=None):
    """Edge matrix E (n_nodes+1 square; last row/col = the logit node) of gradient x (activation - mean) attributions:
    E[i, j] = |sum over batch of d(node_j) / d(node_i) * (node_i - mean_i)| for i a head, j an MLP unit, and
    E[j, out] = |sum grad(metric wrt node_j) * (node_j - mean_j)|. Heads never feed heads (one layer); MLP units feed
    only the output. Returned symmetrised by the caller via ``whiteboxtopo.legibility.graph.symmetrise``."""
    from whiteboxtopo.legibility.discovery import logprob_metric
    metric = metric or logprob_metric
    if means is None: means = view.node_means(tokens)
    n = view.n_nodes; E = np.zeros((n + 1, n + 1))
    logits = view.forward(tokens, record=True); outs = view._node_out
    mval = metric(logits, targets)
    g = torch.autograd.grad(mval, outs, retain_graph=True, allow_unused=True)
    for j in range(n):
        if g[j] is not None: E[j, n] = abs(((outs[j] - means[j]) * g[j]).sum().item())
    # head -> unit edges: d(unit_j)/d(head_i) via a summed unit output (one backward per unit is too costly; use the
    # per-input sum of unit activations weighted by their output gradient — the chain-rule contribution of head i to the metric through unit j)
    for j in range(view.H, n):
        if g[j] is None: continue
        s = ((outs[j] - means[j]) * g[j]).sum()
        gh = torch.autograd.grad(s, outs[:view.H], retain_graph=True, allow_unused=True)
        for i in range(view.H):
            if gh[i] is not None: E[i, j] = abs(((outs[i] - means[i]) * gh[i]).sum().item())
    names = [f"head{i}" for i in range(view.H)] + [f"mlp{j}" for j in range(view.n_mlp)] + ["logits"]
    return E, names


def p0_profile(E, density=0.1, seed=0):
    """(Q, treewidth bound, sparsity) of the attribution graph at a density, per whiteboxtopo.legibility.graph."""
    from whiteboxtopo.legibility import graph as G
    A = G.threshold_density(G.symmetrise(E), density)
    Q, _ = G.modularity_q(A, seed=seed)
    tw = G.treewidth_bound(A); imp = np.abs(E).sum(1)
    return dict(Q=float(Q), treewidth=int(tw), sparsity=float(G.sparsity(imp)), density=density)


# ------------------------------------------------------------------------------------------------ gauge orbit
def gauge_samples(model: ToyTransformer, n=300, seed=0, kind="combined"):
    rng = np.random.default_rng(seed)
    return [tf_gauge.sample(model, rng, kind=kind) for _ in range(n)]


def on_orbit(model: ToyTransformer, fn, n=300, seed=0, kind="combined"):
    """[fn(g·model) for n random gauge elements g] — fn must not mutate the model. Uses tf_gauge.apply (returns a copy)."""
    vals = []
    for g in gauge_samples(model, n, seed, kind):
        vals.append(fn(tf_gauge.apply(model, g)))
    return vals


def cv(values):
    v = np.asarray(values, dtype=float); return float(v.std(0).max() / (np.abs(v).mean() + 1e-12)) if v.ndim > 1 else float(v.std() / (abs(v.mean()) + 1e-12))
