"""Toy attention model (F7) and its gauge group backend.

``ToyTransformer``: token + position embeddings, ``n_layers`` pre-LayerNorm blocks
(causal multi-head attention, ReLU MLP), final LayerNorm, untied unembedding.
Parameters are kept as explicit per-head tensors so that the renamings below
are plain index/matrix operations:

* ``("head", l)``  head permutation                                (perms)
* ``("rot", l)``   per-head invertible basis change R_h            (rots)
                   Q -> Q R, K -> K R^{-T}, V -> V R, O -> R^{-1} O
* ``("mlp", l)``   MLP hidden-unit permutation / ReLU scaling       (perms, scales)
* ``("ln1", l)``, ``("ln2", l)``, ``"ln_f"``  LayerNorm rescalings  (scales)
                   (gamma_i, beta_i) -> a_i (gamma_i, beta_i), row i of the next map / a_i

Every one of these is an exact symmetry of the function (float round-off only).
"""
from __future__ import annotations
import copy
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .group import GaugeElement


class ToyTransformer(nn.Module):
    def __init__(self, vocab: int, seq_len: int, d_model: int = 32, n_heads: int = 4, n_layers: int = 1,
                 d_mlp: Optional[int] = None, n_out: Optional[int] = None, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        d_mlp = d_mlp or 4 * d_model
        assert d_model % n_heads == 0
        self.vocab, self.seq_len, self.d, self.H, self.L = vocab, seq_len, d_model, n_heads, n_layers
        self.dh, self.m = d_model // n_heads, d_mlp
        self.n_out = n_out or vocab
        s = 1.0 / np.sqrt(d_model)
        self.wte = nn.Parameter(torch.randn(vocab, d_model) * s)
        self.wpe = nn.Parameter(torch.randn(seq_len, d_model) * s)
        P = lambda *shape: nn.Parameter(torch.randn(*shape) * s)
        Z = lambda *shape: nn.Parameter(torch.zeros(*shape))
        self.ln1_g = nn.ParameterList([nn.Parameter(torch.ones(d_model)) for _ in range(n_layers)])
        self.ln1_b = nn.ParameterList([Z(d_model) for _ in range(n_layers)])
        self.Wq = nn.ParameterList([P(d_model, n_heads, self.dh) for _ in range(n_layers)])
        self.Wk = nn.ParameterList([P(d_model, n_heads, self.dh) for _ in range(n_layers)])
        self.Wv = nn.ParameterList([P(d_model, n_heads, self.dh) for _ in range(n_layers)])
        self.bq = nn.ParameterList([Z(n_heads, self.dh) for _ in range(n_layers)])
        self.bk = nn.ParameterList([Z(n_heads, self.dh) for _ in range(n_layers)])
        self.bv = nn.ParameterList([Z(n_heads, self.dh) for _ in range(n_layers)])
        self.Wo = nn.ParameterList([P(n_heads, self.dh, d_model) for _ in range(n_layers)])
        self.bo = nn.ParameterList([Z(d_model) for _ in range(n_layers)])
        self.ln2_g = nn.ParameterList([nn.Parameter(torch.ones(d_model)) for _ in range(n_layers)])
        self.ln2_b = nn.ParameterList([Z(d_model) for _ in range(n_layers)])
        self.W1 = nn.ParameterList([P(d_model, d_mlp) for _ in range(n_layers)])
        self.b1 = nn.ParameterList([Z(d_mlp) for _ in range(n_layers)])
        self.W2 = nn.ParameterList([P(d_mlp, d_model) for _ in range(n_layers)])
        self.b2 = nn.ParameterList([Z(d_model) for _ in range(n_layers)])
        self.lnf_g = nn.Parameter(torch.ones(d_model))
        self.lnf_b = nn.Parameter(torch.zeros(d_model))
        self.Wu = P(d_model, self.n_out)
        self.bu = Z(self.n_out)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)))

    @staticmethod
    def _ln(x, g, b, eps=1e-5):
        return F.layer_norm(x, (x.shape[-1],), g, b, eps)

    def forward(self, tokens, return_cache: bool = False):
        """tokens (B, T) long -> logits (B, T, n_out). With ``return_cache`` also returns a dict of
        per-layer residual streams, head outputs (B, T, H, dh) and MLP hidden activations (B, T, m)."""
        B, T = tokens.shape
        x = self.wte[tokens] + self.wpe[:T][None]
        cache = {"resid": [x], "heads": [], "mlp": [], "attn": []}
        for l in range(self.L):
            h = self._ln(x, self.ln1_g[l], self.ln1_b[l])
            q = torch.einsum("btd,dhk->bthk", h, self.Wq[l]) + self.bq[l]
            k = torch.einsum("btd,dhk->bthk", h, self.Wk[l]) + self.bk[l]
            v = torch.einsum("btd,dhk->bthk", h, self.Wv[l]) + self.bv[l]
            att = torch.einsum("bthk,bshk->bhts", q, k) / np.sqrt(self.dh)
            att = att.masked_fill(~self.mask[:T, :T], float("-inf")).softmax(-1)
            z = torch.einsum("bhts,bshk->bthk", att, v)                       # (B,T,H,dh)
            head_out = torch.einsum("bthk,hkd->bthd", z, self.Wo[l])          # per-head contribution
            x = x + head_out.sum(2) + self.bo[l]
            cache["heads"].append(z); cache["attn"].append(att)
            h2 = self._ln(x, self.ln2_g[l], self.ln2_b[l])
            a = torch.relu(h2 @ self.W1[l] + self.b1[l])
            cache["mlp"].append(a)
            x = x + a @ self.W2[l] + self.b2[l]
            cache["resid"].append(x)
        out = self._ln(x, self.lnf_g, self.lnf_b) @ self.Wu + self.bu
        return (out, cache) if return_cache else out


def handles(model) -> bool:
    return isinstance(model, ToyTransformer)


def describe(model: ToyTransformer):
    L, H, dh, d, m = model.L, model.H, model.dh, model.d, model.m
    return {"type": "toy_transformer", "n_layers": L, "n_heads": H, "d_head": dh, "d_model": d, "d_mlp": m,
            "keys": [(k, l) for l in range(L) for k in ("head", "rot", "mlp", "ln1", "ln2")] + ["ln_f"],
            "known_dim": int(L * (H * dh * dh + m + 2 * d) + d),         # GL(dh) rotations + MLP scalings + LN scalings
            "known_dim_orthogonal": int(L * (H * dh * (dh - 1) // 2 + m + 2 * d) + d),
            "n_params": int(sum(p.numel() for p in model.parameters()))}


def _rand_rot(rng, k, orthogonal=True):
    A = rng.normal(size=(k, k))
    if orthogonal:
        Q, R = np.linalg.qr(A)
        return Q * np.sign(np.diag(R))[None, :]
    # well-conditioned general linear element
    U, _, Vt = np.linalg.svd(A)
    s = np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=k))
    return (U * s[None, :]) @ Vt


def sample(model: ToyTransformer, rng, kind="combined", log_range=(0.1, 10.0), orthogonal=True) -> GaugeElement:
    g = GaugeElement(kind=kind)
    lo, hi = np.log(log_range[0]), np.log(log_range[1])
    for l in range(model.L):
        if kind in ("perm", "combined"):
            g.perms[("head", l)] = rng.permutation(model.H)
            g.perms[("mlp", l)] = rng.permutation(model.m)
        if kind in ("scale", "combined"):
            g.scales[("mlp", l)] = np.exp(rng.uniform(lo, hi, size=model.m))
            g.scales[("ln1", l)] = np.exp(rng.uniform(lo, hi, size=model.d))
            g.scales[("ln2", l)] = np.exp(rng.uniform(lo, hi, size=model.d))
        if kind in ("rot", "combined"):
            g.rots[("rot", l)] = [_rand_rot(rng, model.dh, orthogonal) for _ in range(model.H)]
    if kind in ("scale", "combined"):
        g.scales["ln_f"] = np.exp(rng.uniform(lo, hi, size=model.d))
    return g


def uniform_scaling(model, alpha, layer=None) -> GaugeElement:
    g = GaugeElement(kind="scale")
    for l in range(model.L):
        if layer is None or layer == l:
            g.scales[("mlp", l)] = np.full(model.m, alpha)
    return g


def _t(x, like):
    return torch.as_tensor(np.asarray(x), dtype=like.dtype, device=like.device)


@torch.no_grad()
def apply(model: ToyTransformer, g: GaugeElement) -> ToyTransformer:
    new = copy.deepcopy(model)
    for l in range(new.L):
        # rotations first (act within heads, commute with head permutation up to relabelling)
        if ("rot", l) in g.rots:
            for h, R in enumerate(g.rots[("rot", l)]):
                R = _t(R, new.Wq[l]); Rinv = torch.linalg.inv(R)
                new.Wq[l][:, h] = new.Wq[l][:, h] @ R;        new.bq[l][h] = new.bq[l][h] @ R
                new.Wk[l][:, h] = new.Wk[l][:, h] @ Rinv.T;   new.bk[l][h] = new.bk[l][h] @ Rinv.T
                new.Wv[l][:, h] = new.Wv[l][:, h] @ R;        new.bv[l][h] = new.bv[l][h] @ R
                new.Wo[l][h] = Rinv @ new.Wo[l][h]
        if ("head", l) in g.perms:
            p = torch.as_tensor(np.asarray(g.perms[("head", l)]), device=new.Wq[l].device)
            for name in ("Wq", "Wk", "Wv"):
                getattr(new, name)[l].copy_(getattr(new, name)[l][:, p])
            for name in ("bq", "bk", "bv", "Wo"):
                getattr(new, name)[l].copy_(getattr(new, name)[l][p])
        if ("mlp", l) in g.scales:
            a = _t(g.scales[("mlp", l)], new.W1[l])
            new.W1[l].mul_(a[None, :]); new.b1[l].mul_(a); new.W2[l].div_(a[:, None])
        if ("mlp", l) in g.perms:
            p = torch.as_tensor(np.asarray(g.perms[("mlp", l)]), device=new.W1[l].device)
            new.W1[l].copy_(new.W1[l][:, p]); new.b1[l].copy_(new.b1[l][p]); new.W2[l].copy_(new.W2[l][p])
        if ("ln1", l) in g.scales:
            a = _t(g.scales[("ln1", l)], new.ln1_g[l])
            new.ln1_g[l].mul_(a); new.ln1_b[l].mul_(a)
            for name in ("Wq", "Wk", "Wv"):
                getattr(new, name)[l].div_(a[:, None, None])
        if ("ln2", l) in g.scales:
            a = _t(g.scales[("ln2", l)], new.ln2_g[l])
            new.ln2_g[l].mul_(a); new.ln2_b[l].mul_(a); new.W1[l].div_(a[:, None])
    if "ln_f" in g.scales:
        a = _t(g.scales["ln_f"], new.lnf_g)
        new.lnf_g.mul_(a); new.lnf_b.mul_(a); new.Wu.div_(a[:, None])
    return new


@torch.no_grad()
def forward(model: ToyTransformer, tokens):
    tokens = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=model.wte.device)
    return model(tokens).detach().cpu().numpy()


def param_vector(model: ToyTransformer) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().double().numpy()
