"""HF GPT-2 backend for the symmetry group (F11).

Exact renamings of ``GPT2LMHeadModel``:
* ``("head", l)``  head permutation (columns of ``c_attn`` per Q/K/V block, rows of attn ``c_proj``)
* ``("rot", l)``   within-head basis change (GL(d_head) or O(d_head)); Q R, K R^{-T}, V R, R^{-1} O
* ``("mlp", l)``   MLP hidden-unit permutation (columns of ``c_fc``, rows of mlp ``c_proj``)
* ``("ln1", l)``   (gamma, beta) of ``ln_1`` scaled by a, rows of ``c_attn`` divided by a
* ``("ln2", l)``   same for ``ln_2`` / ``c_fc``

NOT exact and therefore NOT sampled: scaling through GELU (GELU is not positively
homogeneous, so (w_in, b) -> a(w_in, b), w_out -> w_out/a changes the function for
a != 1), and ``ln_f`` rescaling (``lm_head`` is tied to ``wte``, so the compensating
column rescale would also change the input embedding). This is the "document exactly
which scalings are exact" requirement of F11. The Conv1D convention is weight (in, out).
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from .group import GaugeElement
from .transformer import _rand_rot


def handles(model) -> bool:
    return type(model).__name__ == "GPT2LMHeadModel"


def _blocks(model):
    return model.transformer.h


def describe(model):
    cfg = model.config
    L, H, d = cfg.n_layer, cfg.n_head, cfg.n_embd
    dh, m = d // H, cfg.n_inner or 4 * d
    return {"type": "gpt2", "n_layers": L, "n_heads": H, "d_head": dh, "d_model": d, "d_mlp": m,
            "keys": [(k, l) for l in range(L) for k in ("head", "rot", "mlp", "ln1", "ln2")],
            "known_dim": int(L * (H * dh * dh + 2 * d)),
            "exact_scalings": ["ln1", "ln2"], "approximate_or_absent": ["mlp (GELU not homogeneous)", "ln_f (tied lm_head)"],
            "n_params": int(sum(p.numel() for p in model.parameters()))}


def sample(model, rng, kind="combined", log_range=(0.1, 10.0), orthogonal=True, layers=None) -> GaugeElement:
    D = describe(model)
    L, H, dh, d, m = D["n_layers"], D["n_heads"], D["d_head"], D["d_model"], D["d_mlp"]
    g = GaugeElement(kind=kind)
    lo, hi = np.log(log_range[0]), np.log(log_range[1])
    for l in (range(L) if layers is None else layers):
        if kind in ("perm", "combined"):
            g.perms[("head", l)] = rng.permutation(H)
            g.perms[("mlp", l)] = rng.permutation(m)
        if kind in ("scale", "combined"):
            g.scales[("ln1", l)] = np.exp(rng.uniform(lo, hi, size=d))
            g.scales[("ln2", l)] = np.exp(rng.uniform(lo, hi, size=d))
        if kind in ("rot", "combined"):
            g.rots[("rot", l)] = [_rand_rot(rng, dh, orthogonal) for _ in range(H)]
    return g


def uniform_scaling(model, alpha, layer=None) -> GaugeElement:
    raise NotImplementedError("GPT-2 has no exact homogeneous per-unit scaling (GELU); use ln1/ln2 scalings")


@torch.no_grad()
def apply(model, g: GaugeElement):
    new = copy.deepcopy(model)
    D = describe(new); H, dh, d = D["n_heads"], D["d_head"], D["d_model"]
    for l, blk in enumerate(_blocks(new)):
        W = blk.attn.c_attn.weight; b = blk.attn.c_attn.bias           # (d, 3d), (3d,)
        Wo = blk.attn.c_proj.weight                                    # (d, d) rows = head*dh
        if ("rot", l) in g.rots:
            for h, R in enumerate(g.rots[("rot", l)]):
                R = torch.as_tensor(R, dtype=W.dtype, device=W.device); Rinv = torch.linalg.inv(R)
                sl = slice(h * dh, (h + 1) * dh)
                q, k, v = sl, slice(d + h * dh, d + (h + 1) * dh), slice(2 * d + h * dh, 2 * d + (h + 1) * dh)
                W[:, q] = W[:, q] @ R;       b[q] = b[q] @ R
                W[:, k] = W[:, k] @ Rinv.T;  b[k] = b[k] @ Rinv.T
                W[:, v] = W[:, v] @ R;       b[v] = b[v] @ R
                Wo[sl, :] = Rinv @ Wo[sl, :]
        if ("head", l) in g.perms:
            p = np.asarray(g.perms[("head", l)])
            col = np.concatenate([np.arange(h * dh, (h + 1) * dh) for h in p])
            idx = torch.as_tensor(np.concatenate([col, d + col, 2 * d + col]), device=W.device)
            W.copy_(W[:, idx]); b.copy_(b[idx])
            Wo.copy_(Wo[torch.as_tensor(col, device=W.device)])
        if ("mlp", l) in g.perms:
            p = torch.as_tensor(np.asarray(g.perms[("mlp", l)]), device=W.device)
            blk.mlp.c_fc.weight.copy_(blk.mlp.c_fc.weight[:, p]); blk.mlp.c_fc.bias.copy_(blk.mlp.c_fc.bias[p])
            blk.mlp.c_proj.weight.copy_(blk.mlp.c_proj.weight[p])
        if ("ln1", l) in g.scales:
            a = torch.as_tensor(g.scales[("ln1", l)], dtype=W.dtype, device=W.device)
            blk.ln_1.weight.mul_(a); blk.ln_1.bias.mul_(a); W.div_(a[:, None])
        if ("ln2", l) in g.scales:
            a = torch.as_tensor(g.scales[("ln2", l)], dtype=W.dtype, device=W.device)
            blk.ln_2.weight.mul_(a); blk.ln_2.bias.mul_(a); blk.mlp.c_fc.weight.div_(a[:, None])
    return new


@torch.no_grad()
def forward(model, tokens):
    tokens = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=next(model.parameters()).device)
    if tokens.ndim == 1:
        tokens = tokens[None]
    return model(tokens).logits.float().cpu().numpy()


def param_vector(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().double().numpy()
